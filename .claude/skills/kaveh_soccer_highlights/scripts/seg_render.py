"""Export one share-quality clip in resumable sub-segments, for clips too long
to render in a single uninterrupted run.

Why this exists (2026-08-23): export runs at roughly 37x realtime on this
laptop when the source .MP4s live on the Google Drive path -- 4K/10-bit HEVC
decode alone is a ~20x realtime floor (see ExportConfig's benchmark comment in
config.py). That puts anything over ~15s of footage past a 10-minute command
timeout, and long unattended jobs on this machine get killed often enough
(sleep, Drive FS stalls, the terminal closing) that a 25-minute single-shot
render is a coin flip. Two of that game's ten picks were 41.8s and 44.6s and
could not be produced by `export-picks` at all.

The fix is to split the clip into segments short enough to finish, render only
the segments that are still missing, and losslessly concat once they are all
present -- so repeated invocations make monotonic progress no matter how many
times a run is interrupted. Each invocation renders `--max-segments` segments
(default 1) and then exits, which keeps a single call comfortably inside a
timeout; just run the same command again until it prints DONE.

Correctness notes worth keeping in mind:
  * Segments are validated with render.is_playable, not by size -- a killed
    encode leaves a big file with no moov atom that no player will open.
    Truncated leftovers are deleted and re-rendered rather than trusted.
  * Encoding goes through render._encode_piece with the project's own
    ExportConfig, so segment settings are byte-for-byte what export-picks
    would have produced; the concat is stream-copied (no re-encode).
  * Chunk-spanning intervals are handled by map_interval_to_chunks exactly as
    export-picks handles them, so a clip crossing a recording boundary splits
    per chunk first and then per segment.
  * Crop and slow-mo (2026-10-18) take the same pick syntax as export-picks.
    The slow-mo replay is simply the LAST segment(s) of the same resumable
    plan -- one per source chunk the window touches, normally one -- so an
    interrupted replay resumes like any other segment. The plan (crop, slow-mo
    window/speed, crf, seg size) is recorded next to the segments; if a rerun
    asks for a different plan the stale segments are discarded rather than
    silently concatenated into a clip with the wrong crop.
  * `trim=N` (2026-10-19) shrinks the live interval to [start + t0, start + t1]
    (picks.trim_window); the trimmed interval is part of the plan, so changing N
    discards the old segments. The scoreboard is rebased onto the trimmed
    window for the live segments; the slow-mo replay stays on the original
    review-clip timeline with the un-rebased score.

Usage (run from repo root, with the project venv). Repeat the same command
until it prints DONE:

  ./.venv/Scripts/python.exe .claude/skills/kaveh_soccer_highlights/scripts/seg_render.py \
      --source-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw" \
      --review-sheet "G:/.../<date>/Review/post_sheet.csv" \
      --clip "r30_31_s2_821a_c025_c026.mp4" \
      --crf 24 \
      --out-dir "C:/local/scratch/export" \
      [--seg-seconds 11.0] [--max-segments 1]

Crop / slow-mo, same grammar as export-picks' pick lines
(`<rank> [crop=left|center|right|full] [trim=N] [slomo|slomo=A-B|slomo=A-B@S|slomo@S]`):

  # one inline pick instead of --clip (the rank resolves against the sheet):
  ... --pick "r07 slomo=6.5-9 crop=left" ...
  # or the same picks file export-picks uses; --clip (a rank or a clip_file)
  # selects which line applies:
  ... --clip r07 --picks-file picks.txt ...

With neither, the sheet's action_region still sets the crop (blank -> full
frame); `--no-crop` ignores it. The output file is named after the sheet's
clip_file either way.

Render to a LOCAL --out-dir, not straight to the Google Drive folder: writing
a large file to the Drive path is what stalled and killed the first attempts
(Drive FS warnings in the Windows event log lined up exactly with the deaths).
Copy the finished clips to the game's flat Share\\ folder afterwards.

Pick --crf per clip from its duration and measured size, not from a rule of
thumb -- see ExportConfig's comment on the ~5x motion-dependent size spread
and the surprisingly steep per-CRF-step factor on static footage. A slow-mo
replay adds output seconds (window / speed: 3s at 0.25x is 12s) to the file,
so budget its size too.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_MAX_ATTEMPTS = 3  # per segment, only for hung (timed-out) encodes

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from soccer_highlights import picks, render  # noqa: E402
from soccer_highlights.config import TrimConfig, load_config  # noqa: E402
from soccer_highlights.discovery import discover_chunks, slice_start_epoch  # noqa: E402
from soccer_highlights.timeline import Interval, map_interval_to_chunks  # noqa: E402


@dataclass(frozen=True)
class LiveSegment:
    source_path: Path
    start: float
    duration: float
    epoch_base: float
    clip_offset: float  # offset of this segment's first frame into the whole clip

    @property
    def output_seconds(self) -> float:
        return self.duration

    def describe(self) -> str:
        return f"{self.duration:.2f}s"


@dataclass(frozen=True)
class ReplaySegment:
    piece: render.ReplayPiece

    @property
    def output_seconds(self) -> float:
        return self.piece.output_seconds

    def describe(self) -> str:
        return f"slo-mo replay {self.piece.duration:.2f}s source -> {self.piece.output_seconds:.2f}s @ {self.piece.speed:g}x"


def _select_pick(
    args: argparse.Namespace, rows: list[dict[str, str]], trim_cfg: TrimConfig
) -> picks.ResolvedPick:
    """Resolve the one pick to render from --pick / --clip [+ --picks-file]."""
    if args.pick:
        text = args.pick
        wanted = None
    else:
        wanted_rows = picks.match_rows(args.clip, rows)
        if len(wanted_rows) != 1:
            raise SystemExit(
                f"--clip {args.clip!r}: "
                + ("no matching clip_file" if not wanted_rows else "ambiguous, matches " + ", ".join(r["clip_file"] for r in wanted_rows))
                + f" in {args.review_sheet}"
            )
        wanted = wanted_rows[0]["clip_file"]
        text = Path(args.picks_file).read_text(encoding="utf-8-sig") if args.picks_file else args.clip
    try:
        resolved = picks.load_picks(text, rows, use_sheet_crop=not args.no_crop, trim_cfg=trim_cfg)
    except picks.PickError as exc:
        raise SystemExit("Bad pick(s), nothing rendered:\n  " + "\n  ".join(exc.errors))
    if wanted is None:
        if len(resolved) != 1:
            raise SystemExit(f"--pick takes exactly one pick line, got {len(resolved)}")
        return resolved[0]
    for pick in resolved:
        if pick.clip_file == wanted:
            return pick
    raise SystemExit(f"{wanted} has no line in {args.picks_file}")


def _write_plan(seg_dir: Path, plan: dict[str, object]) -> None:
    """Record the plan next to the segments; if an earlier run used a different
    one, discard its segments (they would be concatenated with the wrong crop /
    replay / quality otherwise). Whole-file write, so a kill mid-write leaves
    either the old plan or the new one -- never half of one."""
    plan_path = seg_dir / "plan.json"
    try:
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = None
    if previous == plan:
        return
    for stale in seg_dir.glob("seg*.mp4"):
        stale.unlink()
    tmp_path = seg_dir / "plan.json.tmp"
    tmp_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    tmp_path.replace(plan_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True, help="The game's Raw/ folder (DJI_*.MP4 + .LRF)")
    parser.add_argument(
        "--review-sheet", required=True, help="CSV with clip_file/start_seconds/end_seconds (review_sheet or post_sheet)"
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--clip", help="clip_file value (or rank, e.g. r07) to render")
    which.add_argument("--pick", help="One pick line, e.g. 'r07 slomo=6.5-9 crop=left' (instead of --clip)")
    parser.add_argument("--picks-file", help="export-picks' picks file; with --clip, the line for that clip supplies crop/slomo")
    parser.add_argument("--no-crop", action="store_true", help="Ignore the sheet's action_region (full frame unless the pick says crop=)")
    parser.add_argument("--crf", type=int, required=True, help="x264 CRF for this clip (pick per clip, see module docstring)")
    parser.add_argument("--out-dir", required=True, help="Local output dir -- do NOT point this at Google Drive")
    parser.add_argument(
        "--seg-seconds", type=float, default=11.0, help="Max footage seconds per segment (default 11.0, ~7min at 37x)"
    )
    parser.add_argument("--max-segments", type=int, default=1, help="Segments to render this invocation (default 1)")
    args = parser.parse_args()
    if args.picks_file and not args.clip:
        parser.error("--picks-file needs --clip (which line to use)")

    with open(args.review_sheet, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    cfg = load_config()
    pick = _select_pick(args, rows, cfg.trim)
    row = pick.row
    clip_file = pick.clip_file

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / clip_file
    if render.is_playable(final_path):
        print(f"DONE {clip_file} already rendered ({final_path.stat().st_size:,} bytes)")
        return

    cfg.export.crf = args.crf
    chunks = discover_chunks(args.source_dir)
    # `full` is the row's whole interval (the replay is mapped from it); `interval`
    # is what the live segments cover -- the trimmed window for a `trim=N` pick.
    full = Interval(start_seconds=float(row["start_seconds"]), end_seconds=float(row["end_seconds"]))
    live_start, live_end = pick.export_seconds()
    interval = Interval(start_seconds=live_start, end_seconds=live_end)
    slices = map_interval_to_chunks(interval, chunks)
    if pick.trim is not None:
        print(f"trim: {pick.describe()} -> {interval.end_seconds - interval.start_seconds:.1f}s of {full.end_seconds - full.start_seconds:.1f}s", flush=True)
        if pick.trim.warning:
            print(f"!! WARNING {clip_file}: {pick.trim.warning}", flush=True)
    # Same sheet columns export-picks reads, so a clip rendered here gets the
    # identical score overlay it would have got from a single-shot export. A
    # trimmed pick's live segments get it rebased onto the trimmed window
    # (`score`); the replay keeps the original-timeline overlay (`full_score`).
    full_score = render.ScoreOverlay.from_sheet_row(row)
    score = full_score.rebased(pick.trim.start_seconds, pick.trim.end_seconds) if full_score and pick.trim else full_score
    crop = picks.rect_for_region(pick.region, cfg.crop)

    # Flatten the per-chunk slices into sub-segments short enough to finish
    # inside one invocation. Chunk boundaries are honoured first so a segment
    # never straddles two source files.
    # Each segment carries its own wall-clock epoch (its owning chunk's
    # filename timestamp + its offset into that chunk), so the burned-in
    # time-of-day overlay stays continuous across a clip that was cut into
    # segments -- without this, every segment would restart the clock at the
    # clip's own start time, and long clips (the only ones that come through
    # here) would read wrong.
    # `clip_pos` is the running offset into the WHOLE clip (across chunk
    # boundaries and segment splits alike), which is what the score flip is
    # measured against -- every -ss-trimmed segment restarts its own t at 0,
    # so without this the score would change in the wrong segment. Goal
    # clips are stretched to cover their tap and routinely land over the
    # ~15s single-run limit, so this script is the NORMAL path for them.
    segments: list[LiveSegment | ReplaySegment] = []
    clip_pos = 0.0
    for cs in slices:
        remaining = cs.local_end_seconds - cs.local_start_seconds
        pos = cs.local_start_seconds
        while remaining > 1e-6:
            take = min(args.seg_seconds, remaining)
            segments.append(LiveSegment(cs.chunk.mp4_path, pos, take, slice_start_epoch(cs.chunk, pos), clip_pos))
            pos += take
            clip_pos += take
            remaining -= take
    # The replay goes last, after the whole live clip (2026-10-18). Its pieces
    # carry the clip-relative offset of the window (not the end of the clip),
    # because that is what the score flip and the true-time clock key on.
    if pick.slomo is not None:
        segments.extend(ReplaySegment(p) for p in render.replay_pieces(render.build_replay(full, pick.slomo, chunks)))

    seg_dir = out_dir / f".segs_{final_path.stem}"
    seg_dir.mkdir(exist_ok=True)
    seg_paths = [seg_dir / f"seg{i:02d}.mp4" for i in range(len(segments))]
    slomo = pick.slomo
    plan: dict[str, object] = {
        "crop": pick.region,
        "slomo": [slomo.start_seconds, slomo.end_seconds, slomo.speed] if slomo else None,
        "crf": args.crf,
        "seg_seconds": args.seg_seconds,
    }
    if pick.trim is not None:
        # Only for trimmed picks, so segments of an in-progress UNtrimmed render
        # keep matching their old plan.json instead of being discarded.
        plan["interval"] = [round(interval.start_seconds, 3), round(interval.end_seconds, 3)]
    _write_plan(seg_dir, plan)

    rendered = 0
    for i, (seg, seg_path) in enumerate(zip(segments, seg_paths)):
        if render.is_playable(seg_path):
            continue
        if rendered >= args.max_segments:
            break
        seg_path.unlink(missing_ok=True)  # drop a truncated leftover from a killed run
        print(
            f"Rendering seg {i + 1}/{len(segments)} of {clip_file}: {seg.describe()} @ crf{args.crf} crop={pick.region}",
            flush=True,
        )
        # A hung ffmpeg is killed by render.encode_timeout_seconds and retried
        # here (2026-10-13: one segment hung ~41h unattended; the identical
        # re-render took 131s).
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                if isinstance(seg, ReplaySegment):
                    render.encode_replay_piece(seg.piece, seg_path, cfg.export, full_score, crop)
                else:
                    render._encode_piece(
                        seg.source_path, seg.start, seg.duration, seg_path, cfg.export,
                        seg.epoch_base, score, seg.clip_offset, crop,
                    )
                break
            except subprocess.TimeoutExpired:
                seg_path.unlink(missing_ok=True)
                print(f"seg {i + 1} timed out (attempt {attempt}/{_MAX_ATTEMPTS})", flush=True)
                if attempt == _MAX_ATTEMPTS:
                    raise SystemExit(f"segment {i} hung {_MAX_ATTEMPTS} times -- investigate the source window")
        if not render.is_playable(seg_path):
            raise SystemExit(f"segment {i} came out unplayable -- rerun to retry it")
        rendered += 1

    done = sum(1 for p in seg_paths if render.is_playable(p))
    print(f"{done}/{len(seg_paths)} segments ready for {clip_file}")
    if done != len(seg_paths):
        print("Run the same command again to continue.")
        return

    render.concat_clips(seg_paths, final_path, force_reencode=False)
    if not render.is_playable(final_path):
        raise SystemExit("concat produced an unplayable file")
    for p in seg_paths:
        p.unlink(missing_ok=True)
    (seg_dir / "plan.json").unlink(missing_ok=True)
    seg_dir.rmdir()
    size = final_path.stat().st_size
    print(f"DONE {clip_file}: {size:,} bytes ({size / 1e6:.1f} MB)")
    if size >= 50_000_000:
        print("WARNING: over Telegram's 50MB limit -- delete it and rerun with a higher --crf")


if __name__ == "__main__":
    main()
