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

Usage (run from repo root, with the project venv). Repeat the same command
until it prints DONE:

  ./.venv/Scripts/python.exe .claude/skills/kaveh_soccer_highlights/scripts/seg_render.py \
      --source-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw" \
      --review-sheet "G:/.../Tests/pre_label/post_sheet.csv" \
      --clip "r30_31_s2_821a_c025_c026.mp4" \
      --crf 24 \
      --out-dir "C:/local/scratch/export" \
      [--seg-seconds 11.0] [--max-segments 1]

Render to a LOCAL --out-dir, not straight to the Google Drive folder: writing
a large file to the Drive path is what stalled and killed the first attempts
(Drive FS warnings in the Windows event log lined up exactly with the deaths).
Copy the finished clips to the Sharable folder afterwards.

Pick --crf per clip from its duration and measured size, not from a rule of
thumb -- see ExportConfig's comment on the ~5x motion-dependent size spread
and the surprisingly steep per-CRF-step factor on static footage.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from soccer_highlights import render  # noqa: E402
from soccer_highlights.config import load_config  # noqa: E402
from soccer_highlights.discovery import discover_chunks, slice_start_epoch  # noqa: E402
from soccer_highlights.timeline import Interval, map_interval_to_chunks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True, help="The game's Raw/ folder (DJI_*.MP4 + .LRF)")
    parser.add_argument(
        "--review-sheet", required=True, help="CSV with clip_file/start_seconds/end_seconds (review_sheet or post_sheet)"
    )
    parser.add_argument("--clip", required=True, help="clip_file value to render")
    parser.add_argument("--crf", type=int, required=True, help="x264 CRF for this clip (pick per clip, see module docstring)")
    parser.add_argument("--out-dir", required=True, help="Local output dir -- do NOT point this at Google Drive")
    parser.add_argument(
        "--seg-seconds", type=float, default=11.0, help="Max footage seconds per segment (default 11.0, ~7min at 37x)"
    )
    parser.add_argument("--max-segments", type=int, default=1, help="Segments to render this invocation (default 1)")
    args = parser.parse_args()

    with open(args.review_sheet, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["clip_file"] == args.clip]
    if not rows:
        raise SystemExit(f"clip_file {args.clip!r} not found in {args.review_sheet}")
    row = rows[0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / args.clip
    if render.is_playable(final_path):
        print(f"DONE {args.clip} already rendered ({final_path.stat().st_size:,} bytes)")
        return

    cfg = load_config()
    cfg.export.crf = args.crf
    chunks = discover_chunks(args.source_dir)
    interval = Interval(start_seconds=float(row["start_seconds"]), end_seconds=float(row["end_seconds"]))
    slices = map_interval_to_chunks(interval, chunks)

    # Flatten the per-chunk slices into sub-segments short enough to finish
    # inside one invocation. Chunk boundaries are honoured first so a segment
    # never straddles two source files.
    # Each segment carries its own wall-clock epoch (its owning chunk's
    # filename timestamp + its offset into that chunk), so the burned-in
    # time-of-day overlay stays continuous across a clip that was cut into
    # segments -- without this, every segment would restart the clock at the
    # clip's own start time, and long clips (the only ones that come through
    # here) would read wrong.
    segments: list[tuple[Path, float, float, float]] = []
    for cs in slices:
        remaining = cs.local_end_seconds - cs.local_start_seconds
        pos = cs.local_start_seconds
        while remaining > 1e-6:
            take = min(args.seg_seconds, remaining)
            segments.append((cs.chunk.mp4_path, pos, take, slice_start_epoch(cs.chunk, pos)))
            pos += take
            remaining -= take

    seg_dir = out_dir / f".segs_{final_path.stem}"
    seg_dir.mkdir(exist_ok=True)
    seg_paths = [seg_dir / f"seg{i:02d}.mp4" for i in range(len(segments))]

    rendered = 0
    for i, ((src, start, dur, epoch_base), seg_path) in enumerate(zip(segments, seg_paths)):
        if render.is_playable(seg_path):
            continue
        if rendered >= args.max_segments:
            break
        seg_path.unlink(missing_ok=True)  # drop a truncated leftover from a killed run
        print(f"Rendering seg {i + 1}/{len(segments)} of {args.clip}: {dur:.2f}s @ crf{args.crf}", flush=True)
        render._encode_piece(src, start, dur, seg_path, cfg.export, epoch_base)
        if not render.is_playable(seg_path):
            raise SystemExit(f"segment {i} came out unplayable -- rerun to retry it")
        rendered += 1

    done = sum(1 for p in seg_paths if render.is_playable(p))
    print(f"{done}/{len(seg_paths)} segments ready for {args.clip}")
    if done != len(seg_paths):
        print("Run the same command again to continue.")
        return

    render.concat_clips(seg_paths, final_path, force_reencode=False)
    if not render.is_playable(final_path):
        raise SystemExit("concat produced an unplayable file")
    for p in seg_paths:
        p.unlink(missing_ok=True)
    seg_dir.rmdir()
    size = final_path.stat().st_size
    print(f"DONE {args.clip}: {size:,} bytes ({size / 1e6:.1f} MB)")
    if size >= 50_000_000:
        print("WARNING: over Telegram's 50MB limit -- delete it and rerun with a higher --crf")


if __name__ == "__main__":
    main()
