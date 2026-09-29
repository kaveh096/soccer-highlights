"""Render a whole-game timelapse: every recorded chunk sped up into one short
clip, with the real scoreboard/clock overlay burned in before compression --
plus a real-speed audio snippet laid under the result afterward.

Why this exists (2026-09-29, Kaveh's request after liking a 25-min single-chunk
prototype): a literal speed-up of the full ~82 minutes of game footage from the
4K masters would cost ~20x-realtime decode alone -- over a day on this laptop's
2-core/no-HW-decode CPU. The `.LRF` 720p proxy runs close to 1x realtime for
this same setpts-based pipeline (measured on the chunk-1 prototype: 1493s of
source in ~1500s wall time), so this script always sources from `.LRF`, never
the 4K `.MP4` -- the extreme speedup and final <=50MB Telegram-size compression
erase any 4K detail anyway, so there is no visible quality argument for paying
that cost.

Design: draw the scoreboard/clock against REAL time first (reusing
render.py's existing, already-correctness-tested drawtext building blocks
almost verbatim), THEN compress time with `setpts` as the LAST step in the
filter graph. This means the score-flip and clock logic never has to know
about the speedup at all -- it draws at real speed exactly like a normal
export, and setpts squeezes the already-correct frames down afterward. The
one genuine extension needed is a multi-goal generalization of
`render._score_filters` (that function assumes at most one flip per
piece, true for a single candidate clip but false for a whole ~25-minute
chunk, which typically contains several real goals) -- `_score_segments`
below does that, walking `marks.score_events()`'s sorted (time, team) list
into contiguous (start, end, white, black) windows.

Two-phase render, deliberately: phase 1 (per-chunk decode+overlay+speed
compression) is the expensive, ~80-90-minute part and is resumable file-by-
file (`is_playable` skip-if-exists, same idiom as seg_render.py) because
that cost must never be paid twice. Phase 2 (concat the four cheap-to-redo
~40s-total pieces, then mux in a real-speed audio snippet and pick a final
CRF to land under Telegram's 50MB cap) operates on a full-length-but-already-
tiny master, so it's fast to iterate on and safe to re-run repeatedly while
tuning size.

Usage (from repo root, with the project venv):

  ./.venv/Scripts/python.exe .claude/skills/kaveh_soccer_highlights/scripts/timelapse_render.py \\
      --source-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw" \\
      --tally-csv "white_goal=G:/.../<date>/Raw/tally_white_goal.csv" \\
      --tally-csv "black_goal=G:/.../<date>/Raw/tally_black_goal.csv" \\
      --clock-offset-seconds <from ingest-marks, see processing.md Step 0c> \\
      --out-dir "C:/local/scratch/timelapse_full" \\
      --out-duration-seconds 40

Rerun the identical command to resume after an interruption -- already-
rendered chunk pieces are skipped. Once `timelapse_master.mp4` exists,
rerun with `--audio-start-seconds`/`--crf` to retune the final mux without
redoing any of the expensive per-chunk work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from soccer_highlights import marks, render, scoreboard  # noqa: E402
from soccer_highlights.config import load_config  # noqa: E402
from soccer_highlights.discovery import discover_chunks, slice_start_epoch  # noqa: E402

FRAME_WIDTH = 1280  # matches the .LRF proxy's native resolution -- no upscale needed
OUT_FPS = 30


def _score_segments(
    events: list[tuple[float, str]], chunk_start: float, chunk_duration: float
) -> list[tuple[float, float, int, int]]:
    """Contiguous (local_start, local_end, white, black) windows covering
    [0, chunk_duration) of one chunk, generalizing marks.score_for_interval
    (which only ever handles 0-1 events) to the several real goals a whole
    ~25-minute chunk typically contains."""
    white = sum(1 for t, team in events if t < chunk_start and team == "white")
    black = sum(1 for t, team in events if t < chunk_start and team == "black")
    local_events = sorted(
        (t - chunk_start, team) for t, team in events if chunk_start <= t < chunk_start + chunk_duration
    )
    segments: list[tuple[float, float, int, int]] = []
    prev_t = 0.0
    for t, team in local_events:
        segments.append((prev_t, t, white, black))
        white += 1 if team == "white" else 0
        black += 1 if team == "black" else 0
        prev_t = t
    segments.append((prev_t, chunk_duration, white, black))
    return segments


def _load_score_events(tally_csvs: list[str], chunks, clock_offset: float, cfg) -> list[tuple[float, str]]:
    per_category = []
    for spec in tally_csvs:
        category, _, path = spec.partition("=")
        loaded, n_decrements = marks.load_tallies_csv(path.strip(), category.strip())
        per_category.extend(loaded)
        if n_decrements:
            print(f"WARNING: {category} has {n_decrements} unresolved decrement(s) -- see processing.md's gate")
    active = marks.merge_tally_marks(per_category)
    resolved = marks.resolve_marks(active, chunks, [], cfg.marks, cfg.timeline, clock_offset)
    return marks.score_events(resolved)


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def render_chunk_piece(
    chunk, events: list[tuple[float, str]], speed_factor: float, cfg, out_path: Path
) -> None:
    lrf_path = chunk.lrf_path or chunk.mp4_path
    chunk_duration = chunk.duration_seconds
    segments = _score_segments(events, chunk.global_start_seconds, chunk_duration)
    chrome_path, layout = scoreboard.chrome_png(cfg.export, FRAME_WIDTH)

    stages = [
        render._drawtext_score_stage(
            cfg.export, f"{white} - {black}", layout, enable=f"between(t,{start:.3f},{end:.3f})"
        )
        for start, end, white, black in segments
    ]
    epoch_base = slice_start_epoch(chunk, 0.0)
    stages.append(render._drawtext_clock_stage(cfg.export, epoch_base, layout))

    out_duration = chunk_duration / speed_factor
    filter_complex = (
        f"[0:v][1:v]overlay=x={layout.inset}:y={layout.inset}[bg];"
        f"[bg]{','.join(stages)},setpts=PTS/{speed_factor:.6f},fps={OUT_FPS}[vout]"
    )
    _run_ffmpeg(
        [
            "-i", str(lrf_path),
            "-loop", "1",
            "-i", str(chrome_path),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-an",
            "-t", f"{out_duration:.6f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            str(out_path),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--tally-csv", action="append", required=True, metavar="CATEGORY=PATH")
    parser.add_argument("--clock-offset-seconds", type=float, default=0.0)
    parser.add_argument("--out-dir", required=True, help="Local output dir -- do NOT point this at Google Drive")
    parser.add_argument("--out-duration-seconds", type=float, default=40.0)
    parser.add_argument("--audio-start-seconds", type=float, default=None, help="Global game-second to start the real-speed audio snippet from (default: auto, 40s window around a near-cam goal)")
    parser.add_argument("--final-crf", type=int, default=23, help="CRF for the final size-fitting mux pass (re-run cheaply to retune)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    chunks = discover_chunks(args.source_dir)
    total_duration = sum(c.duration_seconds for c in chunks)
    speed_factor = total_duration / args.out_duration_seconds
    print(f"{len(chunks)} chunk(s), {total_duration:.1f}s total footage -> speed factor {speed_factor:.3f}x")

    events = _load_score_events(args.tally_csv, chunks, args.clock_offset_seconds, cfg)
    print(f"{len(events)} real goal event(s) across the whole game: {events}")

    piece_paths = []
    for chunk in chunks:
        piece_path = out_dir / f"piece_{chunk.sequence:03d}.mp4"
        piece_paths.append(piece_path)
        if render.is_playable(piece_path):
            print(f"SKIP chunk {chunk.sequence} (already rendered)")
            continue
        print(f"Rendering chunk {chunk.sequence} ({chunk.duration_seconds:.1f}s -> {chunk.duration_seconds / speed_factor:.2f}s)...")
        piece_path.unlink(missing_ok=True)
        render_chunk_piece(chunk, events, speed_factor, cfg, piece_path)
        if not render.is_playable(piece_path):
            raise SystemExit(f"chunk {chunk.sequence} came out unplayable -- rerun to retry it")

    master_path = out_dir / "timelapse_master.mp4"
    if not render.is_playable(master_path):
        from soccer_highlights.clipping import concat_clips

        concat_clips(list(piece_paths), master_path, force_reencode=False)
        if not render.is_playable(master_path):
            raise SystemExit("concat produced an unplayable master -- delete timelapse_master.mp4 and rerun")
    master_duration_out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(master_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"DONE phase 1: {master_path} ({master_duration_out}s, should be ~{args.out_duration_seconds}s)")

    # Phase 2: real-speed audio snippet + final size-fitting mux.
    audio_start = args.audio_start_seconds
    if audio_start is None:
        # Default: a 40s (or out_duration, whichever fits) window starting a
        # bit before the highest-scored near-cam goal event, for a lively bed.
        goal_time = events[0][0] if events else 0.0
        audio_start = max(0.0, goal_time - 20.0)
    audio_len = min(args.out_duration_seconds, total_duration - audio_start)

    owning_chunk = next(c for c in chunks if c.global_start_seconds <= audio_start < c.global_start_seconds + c.duration_seconds)
    local_start = audio_start - owning_chunk.global_start_seconds
    audio_path = out_dir / "audio_snippet.m4a"
    print(f"Extracting {audio_len:.1f}s real-speed audio from chunk {owning_chunk.sequence} @ {local_start:.1f}s...")
    _run_ffmpeg(
        [
            "-ss", f"{local_start:.3f}", "-t", f"{audio_len:.3f}",
            "-i", str(owning_chunk.mp4_path),
            "-vn", "-c:a", "aac", "-b:a", "160k",
            str(audio_path),
        ]
    )

    final_path = out_dir / "timelapse_final.mp4"
    print(f"Muxing final clip at CRF {args.final_crf}...")
    _run_ffmpeg(
        [
            "-i", str(master_path), "-i", str(audio_path),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(args.final_crf),
            "-c:a", "copy", "-shortest",
            str(final_path),
        ]
    )
    if not render.is_playable(final_path):
        raise SystemExit("final mux produced an unplayable file")
    size = final_path.stat().st_size
    print(f"DONE: {final_path} ({size:,} bytes, {size / 1e6:.1f} MB)")
    if size >= 50_000_000:
        print(f"WARNING: over Telegram's 50MB limit -- rerun with --final-crf {args.final_crf + 4} (cheap, only re-muxes the final ~{args.out_duration_seconds:.0f}s)")


if __name__ == "__main__":
    main()
