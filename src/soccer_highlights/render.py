"""Render small, resized/downsampled review clips -- plus a whole-game skim
and negative-space (uncovered) clips -- from the .LRF proxies only. Meant
for fast human true/false-positive review on modest hardware, not final
output quality."""

from __future__ import annotations

import subprocess
from pathlib import Path

from soccer_highlights.clipping import concat_clips
from soccer_highlights.config import ExportConfig, ReviewConfig
from soccer_highlights.discovery import Chunk
from soccer_highlights.timeline import ChunkSlice


def _scale_filter(max_width: int) -> str:
    # -2 keeps height a multiple of 2 (required by libx264) while preserving
    # the source's aspect ratio, instead of forcing a fixed WxH that would
    # distort a 16:9 source.
    return f"scale={max_width}:-2"


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def is_playable(path: Path) -> bool:
    """True only if ffprobe can read a duration out of `path`.

    Existence plus a non-zero size is NOT enough to call a render finished: a
    process killed mid-encode leaves a large file with no `moov` atom, which
    ffprobe rejects as "moov atom not found" and no player will open. This bit
    hard on 2026-08-23 -- an export killed partway left a 9.5MB corpse that
    cmd_export_picks' resume check happily treated as done, and it would have
    been posted to Telegram if it hadn't been spot-checked. Any resume/skip
    decision about an existing media file should go through this, not through
    `stat().st_size > 0`."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() not in ("", "N/A")


def _encode_piece(source_path: Path, start: float, duration: float, out_path: Path, cfg: ReviewConfig | ExportConfig) -> None:
    audio_args = ["-ac", "1"] if cfg.mono_audio else []
    _run_ffmpeg(
        [
            "-ss", f"{start:.3f}",
            "-i", str(source_path),
            "-t", f"{duration:.3f}",
            "-vf", f"{_scale_filter(cfg.max_width)},fps={cfg.fps}",
            "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf), "-threads", str(cfg.threads),
            "-c:a", "aac", "-strict", "-2", "-b:a", f"{cfg.audio_bitrate_kbps}k", *audio_args,
            str(out_path),
        ]
    )


def render_review_clip(slices: list[ChunkSlice], out_path: Path, tmp_dir: Path, cfg: ReviewConfig) -> None:
    """Render a (possibly chunk-boundary-spanning) interval as one small
    review clip, sourced from each chunk's .LRF proxy."""
    if len(slices) == 1:
        cs = slices[0]
        source_path = cs.chunk.lrf_path or cs.chunk.mp4_path
        _encode_piece(source_path, cs.local_start_seconds, cs.local_end_seconds - cs.local_start_seconds, out_path, cfg)
        return

    part_paths: list[Path] = []
    for i, cs in enumerate(slices):
        source_path = cs.chunk.lrf_path or cs.chunk.mp4_path
        part_path = tmp_dir / f"{out_path.stem}_part{i}.mp4"
        _encode_piece(source_path, cs.local_start_seconds, cs.local_end_seconds - cs.local_start_seconds, part_path, cfg)
        part_paths.append(part_path)

    concat_clips(part_paths, out_path, force_reencode=False)
    for part_path in part_paths:
        part_path.unlink(missing_ok=True)


def render_export_clip(slices: list[ChunkSlice], out_path: Path, tmp_dir: Path, cfg: ExportConfig) -> None:
    """Render a (possibly chunk-boundary-spanning) interval as one
    full-resolution, re-encoded delivery clip, always sourced from the
    full-res .MP4 (never the .LRF proxy) -- unlike render_review_clip,
    this is meant for sharing, not just fast true/false-positive review."""
    if len(slices) == 1:
        cs = slices[0]
        _encode_piece(cs.chunk.mp4_path, cs.local_start_seconds, cs.local_end_seconds - cs.local_start_seconds, out_path, cfg)
        return

    part_paths: list[Path] = []
    for i, cs in enumerate(slices):
        part_path = tmp_dir / f"{out_path.stem}_part{i}.mp4"
        _encode_piece(cs.chunk.mp4_path, cs.local_start_seconds, cs.local_end_seconds - cs.local_start_seconds, part_path, cfg)
        part_paths.append(part_path)

    concat_clips(part_paths, out_path, force_reencode=False)
    for part_path in part_paths:
        part_path.unlink(missing_ok=True)


def render_whole_game_skim(chunks: list[Chunk], out_path: Path, cfg: ReviewConfig) -> None:
    """Concatenate every chunk's .LRF proxy and resize/downsample the whole
    thing into one small file for a quick full skim."""
    inputs: list[str] = []
    filter_inputs: list[str] = []
    for i, chunk in enumerate(chunks):
        source_path = chunk.lrf_path or chunk.mp4_path
        inputs += ["-i", str(source_path)]
        filter_inputs.append(f"[{i}:v:0][{i}:a:0]")
    concat_filter = "".join(filter_inputs) + f"concat=n={len(chunks)}:v=1:a=1[v][a]"
    filter_complex = f"{concat_filter};[v]{_scale_filter(cfg.max_width)},fps={cfg.fps}[vout]"

    _run_ffmpeg(
        [
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[a]",
            "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf), "-threads", str(cfg.threads),
            "-c:a", "aac", "-strict", "-2", "-b:a", f"{cfg.audio_bitrate_kbps}k", "-ac", "1",
            str(out_path),
        ]
    )
