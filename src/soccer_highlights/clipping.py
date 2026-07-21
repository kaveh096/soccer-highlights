"""Slice highlight clips out of the full-resolution source chunks via ffmpeg
stream copy (no re-encode), reading directly from wherever the source files
live (e.g. the camera's memory card) and writing only the small clip output."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from soccer_highlights.timeline import ChunkSlice


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def slice_single(chunk_slice: ChunkSlice, out_path: Path) -> None:
    duration = chunk_slice.local_end_seconds - chunk_slice.local_start_seconds
    _run_ffmpeg(
        [
            "-ss", f"{chunk_slice.local_start_seconds:.3f}",
            "-i", str(chunk_slice.chunk.mp4_path),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out_path),
        ]
    )


def concat_clips(clip_paths: list[Path], out_path: Path, force_reencode: bool = False) -> None:
    if len(clip_paths) == 1:
        clip_paths[0].replace(out_path)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as list_file:
        for clip_path in clip_paths:
            escaped = str(clip_path.resolve()).replace("'", r"'\''")
            list_file.write(f"file '{escaped}'\n")
        list_path = Path(list_file.name)
    try:
        args = ["-f", "concat", "-safe", "0", "-i", str(list_path)]
        args += ["-c", "aac"] if force_reencode else ["-c", "copy"]
        _run_ffmpeg([*args, str(out_path)])
    finally:
        list_path.unlink(missing_ok=True)


def build_highlight_clip(slices: list[ChunkSlice], out_path: Path, tmp_dir: Path) -> None:
    """Slice each per-chunk piece of an interval and join them (if the
    interval spans a chunk boundary) into a single output clip at out_path."""
    if len(slices) == 1:
        slice_single(slices[0], out_path)
        return

    part_paths: list[Path] = []
    for i, chunk_slice in enumerate(slices):
        part_path = tmp_dir / f"{out_path.stem}_part{i}.mp4"
        slice_single(chunk_slice, part_path)
        part_paths.append(part_path)

    concat_clips(part_paths, out_path, force_reencode=False)
    for part_path in part_paths:
        part_path.unlink(missing_ok=True)
