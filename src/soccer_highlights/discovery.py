"""Discover and order DJI recording chunks within a session directory.

DJI Action cameras split long recordings into sequential files named like
``DJI_20260719074839_0001_D.MP4``, each with a matching ``.LRF`` low-res
proxy of identical duration/audio. This module finds those pairs, orders
them by sequence number, and establishes each chunk's offset within the
continuous recording timeline.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

_NAME_RE = re.compile(r"^DJI_(?P<timestamp>\d{14})_(?P<seq>\d+)_D$")

# Recording chunks are back-to-back; if the gap between a chunk's filename
# timestamp and the previous chunk's (duration-based) end exceeds this, warn
# that frames may have been dropped between files.
_MAX_EXPECTED_GAP_SECONDS = 2.0


@dataclass
class Chunk:
    sequence: int
    start_time: datetime
    mp4_path: Path
    lrf_path: Path | None
    duration_seconds: float
    global_start_seconds: float


def _probe_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def discover_chunks(source_dir: str | Path) -> list[Chunk]:
    """Find DJI_*_D.MP4/.LRF pairs in ``source_dir``, ordered by sequence,
    with each chunk's global_start_seconds set assuming continuous recording.
    """
    source_dir = Path(source_dir)
    parsed: list[tuple[int, datetime, Path]] = []
    for mp4_path in source_dir.glob("DJI_*_D.MP4"):
        match = _NAME_RE.match(mp4_path.stem)
        if not match:
            continue
        seq = int(match.group("seq"))
        start_time = datetime.strptime(match.group("timestamp"), "%Y%m%d%H%M%S")
        parsed.append((seq, start_time, mp4_path))
    if not parsed:
        raise FileNotFoundError(f"No DJI_*_D.MP4 files found in {source_dir}")
    parsed.sort(key=lambda item: item[0])

    chunks: list[Chunk] = []
    global_offset = 0.0
    previous_end_wallclock: datetime | None = None
    for seq, start_time, mp4_path in parsed:
        lrf_path = mp4_path.with_suffix(".LRF")
        if not lrf_path.exists():
            lrf_path = None
        duration = _probe_duration_seconds(mp4_path)

        if previous_end_wallclock is not None:
            gap = (start_time - previous_end_wallclock).total_seconds()
            if abs(gap) > _MAX_EXPECTED_GAP_SECONDS:
                print(
                    f"WARNING: {mp4_path.name} starts {gap:.2f}s after the previous "
                    f"chunk's expected end -- possible dropped frames between chunks."
                )

        chunks.append(
            Chunk(
                sequence=seq,
                start_time=start_time,
                mp4_path=mp4_path,
                lrf_path=lrf_path,
                duration_seconds=duration,
                global_start_seconds=global_offset,
            )
        )
        global_offset += duration
        previous_end_wallclock = start_time.fromtimestamp(start_time.timestamp() + duration)

    return chunks


def wallclock_to_global(w: datetime, chunks: list[Chunk]) -> float | None:
    """Inverse of the recorded-time axis global_start_seconds is built on:
    map a wall-clock instant back to a global-timeline offset.

    global_start_seconds accumulates durations alone (continuous-recording
    assumption), so it drifts from real wall-clock time by however long the
    camera was stopped between chunks (41.5 min across one 110.3 min game).
    An externally-timestamped mark (a watch press, a DJI highlight) has to
    be re-anchored through the owning chunk's own filename timestamp instead
    of naively subtracting session start, exactly like cli.py's
    `_wall_clock_tag` does in the other direction.

    ``w`` must be a naive datetime already in the recording's local time
    zone (chunks' start_time is naive, parsed straight from the DJI
    filename with no timezone info -- see soccer_highlights.marks for the
    explicit LA-timezone pinning callers are expected to have done first).

    Returns ``None`` if ``w`` falls before the first chunk, after the last
    chunk, or inside an unrecorded gap between two chunks (camera stopped
    and restarted) -- the caller must report "marked but not recorded"
    rather than silently mapping to the wrong offset."""
    for chunk in chunks:
        chunk_end = chunk.start_time + timedelta(seconds=chunk.duration_seconds)
        if chunk.start_time <= w < chunk_end:
            return chunk.global_start_seconds + (w - chunk.start_time).total_seconds()
        if w < chunk.start_time:
            return None
    return None


def slice_start_epoch(chunk: Chunk, local_start_seconds: float) -> float:
    """Epoch value for the instant `local_start_seconds` into `chunk`, to feed
    ffmpeg drawtext's ``%{pts\\:gmtime\\:...}`` when burning the time of day
    into an exported clip (see soccer_highlights.render).

    Forward counterpart to `wallclock_to_global` -- global/local offset out to
    wall clock, rather than an external timestamp in.

    **Deliberately not a true UTC instant.** The DJI filename timestamp
    `chunk.start_time` already IS the local wall clock a viewer wants to read,
    so it is reinterpreted as if it were UTC: formatting that with `gmtime`
    reproduces exactly those digits. Converting it to a real UTC epoch instead
    and formatting with `gmtime` burns in UTC (caught in testing 2026-08-31: an
    8:46 AM game read "03.46 PM"), and formatting with `localtime` would make
    the output depend on the render machine's OS timezone. This way the burned-in
    clock is machine-independent and needs no DST handling at all."""
    local_dt = chunk.start_time + timedelta(seconds=local_start_seconds)
    return local_dt.replace(tzinfo=timezone.utc).timestamp()
