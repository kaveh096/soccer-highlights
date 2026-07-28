"""Audio extraction via ffmpeg, reading directly from the source file
(camera memory card, local disk, or a cloud-synced drive) with no
intermediate copy."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np


def extract_audio_samples(path: str | Path, sample_rate: int, mono: bool, max_retries: int = 3) -> np.ndarray:
    """Decode the audio track of ``path`` into a float32 array in [-1, 1],
    resampled to ``sample_rate``. Streams raw PCM through a pipe -- never
    writes the source file or a full-length intermediate to disk.

    Retries a few times on failure: reading a source hosted on a
    cloud-synced drive (e.g. Google Drive's virtual filesystem) can fail
    with a transient I/O error rather than a genuinely missing/corrupt
    file -- observed in practice reading a freshly-uploaded, not-yet-
    fully-cached recording (ffmpeg exiting 3221225794 / 0xC0000006
    STATUS_IN_PAGE_ERROR), which a plain retry recovered from.
    """
    channels = 1 if mono else 2
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"WARNING: audio extraction failed for {path} (attempt {attempt}/{max_retries}): {exc}")
            if attempt < max_retries:
                time.sleep(3.0)
    else:
        raise last_error

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if not mono:
        samples = samples.reshape(-1, channels)
    return samples
