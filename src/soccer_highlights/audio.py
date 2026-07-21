"""Audio extraction via ffmpeg, reading directly from the source file
(camera memory card or local disk) with no intermediate copy."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def extract_audio_samples(path: str | Path, sample_rate: int, mono: bool) -> np.ndarray:
    """Decode the audio track of ``path`` into a float32 array in [-1, 1],
    resampled to ``sample_rate``. Streams raw PCM through a pipe -- never
    writes the source file or a full-length intermediate to disk.
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
    result = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if not mono:
        samples = samples.reshape(-1, channels)
    return samples
