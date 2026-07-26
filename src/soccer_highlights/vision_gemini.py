"""Gemini native-video counterpart to vision.py's Claude stills-based
confirm pass. Built to answer one question directly: does sending
continuous video (not discrete extracted frames) help classify a
candidate window, or does Gemini's own frame subsampling (1 FPS by
default, confirmed via its public docs) hit the identical blind spot a
sub-second shot/strike transient created for Claude's evenly-spaced
stills before the peak-anchoring fix?

Reuses vision.py's pure logic and prompt text unmodified for the first,
fair comparison round (same _CONFIRM_PROMPT, same peak-anchored sample
window via _peak_sample_window, same VisionVerdict/_parse_verdict_json/
decide_confirm) -- only the I/O differs: a short video clip instead of
stills, one Gemini generateContent call instead of Claude's Messages API.
Not unit-tested (ffmpeg + network I/O), same split as vision.py's Claude
implementation -- validate against real footage with vision-compare.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from soccer_highlights.config import GeminiConfig
from soccer_highlights.discovery import Chunk
from soccer_highlights.timeline import Interval, map_interval_to_chunks
from soccer_highlights.vision import (
    _CONFIRM_PROMPT,
    _parse_verdict_json,
    _peak_sample_window,
    VisionVerdict,
)

_GENERATE_CONTENT_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _extract_peak_clip(sample_window: Interval, chunks: list[Chunk], max_width: int, out_path: Path) -> Path | None:
    """Cut a short mp4 clip covering `sample_window` from the .LRF proxy.
    If the window happens to span a chunk boundary (rare -- only possible
    right at the edge of a recording chunk), just use the first slice;
    good enough for this comparison, not a production render path."""
    slices = map_interval_to_chunks(sample_window, chunks)
    if not slices:
        return None
    cs = slices[0]
    source_path = cs.chunk.lrf_path or cs.chunk.mp4_path
    duration = cs.local_end_seconds - cs.local_start_seconds
    if duration <= 0:
        return None
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{cs.local_start_seconds:.3f}",
            "-i", str(source_path),
            "-t", f"{duration:.3f}",
            "-vf", f"scale={max_width}:-2",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-strict", "-2", "-b:a", "64k",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def _call_gemini(video_path: Path, prompt: str, cfg: GeminiConfig, api_key: str) -> str:
    video_b64 = base64.standard_b64encode(video_path.read_bytes()).decode("utf-8")
    body = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "video/mp4", "data": video_b64}},
                    {"text": prompt},
                ]
            }
        ]
    }
    url = _GENERATE_CONTENT_URL.format(model=cfg.model)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg.request_timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Gemini API error {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc

    return "".join(
        part.get("text", "")
        for candidate in data.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
    )


def classify_confirm(interval: Interval, chunks: list[Chunk], cfg: GeminiConfig) -> VisionVerdict | None:
    """Gemini counterpart to vision.classify_confirm: same peak-anchored
    sample window and same (unmodified, for a fair first comparison)
    confirm prompt, but the model sees one continuous video clip instead
    of discrete stills."""
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set -- see README's Vision AI section for setup")

    sample_window = _peak_sample_window(interval, cfg)

    try:
        with tempfile.TemporaryDirectory(prefix="soccer_hl_vision_gemini_") as tmp_dir_str:
            clip_path = Path(tmp_dir_str) / "clip.mp4"
            extracted = _extract_peak_clip(sample_window, chunks, cfg.clip_max_width, clip_path)
            if extracted is None:
                return None
            prompt = _CONFIRM_PROMPT.format(duration=sample_window.end_seconds - sample_window.start_seconds)
            raw = _call_gemini(extracted, prompt, cfg, api_key)
            return _parse_verdict_json(raw, mode="confirm")
    except Exception as exc:
        print(f"WARNING: gemini confirm failed for [{interval.start_seconds:.1f}, {interval.end_seconds:.1f}]: {exc}")
        return None
