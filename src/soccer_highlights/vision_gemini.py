"""Gemini native-video counterpart to vision.py's Claude stills-based
confirm pass. Built to answer one question directly: does sending
continuous video (not discrete extracted frames) help classify a
candidate window, or does Gemini's own frame subsampling (1 FPS by
default, confirmed via its public docs) hit the identical blind spot a
sub-second shot/strike transient created for Claude's evenly-spaced
stills before the peak-anchoring fix?

Round 1 (fair baseline) reused vision.py's _CONFIRM_PROMPT unmodified --
stills-specific language ("these frames are sampled from...") sent to a
model that actually receives one continuous clip. Result: Gemini beat
Claude on F1 anyway (0.432 vs 0.409, see README's Vision AI Round 2) --
still the best result found across every variant tried so far.

Round 2 (`_AV_CONFIRM_PROMPT`, defined here but NOT the active default --
see below) explicitly asked the model to use the audio track already
baked into the extracted clip (the ffmpeg cut always includes `-c:a
aac`) alongside the video, on the theory that Claude structurally can't
do that (stills only) so it's a real Gemini-specific lever. Result:
worse, not better (F1 0.432 -> 0.320, FN 5 -> 9) -- same pattern as
Round 3's stricter visual definition (see vision.py): any prompt change
that raises the bar for "count this as a real event" (visual specificity
OR audio+video correlation) loses more recall than it gains precision on
this task. Kept as a documented negative result, not deleted; the
classify_confirm default below still uses the plain (Round 1)
_CONFIRM_PROMPT since that's the best-performing configuration found.

Shares vision.py's pure logic (peak-anchored sample window via
_peak_sample_window, VisionVerdict/_parse_verdict_json/decide_confirm) --
only the I/O and the prompt text differ: a short video clip instead of
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

# Tested (2026-07-25) and NOT adopted -- see this module's docstring and
# README's Vision AI Round 4: made recall worse (F1 0.432 -> 0.320).
# Kept for reference/future revisiting, not used by classify_confirm.
_AV_CONFIRM_PROMPT = """You are reviewing a short video clip, WITH AUDIO, from a Sunday recreational soccer game (16-player, half-court). An audio-based detector flagged a possible shot on target or goal; this clip covers a {duration:.1f}-second window CENTERED ON THE MOMENT OF LOUDEST AUDIO ACTIVITY (the likely instant of ball contact or crowd reaction) -- it is not the whole highlight, just the moment itself.

Use BOTH the video and the audio track together: listen for a sharp ball-strike/contact sound or a crowd reaction (cheering, gasping, shouting) at the same time as watching for real in-play action (an actual shot, save, goal, or the crowd's live reaction to one) -- versus something that can sound similar on audio but isn't real action: a practice/drill shot during a stoppage, players standing around or chatting during a break, warm-up, etc. The actual moment can be brief -- if the clip is ambiguous or doesn't clearly rule out a real event, prefer is_event: true. It's worse to discard a real highlight than to keep one extra false positive.

Also write a short, factual one-line caption suitable for use in a filename, describing only what's visibly or audibly confirmable (e.g. "Shot on goal saved by keeper", "Goal celebration near far post"). Do not invent player names, jersey numbers, or the exact score unless clearly legible/audible.

Respond with ONLY a single JSON object, no other text, no code fence:
{{"is_event": true or false, "confidence": 0.0-1.0, "caption": "short factual caption", "rationale": "one short sentence"}}"""


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


def _call_gemini(
    video_path: Path, prompt: str, cfg: GeminiConfig, api_key: str, video_metadata: dict | None = None
) -> str:
    """video_metadata (e.g. {"fps": 5}) overrides Gemini's default 1fps
    frame-sampling for this part -- optional and unused by classify_confirm
    (the concluded Rounds 1-4 vision-compare path stays exactly as it was);
    label_audit.generate_description passes it, see that module for why."""
    video_b64 = base64.standard_b64encode(video_path.read_bytes()).decode("utf-8")
    part: dict = {"inline_data": {"mime_type": "video/mp4", "data": video_b64}}
    if video_metadata:
        part["video_metadata"] = video_metadata
    body = {"contents": [{"parts": [part, {"text": prompt}]}]}
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
    sample window, same _CONFIRM_PROMPT text as Claude gets (best-
    performing configuration found across Rounds 1-4 -- see this
    module's docstring), but the model sees one continuous video clip
    instead of discrete stills."""
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
