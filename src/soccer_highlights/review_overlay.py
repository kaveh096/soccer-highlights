"""Review-only overlay: the crop-box outline and slow-mo dot burned into the
1280px review clips AFTER Gemini has scored the clean ones (2026-10-18).

Ordering is the point. Gemini must never see an overlay (a drawn box or a dot
would contaminate the score and the action_region read), so pre-label renders
CLEAN clips into a local work dir, Gemini scores those, and only then does this
module re-encode each clip with (a) the outline of the crop window its
`action_region` will be exported with and (b) one small solid dot, top-right,
for ~20 frames centred on `slomo_center_s` (the slow-mo replay's centre). Exports
never get either.

Resume safety lives here too, because this laptop gets interrupted: an
`OverlayLedger` (JSON in the work dir) records which clean clips and which
overlaid clips are finished, keyed by a signature of everything that determines
the output. A step is skipped only when its ledger signature matches AND the file
passes `render.is_playable` (a bare size check passes a killed encode's corpse).
The ledger is written only after a step fully succeeds, and the final file lands
in the Review folder through a temp name + os.replace, so a Drive folder never
holds a half-written clip under its real name.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from soccer_highlights.config import CropConfig, ReviewConfig, ReviewOverlayConfig
from soccer_highlights.crop import crop_rect
from soccer_highlights.render import _run_ffmpeg, encode_timeout_seconds, is_playable


def probe_video(path: Path) -> tuple[int, int, float]:
    """(width, height, duration_seconds) of the first video stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return int(stream["width"]), int(stream["height"]), float(info["format"]["duration"])


def dot_frame_window(slomo_center_s: float, n_frames: int, fps: float) -> tuple[float, float]:
    """[start, end) seconds during which the dot is visible: `n_frames` frames
    centred on the slow-mo centre (truncated by the clip start, not shifted)."""
    half = n_frames / fps / 2.0
    return (max(slomo_center_s - half, 0.0), slomo_center_s + half)


def build_overlay_graph(
    width: int,
    height: int,
    region: str | None,
    slomo_center_s: float | None,
    fps: float,
    crop_cfg: CropConfig,
    cfg: ReviewOverlayConfig,
) -> str | None:
    """The ffmpeg filter graph for one clip, or None when there is nothing to draw.

    The box uses `crop.crop_rect` (the same geometry export crops with) scaled to
    this clip's pixel size; its thickness is `cfg.box_thickness_px` at 1280px
    wide, scaled for other widths. The dot is overlaid from input 1 (a looped
    PNG, see `overlay_review_clip`) with a time-window `enable`. With a dot the
    graph is `[0:v]...[vout]` (use -filter_complex, map [vout]); without one it is
    a plain `drawbox` chain (use -vf)."""
    stages: list[str] = []
    if region:
        x0, y0, w, h = crop_rect(region, crop_cfg)
        thickness = max(1, round(cfg.box_thickness_px * width / 1280))
        stages.append(
            f"drawbox=x={round(x0 * width)}:y={round(y0 * height)}:w={round(w * width)}:h={round(h * height)}"
            f":color=0x{cfg.box_color}@{cfg.box_alpha}:t={thickness}"
        )
    if slomo_center_s is None:
        return ",".join(stages) if stages else None

    start, end = dot_frame_window(slomo_center_s, cfg.dot_frames, fps)
    margin = max(1, round(cfg.dot_margin_fraction * width))
    enable = f"gte(t,{start:.3f})*lt(t,{end:.3f})"
    head = f"[0:v]{','.join(stages)}[bx];[bx]" if stages else "[0:v]"
    return f"{head}[1:v]overlay=x=main_w-overlay_w-{margin}:y={margin}:enable='{enable}'[vout]"


def write_dot_png(path: Path, diameter_px: int, color_hex: str) -> None:
    """A solid dot with a dark rim, anti-aliased by 4x supersampling."""
    scale = 4
    size = diameter_px * scale
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.ellipse((0, 0, size - 1, size - 1), fill=(20, 20, 20, 255))
    rim = max(1, round(diameter_px * 0.16)) * scale
    rgb = tuple(int(color_hex[i : i + 2], 16) for i in (0, 2, 4))
    draw.ellipse((rim, rim, size - 1 - rim, size - 1 - rim), fill=(*rgb, 255))
    im.resize((diameter_px, diameter_px), Image.LANCZOS).save(path)


def overlay_review_clip(
    clean_path: Path,
    out_path: Path,
    region: str | None,
    slomo_center_s: float | None,
    crop_cfg: CropConfig,
    overlay_cfg: ReviewOverlayConfig,
    review_cfg: ReviewConfig,
    tmp_dir: Path,
) -> None:
    """Re-encode `clean_path` to `out_path` with the box (if `region`) and the dot
    (if `slomo_center_s`). Same settings as the review render (crf/preset/threads
    from `review_cfg`), audio stream-copied. A clip with neither is just copied.
    `out_path` is written directly -- callers publish it atomically themselves."""
    width, height, duration = probe_video(clean_path)
    diameter = max(6, round(overlay_cfg.dot_diameter_fraction * width))
    dot_path = tmp_dir / f"dot_{diameter}_{overlay_cfg.dot_color}.png"
    use_dot = slomo_center_s is not None
    if use_dot and not dot_path.exists():
        tmp_dir.mkdir(parents=True, exist_ok=True)
        write_dot_png(dot_path, diameter, overlay_cfg.dot_color)

    graph = build_overlay_graph(width, height, region, slomo_center_s, review_cfg.fps, crop_cfg, overlay_cfg)
    if graph is None:
        shutil.copyfile(clean_path, out_path)
        return

    encode_args = [
        "-c:v", "libx264", "-preset", review_cfg.preset, "-crf", str(review_cfg.crf),
        "-threads", str(review_cfg.threads), "-c:a", "copy",
    ]
    if use_dot:
        # -loop 1 on the PNG and -t as an OUTPUT option: same two requirements as
        # render._encode_piece's scoreboard chrome (see its docstring).
        args = [
            "-i", str(clean_path), "-loop", "1", "-i", str(dot_path),
            "-filter_complex", graph, "-map", "[vout]", "-map", "0:a?",
            "-t", f"{duration:.3f}", *encode_args, str(out_path),
        ]
    else:
        args = ["-i", str(clean_path), "-vf", graph, *encode_args, str(out_path)]
    _run_ffmpeg(args, timeout=encode_timeout_seconds(duration))


def clean_signature(start_seconds: float, end_seconds: float) -> str:
    """Identity of a clean clip: the interval it was cut from (as written to the
    sheet, 0.01s) -- a rerun with different detection settings must re-render."""
    return f"{start_seconds:.2f}-{end_seconds:.2f}"


def overlay_signature(
    start_seconds: float,
    end_seconds: float,
    region: str | None,
    slomo_center_s: float | None,
    cfg: ReviewOverlayConfig,
    crop_cfg: CropConfig,
) -> str:
    """Identity of an overlaid clip: everything that changes its pixels. A region
    arriving late (a failed describe retried on rerun) or a config tweak changes
    this, so the clip is re-overlaid from the clean one; otherwise it is skipped."""
    return "|".join(
        [
            clean_signature(start_seconds, end_seconds),
            region or "-",
            "-" if slomo_center_s is None else f"{slomo_center_s:.1f}",
            repr(cfg),
            repr(crop_cfg),
        ]
    )


class OverlayLedger:
    """JSON record of finished steps: {"clean": {clip_file: sig}, "overlaid": {...}}.
    Saved atomically after every record() so an interrupt loses at most the step
    in flight."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict[str, str]] = {"clean": {}, "overlaid": {}}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                for kind in self._data:
                    self._data[kind] = dict(loaded.get(kind, {}))
            except (OSError, ValueError):
                pass  # unreadable ledger = nothing proven done; redo, never trust

    def matches(self, kind: str, clip_file: str, signature: str) -> bool:
        return self._data[kind].get(clip_file) == signature

    def record(self, kind: str, clip_file: str, signature: str) -> None:
        self._data[kind][clip_file] = signature
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def forget(self, kind: str, clip_file: str) -> None:
        self._data[kind].pop(clip_file, None)


def step_is_done(ledger: OverlayLedger, kind: str, clip_file: str, signature: str, path: Path) -> bool:
    """A step is done only if the ledger says so for THIS signature and the file
    is actually playable."""
    return ledger.matches(kind, clip_file, signature) and is_playable(path)


def publish_file(src: Path, dst: Path) -> None:
    """Copy `src` into `dst`'s folder under a temp name, then os.replace it into
    place -- so a (possibly Google-Drive-backed) Review folder never has a
    half-written clip under its real name. The temp name starts with `.` and is
    simply overwritten if an interrupt leaves one behind."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(f".{dst.name}.part")
    shutil.copyfile(src, part)
    os.replace(part, dst)
