"""Action-region crop geometry -- the single source of truth for BOTH the review
overlay's box outline and export's crop (2026-10-18).

Pure arithmetic on frame fractions, no I/O, so the 1280px review clip and the
3840x2160 source resolve to the same window by construction: multiply the
returned fractions by whatever frame size is in hand. The window is
`CropConfig.width_fraction` x `height_fraction` of the frame (2/3 x 2/3 =
2560x1440 of a 4K source -- a 1.5x crop that matches today's share width with no
upscale). Horizontal position follows the region; vertical position is fixed at
`y_center`.
"""

from __future__ import annotations

from soccer_highlights.config import CropConfig

REGIONS = ("left", "center", "right")
# Regions defined by explicit boxes in `CropConfig.boxes` (end-cam games: the
# near goal and the far end). Valid on a pick / sheet row only when configured.
BOXED_REGIONS = ("near", "far")
ALL_REGIONS = (*REGIONS, *BOXED_REGIONS)
# "full" = no crop; accepted by crop_rect so a caller (export's `crop=full`
# override) can ask for it uniformly, but it is never a valid *detected* region.
FULL = "full"


def crop_rect(region: str, cfg: CropConfig) -> tuple[float, float, float, float]:
    """(x0, y0, w, h) of the crop window as fractions of the frame.

    left -> x0=0, center -> x0=(1-w)/2, right -> x0=1-w (1/6 and 1/3 for a 2/3
    wide window). y0 is centred on `cfg.y_center` and clamped so the window never
    leaves the frame. "full" returns the whole frame. Raises ValueError for an
    unknown region or a window size outside (0, 1]."""
    w, h = cfg.width_fraction, cfg.height_fraction
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        raise ValueError(f"crop window fractions must be in (0, 1]: width={w}, height={h}")
    if region == FULL:
        return (0.0, 0.0, 1.0, 1.0)
    if region in BOXED_REGIONS:
        box = cfg.boxes.get(region)
        if box is None:
            raise ValueError(f"crop region {region!r} has no box in crop.boxes")
        x0, y0, bw, bh = box
        if not (bw > 0 and bh > 0 and x0 >= 0 and y0 >= 0 and x0 + bw <= 1 + 1e-9 and y0 + bh <= 1 + 1e-9):
            raise ValueError(f"crop box {region!r} = {box} is outside the frame")
        return (x0, y0, bw, bh)
    x0_by_region = {"left": 0.0, "center": (1.0 - w) / 2.0, "right": 1.0 - w}
    if region not in x0_by_region:
        raise ValueError(f"unknown crop region {region!r} (expected one of {ALL_REGIONS} or {FULL!r})")
    y0 = min(max(cfg.y_center - h / 2.0, 0.0), 1.0 - h)
    return (x0_by_region[region], y0, w, h)


def candidate_region(
    camera_mode: str, is_near_cam_goal: bool | None, gemini_region: str | None, cfg: CropConfig
) -> str | None:
    """The `action_region` a review candidate gets (2026-10-26).

    End-cam game with both `near` and `far` boxes configured: a goal tap whose
    scoring end is known picks the box deterministically (near goal -> `near`,
    far goal -> `far`); anything else gets NO crop, because Gemini's
    left/center/right thirds describe a different geometry than these boxes.
    Kaveh overrides with `crop=near|far` in his pick. Otherwise (center-cam, or
    no boxes configured) Gemini's region passes through unchanged."""
    if camera_mode == "end" and all(name in cfg.boxes for name in BOXED_REGIONS):
        if is_near_cam_goal is None:
            return None
        return "near" if is_near_cam_goal else "far"
    return gemini_region
