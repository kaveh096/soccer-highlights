import pytest

from soccer_highlights.config import CropConfig
from soccer_highlights.crop import candidate_region, crop_rect


def test_default_window_is_two_thirds_square_of_the_frame():
    cfg = CropConfig()
    assert (cfg.width_fraction, cfg.height_fraction, cfg.y_center) == (2 / 3, 2 / 3, 0.5)


def test_crop_rect_horizontal_position_by_region():
    cfg = CropConfig()
    assert crop_rect("left", cfg)[0] == 0.0
    assert crop_rect("center", cfg)[0] == pytest.approx(1 / 6)
    assert crop_rect("right", cfg)[0] == pytest.approx(1 / 3)
    for region in ("left", "center", "right"):
        x0, y0, w, h = crop_rect(region, cfg)
        assert (w, h) == (2 / 3, 2 / 3)
        assert y0 == pytest.approx(1 / 6)  # y_center 0.5 -> window spans 1/6..5/6
        assert x0 + w <= 1.0 + 1e-9


def test_crop_rect_on_a_4k_source_is_2560x1440():
    x0, y0, w, h = crop_rect("center", CropConfig())
    assert (round(x0 * 3840), round(y0 * 2160), round(w * 3840), round(h * 2160)) == (640, 360, 2560, 1440)


def test_crop_rect_y_center_moves_the_window_and_is_clamped_into_the_frame():
    low = crop_rect("left", CropConfig(y_center=0.6))
    assert low[1] == pytest.approx(0.6 - 1 / 3)
    assert crop_rect("left", CropConfig(y_center=0.0))[1] == 0.0
    assert crop_rect("left", CropConfig(y_center=1.0))[1] == pytest.approx(1 / 3)  # 1 - h


def test_crop_rect_full_is_the_whole_frame_and_unknown_region_raises():
    assert crop_rect("full", CropConfig()) == (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        crop_rect("top", CropConfig())
    with pytest.raises(ValueError):
        crop_rect("left", CropConfig(width_fraction=1.5))


def _boxed_cfg() -> CropConfig:
    return CropConfig(boxes={"near": [0.0, 0.073611, 0.73125, 0.73125], "far": [0.40625, 0.326389, 0.375, 0.375]})


def test_boxed_regions_return_their_configured_rect():
    cfg = _boxed_cfg()
    assert crop_rect("near", cfg) == (0.0, 0.073611, 0.73125, 0.73125)
    x0, y0, w, h = crop_rect("far", cfg)
    assert (round(x0 * 1280), round(y0 * 720), round(w * 1280), round(h * 720)) == (520, 235, 480, 270)


def test_boxed_region_without_a_box_or_outside_the_frame_raises():
    with pytest.raises(ValueError, match="no box"):
        crop_rect("near", CropConfig())
    with pytest.raises(ValueError, match="outside the frame"):
        crop_rect("near", CropConfig(boxes={"near": [0.5, 0.0, 0.73, 0.73]}))


def test_candidate_region_end_cam_uses_the_goal_end_not_gemini():
    cfg = _boxed_cfg()
    assert candidate_region("end", True, "left", cfg) == "near"
    assert candidate_region("end", False, "left", cfg) == "far"
    assert candidate_region("end", None, "left", cfg) is None  # no goal tap, no wide box -> no crop, thirds don't apply


def test_candidate_region_non_goal_clip_gets_wide_when_configured():
    cfg = CropConfig(boxes={**_boxed_cfg().boxes, "wide": [0.35, 0.25, 0.55, 0.55]})
    assert candidate_region("end", None, "left", cfg) == "wide"
    assert candidate_region("end", True, "left", cfg) == "near"  # a goal tap keeps its own side
    assert candidate_region("end", False, "left", cfg) == "far"
    assert crop_rect("wide", cfg) == (0.35, 0.25, 0.55, 0.55)


def test_candidate_region_passes_gemini_through_for_center_cam_or_unconfigured_boxes():
    assert candidate_region("center", True, "right", _boxed_cfg()) == "right"
    assert candidate_region("end", True, "right", CropConfig()) == "right"
    assert candidate_region("end", True, "right", CropConfig(boxes={"near": [0, 0, 0.5, 0.5]})) == "right"
