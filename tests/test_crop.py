import pytest

from soccer_highlights.config import CropConfig
from soccer_highlights.crop import crop_rect


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
