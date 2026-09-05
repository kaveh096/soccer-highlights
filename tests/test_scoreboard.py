from dataclasses import replace

import pytest

from soccer_highlights.config import ExportConfig
from soccer_highlights.render import ScoreOverlay, _score_filters
from soccer_highlights.scoreboard import (
    chrome_png,
    compute_layout,
    pill_label,
    render_chrome,
    sample_clock_text,
)


def _cfg(**kw) -> ExportConfig:
    return replace(ExportConfig(), **kw)


def test_layout_scales_with_frame_width():
    """Geometry is authored for a 2560px frame; a smaller export must get a
    proportionally smaller board, not the same giant one."""
    big = compute_layout(_cfg(), 2560)
    small = compute_layout(_cfg(), 1280)

    assert small.width == pytest.approx(big.width / 2, rel=0.02)
    assert small.font_score == pytest.approx(big.font_score / 2, rel=0.05)


def test_layout_reserves_room_for_a_two_digit_score():
    """The slot is sized for the widest score, so the anchors do not move
    between a 2-1 game and a 10-9 one -- the pills are baked into a static
    PNG and cannot get out of the way."""
    layout = compute_layout(_cfg(), 2560)
    a = compute_layout(_cfg(), 2560)

    assert layout.x_score_center == a.x_score_center
    # The score centre must sit left of the clock, with the BLACK pill between.
    assert layout.x_score_center < layout.x_clock
    assert layout.x_clock < layout.width


def test_panel_is_measured_from_the_configured_clock_format():
    """Sizing the panel for a stand-in string while drawing a longer one runs
    the clock off the end of the panel -- caught in the first real render,
    2026-09-05."""
    short = compute_layout(_cfg(scoreboard_time_format="%I.%M %p"), 2560)
    dated = compute_layout(_cfg(scoreboard_time_format="%b %d %I.%M.%S %p"), 2560)

    assert dated.width > short.width


def test_sample_clock_text_follows_the_format():
    assert sample_clock_text(_cfg(scoreboard_time_format="%I.%M.%S %p")) == "10.46.36 PM"
    assert "Dec" in sample_clock_text(_cfg(scoreboard_time_format="%b %d %I.%M %p"))


def test_pill_labels_are_drawn_in_caps():
    # Config stays human-readable; the design sets the pills in caps. Measuring
    # and drawing both go through this, so they cannot disagree about case.
    assert pill_label("White") == "WHITE"


def test_chrome_png_matches_the_layout_and_is_cached():
    cfg = _cfg()
    layout = compute_layout(cfg, 2560)

    path_a, layout_a = chrome_png(cfg, 2560)
    path_b, _ = chrome_png(cfg, 2560)

    assert path_a == path_b and path_a.exists()
    assert layout_a == layout
    assert render_chrome(cfg, layout).size == (layout.width, layout.height)


def test_score_stages_draw_digits_only_not_the_team_names():
    """The team names live in the PNG's pills. Drawing them again in the
    score text prints the labels on top of themselves -- exactly what the
    first real render did (2026-09-05)."""
    cfg = _cfg()
    layout = compute_layout(cfg, 2560)

    [stage] = _score_filters(cfg, ScoreOverlay(white=2, black=1), 0.0, 10.0, layout)

    assert "2 - 1" in stage
    assert "WHITE" not in stage and "White" not in stage


def test_score_stages_gate_the_flip_on_the_pieces_own_timeline():
    cfg = _cfg()
    layout = compute_layout(cfg, 2560)
    score = ScoreOverlay(white=1, black=0, flip_seconds=4.0, flip_team="white")

    before = _score_filters(cfg, score, 0.0, 3.0, layout)  # piece ends before the flip
    spanning = _score_filters(cfg, score, 3.0, 3.0, layout)  # flip is 1.0s into this piece
    after = _score_filters(cfg, score, 6.0, 3.0, layout)  # piece starts after the flip

    assert len(before) == 1 and "1 - 0" in before[0]
    assert len(spanning) == 2 and "lt(t,1.000)" in spanning[0] and "gte(t,1.000)" in spanning[1]
    assert len(after) == 1 and "2 - 0" in after[0]
