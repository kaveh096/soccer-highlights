"""Pick-line parsing and resolution (2026-10-18): `<rank> [crop=...] [slomo...]`."""

import pytest

from soccer_highlights.picks import (
    DEFAULT_SLOMO_SPEED,
    PickError,
    PickSpec,
    SlowMo,
    clip_ranks,
    default_slomo_window,
    load_picks,
    row_region,
    parse_picks,
    resolve_picks,
)


def _row(clip_file: str, start: float = 100.0, length: float = 14.0, region: str = "", center: str = "") -> dict[str, str]:
    return {
        "clip_file": clip_file,
        "start_seconds": str(start),
        "end_seconds": str(start + length),
        "action_region": region,
        "slomo_center_s": center,
    }


ROWS = [
    _row("r03_s5_746a_c032.mp4", region="left", center="9.0"),
    _row("r05_s4_750a_c040.mp4", region="", center=""),
    _row("r07_s4_752a_c041.mp4", region="right", center="1.0"),
    _row("r30_31_s2_821a_c025_c026.mp4", region="center", center="12.5"),
]


# --- parsing -----------------------------------------------------------------


def test_parse_the_documented_examples():
    text = """
    r03
    r05 crop=left
    r07 slomo
    r07 slomo=6.5-9
    r09 slomo=4-7@0.5 crop=full
    r10 slomo@0.5
    """
    specs = parse_picks(text)

    assert specs[0] == PickSpec(ref="r03")
    assert specs[1] == PickSpec(ref="r05", crop="left")
    assert specs[2] == PickSpec(ref="r07", slomo=True)
    assert specs[3] == PickSpec(ref="r07", slomo=True, window=(6.5, 9.0))
    assert specs[4] == PickSpec(ref="r09", crop="full", slomo=True, window=(4.0, 7.0), speed=0.5)
    assert specs[5] == PickSpec(ref="r10", slomo=True, speed=0.5)


def test_blank_lines_comments_and_semicolons():
    specs = parse_picks("# header\n\nr03  # trailing comment\nr05 crop=left; r07 slomo\n   \n")
    assert [s.ref for s in specs] == ["r03", "r05", "r07"]
    assert specs[1].crop == "left" and specs[2].slomo


def test_token_order_is_free_and_case_of_crop_value_is_ignored():
    assert parse_picks("r01 crop=RIGHT slomo=1-2")[0] == PickSpec(ref="r01", crop="right", slomo=True, window=(1.0, 2.0))


@pytest.mark.parametrize(
    "line, fragment",
    [
        ("r03 bogus", "unknown token"),
        ("r03 crop=top", "bad crop"),
        ("r03 crop=left crop=right", "crop given twice"),
        ("r03 slomo slomo=1-2", "slomo given twice"),
        ("r03 slomo=5", "bad slomo token"),
        ("r03 slomo=9-6", "A < B"),
        ("r03 slomo=4-4", "A < B"),
        ("r03 slomo@2", "speed"),
        ("r03 slomo@0.01", "speed"),
        ("slomo=1-2", "start with a rank"),
        ("crop=left", "start with a rank"),
    ],
)
def test_invalid_lines_are_rejected(line, fragment):
    with pytest.raises(PickError) as exc:
        parse_picks(line)
    assert fragment in str(exc.value)


def test_every_bad_line_is_reported_not_just_the_first():
    text = "r03\nr05 crop=top\nr07 slomo=9-6\nr09 whatever\n"
    with pytest.raises(PickError) as exc:
        parse_picks(text)

    assert len(exc.value.errors) == 3
    assert exc.value.errors[0].startswith("line 2")
    assert exc.value.errors[1].startswith("line 3")
    assert exc.value.errors[2].startswith("line 4")


# --- resolution --------------------------------------------------------------


def test_rank_matches_rank_named_clip_file_and_full_names_and_unpadded_ranks():
    out = resolve_picks(parse_picks("r03\nr5\nr07_s4_752a_c041.mp4"), ROWS)
    assert [p.clip_file for p in out] == ["r03_s5_746a_c032.mp4", "r05_s4_750a_c040.mp4", "r07_s4_752a_c041.mp4"]


def test_merged_clip_answers_to_both_ranks():
    assert clip_ranks("r30_31_s2_821a_c025_c026.mp4") == [30, 31]
    assert clip_ranks("r03_s5_746a_c032.mp4") == [3]
    assert clip_ranks("clip_055.mp4") == []
    assert resolve_picks(parse_picks("r31"), ROWS)[0].clip_file == "r30_31_s2_821a_c025_c026.mp4"


def test_crop_defaults_to_sheet_region_blank_means_full_and_pick_overrides():
    out = {p.clip_file[:3]: p.region for p in resolve_picks(parse_picks("r03\nr05\nr07 crop=left"), ROWS)}
    assert out == {"r03": "left", "r05": "full", "r07": "left"}


def test_crop_full_overrides_sheet_region_and_no_sheet_crop_ignores_it():
    assert resolve_picks(parse_picks("r03 crop=full"), ROWS)[0].region == "full"
    assert resolve_picks(parse_picks("r03"), ROWS, use_sheet_crop=False)[0].region == "full"
    assert resolve_picks(parse_picks("r03 crop=center"), ROWS, use_sheet_crop=False)[0].region == "center"


def test_sheet_without_the_new_columns_still_works():
    old = [{"clip_file": "r01_s5_x.mp4", "start_seconds": "0", "end_seconds": "10"}]
    out = resolve_picks(parse_picks("r01"), old)
    assert out[0].region == "full" and out[0].slomo is None


def test_default_slomo_window_is_2s_before_to_1s_after_the_centre():
    assert default_slomo_window(9.0, 14.0) == (7.0, 10.0)


def test_default_slomo_window_is_clamped_to_the_clip():
    assert default_slomo_window(1.0, 14.0) == (0.0, 2.0)
    assert default_slomo_window(13.5, 14.0) == (11.5, 14.0)
    assert default_slomo_window(40.0, 14.0) is None  # centre far outside the clip


def test_slomo_uses_the_sheet_centre_and_default_speed():
    pick = resolve_picks(parse_picks("r03 slomo"), ROWS)[0]
    assert pick.slomo == SlowMo(7.0, 10.0, DEFAULT_SLOMO_SPEED)
    assert pick.slomo.source_seconds == 3.0
    assert pick.slomo.output_seconds == 12.0


def test_slomo_default_window_near_clip_start_is_clamped():
    pick = resolve_picks(parse_picks("r07 slomo"), ROWS)[0]
    assert pick.slomo == SlowMo(0.0, 2.0, 0.25)


def test_explicit_window_and_speed_win_over_the_centre():
    pick = resolve_picks(parse_picks("r03 slomo=4-7@0.5"), ROWS)[0]
    assert pick.slomo == SlowMo(4.0, 7.0, 0.5)
    assert pick.slomo.output_seconds == 6.0


def test_slomo_without_a_centre_is_an_error_unless_a_window_is_given():
    with pytest.raises(PickError) as exc:
        resolve_picks(parse_picks("r05 slomo"), ROWS)
    assert "slomo_center_s" in str(exc.value)
    assert resolve_picks(parse_picks("r05 slomo=1-3"), ROWS)[0].slomo == SlowMo(1.0, 3.0, 0.25)


def test_window_outside_the_clip_is_an_error():
    with pytest.raises(PickError) as exc:
        resolve_picks(parse_picks("r03 slomo=10-15"), ROWS)  # clip is 14s long
    assert "outside" in str(exc.value)


def test_unknown_ambiguous_and_duplicate_picks_are_all_reported():
    rows = ROWS + [_row("r03_s1_999a_c099.mp4")]
    with pytest.raises(PickError) as exc:
        resolve_picks(parse_picks("r03\nr99\nr05\nr05"), rows)

    messages = "\n".join(exc.value.errors)
    assert "ambiguous" in messages
    assert "'r99'" in messages and "no clip" in messages
    assert "more than once" in messages
    assert len(exc.value.errors) == 3


def test_load_picks_reports_syntax_and_resolution_errors_together():
    with pytest.raises(PickError) as exc:
        load_picks("r03 crop=top\nr99\nr05 slomo", ROWS)
    assert len(exc.value.errors) == 3


def test_near_and_far_are_valid_crop_choices_and_sheet_regions():
    specs = parse_picks("r01 crop=near; r02 crop=FAR")
    assert [s.crop for s in specs] == ["near", "far"]
    assert row_region({"action_region": "near"}) == "near"
    assert row_region({"action_region": "far"}) == "far"
