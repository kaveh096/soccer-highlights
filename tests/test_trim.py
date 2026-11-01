"""`trim=N` picks (2026-10-19): grammar, window maths, scoreboard rebase, slow-mo
defaults, and the export-picks plumbing. No ffmpeg."""

import csv
from datetime import datetime
from pathlib import Path

import pytest

from soccer_highlights import cli, render
from soccer_highlights.config import Config, TrimConfig
from soccer_highlights.discovery import Chunk
from soccer_highlights.picks import (
    PickError,
    PickSpec,
    SlowMo,
    load_picks,
    parse_picks,
    resolve_picks,
    trim_window,
)
from soccer_highlights.render import ScoreOverlay

CFG = TrimConfig()


def _row(
    clip_file: str = "r04_s5_746a_c032.mp4",
    start: float = 1000.0,
    length: float = 80.0,
    flip: str = "",
    team: str = "",
    center: str = "",
) -> dict[str, str]:
    return {
        "clip_file": clip_file,
        "start_seconds": str(start),
        "end_seconds": str(start + length),
        "action_region": "",
        "slomo_center_s": center,
        "score_white": "2",
        "score_black": "1",
        "score_flip_seconds": flip,
        "score_flip_team": team,
    }


# --- grammar -------------------------------------------------------------------


def test_trim_token_parses_as_float_seconds():
    assert parse_picks("r04 trim=58")[0] == PickSpec(ref="r04", trim=58.0)
    assert parse_picks("r04 crop=left trim=31.5 slomo")[0] == PickSpec(ref="r04", crop="left", trim=31.5, slomo=True)


def test_commas_and_case_are_liberal():
    a = parse_picks("R01 trim=31, slomo")
    b = parse_picks("r01 trim=31 slomo")
    assert a == b == [PickSpec(ref="r01", trim=31.0, slomo=True)]
    assert parse_picks("r02, TRIM=20,CROP=Left,SLOMO=1-2@0.5")[0] == PickSpec(
        ref="r02", trim=20.0, crop="left", slomo=True, window=(1.0, 2.0), speed=0.5
    )


@pytest.mark.parametrize(
    "line, fragment",
    [
        ("r04 trim", "bad trim token"),
        ("r04 trim=", "bad trim token"),
        ("r04 trim=abc", "bad trim token"),
        ("r04 trim=-3", "bad trim token"),
        ("r04 trim=0", "greater than 0"),
        ("r04 trim=5 trim=6", "trim given twice"),
        ("trim=5", "start with a rank"),
        ("r04 bogus", "unknown token"),
    ],
)
def test_bad_trim_lines_are_rejected(line, fragment):
    with pytest.raises(PickError) as exc:
        parse_picks(line)
    assert fragment in str(exc.value)


def test_trim_outside_the_clip_is_listed_with_the_other_bad_lines():
    rows = [_row("r04_x.mp4", length=80.0), _row("r05_y.mp4", length=30.0)]
    with pytest.raises(PickError) as exc:
        load_picks("r04 trim=80\nr05 trim=31\nr05x\nr04 crop=top", rows)
    messages = "\n".join(exc.value.errors)
    assert "trim=80 is outside the 80.0s clip" in messages
    assert "line 4" in messages and "bad crop" in messages
    assert "no clip" in messages
    assert len(exc.value.errors) == 4


def test_picks_without_trim_resolve_exactly_as_before():
    row = _row(flip="12.0", team="white", center="9.0")
    plain = resolve_picks(parse_picks("r04 slomo"), [row])[0]
    assert plain.trim is None
    assert plain.export_seconds() == (1000.0, 1080.0)
    assert plain.slomo == SlowMo(5.5, 10.0, 0.25)
    assert plain.describe() == "crop=full slomo=5.5-10@0.25"


# --- trim_window ---------------------------------------------------------------


def test_default_window_is_10s_before_and_4s_after():
    w = trim_window(31.0, 80.0, None, CFG)
    assert (w.start_seconds, w.end_seconds, w.warning) == (21.0, 35.0, None)


def test_flip_inside_the_window_does_not_change_it():
    assert trim_window(31.0, 80.0, 34.0, CFG).end_seconds == 35.0


def test_flip_after_the_window_extends_to_cover_the_tap():
    w = trim_window(58.0, 100.0, 66.0, CFG)  # base 48-62, tap at 66
    assert (w.start_seconds, w.end_seconds, w.warning) == (48.0, 67.5, None)


def test_extension_at_exactly_the_cap_still_follows_the_tap():
    w = trim_window(20.0, 100.0, 35.0, CFG)  # f - N == 15
    assert w.end_seconds == 36.5 and w.warning is None


def test_flip_beyond_the_cap_stops_at_n_plus_15_with_a_loud_warning():
    w = trim_window(20.0, 100.0, 40.0, CFG)  # f - N == 20
    assert (w.start_seconds, w.end_seconds) == (10.0, 35.0)
    assert w.warning is not None and "OUTSIDE" in w.warning and "40" in w.warning


def test_flip_before_the_window_needs_no_extension_and_no_warning():
    w = trim_window(58.0, 100.0, 5.0, CFG)
    assert (w.start_seconds, w.end_seconds, w.warning) == (48.0, 62.0, None)


def test_window_is_clamped_to_the_clip():
    assert trim_window(3.0, 80.0, None, CFG).start_seconds == 0.0
    w = trim_window(78.0, 80.0, None, CFG)
    assert (w.start_seconds, w.end_seconds) == (68.0, 80.0)
    assert trim_window(70.0, 80.0, 79.0, CFG).end_seconds == 80.0  # tap + 1.5 clamped to the clip, flip still inside


@pytest.mark.parametrize("n", [0.0, -1.0, 80.0, 90.0])
def test_trim_window_rejects_n_outside_the_clip(n):
    with pytest.raises(ValueError):
        trim_window(n, 80.0, None, CFG)


def test_window_follows_the_config():
    w = trim_window(30.0, 80.0, None, TrimConfig(lead_seconds=5.0, tail_seconds=2.0))
    assert (w.start_seconds, w.end_seconds) == (25.0, 32.0)


def test_resolved_pick_exports_the_sub_interval_in_session_seconds():
    pick = resolve_picks(parse_picks("r04 trim=58"), [_row(start=1000.0, length=100.0)])[0]
    assert pick.export_seconds() == (1048.0, 1062.0)
    assert "trim=48-62" in pick.describe()


def test_resolved_pick_carries_the_cap_warning():
    pick = resolve_picks(parse_picks("r04 trim=20"), [_row(length=100.0, flip="40.0", team="white")])[0]
    assert pick.trim is not None and pick.trim.warning


# --- scoreboard rebase -------------------------------------------------------------


def test_rebase_flip_inside_window_shifts_the_offset():
    board = ScoreOverlay(2, 1, flip_seconds=66.0, flip_team="white").rebased(48.0, 67.5)
    assert board == ScoreOverlay(2, 1, flip_seconds=18.0, flip_team="white")


def test_rebase_flip_before_window_starts_on_the_post_flip_score():
    white = ScoreOverlay(2, 1, flip_seconds=5.0, flip_team="white").rebased(48.0, 62.0)
    black = ScoreOverlay(2, 1, flip_seconds=5.0, flip_team="black").rebased(48.0, 62.0)
    assert (white.white, white.black, white.flip_seconds) == (3, 1, None)
    assert (black.white, black.black, black.flip_seconds) == (2, 2, None)
    assert white.digits_before == "3 - 1" and black.digits_before == "2 - 2"


def test_rebase_flip_after_window_keeps_the_start_score_and_no_flip():
    board = ScoreOverlay(2, 1, flip_seconds=40.0, flip_team="white").rebased(10.0, 35.0)
    assert board == ScoreOverlay(2, 1)


def test_rebase_without_a_flip_is_the_same_score():
    assert ScoreOverlay(4, 3).rebased(10.0, 20.0) == ScoreOverlay(4, 3)


# --- slow-mo with trim -------------------------------------------------------------


def test_trimmed_slomo_is_centred_on_n_not_on_the_sheet_dot():
    row = _row(length=100.0, center="9.0")
    pick = resolve_picks(parse_picks("r04 trim=58 slomo"), [row])[0]
    assert pick.slomo == SlowMo(54.5, 59.0, 0.25)  # original review-clip timeline


def test_trimmed_slomo_needs_no_sheet_centre():
    pick = resolve_picks(parse_picks("r04 trim=58 slomo"), [_row(length=100.0, center="")])[0]
    assert pick.slomo == SlowMo(54.5, 59.0, 0.25)


def test_untrimmed_slomo_still_uses_the_sheet_centre():
    pick = resolve_picks(parse_picks("r04 slomo"), [_row(length=100.0, center="9.0")])[0]
    assert pick.slomo == SlowMo(5.5, 10.0, 0.25)


def test_trimmed_slomo_default_is_clamped_to_the_clip():
    pick = resolve_picks(parse_picks("r04 trim=1 slomo"), [_row(length=100.0)])[0]
    assert pick.slomo == SlowMo(0.0, 2.0, 0.25)


def test_explicit_slomo_window_is_on_the_original_timeline_and_may_lie_outside_the_trim():
    pick = resolve_picks(parse_picks("r04 trim=58 slomo=10-13"), [_row(length=100.0)])[0]
    assert pick.slomo == SlowMo(10.0, 13.0, 0.25)
    assert pick.trim is not None and pick.trim.start_seconds == 48.0


def test_explicit_slomo_window_is_still_validated_against_the_clip():
    with pytest.raises(PickError) as exc:
        resolve_picks(parse_picks("r04 trim=58 slomo=90-110"), [_row(length=100.0)])
    assert "outside" in str(exc.value)


# --- export-picks plumbing ------------------------------------------------------------


def test_export_picks_cuts_the_trimmed_interval_and_rebases_the_score(monkeypatch, tmp_path):
    chunk = Chunk(
        sequence=1,
        start_time=datetime(2026, 10, 18, 9, 51, 48),
        mp4_path=Path("DJI_1.MP4"),
        lrf_path=None,
        duration_seconds=5000.0,
        global_start_seconds=0.0,
    )
    sheet = tmp_path / "sheet.csv"
    row = _row(start=1000.0, length=100.0, flip="66.0", team="white", center="")
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    calls: list[tuple] = []
    monkeypatch.setattr(cli, "discover_chunks", lambda source_dir: [chunk])
    monkeypatch.setattr(render, "render_export_clip", lambda *a, **k: calls.append(a))

    cli.cmd_export_picks(Config(), str(sheet), "R04 trim=58, slomo", str(tmp_path / "out"), None)

    ((slices, out_path, _tmp, _cfg, live_score, _crop, replay, full_score),) = calls
    assert [(s.local_start_seconds, s.local_end_seconds) for s in slices] == [(1048.0, 1067.5)]
    assert live_score == ScoreOverlay(2, 1, flip_seconds=18.0, flip_team="white")
    assert full_score == ScoreOverlay(2, 1, flip_seconds=66.0, flip_team="white")
    # Replay: window [54.5, 59] on the original timeline, mapped from the full row interval.
    (piece,) = render.replay_pieces(replay)
    assert (piece.start, piece.duration, piece.clip_offset) == (1054.5, 4.5, 54.5)


def test_export_picks_without_trim_passes_the_whole_row_interval(monkeypatch, tmp_path):
    chunk = Chunk(
        sequence=1,
        start_time=datetime(2026, 10, 18, 9, 51, 48),
        mp4_path=Path("DJI_1.MP4"),
        lrf_path=None,
        duration_seconds=5000.0,
        global_start_seconds=0.0,
    )
    sheet = tmp_path / "sheet.csv"
    row = _row(start=1000.0, length=100.0, flip="66.0", team="white")
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    calls: list[tuple] = []
    monkeypatch.setattr(cli, "discover_chunks", lambda source_dir: [chunk])
    monkeypatch.setattr(render, "render_export_clip", lambda *a, **k: calls.append(a))

    cli.cmd_export_picks(Config(), str(sheet), "r04", str(tmp_path / "out"), None)

    ((slices, _out, _tmp, _cfg, live_score, _crop, replay, full_score),) = calls
    assert [(s.local_start_seconds, s.local_end_seconds) for s in slices] == [(1000.0, 1100.0)]
    assert live_score == full_score == ScoreOverlay(2, 1, flip_seconds=66.0, flip_team="white")
    assert replay is None
