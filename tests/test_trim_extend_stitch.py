"""`trim=A-B`, `extend=N` and `stitch=A-B@CROP[@S]` picks (2026-11-20): grammar,
validation, window maths, the replay/filter-graph arguments and the export-picks /
seg_render plumbing. ffmpeg itself is never run here (see the smoke test in the
session notes); `_run_ffmpeg` is replaced by a recorder."""

import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from soccer_highlights import cli, render, scoreboard
from soccer_highlights.config import Config, CropConfig, ExportConfig, TrimConfig
from soccer_highlights.crop import ALL_REGIONS
from soccer_highlights.discovery import Chunk
from soccer_highlights.picks import (
    CROP_CHOICES,
    MAX_EXTEND_SECONDS,
    PickError,
    PickSpec,
    SlowMo,
    Stitch,
    check_crops,
    explicit_trim_window,
    load_picks,
    parse_picks,
    resolve_picks,
    trim_window,
)
from soccer_highlights.render import AudioParams, ScoreOverlay
from soccer_highlights.timeline import Interval

BOXES = {"near": [0.0, 0.2, 0.5, 0.5], "far": [0.5, 0.25, 0.4, 0.4], "wide": [0.0, 0.0, 1.0, 1.0]}


def _row(
    clip_file: str = "r55_s5_746a_c032.mp4",
    start: float = 1000.0,
    length: float = 40.0,
    flip: str = "",
    team: str = "",
    center: str = "",
    region: str = "wide",
) -> dict[str, str]:
    return {
        "clip_file": clip_file,
        "start_seconds": str(start),
        "end_seconds": str(start + length),
        "action_region": region,
        "slomo_center_s": center,
        "score_white": "2",
        "score_black": "1",
        "score_flip_seconds": flip,
        "score_flip_team": team,
    }


def _chunk(seq: int, global_start: float, duration: float) -> Chunk:
    return Chunk(
        sequence=seq,
        start_time=datetime(2026, 11, 15, 9, 44, 50),
        mp4_path=Path(f"DJI_{seq}.MP4"),
        lrf_path=None,
        duration_seconds=duration,
        global_start_seconds=global_start,
    )


# --- grammar -------------------------------------------------------------------


def test_trim_a_b_parses_as_a_range_and_trim_n_is_unchanged():
    assert parse_picks("r01 trim=20-45")[0] == PickSpec(ref="r01", trim_range=(20.0, 45.0))
    assert parse_picks("r01 trim=0-25.5")[0] == PickSpec(ref="r01", trim_range=(0.0, 25.5))
    assert parse_picks("r01 trim=58")[0] == PickSpec(ref="r01", trim=58.0)
    assert parse_picks("r01 slomo trim=0-25")[0] == PickSpec(ref="r01", slomo=True, trim_range=(0.0, 25.0))


def test_extend_parses():
    assert parse_picks("r23 extend=6")[0] == PickSpec(ref="r23", extend=6.0)
    assert parse_picks("r23 EXTEND=2.5, trim=3-9")[0] == PickSpec(ref="r23", extend=2.5, trim_range=(3.0, 9.0))
    assert parse_picks(f"r23 extend={MAX_EXTEND_SECONDS:g}")[0].extend == MAX_EXTEND_SECONDS


def test_stitch_parses_with_and_without_speed():
    assert parse_picks("r55 stitch=20-26@far")[0] == PickSpec(ref="r55", stitch=Stitch(20.0, 26.0, "far", 1.0))
    assert parse_picks("r55 stitch=20-26@far@0.5")[0] == PickSpec(ref="r55", stitch=Stitch(20.0, 26.0, "far", 0.5))
    assert parse_picks("R55, STITCH=20.5-26@Left@1")[0].stitch == Stitch(20.5, 26.0, "left", 1.0)
    for crop in CROP_CHOICES:
        assert parse_picks(f"r55 stitch=1-2@{crop}")[0].stitch.crop == crop
    assert set(CROP_CHOICES) == {*ALL_REGIONS, "full"}


def test_tokens_combine_in_any_order():
    a = parse_picks("r55 crop=wide trim=5-35 extend=4 stitch=20-26@far@0.5")
    b = parse_picks("r55 stitch=20-26@far@0.5, extend=4 trim=5-35 crop=wide")
    assert a == b
    assert a[0] == PickSpec(
        ref="r55", crop="wide", trim_range=(5.0, 35.0), extend=4.0, stitch=Stitch(20.0, 26.0, "far", 0.5)
    )


@pytest.mark.parametrize(
    "line, fragment",
    [
        ("r01 trim=20-10", "must have A < B"),
        ("r01 trim=10-10", "must have A < B"),
        ("r01 trim=-5-10", "bad trim token"),
        ("r01 trim=5-", "bad trim token"),
        ("r01 trim=5-9 trim=3", "trim given twice"),
        ("r01 trim=3 trim=5-9", "trim given twice"),
        ("r01 trim=0", "greater than 0"),
        ("r23 extend=0", "greater than 0"),
        ("r23 extend=30.5", "over the 30s cap"),
        ("r23 extend=31", "over the 30s cap"),
        ("r23 extend", "bad extend token"),
        ("r23 extend=-2", "bad extend token"),
        ("r23 extend=2 extend=3", "extend given twice"),
        ("r55 stitch=20-26", "bad stitch token"),
        ("r55 stitch=20-26@", "bad stitch token"),
        ("r55 stitch=20-26@top", "bad stitch crop"),
        ("r55 stitch=26-20@far", "must have A < B"),
        ("r55 stitch=20-20@far", "must have A < B"),
        ("r55 stitch=20-26@far@0", "greater than 0 and at most 1"),
        ("r55 stitch=20-26@far@1.5", "greater than 0 and at most 1"),
        ("r55 stitch=20-26@far@abc", "bad stitch token"),
        ("r55 stitch=1-2@far stitch=3-4@near", "stitch given twice"),
        ("r55 stitch=20-26@far slomo", "stitch and slomo cannot be combined yet"),
        ("r55 slomo=1-2 stitch=20-26@far", "stitch and slomo cannot be combined yet"),
        ("extend=6", "start with a rank"),
        ("stitch=1-2@far", "start with a rank"),
    ],
)
def test_bad_new_tokens_are_rejected(line, fragment):
    with pytest.raises(PickError) as exc:
        parse_picks(line)
    assert fragment in str(exc.value)


def test_all_errors_are_reported_together():
    rows = [_row("r55_x.mp4", length=40.0)]
    text = "r55 trim=20-10\nr55 extend=99\nr55 stitch=1-2@far slomo\nr56\nr55 stitch=30-45@far"
    with pytest.raises(PickError) as exc:
        load_picks(text, rows)
    messages = "\n".join(exc.value.errors)
    assert len(exc.value.errors) == 5
    for fragment in ("A < B", "cap", "cannot be combined", "no clip", "outside the 40.0s clip"):
        assert fragment in messages


# --- explicit trim window maths -----------------------------------------------------


def test_explicit_window_is_exactly_a_b():
    w = explicit_trim_window(10.0, 25.0, 40.0, None)
    assert (w.start_seconds, w.end_seconds, w.warning) == (10.0, 25.0, None)


def test_explicit_window_is_never_stretched_to_the_score_tap():
    """trim=N would stretch [48, 62] to the tap at 66; trim=A-B must not."""
    assert trim_window(58.0, 100.0, 66.0, TrimConfig()).end_seconds == 67.5
    w = explicit_trim_window(48.0, 62.0, 100.0, 66.0)
    assert (w.start_seconds, w.end_seconds) == (48.0, 62.0)
    assert w.warning is not None and "will not flip" in w.warning and "66" in w.warning


def test_flip_inside_or_before_the_explicit_window_is_quiet():
    assert explicit_trim_window(10.0, 25.0, 40.0, 20.0).warning is None
    assert explicit_trim_window(10.0, 25.0, 40.0, 3.0).warning is None


def test_flip_exactly_at_b_warns_because_it_is_not_on_screen():
    assert explicit_trim_window(10.0, 25.0, 40.0, 25.0).warning is not None


def test_end_past_the_clip_is_clamped_with_a_warning():
    w = explicit_trim_window(10.0, 55.0, 40.0, None)
    assert (w.start_seconds, w.end_seconds) == (10.0, 40.0)
    assert w.warning is not None and "clamped" in w.warning and "extend" in w.warning


def test_clamp_and_flip_warnings_are_both_kept():
    w = explicit_trim_window(10.0, 55.0, 40.0, 45.0)
    assert w.warning is not None and "clamped" in w.warning and "will not flip" in w.warning


def test_start_at_or_after_the_end_of_the_clip_is_an_error():
    with pytest.raises(ValueError, match="starts at or after"):
        explicit_trim_window(40.0, 50.0, 40.0, None)


def test_extension_covering_the_flip_keeps_the_warning_quiet():
    assert explicit_trim_window(10.0, 25.0, 40.0, 28.0, extend_s=6.0).warning is None
    assert explicit_trim_window(10.0, 25.0, 40.0, 33.0, extend_s=6.0).warning is not None
    assert trim_window(20.0, 100.0, 40.0, TrimConfig(), extend_s=4.0).warning is not None
    assert trim_window(20.0, 100.0, 40.0, TrimConfig(), extend_s=6.0).warning is None


def test_resolve_attaches_the_explicit_window_and_its_warning():
    row = _row(length=80.0, flip="66.0", team="white")
    pick = resolve_picks(parse_picks("r55 trim=48-62"), [row])[0]
    assert pick.trim is not None and (pick.trim.start_seconds, pick.trim.end_seconds) == (48.0, 62.0)
    assert pick.trim.warning is not None
    assert pick.export_seconds() == (1048.0, 1062.0)
    assert "trim=48-62" in pick.describe()


def test_resolve_rejects_an_explicit_start_outside_the_clip():
    with pytest.raises(PickError) as exc:
        resolve_picks(parse_picks("r55 trim=45-50"), [_row(length=40.0)])
    assert "starts at or after the end of the 40.0s clip" in str(exc.value)


def test_resolve_clamps_an_explicit_end():
    pick = resolve_picks(parse_picks("r55 trim=10-90"), [_row(length=40.0)])[0]
    assert pick.export_seconds() == (1010.0, 1040.0)
    assert pick.trim is not None and "clamped" in (pick.trim.warning or "")


# --- extend --------------------------------------------------------------------------


def test_extend_alone_adds_seconds_after_the_whole_review_clip():
    pick = resolve_picks(parse_picks("r55 extend=6"), [_row(length=40.0)])[0]
    assert pick.trim is None
    assert pick.export_seconds() == (1000.0, 1046.0)
    assert pick.live_window() is None
    assert "extend=6" in pick.describe()


def test_extend_follows_a_trim_window():
    pick = resolve_picks(parse_picks("r55 trim=10-25 extend=6"), [_row(length=40.0)])[0]
    assert pick.export_seconds() == (1010.0, 1031.0)
    assert pick.live_window() == (10.0, 31.0)


def test_picks_without_new_tokens_resolve_exactly_as_before():
    pick = resolve_picks(parse_picks("r55"), [_row(length=40.0)])[0]
    assert (pick.extend_seconds, pick.stitch, pick.trim) == (0.0, None, None)
    assert pick.export_seconds() == (1000.0, 1040.0)
    assert pick.describe() == "crop=wide"


def test_score_flip_inside_the_extension_is_still_on_screen():
    """Explicit trim ends at 25 but the 6s extension reaches 31: a flip at 28 is shown."""
    pick = resolve_picks(parse_picks("r55 trim=10-25 extend=6"), [_row(length=40.0, flip="28.0", team="white")])[0]
    score = ScoreOverlay.from_sheet_row(pick.row)
    assert score is not None and pick.live_window() is not None
    assert score.rebased(*pick.live_window()) == ScoreOverlay(2, 1, flip_seconds=18.0, flip_team="white")
    assert score.rebased(10.0, 25.0) == ScoreOverlay(2, 1)  # without the extension it would be lost


def test_extend_across_a_chunk_boundary_splits_per_chunk_and_the_clock_follows():
    from soccer_highlights.discovery import slice_start_epoch
    from soccer_highlights.timeline import map_interval_to_chunks

    chunks = [_chunk(1, 0.0, 1040.0), _chunk(2, 1040.0, 500.0)]
    pick = resolve_picks(parse_picks("r55 extend=20"), [_row(length=30.0)])[0]
    start, end = pick.export_seconds()
    interval = Interval(start, end)
    slices = map_interval_to_chunks(interval, chunks)

    assert [(s.chunk.sequence, s.local_start_seconds, s.local_end_seconds) for s in slices] == [(1, 1000.0, 1040.0), (2, 0.0, 10.0)]
    assert render.footage_shortfall(interval, slices) == 0.0
    # Each side of the boundary gets its own chunk's wall-clock epoch.
    assert slice_start_epoch(slices[1].chunk, 0.0) != slice_start_epoch(slices[0].chunk, 1000.0)
    # Footage that ends early is reported, not hidden.
    short = map_interval_to_chunks(interval, [_chunk(1, 0.0, 1040.0)])
    assert render.footage_shortfall(interval, short) == pytest.approx(10.0)


# --- stitch / slomo-with-trim resolution ----------------------------------------------------


def test_stitch_resolves_and_uses_the_row_crop_for_the_live_part():
    pick = resolve_picks(parse_picks("r55 stitch=20-26@far@0.5"), [_row(length=40.0, region="wide")])[0]
    assert pick.region == "wide"
    assert pick.stitch == Stitch(20.0, 26.0, "far", 0.5)
    assert pick.slomo is None
    assert pick.describe() == "crop=wide stitch=20-26@far@0.5"
    assert resolve_picks(parse_picks("r55 stitch=20-26@far"), [_row(length=40.0)])[0].describe() == "crop=wide stitch=20-26@far"


@pytest.mark.parametrize("line, fragment", [("r55 stitch=30-45@far", "outside the 40.0s clip"), ("r55 stitch=5-5.1@far", "shorter than")])
def test_stitch_window_must_be_inside_the_clip(line, fragment):
    with pytest.raises(PickError) as exc:
        resolve_picks(parse_picks(line), [_row(length=40.0)])
    assert fragment in str(exc.value)


def test_slomo_with_explicit_trim_defaults_to_the_sheet_dot_when_inside_the_window():
    pick = resolve_picks(parse_picks("r55 slomo trim=0-25"), [_row(length=40.0, center="12.0")])[0]
    assert pick.slomo == SlowMo(8.5, 13.0, 0.25)


@pytest.mark.parametrize("center", ["30.0", ""])
def test_slomo_with_explicit_trim_fails_when_the_dot_is_outside_or_missing(center):
    with pytest.raises(PickError) as exc:
        resolve_picks(parse_picks("r55 slomo trim=0-25"), [_row(length=40.0, center=center)])
    assert "slomo=A-B" in str(exc.value) and "trim=0-25" in str(exc.value)


def test_slomo_with_explicit_trim_and_explicit_window_needs_no_dot():
    pick = resolve_picks(parse_picks("r55 slomo=30-33 trim=0-25"), [_row(length=40.0, center="")])[0]
    assert pick.slomo == SlowMo(30.0, 33.0, 0.25)


def test_trim_n_slomo_still_centres_on_n():
    pick = resolve_picks(parse_picks("r55 trim=30 slomo"), [_row(length=40.0, center="5.0")])[0]
    assert pick.slomo == SlowMo(26.5, 31.0, 0.25)


def test_check_crops_resolves_both_the_live_and_the_stitch_crop():
    pick = resolve_picks(parse_picks("r55 stitch=20-26@far"), [_row(length=40.0, region="wide")])[0]
    check_crops(pick, CropConfig(boxes=BOXES))
    with pytest.raises(ValueError, match="'far'"):
        check_crops(pick, CropConfig(boxes={"wide": BOXES["wide"]}))


# --- replay mapping and filter graph ------------------------------------------------------------


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    recorded: list[list[str]] = []
    monkeypatch.setattr(render, "_run_ffmpeg", lambda args, timeout=None: recorded.append(list(args)))
    monkeypatch.setattr(render, "probe_audio", lambda path: AudioParams(48000, "stereo"))
    return recorded


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _stitch_pieces(line: str, crop_cfg: CropConfig, chunks: list[Chunk] | None = None):
    pick = resolve_picks(parse_picks(line), [_row(length=40.0)])[0]
    full = Interval(1000.0, 1040.0)
    replay = render.build_pick_replay(pick, full, chunks or [_chunk(1, 0.0, 5000.0)], crop_cfg)
    assert replay is not None
    return pick, replay, render.replay_pieces(replay)


def test_stitch_replay_is_the_review_window_on_the_source_with_its_own_crop():
    cfg = CropConfig(boxes=BOXES)
    _, replay, (piece,) = _stitch_pieces("r55 stitch=20-26@far@0.5", cfg)

    assert piece.start == pytest.approx(1020.0)
    assert piece.duration == pytest.approx(6.0)
    assert piece.clip_offset == pytest.approx(20.0)  # score / clock keyed on the original timeline
    assert piece.speed == 0.5 and piece.output_seconds == pytest.approx(12.0)
    assert piece.own_crop and piece.crop == pytest.approx(tuple(BOXES["far"]))
    assert piece.show_speed is True
    assert replay.own_crop


def test_stitch_crop_is_resolved_through_the_config_boxes():
    moved = CropConfig(boxes={**BOXES, "far": [0.1, 0.1, 0.3, 0.3]})
    _, _, (piece,) = _stitch_pieces("r55 stitch=20-26@far", moved)
    assert piece.crop == pytest.approx((0.1, 0.1, 0.3, 0.3))
    _, _, (full_piece,) = _stitch_pieces("r55 stitch=20-26@full", moved)
    assert full_piece.own_crop and full_piece.crop is None


def test_stitch_at_normal_speed_hides_the_speed_in_the_badge():
    _, _, (normal,) = _stitch_pieces("r55 stitch=20-26@far", CropConfig(boxes=BOXES))
    _, _, (slowed,) = _stitch_pieces("r55 stitch=20-26@far@0.5", CropConfig(boxes=BOXES))
    assert normal.speed == 1.0 and normal.show_speed is False
    assert slowed.show_speed is True


def test_stitch_window_straddling_a_chunk_boundary_splits_into_two_pieces():
    chunks = [_chunk(1, 0.0, 1023.0), _chunk(2, 1023.0, 500.0)]
    _, _, (first, second) = _stitch_pieces("r55 stitch=20-26@far", CropConfig(boxes=BOXES), chunks)
    assert (first.duration, second.duration) == (pytest.approx(3.0), pytest.approx(3.0))
    assert (first.clip_offset, second.clip_offset) == (pytest.approx(20.0), pytest.approx(23.0))
    assert first.own_crop and second.own_crop and first.crop == second.crop


def test_slomo_replay_pieces_still_use_the_live_crop():
    pick = resolve_picks(parse_picks("r55 slomo=5-8"), [_row(length=40.0)])[0]
    replay = render.build_pick_replay(pick, Interval(1000.0, 1040.0), [_chunk(1, 0.0, 5000.0)], CropConfig())
    assert replay is not None
    (piece,) = render.replay_pieces(replay)
    assert (piece.own_crop, piece.crop, piece.show_speed) == (False, None, True)


def test_no_replay_for_a_pick_without_slomo_or_stitch():
    pick = resolve_picks(parse_picks("r55 extend=3"), [_row(length=40.0)])[0]
    assert render.build_pick_replay(pick, Interval(1000.0, 1040.0), [_chunk(1, 0.0, 5000.0)], CropConfig()) is None


def test_stitch_with_no_footage_for_the_window_fails_clearly():
    pick = resolve_picks(parse_picks("r55 stitch=20-26@far"), [_row(length=40.0)])[0]
    with pytest.raises(ValueError, match="stitch window 20-26s maps onto no source footage"):
        render.build_pick_replay(pick, Interval(1000.0, 1040.0), [_chunk(1, 0.0, 10.0)], CropConfig(boxes=BOXES))


BOARD = ScoreOverlay(2, 1, flip_seconds=22.0, flip_team="white")


def _encode_stitch(calls, line: str, score=BOARD, **kw):
    crop_cfg = CropConfig(boxes=BOXES)
    _, _, (piece,) = _stitch_pieces(line, crop_cfg)
    render.encode_replay_piece(piece, Path("o.mp4"), ExportConfig(**kw), score, crop=(0.0, 0.0, 1.0, 1.0))
    return calls[-1], piece


def test_stitch_filter_graph_has_the_stitch_crop_speed_score_and_muted_audio_when_slowed(calls):
    args, piece = _encode_stitch(calls, "r55 stitch=20-26@far@0.5")
    fc = _arg(args, "-filter_complex")

    # The stitch's own box (far), not the live crop passed to encode_replay_piece.
    assert "crop=w=round(iw*0.400000/2)*2:h=round(ih*0.400000/2)*2:x=round(iw*0.500000/2)*2:y=round(ih*0.250000/2)*2" in fc
    assert "iw*1.000000" not in fc
    # True-frame stretch by 1/speed, then fps; overlays are drawn before it.
    assert fc.index("drawtext=") < fc.index("setpts=(PTS-STARTPTS)/0.5") < fc.index("fps=30.0")
    # Score as of the replay's source moment: flip at clip t=22 is 2s into a window starting at 20.
    assert "enable='lt(t,2.000)'" in fc and "enable='gte(t,2.000)'" in fc
    # Slowed (0.5x): muted -- silent AAC in the live part's layout, source audio never mapped.
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in args
    assert "0:a?" not in args and "0:a" not in args and "2:a" in args
    assert _arg(args, "-c:a") == "aac"
    # 6s of source at 0.5x.
    assert _arg(args, "-t") == "12.000"
    # REPLAY badge: blink overlay present, speed printed (the PNG that carries it).
    assert "enable='lt(mod(t*1.25,1),0.6)'" in fc
    badge_png = scoreboard.replay_badge_pngs(ExportConfig(), 2560, 0.5)[0]
    assert str(badge_png) in args


def test_stitch_at_normal_speed_uses_the_speedless_badge_and_unstretched_pts(calls):
    args, piece = _encode_stitch(calls, "r55 stitch=20-26@far")
    fc = _arg(args, "-filter_complex")

    assert "setpts=(PTS-STARTPTS)/1," in fc
    assert _arg(args, "-t") == "6.000"
    # Normal speed keeps the source audio (Kaveh, 2026-11-21): no silent track.
    assert "anullsrc" not in " ".join(args) and "0:a?" in args and "2:a" not in args
    no_speed_png, _, no_speed_layout = scoreboard.replay_badge_pngs(ExportConfig(), 2560, 1.0, False)
    with_speed_png, _, with_speed_layout = scoreboard.replay_badge_pngs(ExportConfig(), 2560, 1.0, True)
    assert str(no_speed_png) in args and str(with_speed_png) not in args
    assert no_speed_layout.width < with_speed_layout.width
    assert no_speed_layout.height == with_speed_layout.height


def test_stitch_without_a_scoreboard_is_the_clock_only_branch_with_its_crop(calls):
    args, _ = _encode_stitch(calls, "r55 stitch=20-26@far@0.5", score=None)
    vf = _arg(args, "-vf")
    assert vf.startswith("crop=") and "iw*0.400000" in vf
    assert vf.index("drawtext=") < vf.index("setpts=") and vf.endswith("fps=30.0")
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in args and "1:a" in args


def test_stitch_at_normal_speed_without_a_scoreboard_maps_the_source_audio(calls):
    args, _ = _encode_stitch(calls, "r55 stitch=20-26@far", score=None)
    assert "anullsrc" not in " ".join(args) and "0:a?" in args and "1:a" not in args


def test_stitch_full_crop_adds_no_crop_filter(calls):
    args, _ = _encode_stitch(calls, "r55 stitch=20-26@full", score=None)
    assert "crop=" not in _arg(args, "-vf")


def test_slomo_replay_badge_args_are_unchanged_by_the_stitch_work(calls):
    """A slow-mo piece still passes the default (speed-showing) badge."""
    pick = resolve_picks(parse_picks("r55 slomo=5-8"), [_row(length=40.0)])[0]
    replay = render.build_pick_replay(pick, Interval(1000.0, 1040.0), [_chunk(1, 0.0, 5000.0)], CropConfig())
    (piece,) = render.replay_pieces(replay)
    render.encode_replay_piece(piece, Path("o.mp4"), ExportConfig(), BOARD, None)
    assert str(scoreboard.replay_badge_pngs(ExportConfig(), 2560, 0.25)[0]) in calls[-1]


def test_render_export_clip_appends_a_stitch_with_its_own_crop_and_the_live_crop_for_the_live_part(monkeypatch, tmp_path):
    chunk = _chunk(1, 0.0, 5000.0)
    encoded: list[tuple] = []
    concatenated: list[list[Path]] = []

    def fake_encode(source, start, duration, out, cfg, epoch=None, score=None, piece_start=0.0, crop=None, slomo_speed=None, show_speed=True):
        encoded.append((start, duration, piece_start, crop, slomo_speed, show_speed))

    monkeypatch.setattr(render, "_encode_piece", fake_encode)
    monkeypatch.setattr(render, "concat_clips", lambda parts, out, force_reencode=False: concatenated.append(list(parts)))
    cfg = CropConfig(boxes=BOXES)
    pick = resolve_picks(parse_picks("r55 stitch=20-26@far"), [_row(length=40.0)])[0]
    replay = render.build_pick_replay(pick, Interval(1000.0, 1040.0), [chunk], cfg)
    live = (0.0, 0.2, 0.5, 0.5)
    start, end = pick.export_seconds()
    from soccer_highlights.timeline import map_interval_to_chunks

    render.render_export_clip(
        map_interval_to_chunks(Interval(start, end), [chunk]), tmp_path / "r55.mp4", tmp_path, ExportConfig(), None, live, replay
    )

    assert encoded[0] == (1000.0, 40.0, 0.0, live, None, True)
    assert encoded[1] == (pytest.approx(1020.0), pytest.approx(6.0), pytest.approx(20.0), pytest.approx(tuple(BOXES["far"])), 1.0, False)
    assert [p.name for p in concatenated[0]] == ["r55_part0.mp4", "r55_replay0.mp4"]


# --- export-picks plumbing -----------------------------------------------------------------------------


def _write_sheet(path: Path, row: dict[str, str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _run_export(monkeypatch, tmp_path, line: str, row: dict[str, str], chunks: list[Chunk], cfg: Config | None = None):
    sheet = tmp_path / "sheet.csv"
    _write_sheet(sheet, row)
    calls: list[tuple] = []
    monkeypatch.setattr(cli, "discover_chunks", lambda source_dir: chunks)
    monkeypatch.setattr(render, "render_export_clip", lambda *a, **k: calls.append(a))
    if cfg is None:
        cfg = Config()
        cfg.crop.boxes = dict(BOXES)
    cli.cmd_export_picks(cfg, str(sheet), line, str(tmp_path / "out"), None)
    return calls


def test_export_picks_explicit_trim_cuts_exactly_a_b_and_warns_about_the_late_tap(monkeypatch, tmp_path, capsys):
    row = _row(start=1000.0, length=80.0, flip="66.0", team="white")
    ((slices, _out, _tmp, _cfg, live_score, _crop, replay, full_score),) = _run_export(
        monkeypatch, tmp_path, "r55 trim=48-62", row, [_chunk(1, 0.0, 5000.0)]
    )

    assert [(s.local_start_seconds, s.local_end_seconds) for s in slices] == [(1048.0, 1062.0)]
    assert live_score == ScoreOverlay(2, 1)  # flip at 66 is after B: the counter never changes
    assert full_score == ScoreOverlay(2, 1, flip_seconds=66.0, flip_team="white")
    assert replay is None
    out = capsys.readouterr().out
    assert "!! WARNING" in out and "will not flip" in out


def test_export_picks_extend_runs_into_the_next_chunk(monkeypatch, tmp_path, capsys):
    row = _row(start=1000.0, length=30.0, flip="25.0", team="black")
    chunks = [_chunk(1, 0.0, 1040.0), _chunk(2, 1040.0, 500.0)]
    ((slices, _out, _tmp, _cfg, live_score, _crop, replay, _full),) = _run_export(monkeypatch, tmp_path, "r55 extend=20", row, chunks)
    assert [(s.chunk.sequence, s.local_start_seconds, s.local_end_seconds) for s in slices] == [(1, 1000.0, 1040.0), (2, 0.0, 10.0)]
    assert live_score == ScoreOverlay(2, 1, flip_seconds=25.0, flip_team="black")
    assert replay is None
    assert "WARNING" not in capsys.readouterr().out


def test_export_picks_warns_when_extend_runs_past_the_footage(monkeypatch, tmp_path, capsys):
    _run_export(monkeypatch, tmp_path, "r55 extend=20", _row(start=1000.0, length=30.0), [_chunk(1, 0.0, 1040.0)])
    assert "extend=20 runs past the available source footage" in capsys.readouterr().out


def test_export_picks_stitch_builds_a_replay_with_its_own_box_and_the_unrebased_score(monkeypatch, tmp_path):
    cfg = Config()
    cfg.crop.boxes = dict(BOXES)
    row = _row(start=1000.0, length=40.0, flip="22.0", team="white", region="wide")
    ((slices, _out, _tmp, _cfg, live_score, crop, replay, full_score),) = _run_export(
        monkeypatch, tmp_path, "r55 stitch=20-26@far@0.5", row, [_chunk(1, 0.0, 5000.0)], cfg
    )

    assert [(s.local_start_seconds, s.local_end_seconds) for s in slices] == [(1000.0, 1040.0)]
    assert crop == pytest.approx(tuple(BOXES["wide"]))
    (piece,) = render.replay_pieces(replay)
    assert piece.crop == pytest.approx(tuple(BOXES["far"])) and piece.own_crop and piece.speed == 0.5
    assert (piece.start, piece.duration, piece.clip_offset) == (1020.0, 6.0, 20.0)
    assert full_score == ScoreOverlay(2, 1, flip_seconds=22.0, flip_team="white")
    assert live_score == full_score


def test_export_picks_reports_an_unconfigured_stitch_box_before_rendering(monkeypatch, tmp_path):
    cfg = Config()
    cfg.crop.boxes = {"wide": BOXES["wide"]}
    with pytest.raises(SystemExit) as exc:
        _run_export(monkeypatch, tmp_path, "r55 stitch=20-26@far", _row(), [_chunk(1, 0.0, 5000.0)], cfg)
    assert "nothing rendered" in str(exc.value) and "'far'" in str(exc.value)


def test_export_picks_reports_every_new_validation_error_together(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run_export(
            monkeypatch, tmp_path, "r55 trim=20-10\nr55 stitch=1-2@far slomo\nr55 extend=31", _row(), [_chunk(1, 0.0, 5000.0)]
        )
    text = str(exc.value)
    assert "A < B" in text and "cannot be combined" in text and "cap" in text


# --- seg_render plumbing --------------------------------------------------------------------------------


@pytest.fixture
def seg_render():
    path = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "kaveh_soccer_highlights" / "scripts" / "seg_render.py"
    spec = importlib.util.spec_from_file_location("seg_render_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _run_seg(seg_render, monkeypatch, tmp_path, pick: str, row: dict[str, str], chunks: list[Chunk], max_segments: int = 99):
    sheet = tmp_path / "sheet.csv"
    _write_sheet(sheet, row)
    cfg = Config()
    cfg.crop.boxes = dict(BOXES)
    encoded: list[dict] = []

    def fake_encode(source, start, duration, out, cfg_, epoch=None, score=None, piece_start=0.0, crop=None, slomo_speed=None, show_speed=True):
        encoded.append(
            {"start": start, "duration": duration, "epoch": epoch, "score": score, "piece_start": piece_start,
             "crop": crop, "speed": slomo_speed, "show_speed": show_speed}
        )
        Path(out).write_bytes(b"x")

    monkeypatch.setattr(seg_render, "load_config", lambda: cfg)
    monkeypatch.setattr(seg_render, "discover_chunks", lambda source_dir: chunks)
    monkeypatch.setattr(render, "_encode_piece", fake_encode)
    monkeypatch.setattr(render, "is_playable", lambda p: Path(p).exists() and Path(p).stat().st_size > 0)
    concatenated: list[list[str]] = []
    monkeypatch.setattr(render, "concat_clips", lambda parts, out, force_reencode=False: (concatenated.append([p.name for p in parts]), Path(out).write_bytes(b"x")))
    monkeypatch.setattr(
        sys, "argv",
        ["seg_render.py", "--source-dir", "unused", "--review-sheet", str(sheet), "--pick", pick, "--crf", "24",
         "--out-dir", str(tmp_path / "out"), "--seg-seconds", "11", "--max-segments", str(max_segments)],
    )
    seg_render.main()
    return encoded, concatenated


def test_seg_render_stitch_is_the_last_resumable_segment_and_is_in_the_plan(seg_render, monkeypatch, tmp_path, capsys):
    row = _row(start=1000.0, length=30.0, flip="22.0", team="white")
    encoded, concatenated = _run_seg(
        seg_render, monkeypatch, tmp_path, "r55 stitch=20-26@far@0.5", row, [_chunk(1, 0.0, 5000.0)], max_segments=3
    )
    # 30s live = 11 + 11 + 8 -> 4th segment (the stitch) is still pending after 3.
    assert [round(e["duration"], 2) for e in encoded] == [11.0, 11.0, 8.0]
    seg_dir = tmp_path / "out" / ".segs_r55_s5_746a_c032"
    plan = json.loads((seg_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["stitch"] == [20.0, 26.0, "far", 0.5, [0.5, 0.25, 0.4, 0.4]]
    assert "extend" not in plan and "interval" not in plan and plan["slomo"] is None

    capsys.readouterr()
    seg_render.sys.argv[seg_render.sys.argv.index("--max-segments") + 1] = "5"
    seg_render.main()
    out = capsys.readouterr().out
    assert "stitch replay 6.00s source -> 12.00s @ 0.5x" in out and "DONE" in out
    replay_call = encoded[-1]
    assert replay_call["speed"] == 0.5 and replay_call["crop"] == pytest.approx(tuple(BOXES["far"]))
    assert replay_call["score"] == ScoreOverlay(2, 1, flip_seconds=22.0, flip_team="white")
    assert replay_call["piece_start"] == 20.0 and replay_call["show_speed"] is True
    assert concatenated == [["seg00.mp4", "seg01.mp4", "seg02.mp4", "seg03.mp4"]]


def test_seg_render_extend_crosses_the_chunk_boundary_and_changes_the_plan(seg_render, monkeypatch, tmp_path):
    row = _row(start=1000.0, length=30.0)
    chunks = [_chunk(1, 0.0, 1040.0), _chunk(2, 1040.0, 500.0)]
    encoded, _ = _run_seg(seg_render, monkeypatch, tmp_path, "r55 extend=20", row, chunks, max_segments=1)

    assert [round(e["duration"], 2) for e in encoded] == [11.0]
    plan = json.loads((tmp_path / "out" / ".segs_r55_s5_746a_c032" / "plan.json").read_text(encoding="utf-8"))
    assert plan["interval"] == [1000.0, 1050.0]
    # Running the rest: 11, 11, 11, 7 (chunk 1) then 10 (chunk 2), epoch per chunk.
    encoded, _ = _run_seg(seg_render, monkeypatch, tmp_path, "r55 extend=20", row, chunks, max_segments=99)
    assert [round(e["duration"], 2) for e in encoded] == [11.0, 11.0, 7.0, 10.0]
    assert encoded[-1]["piece_start"] == pytest.approx(40.0)
    assert encoded[-1]["epoch"] != encoded[0]["epoch"]


def test_seg_render_explicit_trim_interval_and_score_follow_the_window(seg_render, monkeypatch, tmp_path):
    row = _row(start=1000.0, length=80.0, flip="20.0", team="white")
    encoded, _ = _run_seg(seg_render, monkeypatch, tmp_path, "r55 trim=10-25", row, [_chunk(1, 0.0, 5000.0)], max_segments=1)

    assert encoded[0]["start"] == pytest.approx(1010.0) and encoded[0]["duration"] == pytest.approx(11.0)
    assert encoded[0]["score"] == ScoreOverlay(2, 1, flip_seconds=10.0, flip_team="white")
    plan = json.loads((tmp_path / "out" / ".segs_r55_s5_746a_c032" / "plan.json").read_text(encoding="utf-8"))
    assert plan["interval"] == [1010.0, 1025.0]


def test_seg_render_discards_stale_segments_when_the_stitch_changes(seg_render, monkeypatch, tmp_path):
    row = _row(start=1000.0, length=30.0)
    chunks = [_chunk(1, 0.0, 5000.0)]
    _run_seg(seg_render, monkeypatch, tmp_path, "r55 stitch=20-26@far", row, chunks, max_segments=1)
    seg_dir = tmp_path / "out" / ".segs_r55_s5_746a_c032"
    assert (seg_dir / "seg00.mp4").exists()

    sys.argv[sys.argv.index("--pick") + 1] = "r55 stitch=20-26@near"
    # Only the plan write matters here: it must delete the old segments.
    plan = json.loads((seg_dir / "plan.json").read_text(encoding="utf-8"))
    changed = {**plan, "stitch": [20.0, 26.0, "near", 1.0, [0.0, 0.2, 0.5, 0.5]]}
    seg_render._write_plan(seg_dir, changed)
    assert not list(seg_dir.glob("seg*.mp4"))
    assert json.loads((seg_dir / "plan.json").read_text(encoding="utf-8")) == changed


def test_seg_render_rejects_stitch_with_slomo_before_rendering(seg_render, monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run_seg(seg_render, monkeypatch, tmp_path, "r55 stitch=20-26@far slomo", _row(), [_chunk(1, 0.0, 5000.0)])
    assert "cannot be combined" in str(exc.value)
