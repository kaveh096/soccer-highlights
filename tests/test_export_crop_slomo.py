"""Crop + slow-mo replay in the export path (2026-10-18).

ffmpeg itself is never run here: `_run_ffmpeg` is replaced by a recorder and
the assertions are on the argument lists / filter strings, which is where every
past export bug (see _encode_piece's docstring) actually lived."""

import re
from datetime import datetime
from pathlib import Path

import pytest

from soccer_highlights import render
from soccer_highlights.config import ExportConfig
from soccer_highlights.crop import CropConfig, crop_rect
from soccer_highlights.discovery import Chunk
from soccer_highlights.picks import SlowMo, rect_for_region
from soccer_highlights.render import AudioParams, ScoreOverlay
from soccer_highlights.timeline import ChunkSlice, Interval

EPOCH = 1700000000.0


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    recorded: list[list[str]] = []
    monkeypatch.setattr(render, "_run_ffmpeg", lambda args, timeout=None: recorded.append(list(args)))
    monkeypatch.setattr(render, "probe_audio", lambda path: AudioParams(48000, "stereo"))
    return recorded


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _encode(calls, **kw):
    score = kw.pop("score", None)
    render._encode_piece(Path("src.mp4"), 12.5, 5.0, Path("o.mp4"), ExportConfig(), EPOCH, score, **kw)
    return calls[-1]


BOARD = ScoreOverlay(2, 1, flip_seconds=3.0, flip_team="white")


# --- crop geometry -----------------------------------------------------------


@pytest.mark.parametrize("region, x", [("left", 0), ("center", 640), ("right", 1280)])
def test_crop_filter_is_a_2560x1440_window_on_a_4k_source(region, x):
    """Evaluate the emitted expressions on a 3840x2160 frame: the window must
    land on exact even pixels, not 2559.99 truncated to 2559."""
    flt = render._crop_filter(rect_for_region(region, CropConfig()))
    vals = {k: eval(expr, {"round": round}, {"iw": 3840, "ih": 2160}) for k, expr in re.findall(r"([whxy])=([^:]+)", flt)}

    assert vals == {"w": 2560, "h": 1440, "x": x, "y": 360}


def test_crop_rect_semantics_match_the_contract():
    cfg = CropConfig()
    assert crop_rect("left", cfg) == pytest.approx((0, 1 / 6, 2 / 3, 2 / 3))
    assert crop_rect("center", cfg) == pytest.approx((1 / 6, 1 / 6, 2 / 3, 2 / 3))
    assert crop_rect("right", cfg) == pytest.approx((1 / 3, 1 / 6, 2 / 3, 2 / 3))
    assert rect_for_region("full", cfg) is None
    with pytest.raises(ValueError):
        crop_rect("top", cfg)


def test_crop_filter_string_per_region_is_stable():
    assert render._crop_filter(rect_for_region("right", CropConfig())) == (
        "crop=w=round(iw*0.666667/2)*2:h=round(ih*0.666667/2)*2:x=round(iw*0.333333/2)*2:y=round(ih*0.166667/2)*2"
    )


def test_crop_comes_first_then_scale_then_fps_then_the_clock_in_the_clock_only_branch(calls):
    args = _encode(calls, crop=rect_for_region("left", CropConfig()))
    vf = _arg(args, "-vf")

    assert vf.startswith("crop=")
    assert vf.index("crop=") < vf.index("scale=2560:-2") < vf.index("fps=30.0") < vf.index("drawtext=")


def test_crop_comes_first_in_the_scoreboard_branch_and_overlays_stay_after_it(calls):
    args = _encode(calls, score=BOARD, crop=rect_for_region("center", CropConfig()))
    fc = _arg(args, "-filter_complex")

    assert fc.startswith("[0:v]crop=")
    assert fc.index("crop=") < fc.index("scale=2560:-2") < fc.index("fps=30.0") < fc.index("overlay=") < fc.index("drawtext=")


# --- regression guard: nothing changes without crop/slomo -------------------


def _normalised(args: list[str]) -> list[str]:
    return [re.sub(r"[^\s']*soccer_hl_scoreboard_[0-9a-f]+\.png", "<CHROME>", a) for a in args]


def test_live_clock_only_args_are_unchanged_without_crop_or_slomo(calls):
    args = _encode(calls)
    assert args == [
        "-ss", "12.500", "-i", "src.mp4", "-t", "5.000",
        "-vf", "scale=2560:-2,fps=30.0,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='%{pts\\:gmtime\\:1700000000.000\\:%b %d %I.%M.%S %p}':x=24:y=24:fontsize=42:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-threads", "0",
        "-c:a", "aac", "-strict", "-2", "-b:a", "192k", "o.mp4",
    ]  # fmt: skip


def test_live_scoreboard_args_are_unchanged_without_crop_or_slomo(calls):
    args = _normalised(_encode(calls, score=BOARD, piece_start=0.0))
    assert args == [
        "-ss", "12.500", "-i", "src.mp4", "-loop", "1", "-i", "<CHROME>",
        "-filter_complex",
        "[0:v]scale=2560:-2,fps=30.0[v];[v][1:v]overlay=x=28:y=28[bg];[bg]drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='2 - 1':x=344-text_w/2:y=90-text_h/2:fontsize=64:fontcolor=0xF7F9FC:enable='lt(t,3.000)',drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='3 - 1':x=344-text_w/2:y=90-text_h/2:fontsize=64:fontcolor=0xF7F9FC:enable='gte(t,3.000)',drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='%{pts\\:gmtime\\:1700000000.000\\:%I.%M.%S %p}':x=693:y=90-text_h/2:fontsize=46:fontcolor=0xCBD2DC[vout]",
        "-map", "[vout]", "-map", "0:a?", "-t", "5.000",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-threads", "0",
        "-c:a", "aac", "-strict", "-2", "-b:a", "192k", "o.mp4",
    ]  # fmt: skip


def test_full_pick_adds_nothing_to_the_filter(calls):
    """`full` resolves to a None rect; it must be indistinguishable from today."""
    a = _encode(calls)
    b = _encode(calls, crop=rect_for_region("full", CropConfig()))
    assert a == b


# --- slow-mo replay chunk -----------------------------------------------------


def test_replay_stretches_pts_before_fps_and_after_the_overlays_clock_only(calls):
    args = _encode(calls, slomo_speed=0.25, crop=rect_for_region("right", CropConfig()))
    vf = _arg(args, "-vf")

    # fps is last (after the stretch); the clock is drawn on TRUE pts before it.
    assert vf.index("crop=") < vf.index("scale=") < vf.index("drawtext=") < vf.index("setpts=") < vf.index("fps=30.0")
    assert "setpts=(PTS-STARTPTS)/0.25" in vf
    assert vf.endswith("fps=30.0")
    assert "minterpolate" not in vf and "framerate" not in vf
    # 5s of source at 0.25x -> 20s out; -t is an OUTPUT option (after the maps).
    assert _arg(args, "-t") == "20.000"
    assert args.index("-t") > args.index("-map")


def test_replay_stretches_pts_before_fps_and_after_the_overlays_scoreboard(calls):
    args = _encode(calls, score=BOARD, slomo_speed=0.25, crop=rect_for_region("right", CropConfig()))
    fc = _arg(args, "-filter_complex")

    assert fc.startswith("[0:v]crop=")
    assert fc.index("overlay=") < fc.rindex("drawtext=") < fc.index("setpts=") < fc.index("fps=30.0")
    # The REPLAY badge goes on after the stretch and fps, so it is the last stage.
    assert fc.index("fps=30.0") < fc.index("[sv][3:v]overlay=") < fc.index("[sb][4:v]overlay=")
    assert fc.endswith("[vout]")
    assert _arg(args, "-t") == "20.000"
    # -t stays an OUTPUT option (after the filter and maps), never between inputs.
    assert args.index("-t") > args.index("-map")
    assert args[args.index("-loop") + 1] == "1"


def test_clock_is_not_rescaled_for_the_stretch(calls):
    """The clock expression must be identical to the live part's: it is drawn
    from the unstretched pts, so it reads true wall-clock and ticks slowly."""
    live = _arg(_encode(calls, score=BOARD), "-filter_complex")
    replay = _arg(_encode(calls, score=BOARD, slomo_speed=0.25), "-filter_complex")

    clock = re.search(r"drawtext=[^,\[]*pts[^,\[]*", live).group(0)
    assert clock in replay
    assert replay.index(clock) < replay.index("setpts=")


def test_score_flip_is_rebased_onto_the_replay_window(calls):
    """Flip at clip t=7.0; the replay starts at clip t=6.0, so the flip is 1.0s into the replay's TRUE-time axis (the score
    is drawn before the stretch, so no pts maths is needed)."""
    score = ScoreOverlay(2, 1, flip_seconds=7.0, flip_team="white")
    args = _encode(calls, score=score, piece_start=6.0, slomo_speed=0.25)
    fc = _arg(args, "-filter_complex")
    # duration passed is 5.0 source seconds here
    assert "enable='lt(t,1.000)'" in fc and "enable='gte(t,1.000)'" in fc


def test_replay_carries_a_silent_track_matching_the_live_audio(calls):
    args = _encode(calls, slomo_speed=0.25)

    i = args.index("lavfi")
    assert args[i - 1] == "-f"
    assert args[i + 1] == "-i"
    assert args[i + 2] == "anullsrc=channel_layout=stereo:sample_rate=48000"
    # Source audio is never mapped (it is not valid slow-mo audio).
    assert "0:a?" not in args and "0:a" not in args
    assert args[args.index("-map") + 1] == "0:v:0"
    assert "1:a" in args
    assert "-c:a" in args and _arg(args, "-b:a") == "192k"


def test_replay_silent_track_in_the_scoreboard_branch_is_input_two(calls):
    args = _encode(calls, score=BOARD, slomo_speed=0.25)
    # source, chrome, silent track, then the badge and its dot.
    assert args.count("-i") == 5
    assert "2:a" in args and "0:a?" not in args


# --- REPLAY badge (2026-10-19) -------------------------------------------------


def _replay_with(cfg, calls, score=BOARD):
    render._encode_piece(Path("src.mp4"), 12.5, 5.0, Path("o.mp4"), cfg, EPOCH, score, slomo_speed=0.25)
    return calls[-1]


def test_replay_badge_sits_under_the_board_and_the_dot_blinks_in_output_time(calls):
    fc = _arg(_replay_with(ExportConfig(), calls), "-filter_complex")

    # Board at (inset, inset); badge directly under it, left edges aligned.
    board_xy = re.search(r"\[v\]\[1:v\]overlay=x=(\d+):y=(\d+)", fc).groups()
    badge_xy = re.search(r"\[sv\]\[3:v\]overlay=x=(\d+):y=(\d+)\[sb\]", fc).groups()
    assert badge_xy[0] == board_xy[0] and int(badge_xy[1]) > int(board_xy[1])
    # Blink is `t`-based AFTER setpts/fps, so 1.25Hz means 1.25 flashes per OUTPUT second.
    assert "[sb][4:v]overlay=" in fc and "enable='lt(mod(t*1.25,1),0.6)'" in fc
    assert fc.index("setpts=") < fc.index("enable='lt(mod(t*1.25")


def test_replay_badge_inputs_are_looped_pngs_after_the_silent_track(calls):
    args = _replay_with(ExportConfig(), calls)

    loops = [i for i, a in enumerate(args) if a == "-loop"]
    assert len(loops) == 3  # chrome, badge, dot
    assert args[loops[1] + 3].endswith(".png") and "badge_dot" in args[loops[2] + 3]
    # The silent track is still mapped from input 2.
    assert "2:a" in args


def test_replay_badge_input_indexes_shift_when_there_is_no_silent_track(calls, monkeypatch):
    monkeypatch.setattr(render, "probe_audio", lambda path: None)
    fc = _arg(_replay_with(ExportConfig(), calls), "-filter_complex")
    assert "[sv][2:v]overlay=" in fc and "[sb][3:v]overlay=" in fc


def test_live_part_never_gets_a_replay_badge(calls):
    args = _encode(calls, score=BOARD)
    assert args.count("-i") == 2
    assert "badge" not in _arg(args, "-filter_complex") and "[sv]" not in _arg(args, "-filter_complex")


def test_replay_badge_can_be_turned_off(calls):
    args = _replay_with(ExportConfig(replay_badge=False), calls)
    assert args.count("-i") == 3
    assert "[sv]" not in _arg(args, "-filter_complex")


def test_replay_without_a_scoreboard_has_no_badge(calls):
    args = _encode(calls, slomo_speed=0.25)
    assert not any(a.endswith(".png") for a in args)


def test_replay_audio_follows_mono_export_setting(calls, monkeypatch):
    monkeypatch.setattr(render, "probe_audio", lambda path: AudioParams(44100, "mono"))
    cfg = ExportConfig(mono_audio=True)
    render._encode_piece(Path("src.mp4"), 1.0, 2.0, Path("o.mp4"), cfg, EPOCH, None, slomo_speed=0.5)
    args = calls[-1]
    assert "anullsrc=channel_layout=mono:sample_rate=44100" in args
    assert args[args.index("-ac") + 1] == "1"
    assert _arg(args, "-t") == "4.000"


def test_source_without_audio_gets_no_audio_in_the_replay_either(calls, monkeypatch):
    monkeypatch.setattr(render, "probe_audio", lambda path: None)
    args = _encode(calls, slomo_speed=0.25)
    assert "lavfi" not in args and "1:a" not in args


def test_replay_timeout_is_sized_for_the_output_not_the_source(calls, monkeypatch):
    seen: dict[str, float | None] = {}
    monkeypatch.setattr(render, "_run_ffmpeg", lambda args, timeout=None: seen.update(timeout=timeout))
    render._encode_piece(Path("s.mp4"), 0.0, 10.0, Path("o.mp4"), ExportConfig(), EPOCH, None, slomo_speed=0.25)
    assert seen["timeout"] == render.encode_timeout_seconds(40.0)


# --- replay mapping and assembly ---------------------------------------------


def _chunk(seq: int, global_start: float, duration: float) -> Chunk:
    return Chunk(
        sequence=seq,
        start_time=datetime(2026, 10, 18, 9, 51, 48),
        mp4_path=Path(f"DJI_{seq}.MP4"),
        lrf_path=None,
        duration_seconds=duration,
        global_start_seconds=global_start,
    )


def test_build_replay_maps_clip_relative_seconds_onto_the_source():
    chunks = [_chunk(1, 0.0, 1000.0)]
    replay = render.build_replay(Interval(500.0, 514.0), SlowMo(7.0, 10.0, 0.25), chunks)
    (piece,) = render.replay_pieces(replay)

    assert piece.start == pytest.approx(507.0)
    assert piece.duration == pytest.approx(3.0)
    assert piece.clip_offset == pytest.approx(7.0)
    assert piece.output_seconds == pytest.approx(12.0)


def test_build_replay_splits_a_window_that_straddles_a_chunk_boundary():
    chunks = [_chunk(1, 0.0, 508.0), _chunk(2, 508.0, 500.0)]
    replay = render.build_replay(Interval(500.0, 514.0), SlowMo(7.0, 10.0, 0.25), chunks)
    first, second = render.replay_pieces(replay)

    assert (first.start, first.duration, first.clip_offset) == (pytest.approx(507.0), pytest.approx(1.0), pytest.approx(7.0))
    assert (second.start, second.duration, second.clip_offset) == (pytest.approx(0.0), pytest.approx(2.0), pytest.approx(8.0))


def test_render_export_clip_appends_the_replay_and_passes_crop_to_both(monkeypatch, tmp_path):
    chunk = _chunk(1, 0.0, 1000.0)
    encoded: list[tuple] = []
    concatenated: list[list[Path]] = []

    def fake_encode(source, start, duration, out, cfg, epoch=None, score=None, piece_start=0.0, crop=None, slomo_speed=None):
        encoded.append((start, duration, piece_start, crop, slomo_speed))

    monkeypatch.setattr(render, "_encode_piece", fake_encode)
    monkeypatch.setattr(render, "concat_clips", lambda parts, out, force_reencode=False: concatenated.append(list(parts)))
    rect = rect_for_region("right", CropConfig())
    replay = render.build_replay(Interval(500.0, 514.0), SlowMo(7.0, 10.0, 0.25), [chunk])

    render.render_export_clip(
        [ChunkSlice(chunk, 500.0, 514.0)], tmp_path / "r07.mp4", tmp_path, ExportConfig(), None, rect, replay
    )

    assert encoded == [(500.0, 14.0, 0.0, rect, None), (pytest.approx(507.0), pytest.approx(3.0), pytest.approx(7.0), rect, 0.25)]
    assert [p.name for p in concatenated[0]] == ["r07_part0.mp4", "r07_replay0.mp4"]


def test_render_export_clip_without_replay_encodes_straight_to_the_output(monkeypatch, tmp_path):
    chunk = _chunk(1, 0.0, 1000.0)
    encoded: list[Path] = []
    monkeypatch.setattr(render, "_encode_piece", lambda source, start, duration, out, *a, **k: encoded.append(out))
    monkeypatch.setattr(render, "concat_clips", lambda *a, **k: pytest.fail("no concat expected"))

    render.render_export_clip([ChunkSlice(chunk, 10.0, 20.0)], tmp_path / "x.mp4", tmp_path, ExportConfig())

    assert encoded == [tmp_path / "x.mp4"]
