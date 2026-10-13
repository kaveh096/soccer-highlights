from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from soccer_highlights import camera_position as cp
from soccer_highlights.discovery import Chunk

H, W = cp.FRAME_H, cp.FRAME_W
GRASS = (70, 120, 55)
WHITE_SHIRT = (235, 235, 230)
TEAL_SHIRT = (20, 140, 150)
SHORTS = (30, 30, 40)


def _frame(goal_dx: int = 0) -> np.ndarray:
    """Grass with a red pop-up goal (red frame, pale net) at the lower left."""
    f = np.empty((H, W, 3), np.uint8)
    f[:] = GRASS
    f[:130] = (60, 110, 200)  # sky above the horizon band
    x0, x1, y0, y1 = 70 + goal_dx, 150 + goal_dx, 170, 285
    f[y0:y1, x0:x1] = (215, 215, 215)  # net
    for sl in (
        (slice(y0, y1), slice(x0, x0 + 6)),
        (slice(y0, y1), slice(x1 - 6, x1)),
        (slice(y0, y0 + 6), slice(x0, x1)),
    ):
        f[sl] = (200, 30, 30)
    return f


def _person(f: np.ndarray, x: int, shirt: tuple[int, int, int], feet_y: int = 280) -> None:
    f[feet_y - 55 : feet_y - 28, x : x + 14] = shirt  # torso
    f[feet_y - 28 : feet_y, x : x + 14] = SHORTS  # shorts and legs
    f[feet_y - 62 : feet_y - 55, x + 3 : x + 11] = (200, 160, 130)  # head


def _clip(n: int, shirt: tuple[int, int, int], present_every: int = 1) -> np.ndarray:
    """n frames; a person at the goal in every ``present_every``-th one, at a
    different x each time so the median background does not absorb them."""
    frames = []
    for i in range(n):
        f = _frame()
        if i % present_every == 0:
            _person(f, 160 + (i * 9) % 90, shirt)
        frames.append(f)
    return np.stack(frames)


def test_white_shirt_at_the_goal_reads_as_white_defending():
    result = cp.analyse_chunk(_clip(24, WHITE_SHIRT))

    assert result.goal is not None and result.goal.locator == "red_frame"
    assert result.verdict == "white_defends_near_goal"
    assert result.white_votes == result.n_votes >= cp.MIN_VOTES


def test_coloured_shirt_reads_as_not_white_whatever_the_colour():
    for shirt in (TEAL_SHIRT, (240, 120, 20), (30, 30, 30), (170, 30, 110)):
        result = cp.analyse_chunk(_clip(24, shirt))

        assert result.verdict == "nonwhite_defends_near_goal", shirt
        assert result.white_votes == 0


def test_blob_inside_the_goal_is_ignored_rather_than_voted_dark():
    # The first prototype voted "dark" on net shadow: a wide, short foreground
    # blob inside the footprint, nearer the goal than the real person.
    frames = []
    for i in range(24):
        f = _frame()
        sx = 85 + (i * 7) % 40
        f[250:285, sx : sx + 30] = (15, 15, 15)  # shadow in the net
        _person(f, 160 + (i * 9) % 90, WHITE_SHIRT)
        frames.append(f)

    result = cp.analyse_chunk(np.stack(frames))

    assert result.verdict == "white_defends_near_goal"
    assert result.nonwhite_votes == 0


def test_only_a_net_shadow_gives_no_votes_at_all():
    frames = []
    for i in range(24):
        f = _frame()
        sx = 85 + (i * 7) % 40
        f[250:285, sx : sx + 30] = (15, 15, 15)
        frames.append(f)

    result = cp.analyse_chunk(np.stack(frames))

    assert result.n_votes == 0
    assert result.verdict == "inconclusive"


def test_too_few_votes_is_inconclusive_not_a_guess():
    result = cp.analyse_chunk(_clip(30, WHITE_SHIRT, present_every=6))

    assert 0 < result.n_votes < cp.MIN_VOTES
    assert result.verdict == "inconclusive"
    assert "votes" in result.reason


def test_no_goal_in_view_says_so():
    frames = np.stack([np.full((H, W, 3), GRASS, np.uint8)] * 10)

    result = cp.analyse_chunk(frames)

    assert result.goal is None
    assert result.verdict == "inconclusive"
    assert "--goal-box" in result.reason


def test_manual_goal_box_override_is_used():
    frames = np.stack([np.full((H, W, 3), GRASS, np.uint8)] * 10)

    result = cp.analyse_chunk(frames, goal_override=(0.1, 0.45, 0.25, 0.8))

    assert result.goal is not None and result.goal.locator == "manual"


def test_white_metal_goal_is_found_when_there_is_no_red():
    f = np.empty((H, W, 3), np.uint8)
    f[:] = GRASS
    f[:130] = (60, 110, 200)
    f[170:290, 10:16] = (245, 245, 245)  # post
    f[170:176, 10:170] = (245, 245, 245)  # crossbar
    f[170:290, 164:170] = (245, 245, 245)  # far post

    goal = cp.locate_goal(f)

    assert goal is not None and goal.locator == "white_frame"
    assert goal.x1 > 100 and goal.y1 > 250


def test_classify_votes_bands():
    assert cp.classify_votes(30, 2)[0] == "white_defends_near_goal"
    assert cp.classify_votes(2, 30)[0] == "nonwhite_defends_near_goal"
    assert cp.classify_votes(14, 16)[0] == "inconclusive"  # ambiguous band
    assert cp.classify_votes(cp.MIN_VOTES - 1, 0)[0] == "inconclusive"


def test_a_camera_bump_mid_chunk_is_aligned_out():
    frames = []
    for i in range(24):
        shift = 0 if i < 12 else 45
        f = _frame(goal_dx=shift)
        _person(f, 160 + shift + (i * 9) % 90, WHITE_SHIRT)
        frames.append(f)
    frames_arr = np.stack(frames)

    aligned, dropped = cp.align_frames(frames_arr)
    centres = [cp._red_goal_centre(f) for f in aligned]

    assert dropped == 0
    assert max(c[0] for c in centres) - min(c[0] for c in centres) <= 2
    assert cp.analyse_chunk(frames_arr).verdict == "white_defends_near_goal"


def test_a_steady_camera_is_left_untouched():
    frames = _clip(10, WHITE_SHIRT)

    aligned, dropped = cp.align_frames(frames)

    assert aligned is frames and dropped == 0


# ---- game level -----------------------------------------------------------


def _chunk(seq: int, start: datetime, duration: float, global_start: float) -> Chunk:
    return Chunk(seq, start, Path(f"c{seq}.MP4"), None, duration, global_start)


def _result(seq: int, white: int, nonwhite: int) -> cp.ChunkResult:
    verdict, reason = cp.classify_votes(white, nonwhite)
    return cp.ChunkResult(seq, f"c{seq}", 36, white + nonwhite, white, nonwhite, None, verdict, reason)


def _oct_game() -> list[Chunk]:
    t1 = datetime(2026, 10, 11, 9, 59, 49)
    t2 = datetime(2026, 10, 11, 10, 52, 34)  # 28 min after chunk 1 ends: halftime
    t3 = t2 + timedelta(seconds=1493)  # back to back
    return [
        _chunk(1, t1, 1448.0, 0.0),
        _chunk(2, t2, 1493.0, 1448.0),
        _chunk(3, t3, 557.0, 1448.0 + 1493.0),
    ]


def test_white_first_half_means_camera_behind_white_goal_and_black_shots_captured():
    results = [_result(1, 30, 2), _result(2, 2, 28), _result(3, 1, 20)]

    game = cp.infer_game(results, _oct_game())

    assert game.halftime_seconds == 1448.0
    assert game.behind_goal_first_half == "white"
    # The flag takes the ATTACKING team -- the opposite of the goal owner.
    assert game.near_cam_team_first_half == "black"
    assert game.flip_as_expected is True
    assert game.confidence == "high"


def test_not_white_first_half_means_camera_behind_black_goal():
    results = [_result(1, 1, 30), _result(2, 25, 3), _result(3, 18, 1)]

    game = cp.infer_game(results, _oct_game())

    assert game.behind_goal_first_half == "black"
    assert game.near_cam_team_first_half == "white"
    assert game.flip_as_expected is True


def test_same_team_in_both_halves_is_inconclusive_with_a_reason():
    results = [_result(1, 30, 2), _result(2, 28, 3), _result(3, 20, 1)]

    game = cp.infer_game(results, _oct_game())

    assert game.flip_as_expected is False
    assert game.behind_goal_first_half == "inconclusive"
    assert game.near_cam_team_first_half == "inconclusive"
    assert any("SAME team" in w for w in game.warnings)


def test_unreadable_second_half_keeps_the_first_half_answer_at_medium_confidence():
    results = [_result(1, 30, 2), _result(2, 3, 4), _result(3, 0, 2)]  # thin second-half votes

    game = cp.infer_game(results, _oct_game())

    assert game.behind_goal_first_half == "white"
    assert game.flip_as_expected is None
    assert game.confidence == "medium"
    assert any("unverified" in w for w in game.warnings)


def test_inconclusive_first_half_gives_no_suggestion():
    results = [_result(1, 5, 4), _result(2, 2, 28), _result(3, 1, 20)]

    game = cp.infer_game(results, _oct_game())

    assert game.near_cam_team_first_half == "inconclusive"
    assert game.confidence == "none"


def test_single_chunk_game_warns_that_the_swap_cannot_be_checked():
    chunks = [_chunk(1, datetime(2026, 10, 11, 9, 59, 49), 1448.0, 0.0)]

    game = cp.infer_game([_result(1, 30, 2)], chunks)

    assert game.halftime_seconds is None
    assert game.near_cam_team_first_half == "black"
    assert game.confidence == "medium"
    assert any("halftime" in w for w in game.warnings)


def _continuous_game() -> list[Chunk]:
    """Sep-27 shape: four back-to-back chunks, no recording gap to place halftime with."""
    t = datetime(2026, 9, 27, 10, 0, 26)
    chunks = []
    for seq in range(1, 5):
        chunks.append(_chunk(seq, t, 1490.0, (seq - 1) * 1490.0))
        t += timedelta(seconds=1490)
    return chunks


def test_continuous_recording_never_takes_a_later_chunk_as_the_first_half_answer():
    # Regression (Sep-27): chunk 1 unreadable, chunk 4 (really the second half)
    # reads not-white. Treating all chunks as "first half" returned a confident
    # WRONG answer; only the first chunk may answer the first half.
    results = [_result(1, 0, 0), _result(2, 0, 0), _result(3, 0, 0), _result(4, 0, 24)]

    game = cp.infer_game(results, _continuous_game())

    assert game.halftime_seconds is None
    assert game.behind_goal_first_half == "inconclusive"
    assert game.near_cam_team_first_half == "inconclusive"


def test_continuous_recording_uses_the_first_chunk_and_a_later_opposite_as_swap_evidence():
    results = [_result(1, 28, 2), _result(2, 25, 4), _result(3, 1, 22), _result(4, 0, 24)]

    game = cp.infer_game(results, _continuous_game())

    assert game.behind_goal_first_half == "white"
    assert game.near_cam_team_first_half == "black"
    assert game.flip_as_expected is True
    assert any("no halftime gap" in w for w in game.warnings)


def test_conclusive_chunks_disagreeing_inside_the_first_half_is_inconclusive():
    # A wrong gap picked as halftime puts a second-half chunk in the "first half".
    t1 = datetime(2026, 10, 11, 9, 0, 0)
    chunks = [_chunk(1, t1, 1400.0, 0.0), _chunk(2, t1 + timedelta(seconds=1400), 1400.0, 1400.0),
              _chunk(3, t1 + timedelta(seconds=1400 + 1400 + 1800), 1400.0, 2800.0)]  # fmt: skip
    results = [_result(1, 28, 2), _result(2, 1, 25), _result(3, 2, 20)]

    game = cp.infer_game(results, chunks)

    assert game.behind_goal_first_half == "inconclusive"
    assert any("disagree" in w for w in game.warnings)


def test_votes_voided_by_a_guard_are_not_combined_into_a_half():
    # Regression (Sep-27, second pass): the overcast guard marked chunk 1
    # inconclusive but game-level combining re-derived "not-white" from its raw votes.
    voided = _result(1, 0, 25)
    voided.verdict, voided.usable = "inconclusive", False

    game = cp.infer_game([voided, _result(2, 0, 0), _result(3, 0, 0), _result(4, 0, 24)], _continuous_game())

    assert game.behind_goal_first_half == "inconclusive"
    assert game.near_cam_team_first_half == "inconclusive"


def test_overcast_scene_is_never_a_definite_verdict():
    sky = (170, 170, 175)  # white-grey sky
    frames = _clip(24, WHITE_SHIRT)
    frames[:, :130] = sky

    result = cp.analyse_chunk(frames)

    assert result.verdict == "inconclusive"
    assert result.usable is False
    assert "overcast" in result.reason


def test_contact_sheet_renders_without_frames_or_votes():
    sheet = cp.contact_sheet([_result(1, 0, 0)])
    assert sheet.size[0] > 0

    result = cp.analyse_chunk(_clip(24, WHITE_SHIRT), sequence=1)
    assert cp.contact_sheet([result]).size[1] > 100


@pytest.mark.parametrize("n", [0, 3])
def test_too_few_frames_is_inconclusive(n):
    frames = np.zeros((n, H, W, 3), np.uint8)

    assert cp.analyse_chunk(frames).verdict == "inconclusive"
