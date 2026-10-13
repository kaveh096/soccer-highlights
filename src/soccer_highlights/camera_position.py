"""Infer which team defends the goal the camera sits behind, from shirt colour.

The camera sits behind one goal for a whole game (it does not move; the TEAMS
swap ends at halftime, so the team behind whose goal it sits flips). The
person standing nearest that goal is, most of the time, the goalkeeper or a
defender of the team defending it -- so the white-vs-not-white colour of that
person's torso, voted over many frames, says which team that is.

Deliberately **white vs not-white**, not white vs black: the "black" team's
shirts are teal, orange, magenta or dark depending on the week, and white is
the only colour that is reliably distinct.

Pure numpy/scipy/PIL, no cv2. The voting code takes an ``(N, H, W, 3)`` uint8
array, so it is testable with synthetic frames and no ffmpeg; `sample_frames`
is the thin ffmpeg-seeking wrapper that produces such an array from a ``.LRF``.

The result is **advisory**: `infer_game` prefers ``inconclusive`` (with the
reason) to a definite answer it cannot back, and `cmd_infer_camera` always
writes a contact sheet so a human can eyeball the boxed people before trusting
it. A wrong ``--near-cam-team-first-half`` inverts the review tiering
(Oct-04 2026), which is exactly the failure this must not hide.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from soccer_highlights.discovery import Chunk, detect_halftime_seconds

FRAME_W = 640
FRAME_H = 360

Verdict = Literal["white_defends_near_goal", "nonwhite_defends_near_goal", "inconclusive"]
Team = Literal["white", "black", "inconclusive"]

# --- thresholds, tuned on Oct-11 / Oct-04 / Sep-27 chunks (see processing.md Step 0d) ---
MIN_VOTES = 12  # fewer person-at-goal detections than this -> no verdict
WHITE_FRACTION_MIN = 0.60  # >= this share of votes white -> white defends the near goal
NONWHITE_FRACTION_MAX = 0.30  # <= this -> not-white defends; the band between is inconclusive
TORSO_WHITE_MIN = 0.35  # share of a torso's pixels that must be white to call that person white
_MIN_TORSO_PIXELS = 20

_FOREGROUND_DIFF = 90  # summed |frame - background| over RGB
_SKY_FRACTION = 0.35  # rows above this share of the frame height are never people we vote on
_MIN_PERSON_H = 30  # px at 360p
_MAX_PERSON_H = 200
_MAX_PERSON_W = 110
_MAX_GOAL_DISTANCE = 220.0


@dataclass(frozen=True)
class GoalBox:
    """The near goal's footprint in frame pixels (x1/y1 inclusive)."""

    x0: float
    y0: float
    x1: float
    y1: float
    locator: str  # "red_frame" | "white_frame" | "manual"

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def contains_fraction(self, xs: np.ndarray, ys: np.ndarray) -> float:
        inside = (xs >= self.x0) & (xs <= self.x1) & (ys >= self.y0) & (ys <= self.y1)
        return float(inside.mean())


@dataclass(frozen=True)
class PersonVote:
    frame_index: int
    is_white: bool
    white_fraction: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in frame pixels


@dataclass
class ChunkResult:
    sequence: int
    path: str
    n_frames: int
    n_votes: int
    white_votes: int
    nonwhite_votes: int
    goal: GoalBox | None
    verdict: Verdict
    reason: str = ""
    votes: list[PersonVote] = field(default_factory=list, repr=False)
    frames: np.ndarray | None = field(default=None, repr=False)  # the (aligned) frames the votes index into
    # False when a guard (overcast sky, no goal) voided the votes: they are shown but never combined.
    usable: bool = True

    @property
    def white_fraction(self) -> float | None:
        return self.white_votes / self.n_votes if self.n_votes else None


@dataclass
class GameResult:
    behind_goal_first_half: Team
    near_cam_team_first_half: Team  # the ATTACKING team: opposite of whose goal the camera is behind
    halftime_seconds: float | None
    flip_as_expected: bool | None  # None = could not be checked
    confidence: Literal["high", "medium", "low", "none"]
    first_half: Verdict
    second_half: Verdict | None
    warnings: list[str]
    chunks: list[ChunkResult]


# ---------------------------------------------------------------- sampling


def sample_frames(
    path: str | Path,
    duration_seconds: float,
    n_frames: int = 36,
    ffmpeg: str = "ffmpeg",
) -> np.ndarray:
    """``n_frames`` evenly spread 640x360 RGB frames from ``path``, by seeking.

    Seeking (one short ffmpeg call per frame) rather than a full decode pass:
    over the Google Drive path a fps-filter pass has to read the whole file,
    while each seek touches only a few MB. Runs at below-normal priority on
    Windows so it doesn't fight a render for this laptop's CPU. Frames that
    fail to decode are skipped, so the result may hold fewer than ``n_frames``.
    """
    flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if sys.platform == "win32" else 0
    frames: list[np.ndarray] = []
    for k in range(n_frames):
        t = duration_seconds * (0.02 + 0.96 * k / max(n_frames - 1, 1))
        cmd = [
            ffmpeg, "-v", "error", "-ss", f"{t:.1f}", "-i", str(path), "-an", "-sn",
            "-frames:v", "1", "-vf", f"scale={FRAME_W}:{FRAME_H}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]  # fmt: skip
        raw = subprocess.run(cmd, capture_output=True, creationflags=flags).stdout
        if len(raw) == FRAME_W * FRAME_H * 3:
            frames.append(np.frombuffer(raw, np.uint8).reshape(FRAME_H, FRAME_W, 3))
    if not frames:
        return np.zeros((0, FRAME_H, FRAME_W, 3), np.uint8)
    return np.stack(frames)


# ---------------------------------------------------------------- goal locator


def median_background(frames: np.ndarray) -> np.ndarray:
    return np.median(frames, axis=0).astype(np.uint8)


def _component_box(mask: np.ndarray, dilate: int, min_pixels: int) -> tuple[float, float, float, float] | None:
    """Bounding box (2nd-98th percentile) of the largest connected blob of ``mask``
    after dilation joins thin parts (posts + net). None if nothing big enough."""
    labels, n = ndi.label(ndi.binary_dilation(mask, iterations=dilate))
    if n == 0:
        return None
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    if sizes[best - 1] < min_pixels:
        return None
    ys, xs = np.nonzero(mask & (labels == best))
    return (
        float(np.percentile(xs, 2)),
        float(np.percentile(ys, 2)),
        float(np.percentile(xs, 98)),
        float(np.percentile(ys, 98)),
    )


def _red_mask(image: np.ndarray) -> np.ndarray:
    """Red goal-frame pixels. Relative (r dominates g and b), not absolute:
    under Sep-27's overcast the frame is a dull (120, 80, 80), not (200, 40, 40).
    Rows above the horizon band are blanked (red roofs, signs)."""
    r, g, b = (image[..., i].astype(int) for i in range(3))
    mask = (r > 100) & (r > 1.6 * g) & (r > 1.6 * b)
    mask[: int(image.shape[0] * 0.40)] = False
    return mask


def _red_goal_centre(frame: np.ndarray) -> tuple[float, float] | None:
    box = _component_box(_red_mask(frame), dilate=3, min_pixels=150)
    if box is None:
        return None
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


_ALIGN_PAD = 130
_ALIGN_JITTER_PX = 4  # centroid noise (a player occluding part of the frame) below this is not camera motion
_ALIGN_MIN_FOUND = 0.6


def align_frames(frames: np.ndarray) -> tuple[np.ndarray, int]:
    """Translate frames so the red goal sits where it does in the median frame.

    The camera is bumped or re-aimed sometimes mid-game (Sep-27 chunk 1: the
    goal jumps 46 px between two samples) and a median background over two
    camera positions is garbage. Only engages when a red goal is found in
    most frames AND moves by more than centroid noise; otherwise returns the
    frames untouched. Frames the goal can't be found in (when aligning) are
    dropped rather than guessed at. Returns (frames, number dropped).
    """
    centres = [_red_goal_centre(f) for f in frames]
    found = [c for c in centres if c is not None]
    if len(found) < _ALIGN_MIN_FOUND * len(frames):
        return frames, 0
    ref_x = float(np.median([c[0] for c in found]))
    ref_y = float(np.median([c[1] for c in found]))
    if max(max(abs(c[0] - ref_x), abs(c[1] - ref_y)) for c in found) <= _ALIGN_JITTER_PX:
        return frames, 0
    aligned: list[np.ndarray] = []
    h, w = frames.shape[1:3]
    for frame, centre in zip(frames, centres):
        if centre is None:
            continue
        dx, dy = round(ref_x - centre[0]), round(ref_y - centre[1])
        if abs(dx) > _ALIGN_PAD or abs(dy) > _ALIGN_PAD:
            continue
        if abs(dx) <= _ALIGN_JITTER_PX and abs(dy) <= _ALIGN_JITTER_PX:
            dx = dy = 0
        padded = np.pad(frame, ((_ALIGN_PAD, _ALIGN_PAD), (_ALIGN_PAD, _ALIGN_PAD), (0, 0)), mode="edge")
        aligned.append(padded[_ALIGN_PAD - dy : _ALIGN_PAD - dy + h, _ALIGN_PAD - dx : _ALIGN_PAD - dx + w])
    if not aligned:
        return frames, 0
    return np.stack(aligned), len(frames) - len(aligned)


def locate_goal(background: np.ndarray, override: tuple[float, float, float, float] | None = None) -> GoalBox | None:
    """Find the near goal in the static background.

    ``override`` is a box in FRACTIONS of the frame (x0, y0, x1, y1).
    Otherwise: the red pop-up goal first (red frame pixels), then a white
    metal goal (thin bright-white posts/crossbar below the horizon). Returns
    None when neither is found -- the caller must report that, not guess.
    """
    h, w = background.shape[:2]
    if override is not None:
        x0, y0, x1, y1 = override
        return GoalBox(x0 * w, y0 * h, x1 * w, y1 * h, "manual")

    box = _component_box(_red_mask(background), dilate=4, min_pixels=100)
    if box is not None and box[3] - box[1] >= 30:
        x0, y0, x1, y1 = box
        # Under overcast the pale frame can show red on one pole only (Sep-27:
        # 13 px wide, 54 tall). A goal is wider than that, and the footprint
        # is what rejects net shadow, so grow a too-narrow box both ways.
        width, height = x1 - x0, y1 - y0
        if width < 0.6 * height:
            grow = height - width
            # The net is the pale side of the pole: grow towards whichever side has more pale pixels.
            pale = (background.min(axis=2) > 110) & (background.max(axis=2) - background.min(axis=2) < 40)
            rows = slice(int(y0), int(y1) + 1)
            left = int(pale[rows, max(0, int(x0 - grow)) : int(x0)].sum())
            right = int(pale[rows, int(x1) + 1 : int(x1 + grow) + 1].sum())
            x0 = x0 - (grow if left >= right else grow / 4)
            x1 = x1 + (grow if right >= left else grow / 4)
        return GoalBox(max(0.0, x0), y0, min(float(w - 1), x1), y1, "red_frame")

    white = (background.min(axis=2) > 195) & (background.max(axis=2) - background.min(axis=2) < 35)
    white[: int(h * 0.45)] = False  # sky, clouds and distant buildings
    box = _component_box(white, dilate=3, min_pixels=150)
    if box is not None and box[2] - box[0] >= 0.06 * w and box[3] - box[1] >= 0.10 * h:
        return GoalBox(*box, "white_frame")
    return None


# ---------------------------------------------------------------- voting


def _torso_white_fraction(frame: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> float | None:
    top, bottom = ys.min(), ys.max()
    height = bottom - top
    torso = (ys >= top + 0.12 * height) & (ys <= top + 0.50 * height)
    if torso.sum() < _MIN_TORSO_PIXELS:
        return None
    px = frame[ys[torso], xs[torso]].astype(int)
    white = (px.min(axis=1) > 165) & (px.max(axis=1) - px.min(axis=1) < 45)
    return float(white.mean())


def _nearest_person(
    foreground: np.ndarray, goal: GoalBox
) -> tuple[np.ndarray, np.ndarray] | None:
    labels, n = ndi.label(foreground)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for k, slices in enumerate(ndi.find_objects(labels), start=1):
        if slices is None:
            continue
        height = slices[0].stop - slices[0].start - 1
        width = slices[1].stop - slices[1].start - 1
        if not (_MIN_PERSON_H <= height <= _MAX_PERSON_H) or width > _MAX_PERSON_W:
            continue
        ys, xs = np.nonzero(labels[slices] == k)
        ys = ys + slices[0].start
        xs = xs + slices[1].start
        # Net shadow / net flapping in the wind shows up as foreground INSIDE
        # the goal footprint (the first prototype's bug: it voted "dark" on the
        # shadow; on Sep-27 a net edge voted 26/26 "not-white"). A keeper
        # standing inside the mouth is lost too, but a person right beside the
        # goal still votes, and fewer votes degrade to inconclusive, not wrong.
        if goal.contains_fraction(xs, ys) > 0.5:
            continue
        dx = max(0.0, goal.x0 - xs.mean(), xs.mean() - goal.x1)
        dy = abs(ys.max() - goal.y1)
        dist = float(np.hypot(dx, 2.0 * dy))
        if dist > _MAX_GOAL_DISTANCE:
            continue
        if best is None or dist < best[0]:
            best = (dist, ys, xs)
    return None if best is None else (best[1], best[2])


def vote_frames(frames: np.ndarray, goal: GoalBox, background: np.ndarray | None = None) -> list[PersonVote]:
    """One vote per frame in which a person-sized blob stands near ``goal``."""
    if background is None:
        background = median_background(frames)
    h = frames.shape[1]
    bg = background.astype(int)
    votes: list[PersonVote] = []
    for i, frame in enumerate(frames):
        diff = np.abs(frame.astype(int) - bg).sum(axis=2)
        fg = diff > _FOREGROUND_DIFF
        fg[: int(h * _SKY_FRACTION)] = False
        fg = ndi.binary_closing(ndi.binary_opening(fg), iterations=2)
        person = _nearest_person(fg, goal)
        if person is None:
            continue
        ys, xs = person
        frac = _torso_white_fraction(frame, ys, xs)
        if frac is None:
            continue
        votes.append(
            PersonVote(
                frame_index=i,
                is_white=frac >= TORSO_WHITE_MIN,
                white_fraction=frac,
                box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            )
        )
    return votes


def classify_votes(white_votes: int, nonwhite_votes: int) -> tuple[Verdict, str]:
    n = white_votes + nonwhite_votes
    if n < MIN_VOTES:
        return "inconclusive", f"only {n} person-at-goal votes (need {MIN_VOTES})"
    fraction = white_votes / n
    if fraction >= WHITE_FRACTION_MIN:
        return "white_defends_near_goal", ""
    if fraction <= NONWHITE_FRACTION_MAX:
        return "nonwhite_defends_near_goal", ""
    return "inconclusive", (
        f"white share {fraction:.0%} is between {NONWHITE_FRACTION_MAX:.0%} and {WHITE_FRACTION_MIN:.0%}"
    )


_OVERCAST_SKY_MIN_CHANNEL = 110  # blue sky reads ~60 in the min channel; a white/grey sky 150+


def _is_overcast(background: np.ndarray) -> bool:
    sky = background[: int(background.shape[0] * 0.30)].min(axis=2)
    return float(np.percentile(sky, 50)) > _OVERCAST_SKY_MIN_CHANNEL


def analyse_chunk(
    frames: np.ndarray,
    sequence: int = 0,
    path: str = "",
    goal_override: tuple[float, float, float, float] | None = None,
) -> ChunkResult:
    """Locate the near goal in ``frames``' background and vote on who stands at it."""
    n_frames = len(frames)
    if n_frames < 5:
        return ChunkResult(sequence, path, n_frames, 0, 0, 0, None, "inconclusive", "too few frames could be read")
    frames, dropped = align_frames(frames)
    background = median_background(frames)
    goal = locate_goal(background, goal_override)
    if goal is None:
        return ChunkResult(
            sequence, path, n_frames, 0, 0, 0, None, "inconclusive",
            "no near goal found (neither a red nor a white frame) -- pass --goal-box",
        )  # fmt: skip
    votes = vote_frames(frames, goal, background)
    white = sum(1 for v in votes if v.is_white)
    verdict, reason = classify_votes(white, len(votes) - white)
    usable = True
    if _is_overcast(background):
        # Metered off a bright grey sky the ground and shirts sit ~2x darker
        # (Sep-27) and the absolute white threshold no longer means "white".
        # The votes are void, not just thin: they must not be combined later.
        verdict, reason, usable = "inconclusive", "overcast scene: shirt colour not validated under a grey sky", False
    if len(frames) != n_frames:
        note = f"frames re-aligned on the goal ({dropped} dropped) -- camera was bumped or the goal was occluded"
        reason = f"{reason}; {note}" if reason else note
    return ChunkResult(
        sequence, path, len(frames), len(votes), white, len(votes) - white, goal, verdict, reason, votes, frames, usable
    )


# ---------------------------------------------------------------- game level


def _combine(results: list[ChunkResult]) -> tuple[Verdict, str]:
    usable = [r for r in results if r.usable]
    if results and not usable:
        return "inconclusive", f"no usable chunk ({results[0].reason or 'votes voided'})"
    white = sum(r.white_votes for r in usable)
    nonwhite = sum(r.nonwhite_votes for r in usable)
    return classify_votes(white, nonwhite)


def _opposite(verdict: Verdict) -> Verdict:
    return {
        "white_defends_near_goal": "nonwhite_defends_near_goal",
        "nonwhite_defends_near_goal": "white_defends_near_goal",
    }.get(verdict, "inconclusive")  # type: ignore[return-value]


def infer_game(chunk_results: list[ChunkResult], chunks: list[Chunk]) -> GameResult:
    """Combine per-chunk results into the first-half answer.

    ``chunks`` supplies the halftime boundary (`detect_halftime_seconds`, the
    single largest recording gap). The first half's verdict is the answer;
    the second half is only a check -- teams swap ends, so it must read as
    the OPPOSITE. A same-sign second half, or a chunk-to-chunk flip that is
    not at the detected halftime, makes the whole result ``inconclusive``.
    """
    by_seq = {c.sequence: c for c in chunks}
    halftime = detect_halftime_seconds(chunks)
    warnings: list[str] = []
    ordered = sorted(chunk_results, key=lambda r: r.sequence)

    first_disagrees = False
    if halftime is None:
        # Sep-27 was recorded continuously, so there is no gap to place the
        # halves with, and "all chunks" mixes both halves. Only the FIRST chunk
        # is certainly first half; a later chunk can only serve as evidence of
        # the swap (an opposite verdict), never as the first-half answer.
        first, second = ordered[:1], ordered[1:]
        warnings.append(
            "no halftime gap detected (continuous recording) -- only the first chunk is used for the "
            "first-half answer; later chunks can only confirm the swap"
        )
    else:
        first = [r for r in ordered if by_seq[r.sequence].global_start_seconds < halftime]
        second = [r for r in ordered if by_seq[r.sequence].global_start_seconds >= halftime]

    first_verdict, first_reason = _combine(first)
    second_verdict: Verdict | None = None
    flip: bool | None = None
    if halftime is None:
        later = {r.verdict for r in second if r.verdict != "inconclusive"}
        if first_verdict != "inconclusive" and _opposite(first_verdict) in later:
            second_verdict, flip = _opposite(first_verdict), True  # type: ignore[assignment]
    else:
        second_verdict = _combine(second)[0] if second else None
        if second_verdict is not None and second_verdict != "inconclusive" and first_verdict != "inconclusive":
            flip = second_verdict == _opposite(first_verdict)

    # Chunk-level consistency: inside a half every conclusive chunk should agree.
    # Disagreement in the FIRST half means the halves are mis-placed (wrong gap
    # picked as halftime) -- that answer cannot be trusted at all.
    for label, half in (("first", first), ("second", second if halftime is not None else [])):
        seen = {r.verdict for r in half if r.verdict != "inconclusive"}
        if len(seen) > 1:
            warnings.append(f"chunks disagree inside the {label} half: " + ", ".join(
                f"#{r.sequence} {r.verdict}" for r in half))  # fmt: skip
            first_disagrees = first_disagrees or label == "first"

    confidence: Literal["high", "medium", "low", "none"]
    if first_disagrees:
        first_verdict = "inconclusive"
        confidence = "none"
    elif first_verdict == "inconclusive":
        warnings.append(f"first half inconclusive: {first_reason}")
        confidence = "none"
    elif flip is False:
        warnings.append(
            "second half reads the SAME team at the near goal as the first half; teams swap ends, so "
            "expected the opposite -- the camera may have moved, teams may not have swapped, or the "
            "inference is wrong. Look at the contact sheet before trusting either half."
        )
        confidence = "none"
    elif flip is True and not any("disagree" in w for w in warnings):
        confidence = "high"
    else:
        if second_verdict is None or second_verdict == "inconclusive":
            warnings.append("second half could not be read, so the swap at halftime is unverified")
        confidence = "medium" if not any("disagree" in w for w in warnings) else "low"

    definite = confidence in ("high", "medium", "low")
    behind: Team = "inconclusive"
    if definite:
        behind = "white" if first_verdict == "white_defends_near_goal" else "black"
    # The flag takes the ATTACKING team: the opposite of whose goal it sits behind.
    near_cam: Team = {"white": "black", "black": "white"}.get(behind, "inconclusive")  # type: ignore[assignment]
    return GameResult(
        behind_goal_first_half=behind,
        near_cam_team_first_half=near_cam,
        halftime_seconds=halftime,
        flip_as_expected=flip,
        confidence=confidence,
        first_half=first_verdict,
        second_half=second_verdict,
        warnings=warnings,
        chunks=ordered,
    )


# ---------------------------------------------------------------- contact sheet


def contact_sheet(results: list[ChunkResult], per_chunk: int = 6, tile_w: int = 320) -> Image.Image:
    """A grid, one row per chunk: frames in which a person was voted, boxed and
    labelled white / not-white. Cropped to the goal's side of the frame so the
    people are big enough to judge by eye."""
    rows: list[Image.Image] = []
    tile_h = int(tile_w * 0.55)
    for result in results:
        frames = result.frames
        row = Image.new("RGB", (tile_w * per_chunk, tile_h + 18), (20, 20, 20))
        ImageDraw.Draw(row).text(
            (4, 3),
            f"chunk #{result.sequence}: {result.verdict}  votes {result.white_votes} white / "
            f"{result.nonwhite_votes} not-white",
            fill=(255, 255, 0),
        )
        votes = result.votes
        if frames is None or not votes:
            rows.append(row)
            continue
        picks = [votes[int(i)] for i in np.linspace(0, len(votes) - 1, min(per_chunk, len(votes)))]
        for col, vote in enumerate(picks):
            frame = Image.fromarray(frames[vote.frame_index])
            draw = ImageDraw.Draw(frame)
            x0, y0, x1, y1 = vote.box
            colour = (0, 255, 0) if vote.is_white else (255, 0, 255)
            draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)
            if result.goal is not None:
                g = result.goal
                draw.rectangle([g.x0, g.y0, g.x1, g.y1], outline=(255, 255, 0), width=1)
            tile = frame.crop((0, int(FRAME_H * 0.3), FRAME_W // 2 + 60, FRAME_H)).resize((tile_w, tile_h))
            ImageDraw.Draw(tile).text(
                (4, 4), f"f{vote.frame_index} {'WHITE' if vote.is_white else 'not-white'}", fill=colour
            )
            row.paste(tile, (col * tile_w, 18))
        rows.append(row)
    if not rows:
        return Image.new("RGB", (tile_w, 40), (20, 20, 20))
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (20, 20, 20))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet
