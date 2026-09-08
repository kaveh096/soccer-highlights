"""Ingest live-tagged "marks" -- wall-clock timestamp + category events from
any external capture path (a Wear OS watch, a hand-written CSV, eventually
DJI highlight marks) -- and map them onto the recording's global timeline,
to be UNIONED with (never a replacement for) audio-detected candidates.

This is source-agnostic by design: every candidate capture path (Tallies, a
custom Wear OS app, DJI marks) reduces to the same thing -- a wall-clock
timestamp and a category -- so this module can be built and tested against
a hand-written CSV with no watch, no app, and no game (2026-08-30/31
session). See the kaveh_soccer_highlights skill and this session's brief
for the full design history.

v1 is report-only (`ingest-marks` in cli.py): it proves out the
wallclock->global mapping and the audio-peak-snap/fallback-window logic
against real data before wiring marks into pre-label's actual rendering
path.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = timezone.utc

from soccer_highlights.config import MarksConfig, TimelineConfig
from soccer_highlights.discovery import Chunk, wallclock_to_global
from soccer_highlights.timeline import GlobalPeak, Interval, merge_intervals

# DJI filename timestamps carry no timezone, and this rig only ever films
# in one place -- pin the assumption explicitly (see SKILL.md's
# timezone/DST trap) rather than guessing per-mark. A mark's own tz-aware
# timestamp is converted to this zone, then stripped to naive, before being
# compared against a chunk's naive (assumed-already-local) start_time.
RECORDING_TZ = ZoneInfo("America/Los_Angeles")

CATEGORIES = frozenset({"white_goal", "black_goal", "moment"})
_UNDO = "undo"


@dataclass
class Mark:
    sequence: int
    timestamp: datetime  # tz-aware, as parsed from the CSV
    category: str  # one of CATEGORIES, or "undo"


@dataclass
class ResolvedMark:
    mark: Mark
    global_seconds: float | None  # None if it landed in an unrecorded gap
    interval: Interval | None  # None iff global_seconds is None
    # "audio_peak": snapped to an existing audio-detected peak nearby --
    #   audio would likely have found this one anyway.
    # "fixed_window": no nearby audio peak -- this mark IS the recall gain.
    # "unrecorded_gap": landed between two chunks; no clip is possible.
    anchor: str


def load_marks_csv(path: str | Path) -> list[Mark]:
    """Parse a marks CSV: columns `timestamp` (ISO 8601 with an explicit
    UTC offset), `category` (white_goal/black_goal/moment/undo), `sequence`
    (monotonic int, for ordering/dedupe independent of any timestamp
    collision or clock skew). Matches the data shape the custom Wear OS app
    PRD specifies, so this parser doesn't change when Tallies/a custom app/
    DJI marks eventually replace the hand-written CSV -- only however that
    source's export gets converted to this shape does."""
    path = Path(path)
    marks: list[Mark] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            category = row["category"].strip()
            if category != _UNDO and category not in CATEGORIES:
                raise ValueError(
                    f"{path}: unknown category {category!r} at sequence {row['sequence']!r} "
                    f"(expected one of {sorted(CATEGORIES)} or {_UNDO!r})"
                )
            timestamp = datetime.fromisoformat(row["timestamp"].strip())
            if timestamp.tzinfo is None:
                raise ValueError(
                    f"{path}: timestamp {row['timestamp']!r} at sequence {row['sequence']!r} has no UTC "
                    "offset -- wall-clock marks must be unambiguous about timezone (see SKILL.md's DST trap)"
                )
            marks.append(Mark(sequence=int(row["sequence"]), timestamp=timestamp, category=category))
    marks.sort(key=lambda m: m.sequence)
    return marks


def load_tallies_csv(path: str | Path, category: str) -> list[Mark]:
    """Parse one CSV exported by the Tallies Wear OS app into Marks.

    Verified against a real export (2026-09-02 bench test). Tallies writes
    `timestamp,count,count_change,action`, ONE FILE PER COUNTER -- hence the
    explicit `category` argument: the counter's identity is in the filename
    (a free-form user-typed string like "soccer tally test - white goal -
    Sep 2.csv"), which is far too fragile to parse, so the caller states it.

    `timestamp` is epoch MILLISECONDS -- unambiguous by construction, with
    no timezone or DST hazard at all (a nicer property than the ISO-8601
    string the custom-app PRD had assumed). Resolution is milliseconds, and
    two taps 0.31s apart were both recorded distinctly in the bench test.

    Only `action == "CLICK"` rows are presses. `TALLY_CREATED` is setup
    noise. `EDIT_TALLY` is a MANUAL edit of the counter value -- it is not a
    press and gets no mark, but it does mean the displayed total was hand-
    adjusted, so it is warned about: after one, the counter total can no
    longer be trusted as a checksum against the number of press rows.

    A negative `count_change` is Tallies' minus button. **Not auto-resolved
    as an undo** (changed 2026-09-07, was pop-the-most-recently-active-press
    before): Sep-06's real data showed a decrement landing 2.5 minutes after
    the press it was meant to cancel, with nothing else in between for that
    stack-based pop to trip over by luck -- but there is no guarantee of
    that in general, and a late decrement silently cancelling the WRONG
    earlier press (if something else in the same category happened in
    between) would be worse than not resolving it at all. Kaveh's call:
    keep every positive press as a real Mark unconditionally, and identify
    the actual mis-taps by hand during candidate review, against real
    footage, rather than have this module guess. Decrement rows are
    counted and warned about, not silently dropped without a trace."""
    path = Path(path)
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r} (expected one of {sorted(CATEGORIES)})")

    active: list[Mark] = []
    saw_edit = False
    n_decrements = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            action = row["action"].strip()
            if action == "EDIT_TALLY":
                saw_edit = True
                continue
            if action != "CLICK":
                continue
            change = int(row["count_change"])
            timestamp = datetime.fromtimestamp(int(row["timestamp"]) / 1000, tz=UTC)
            if change > 0:
                # |change| > 1 at a single instant can't be split into
                # separate press times, so it collapses to one mark.
                if change > 1:
                    print(f"WARNING: {path.name}: count_change={change} at {timestamp} -- recording a single mark")
                active.append(Mark(sequence=0, timestamp=timestamp, category=category))
            elif change < 0:
                n_decrements += 1

    if n_decrements:
        print(
            f"WARNING: {path.name}: {n_decrements} decrement(s) (Tallies' minus button) were NOT auto-resolved -- "
            "every positive press is still kept as a Mark. Identify and remove any real mis-taps by hand during "
            "candidate review."
        )
    if saw_edit:
        print(
            f"WARNING: {path.name} contains an EDIT_TALLY row -- the counter total was manually edited, so it "
            "cannot be cross-checked against the number of press rows (the presses themselves are still fine)."
        )
    return active


def merge_tally_marks(per_category: list[Mark]) -> list[Mark]:
    """Merge Marks from several per-counter Tallies files into one ordered
    list, assigning `sequence` by timestamp. Tallies' epoch-millisecond
    timestamps are globally comparable across files, so time order IS the
    true press order -- unlike the custom-app PRD's design, where a
    monotonic per-device sequence number was needed because a single file's
    ordering couldn't be trusted."""
    ordered = sorted(per_category, key=lambda m: m.timestamp)
    return [Mark(sequence=i, timestamp=m.timestamp, category=m.category) for i, m in enumerate(ordered, start=1)]


def resolve_undos(marks: list[Mark]) -> list[Mark]:
    """Apply undo rows: each `undo` cancels the most recently-tapped mark
    still active before it, by sequence order -- never by mutating the
    source file (the PRD's append-only rule: undo is a row, not an edit).
    An undo with nothing left to cancel is dropped silently rather than
    raising -- a double-undo or a clock hiccup on the watch shouldn't kill
    the whole ingest run."""
    active: list[Mark] = []
    for mark in marks:
        if mark.category == _UNDO:
            if active:
                active.pop()
        else:
            active.append(mark)
    return active


def resolve_marks(
    marks: list[Mark],
    chunks: list[Chunk],
    audio_peaks: list[GlobalPeak],
    marks_cfg: MarksConfig,
    timeline_cfg: TimelineConfig,
    clock_offset_seconds: float = 0.0,
) -> list[ResolvedMark]:
    """Map each (already undo-resolved) mark to a global-timeline interval.

    `clock_offset_seconds` is ADDED to every mark's timestamp before
    mapping, to correct a measured skew between the watch's clock and the
    camera's RTC (which is what stamps the DJI filenames the whole timeline
    is anchored on). Get the value from `measure_clock_offset` -- it is
    deliberately never applied automatically, because silently shifting
    every mark by a wrongly-measured offset would corrupt the whole run
    while still looking plausible. Note it is applied to the wall clock
    BEFORE the gap check, since a shift can legitimately move a mark into
    or out of an unrecorded gap.

    Marks are trailing, so both windows are asymmetric, but they are two
    DIFFERENT windows and the distinction matters (see MarksConfig):

    - the *snap search* window (`snap_*`) is tight, sized to the actual
      press delay. Anchoring on the loudest peak across a too-wide window
      is how you confidently cut the wrong clip.
    - the *fallback clip* window (`goal_*`/`moment_*`) is wide, used only
      when no peak is found at all -- there, over-long beats under-long.

    Rule: search the tight window for an existing audio peak; if one
    exists, anchor there and apply the normal `timeline_cfg`
    lookback/post_peak -- audio gives a more precise boundary than a press
    timestamp does. If none exists, fall back to the wide category window.
    Which path fired is recorded on every ResolvedMark -- that flag IS the
    recall measurement this whole feature exists to produce."""
    sorted_peaks = sorted(audio_peaks, key=lambda p: p.time_seconds)
    resolved: list[ResolvedMark] = []
    for mark in marks:
        local_ts = mark.timestamp.astimezone(RECORDING_TZ).replace(tzinfo=None) + timedelta(
            seconds=clock_offset_seconds
        )
        global_seconds = wallclock_to_global(local_ts, chunks)
        if global_seconds is None:
            resolved.append(ResolvedMark(mark=mark, global_seconds=None, interval=None, anchor="unrecorded_gap"))
            continue

        if mark.category == "moment":
            lookback, lookahead = marks_cfg.moment_lookback_seconds, marks_cfg.moment_lookahead_seconds
        else:
            lookback, lookahead = marks_cfg.goal_lookback_seconds, marks_cfg.goal_lookahead_seconds
        window_start = max(0.0, global_seconds - lookback)
        window_end = global_seconds + lookahead

        snap_start = max(0.0, global_seconds - marks_cfg.snap_lookback_seconds)
        snap_end = global_seconds + marks_cfg.snap_lookahead_seconds
        nearby = [p for p in sorted_peaks if snap_start <= p.time_seconds <= snap_end]
        if nearby:
            peak = max(nearby, key=lambda p: p.score)
            interval = Interval(
                start_seconds=max(0.0, peak.time_seconds - timeline_cfg.lookback_seconds),
                end_seconds=peak.time_seconds + timeline_cfg.post_peak_seconds,
                peaks=[peak],
            )
            anchor = "audio_peak"
        else:
            interval = Interval(start_seconds=window_start, end_seconds=window_end, peaks=[])
            anchor = "fixed_window"
        resolved.append(ResolvedMark(mark=mark, global_seconds=global_seconds, interval=interval, anchor=anchor))
    return resolved


def _merge_same_category_fixed_windows(
    extra: list[tuple[Interval, str, ResolvedMark]],
) -> list[tuple[Interval, str, ResolvedMark]]:
    """Merge fixed-window fallback candidates from the SAME tally category
    when their clip windows touch or overlap -- two taps close enough in
    time that their independent fallback windows collide should produce one
    clip, not two near-identical ones (Kaveh, 2026-09-06, after two
    black-goal taps 6.8s apart rendered as two 90%-overlapping clips on a
    real game). The earliest tap's owner represents the merged group --
    `score_for_interval`'s existing "N goal taps inside one clip" warning
    fires naturally whenever a merge folds in more than one real goal event,
    which is the intended surfacing of a likely double-tap; no separate
    warning is added here. Cross-category pairs are never merged: a
    white_goal and a black_goal close together are still two different
    events."""
    by_category: dict[str, list[tuple[Interval, str, ResolvedMark]]] = {}
    for item in extra:
        by_category.setdefault(item[2].mark.category, []).append(item)

    merged: list[tuple[Interval, str, ResolvedMark]] = []
    for group in by_category.values():
        ordered = sorted(group, key=lambda item: item[0].start_seconds)
        current_interval, current_source, current_owner = ordered[0]
        for interval, source, owner in ordered[1:]:
            if interval.start_seconds <= current_interval.end_seconds:
                current_interval = Interval(
                    start_seconds=current_interval.start_seconds,
                    end_seconds=max(current_interval.end_seconds, interval.end_seconds),
                    peaks=current_interval.peaks + interval.peaks,
                )
            else:
                merged.append((current_interval, current_source, current_owner))
                current_interval, current_source, current_owner = interval, source, owner
        merged.append((current_interval, current_source, current_owner))
    return merged


def union_with_audio_detailed(
    audio_intervals: list[Interval], resolved: list[ResolvedMark]
) -> list[tuple[Interval, str, ResolvedMark | None]]:
    """`union_with_audio`, but also reporting WHICH mark each interval came
    from (None for an audio candidate no mark touched).

    The association is what the score-counter overlay needs: a `both` row's
    clip has to be extended to cover its own tap, and only the owning mark
    knows where that tap is. Kept as the single implementation, with
    `union_with_audio` as a thin wrapper over it, so the peak-matching rule
    lives in exactly one place."""
    sources = ["audio"] * len(audio_intervals)
    owners: list[ResolvedMark | None] = [None] * len(audio_intervals)
    extra: list[tuple[Interval, str, ResolvedMark]] = []
    for r in resolved:
        if r.anchor == "audio_peak" and r.interval is not None:
            peak_time = r.interval.peaks[0].time_seconds
            for i, candidate in enumerate(audio_intervals):
                if any(p.time_seconds == peak_time for p in candidate.peaks):
                    sources[i] = "both"
                    owners[i] = r
                    break
        elif r.anchor == "fixed_window" and r.interval is not None:
            extra.append((r.interval, "mark", r))

    combined: list[tuple[Interval, str, ResolvedMark | None]] = list(
        zip(audio_intervals, sources, owners)
    ) + _merge_same_category_fixed_windows(extra)
    combined.sort(key=lambda triple: triple[0].start_seconds)
    return combined


def extend_for_score_flip(
    combined: list[tuple[Interval, str, ResolvedMark | None]], marks_cfg: MarksConfig
) -> int:
    """Extend peak-anchored clips so the tap that scores them falls INSIDE
    the clip, giving the burned-in score somewhere to visibly flip.

    This is the one place marks MODIFY an audio interval rather than just
    annotating it. It stays inside the recall-first rule -- an interval is
    only ever extended, never shortened or dropped -- but it does mean a
    `both` clip is longer than audio detection alone would have made it.

    Only `both` rows need this. A `fixed_window` clip already ends
    `goal_lookahead_seconds` after its own tap by construction, so the tap
    is always inside it.

    The cap matters: without one, a forgotten press minutes later would
    stretch a 14s highlight into a several-minute clip. Past the cap the
    clip is left alone and `score_for_interval` falls back to flipping at
    the clip's midpoint instead. Returns how many intervals were extended."""
    extended = 0
    for interval, source, owner in combined:
        if source != "both" or owner is None or owner.global_seconds is None or not interval.peaks:
            continue
        peak_time = owner.interval.peaks[0].time_seconds if owner.interval and owner.interval.peaks else None
        if peak_time is None or owner.global_seconds > peak_time + marks_cfg.score_flip_cap_seconds:
            continue
        new_end = owner.global_seconds + marks_cfg.score_flip_tail_seconds
        if new_end > interval.end_seconds:
            interval.end_seconds = new_end
            extended += 1
    return extended


def union_with_audio(
    audio_intervals: list[Interval], resolved: list[ResolvedMark]
) -> tuple[list[Interval], list[str]]:
    """Union mark-derived intervals with audio candidates, NEVER replacing
    them (the project's standing recall-first rule: if the watch is
    forgotten, this must degrade to exactly today's audio-only behavior).

    Returns the combined intervals and a parallel list of provenance
    labels, which is the whole point -- `mark` rows are events audio missed
    entirely, so counting them IS the recall measurement:

    - ``audio``: an audio candidate no mark corroborated.
    - ``both``:  an audio candidate a mark landed on. Provenance is decided
      by the mark's *anchor*, not by geometric overlap -- a `fixed_window`
      mark's fallback window is 60s wide and would frequently overlap some
      unrelated earlier candidate, which would wrongly credit audio with
      having found it.
    - ``mark``:  a mark with no audio peak in its (tight) snap window --
      a genuinely new candidate that audio detection missed.

    Marks that landed in an unrecorded gap contribute nothing; they can't
    be rendered and are reported separately."""
    combined = union_with_audio_detailed(audio_intervals, resolved)
    return [iv for iv, _, _ in combined], [src for _, src, _ in combined]


def tap_claim(mark_category: str) -> str | None:
    """A one-line natural-language claim for
    `label_audit.generate_description`'s `tap_context` parameter (2026-09-07),
    describing what a tally category means for describe-prompt purposes.
    None for no tap at all -- there's nothing to tell Gemini."""
    if mark_category == "white_goal":
        return "the white team scored a goal"
    if mark_category == "black_goal":
        return "the dark/black team scored a goal"
    if mark_category == "moment":
        return "something notable happened here, but it was NOT tapped as a goal"
    return None


@dataclass
class ScoreColumns:
    """The running score as it should appear on one clip. Written into the
    review sheet so it can be eyeballed -- and corrected by hand -- before
    any 37x-realtime render, rather than being re-derived at export time."""

    white: int  # score at the clip's START, before any goal inside it
    black: int
    flip_seconds: float | None  # offset INTO the clip where the score changes
    flip_team: str  # "white" / "black", or "" when there is no flip


_GOAL_CATEGORIES = {"white_goal": "white", "black_goal": "black"}


def score_events(resolved: list[ResolvedMark]) -> list[tuple[float, str]]:
    """Goal taps as (global_seconds, "white"|"black"), in time order.

    The TAP time is the score's timeline position, deliberately -- not the
    audio peak. A real broadcast's score graphic also updates a beat after
    the ball goes in, and using the tap makes the behavior identical whether
    or not audio happened to find a peak (Kaveh's call, 2026-09-05). It also
    means the overlay doubles as feedback on how fast he tapped.

    `moment` marks never score. Marks that fell in an unrecorded gap are
    skipped -- there is no timeline position to place them at. Undos are
    expected to have been resolved already (`resolve_undos` /
    `load_tallies_csv`), so a cancelled goal never reaches here."""
    events = [
        (r.global_seconds, _GOAL_CATEGORIES[r.mark.category])
        for r in resolved
        if r.mark.category in _GOAL_CATEGORIES and r.global_seconds is not None
    ]
    events.sort(key=lambda e: e[0])
    return events


def score_for_interval(
    interval: Interval, events: list[tuple[float, str]], owner: ResolvedMark | None = None
) -> ScoreColumns:
    """Running score for one clip, plus where (if anywhere) it flips.

    ALL of the placement policy lives here rather than in the renderer, so
    that every decision is visible as a plain number in the review sheet and
    can be hand-corrected before export.

    `owner` is the mark this clip came from, if any (see
    `union_with_audio_detailed`). It exists for one case: a goal whose tap
    landed OUTSIDE its own clip, because the press was slow enough that
    `extend_for_score_flip` would have blown its cap covering it. That clip
    still shows a goal, so the score still has to move -- it just has no
    trustworthy instant to move at. The flip then goes at the clip's
    MIDPOINT: deterministic, and dependent on neither audio nor press
    timing, which are precisely the two things that failed in that case."""
    white = sum(1 for t, team in events if t < interval.start_seconds and team == "white")
    black = sum(1 for t, team in events if t < interval.start_seconds and team == "black")

    inside = [(t, team) for t, team in events if interval.start_seconds <= t <= interval.end_seconds]
    if len(inside) > 1:
        print(
            f"WARNING: {len(inside)} goal taps inside one clip "
            f"({interval.start_seconds:.1f}-{interval.end_seconds:.1f}s) -- only the first is drawn"
        )
    if inside:
        tap_time, team = inside[0]
        return ScoreColumns(white=white, black=black, flip_seconds=tap_time - interval.start_seconds, flip_team=team)

    if owner is not None and owner.mark.category in _GOAL_CATEGORIES and owner.global_seconds is not None:
        midpoint = (interval.end_seconds - interval.start_seconds) / 2
        return ScoreColumns(
            white=white, black=black, flip_seconds=midpoint, flip_team=_GOAL_CATEGORIES[owner.mark.category]
        )
    return ScoreColumns(white=white, black=black, flip_seconds=None, flip_team="")


_OTHER_TEAM = {"white": "black", "black": "white"}


def is_near_cam_goal(
    mark_category: str,
    global_seconds: float,
    halftime_seconds: float | None,
    near_cam_team_first_half: str | None,
) -> bool:
    """Whether a white_goal/black_goal tap happened at the camera's own end
    of the field, per the per-game camera setup Kaveh gives directly
    (2026-09-07 -- "Sep 6, cam was behind black team's goal capturing white
    team's shots. At half time, it is swapped"). Camera position isn't
    derivable from the footage or from Gemini, so this is the authoritative
    signal for a genuine goal tap -- see `review_tier`, which stops
    consulting `goal_this_end` at all once a goal-category mark and a
    complete camera config are both available.

    Returns False whenever the per-game input is incomplete (no halftime
    boundary detected, or `near_cam_team_first_half` not given) -- the
    caller is expected to treat that as "no info" (fall back to
    goal_this_end via `review_tier`'s `is_near_cam_goal=None`), not as "far
    end", so this function itself never needs to distinguish the two."""
    if mark_category not in _GOAL_CATEGORIES or near_cam_team_first_half is None or halftime_seconds is None:
        return False
    scoring_team = _GOAL_CATEGORIES[mark_category]
    near_cam_team = near_cam_team_first_half if global_seconds < halftime_seconds else _OTHER_TEAM[near_cam_team_first_half]
    return scoring_team == near_cam_team


def review_tier(
    mark_category: str,
    gemini_score: int,
    goal_this_end: bool,
    is_near_cam_goal: bool | None = None,
    is_sync_clap: bool = False,
) -> int:
    """Bucket a pre-label candidate into a review-order tier -- lower reviews
    first. Revised 2026-09-07 after reviewing Sep-06's real ranked sheet:
    Kaveh found Gemini false-positive goal calls on `moment`-tagged clips he
    knew weren't goals, and asked that **taps beat Gemini** whenever they
    actively disagree, using the per-game camera setup rather than
    `goal_this_end` to judge a genuine goal tap's visibility:

    1. A genuine goal tap (`white_goal`/`black_goal`) at the camera's own
       end of the field, per `is_near_cam_goal` -- Gemini is NOT consulted
       here at all once a real goal tap exists. `is_near_cam_goal=None`
       (the per-game camera config wasn't given) falls back to
       `goal_this_end`, i.e. today's pre-2026-09-07 behavior, unchanged.
    2. A `moment`-tagged, non-goal candidate Gemini still rates >=3 -- but
       NEVER a case where `goal_this_end=True` contradicts the moment tap;
       an explicit non-goal tap overrides Gemini's goal claim outright, it
       doesn't just fail to promote it (2026-09-07: r03/r07 on Sep-06 were
       exactly this -- Kaveh's own moment tap said "not a goal", Gemini said
       "goal", and Gemini was wrong both times).
    3. Any other candidate Gemini rates >=4 (a highlight regardless of
       source) -- including a goal tap that failed its near-cam check, or
       an untapped clip with no `goal_this_end` signal to lean on.
    4. Any other `moment`-tagged candidate, whatever its score.
    5. Everything else.

    An untapped clip (`mark_category=""`) has nothing to "value over
    Gemini" -- there's no tap, so `goal_this_end` alone still decides tier 1
    for it, same as before.

    `is_sync_clap` (2026-09-06): the clap-sync `moment` tap at kickoff
    (Step 0c) is a timing reference, not a real highlight candidate --
    never earns the tier 2/4 moment boost even though its mark_category is
    genuinely "moment". A caller identifies it as the game's
    chronologically-first `moment` mark."""
    if mark_category in _GOAL_CATEGORIES:
        if is_near_cam_goal is None:
            if goal_this_end:
                return 1
        elif is_near_cam_goal:
            return 1
    elif mark_category == "moment":
        if not is_sync_clap and gemini_score >= 3:
            return 2
    elif goal_this_end:
        return 1

    if gemini_score >= 4:
        return 3
    if mark_category == "moment" and not is_sync_clap:
        return 4
    return 5


def review_sort_key(
    mark_category: str,
    gemini_score: int,
    goal_this_end: bool,
    start_seconds: float,
    is_near_cam_goal: bool | None = None,
    is_sync_clap: bool = False,
) -> tuple[int, int, float]:
    """Sort key for `name-candidates`' review-order rename: tier first, then
    descending Gemini score within the tier, then chronological order as a
    final, fully deterministic tiebreak."""
    tier = review_tier(mark_category, gemini_score, goal_this_end, is_near_cam_goal, is_sync_clap)
    return (tier, -gemini_score, start_seconds)


@dataclass
class ClockOffset:
    mark: ResolvedMark
    peak: GlobalPeak
    offset_seconds: float  # add this to every mark to align it with the camera


def measure_clock_offset(
    resolved: list[ResolvedMark], audio_peaks: list[GlobalPeak], window_seconds: float
) -> ClockOffset | None:
    """Measure watch-clock vs. camera-RTC skew from the clap-sync ritual.

    Kaveh claps loudly in front of the camera and taps `moment` at the same
    instant, so the two are simultaneous by construction: any difference
    between the clap's audio transient and the mark's mapped position IS
    the clock offset. A clap is a sharp, isolated transient, which is
    precisely what `onset_flux` detects best, so it should be the loudest
    thing in its neighbourhood.

    Uses the FIRST `moment` mark, since the ritual happens at kickoff. This
    is a heuristic -- if `moment` also gets used mid-game, only the first
    one is treated as the clap.

    Returns None if there is no `moment` mark, or no audio peak within
    ±window_seconds of it. A None with marks present is itself a signal:
    either the clap wasn't detected, or the skew is larger than the search
    window (i.e. the camera RTC sync did not take)."""
    moments = [r for r in resolved if r.mark.category == "moment" and r.global_seconds is not None]
    if not moments:
        return None
    clap = min(moments, key=lambda r: r.mark.timestamp)
    nearby = [p for p in audio_peaks if abs(p.time_seconds - clap.global_seconds) <= window_seconds]
    if not nearby:
        return None
    peak = max(nearby, key=lambda p: p.score)
    return ClockOffset(mark=clap, peak=peak, offset_seconds=peak.time_seconds - clap.global_seconds)


def write_ingest_report_csv(resolved: list[ResolvedMark], out_path: str | Path) -> None:
    out_path = Path(out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sequence", "category", "timestamp_local", "global_seconds", "anchor", "window_start", "window_end"]
        )
        for r in resolved:
            local = r.mark.timestamp.astimezone(RECORDING_TZ)
            writer.writerow(
                [
                    r.mark.sequence,
                    r.mark.category,
                    local.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{r.global_seconds:.2f}" if r.global_seconds is not None else "",
                    r.anchor,
                    f"{r.interval.start_seconds:.2f}" if r.interval is not None else "",
                    f"{r.interval.end_seconds:.2f}" if r.interval is not None else "",
                ]
            )
