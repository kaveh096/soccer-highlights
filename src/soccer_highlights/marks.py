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
from soccer_highlights.timeline import GlobalPeak, Interval

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

    A negative `count_change` is Tallies' minus button, which is this
    capture path's undo. It is resolved HERE, within this one counter's
    file, rather than by `resolve_undos` -- with one counter per file, a
    minus on the white counter must cancel a white goal, never "whatever
    was tapped most recently across all three counters", which is what a
    global stack would do."""
    path = Path(path)
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r} (expected one of {sorted(CATEGORIES)})")

    active: list[Mark] = []
    saw_edit = False
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
            elif change < 0 and active:
                active.pop()

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
    sources = ["audio"] * len(audio_intervals)
    extra: list[Interval] = []
    for r in resolved:
        if r.anchor == "audio_peak" and r.interval is not None:
            peak_time = r.interval.peaks[0].time_seconds
            for i, candidate in enumerate(audio_intervals):
                if any(p.time_seconds == peak_time for p in candidate.peaks):
                    sources[i] = "both"
                    break
        elif r.anchor == "fixed_window" and r.interval is not None:
            extra.append(r.interval)

    combined = list(zip(audio_intervals, sources)) + [(iv, "mark") for iv in extra]
    combined.sort(key=lambda pair: pair[0].start_seconds)
    return [iv for iv, _ in combined], [src for _, src in combined]


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
