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
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
) -> list[ResolvedMark]:
    """Map each (already undo-resolved) mark to a global-timeline interval.

    Marks are trailing, so the search/fallback window is asymmetric and
    generous (`marks_cfg`), unlike an audio peak's own tight
    lookback/post_peak. Rule: search the mark's window for an existing
    audio peak; if one exists, anchor there and apply the normal (tighter)
    `timeline_cfg` lookback/post_peak -- audio gives a more precise
    boundary than a press timestamp does. If none exists, fall back to the
    full asymmetric window. Which path fired is recorded on every
    ResolvedMark -- that flag IS the recall measurement this whole feature
    exists to produce."""
    sorted_peaks = sorted(audio_peaks, key=lambda p: p.time_seconds)
    resolved: list[ResolvedMark] = []
    for mark in marks:
        local_ts = mark.timestamp.astimezone(RECORDING_TZ).replace(tzinfo=None)
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

        nearby = [p for p in sorted_peaks if window_start <= p.time_seconds <= window_end]
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
