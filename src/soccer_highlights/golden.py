"""Build a reusable "golden" set of ground-truth event timestamps from a
fully human-labeled batch-review round, and score candidate detection
output against it without rendering any clips or asking a human to
re-watch anything.

Once a round is fully labeled -- every strategy sheet's `verdict` filled
in, and every negative-space `FN` row's `notes` carrying an exact "N
seconds in" offset -- the underlying real-world event times can be
extracted once and reused indefinitely: further tuning rounds only need to
run detection (audio-only, no rendering) and compare its intervals against
these known timestamps.

A TP clip's exact "moment" isn't in the CSV, so it's approximated by the
clip's strongest detected peak time (from that strategy's events.json,
same row order as the review sheet) -- close enough given round 2's short,
tightly-anchored clips. The same real event is often independently caught
by several strategies with slightly different peak times, so nearby
anchors (within `cluster_window_seconds`) are collapsed to one event.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from soccer_highlights.timeline import Interval

_OFFSET_RE = re.compile(r"(\d+(?:\.\d+)?)\s*sec")


@dataclass
class GoldenEvent:
    time_seconds: float
    sources: list[str]  # clip(s) this was derived from, for traceability


def _read_sheet(sheet_path: Path) -> list[dict]:
    with open(sheet_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_events(events_path: Path) -> list[dict]:
    with open(events_path, encoding="utf-8") as f:
        return json.load(f)


def _tp_anchor_times(review_root: Path) -> list[tuple[float, str]]:
    anchors: list[tuple[float, str]] = []
    for d in sorted(review_root.iterdir()):
        if not d.is_dir() or d.name == "negatives":
            continue
        sheet_path = d / "review_sheet.csv"
        events_path = d / "events.json"
        if not sheet_path.exists() or not events_path.exists():
            continue
        rows = _read_sheet(sheet_path)
        events = _read_events(events_path)
        for row, event in zip(rows, events):
            if row["verdict"].strip().upper() != "TP":
                continue
            peaks = event.get("peaks", [])
            if not peaks:
                continue
            best = max(peaks, key=lambda p: p["score"])
            anchors.append((best["time_seconds"], f"{d.name}/{row['clip_file']}"))
    return anchors


def _fn_times(review_root: Path) -> list[tuple[float, str]]:
    negatives_sheet = review_root / "negatives" / "review_sheet.csv"
    if not negatives_sheet.exists():
        return []
    times: list[tuple[float, str]] = []
    for row in _read_sheet(negatives_sheet):
        if row["verdict"].strip().upper() != "FN":
            continue
        match = _OFFSET_RE.search(row["notes"])
        if not match:
            raise ValueError(
                f"FN row {row['clip_file']} has no parseable 'N seconds' offset in notes: {row['notes']!r}"
            )
        offset = float(match.group(1))
        times.append((float(row["start_seconds"]) + offset, f"negatives/{row['clip_file']}"))
    return times


def build_golden_events(review_root: Path, cluster_window_seconds: float = 12.0) -> list[GoldenEvent]:
    """Combine every TP clip's detected peak time and every FN's exact
    offset into one deduplicated list of real-world event timestamps."""
    raw = sorted(_tp_anchor_times(review_root) + _fn_times(review_root))
    if not raw:
        return []

    clusters: list[list[tuple[float, str]]] = [[raw[0]]]
    for time_seconds, source in raw[1:]:
        if time_seconds - clusters[-1][-1][0] <= cluster_window_seconds:
            clusters[-1].append((time_seconds, source))
        else:
            clusters.append([(time_seconds, source)])

    return [
        GoldenEvent(
            time_seconds=sum(t for t, _ in cluster) / len(cluster),
            sources=[s for _, s in cluster],
        )
        for cluster in clusters
    ]


def save_golden_events(events: list[GoldenEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"time_seconds": round(e.time_seconds, 2), "sources": e.sources} for e in events]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_golden_events(path: Path) -> list[GoldenEvent]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [GoldenEvent(time_seconds=d["time_seconds"], sources=d["sources"]) for d in data]


@dataclass
class GoldenScore:
    total_clips: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    total_duration_seconds: float
    max_clip_duration_seconds: float
    mean_clip_duration_seconds: float | None


def score_intervals_against_golden(intervals: list[Interval], golden_events: list[GoldenEvent]) -> GoldenScore:
    """Score a candidate strategy's merged highlight intervals against a
    golden event set: a clip is a TP if it contains >=1 golden event, FP
    otherwise; a golden event is "found" if any clip contains it.

    Containment alone is gameable: a config loose enough to merge into a
    handful of multi-hundred-second blobs trivially "contains" almost
    every event and looks like perfect precision/recall while being
    useless (defeats the whole point of short, focused clips). Always
    check `max_clip_duration_seconds`/`total_duration_seconds` alongside
    precision/recall/F1 -- don't rank candidates on F1 alone."""
    tp_clips = 0
    fp_clips = 0
    found = [False] * len(golden_events)
    durations = [iv.end_seconds - iv.start_seconds for iv in intervals]

    for interval in intervals:
        contains_any = False
        for i, event in enumerate(golden_events):
            if interval.start_seconds <= event.time_seconds <= interval.end_seconds:
                contains_any = True
                found[i] = True
        if contains_any:
            tp_clips += 1
        else:
            fp_clips += 1

    false_negatives = found.count(False)
    precision = tp_clips / (tp_clips + fp_clips) if intervals else None
    recall = (len(golden_events) - false_negatives) / len(golden_events) if golden_events else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return GoldenScore(
        total_clips=len(intervals),
        true_positives=tp_clips,
        false_positives=fp_clips,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        total_duration_seconds=sum(durations),
        max_clip_duration_seconds=max(durations) if durations else 0.0,
        mean_clip_duration_seconds=(sum(durations) / len(durations)) if durations else None,
        f1=f1,
    )
