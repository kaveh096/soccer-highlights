"""Turn a batch-review run's clips into fillable scoring sheets, and turn
filled-in sheets back into precision/recall/F1 per strategy.

Workflow:
  1. After `batch-review`, run `review-sheet` to generate one CSV per
     strategy (+ one for negatives) next to that strategy's clips.
  2. Watch the clips, fill in the `verdict` column:
       - strategy sheets: TP (real shot-on-target) or FP (not one)
       - negatives sheet: FN (a real event fell in this uncovered gap) or
         TN (correctly nothing there)
  3. Run `score` to compute precision/recall/F1 per strategy. Ground truth
     is defined as the union of every clip marked TP across every
     strategy (merged when overlapping), plus each FN-marked negative gap
     -- so recall is relative to everything found across the whole batch,
     not an independent ground truth (see session notes on why that's a
     limitation, not just a simplification).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from soccer_highlights.timeline import Interval, merge_intervals

STRATEGY_VERDICTS = {"TP", "FP"}
NEGATIVE_VERDICTS = {"FN"}
ALL_VERDICTS = {"TP", "FP", "TN", "FN"}
INTERESTING_VERDICTS = {"TP", "FN"}  # verdicts confirming a real event was present


def _read_events(events_path: Path) -> list[dict]:
    with open(events_path, encoding="utf-8") as f:
        return json.load(f)


def _read_sheet(sheet_path: Path) -> list[dict]:
    with open(sheet_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class PriorLabel:
    interval: Interval
    verdict: str  # one of ALL_VERDICTS
    notes: str
    source: str  # e.g. "crowd_loose/clip_004.mp4"


def load_prior_labels(prior_root: Path) -> list[PriorLabel]:
    """Read every labeled clip (TP/FP/TN/FN) out of a previous round's
    review_root, for use as a time-overlap basis to guess verdicts on a
    freshly re-run batch."""
    labels: list[PriorLabel] = []
    for sub in sorted(prior_root.iterdir()):
        sheet_path = sub / "review_sheet.csv"
        if not sub.is_dir() or not sheet_path.exists():
            continue
        for row in _read_sheet(sheet_path):
            verdict = row["verdict"].strip().upper()
            if verdict not in ALL_VERDICTS:
                continue
            labels.append(
                PriorLabel(
                    interval=Interval(
                        start_seconds=float(row["start_seconds"]), end_seconds=float(row["end_seconds"])
                    ),
                    verdict=verdict,
                    notes=row.get("notes", "").strip(),
                    source=f"{sub.name}/{row['clip_file']}",
                )
            )
    return labels


def _guess_verdict(interval: Interval, prior_labels: list[PriorLabel]) -> tuple[str, str]:
    """Guess TP/FP for a new interval from time-overlapping prior labels.
    Returns (guess, basis) -- guess is "" if there's nothing to go on, or
    "AMBIGUOUS" if overlapping prior labels disagree on whether something
    interesting was there."""
    overlapping = [p for p in prior_labels if _overlaps(interval, p.interval)]
    if not overlapping:
        return "", "no overlapping prior label"

    classes = {p.verdict in INTERESTING_VERDICTS for p in overlapping}
    basis = "; ".join(f"{p.source}={p.verdict}" + (f" ({p.notes})" if p.notes else "") for p in overlapping)

    if classes == {True}:
        return "TP", basis
    if classes == {False}:
        return "FP", basis
    return "AMBIGUOUS", basis


def generate_review_sheet(strategy_dir: Path, prior_labels: list[PriorLabel] | None = None) -> Path:
    """Build (or overwrite) a fillable review_sheet.csv for one strategy
    (or the negatives) directory, from its events.json + clip_NNN.mp4 files.

    If `prior_labels` is given (from a previous round's `load_prior_labels`),
    each row also gets a `guess`/`guess_basis` column: a best-effort TP/FP
    prediction from time-overlap with the old labels, for the human to
    verify rather than re-label from scratch. This never touches the
    `verdict` column itself -- that stays blank until a human confirms it."""
    events = _read_events(strategy_dir / "events.json")
    sheet_path = strategy_dir / "review_sheet.csv"
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["clip_file", "start_seconds", "end_seconds", "duration_seconds", "max_peak_score", "verdict", "notes"]
        if prior_labels is not None:
            header += ["guess", "guess_basis"]
        writer.writerow(header)
        for i, event in enumerate(events, start=1):
            max_score = max((p["score"] for p in event.get("peaks", [])), default=0.0)
            row = [
                f"clip_{i:03d}.mp4",
                event["start_seconds"],
                event["end_seconds"],
                event["duration_seconds"],
                round(max_score, 4),
                "",
                "",
            ]
            if prior_labels is not None:
                interval = Interval(start_seconds=event["start_seconds"], end_seconds=event["end_seconds"])
                guess, basis = _guess_verdict(interval, prior_labels)
                row += [guess, basis]
            writer.writerow(row)
    return sheet_path


def generate_all_review_sheets(review_root: Path, prior_root: Path | None = None) -> list[Path]:
    prior_labels = load_prior_labels(prior_root) if prior_root is not None else None
    sheets = []
    for sub in sorted(review_root.iterdir()):
        if sub.is_dir() and (sub / "events.json").exists():
            sheets.append(generate_review_sheet(sub, prior_labels))
    return sheets


@dataclass
class StrategyScore:
    name: str
    total_clips: int
    labeled_clips: int
    true_positives: int
    false_positives: int
    precision: float | None
    recall: float | None
    f1: float | None


def _overlaps(a: Interval, b: Interval) -> bool:
    return a.start_seconds < b.end_seconds and a.end_seconds > b.start_seconds


def score_all(review_root: Path) -> tuple[list[StrategyScore], list[Interval]]:
    strategy_dirs = [
        d
        for d in sorted(review_root.iterdir())
        if d.is_dir() and d.name != "negatives" and (d / "review_sheet.csv").exists()
    ]

    tp_intervals_by_strategy: dict[str, list[Interval]] = {}
    fp_counts: dict[str, int] = {}
    labeled_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}

    for d in strategy_dirs:
        rows = _read_sheet(d / "review_sheet.csv")
        total_counts[d.name] = len(rows)
        tp_intervals: list[Interval] = []
        fp = 0
        labeled = 0
        for row in rows:
            verdict = row["verdict"].strip().upper()
            if verdict not in STRATEGY_VERDICTS:
                continue
            labeled += 1
            if verdict == "TP":
                tp_intervals.append(
                    Interval(start_seconds=float(row["start_seconds"]), end_seconds=float(row["end_seconds"]))
                )
            else:
                fp += 1
        tp_intervals_by_strategy[d.name] = tp_intervals
        fp_counts[d.name] = fp
        labeled_counts[d.name] = labeled

    all_tp = [iv for intervals in tp_intervals_by_strategy.values() for iv in intervals]

    negatives_dir = review_root / "negatives"
    negatives_sheet = negatives_dir / "review_sheet.csv"
    if negatives_sheet.exists():
        for row in _read_sheet(negatives_sheet):
            if row["verdict"].strip().upper() in NEGATIVE_VERDICTS:
                all_tp.append(Interval(start_seconds=float(row["start_seconds"]), end_seconds=float(row["end_seconds"])))

    ground_truth = merge_intervals(all_tp, min_gap_seconds=0.0, min_interval_seconds=0.0)

    scores = []
    for d in strategy_dirs:
        name = d.name
        tp_intervals = tp_intervals_by_strategy[name]
        fp = fp_counts[name]
        tp = len(tp_intervals)
        precision = tp / (tp + fp) if (tp + fp) else None

        covered_gt = sum(1 for gt in ground_truth if any(_overlaps(iv, gt) for iv in tp_intervals))
        recall = covered_gt / len(ground_truth) if ground_truth else None

        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)

        scores.append(
            StrategyScore(
                name=name,
                total_clips=total_counts[name],
                labeled_clips=labeled_counts[name],
                true_positives=tp,
                false_positives=fp,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    return scores, ground_truth


def format_score_report(scores: list[StrategyScore], ground_truth: list[Interval]) -> str:
    lines = [f"Ground truth events (union of all TP + FN-flagged negatives): {len(ground_truth)}", ""]
    header = f"{'strategy':<16}{'labeled/total':<16}{'TP':<6}{'FP':<6}{'precision':<12}{'recall':<12}{'F1':<8}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in scores:
        precision_str = f"{s.precision:.2f}" if s.precision is not None else "n/a"
        recall_str = f"{s.recall:.2f}" if s.recall is not None else "n/a"
        f1_str = f"{s.f1:.2f}" if s.f1 is not None else "n/a"
        lines.append(
            f"{s.name:<16}{f'{s.labeled_clips}/{s.total_clips}':<16}{s.true_positives:<6}{s.false_positives:<6}"
            f"{precision_str:<12}{recall_str:<12}{f1_str:<8}"
        )
    return "\n".join(lines)
