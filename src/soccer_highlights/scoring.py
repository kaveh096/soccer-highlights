"""Turn a batch-review run's clips into fillable scoring sheets, and turn
filled-in sheets back into precision/recall/F1 per strategy.

Workflow:
  1. After `batch-review`, run `review-sheet` to generate one CSV per
     strategy (+ one for negatives) next to that strategy's clips.
  2. Watch the clips, fill in the `verdict` column:
       - strategy sheets: TP (real shot-on-target) or FP (not one)
       - negatives sheet: MISS (a real event fell in this uncovered gap)
         or leave blank (nothing there)
  3. Run `score` to compute precision/recall/F1 per strategy. Ground truth
     is defined as the union of every clip marked TP across every
     strategy (merged when overlapping), plus each MISS-marked negative
     gap -- so recall is relative to everything found across the whole
     batch, not an independent ground truth (see session notes on why
     that's a limitation, not just a simplification).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from soccer_highlights.timeline import Interval, merge_intervals

STRATEGY_VERDICTS = {"TP", "FP"}
NEGATIVE_VERDICTS = {"MISS"}


def _read_events(events_path: Path) -> list[dict]:
    with open(events_path, encoding="utf-8") as f:
        return json.load(f)


def generate_review_sheet(strategy_dir: Path) -> Path:
    """Build (or overwrite) a fillable review_sheet.csv for one strategy
    (or the negatives) directory, from its events.json + clip_NNN.mp4 files."""
    events = _read_events(strategy_dir / "events.json")
    sheet_path = strategy_dir / "review_sheet.csv"
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["clip_file", "start_seconds", "end_seconds", "duration_seconds", "max_peak_score", "verdict", "notes"]
        )
        for i, event in enumerate(events, start=1):
            max_score = max((p["score"] for p in event.get("peaks", [])), default=0.0)
            writer.writerow(
                [
                    f"clip_{i:03d}.mp4",
                    event["start_seconds"],
                    event["end_seconds"],
                    event["duration_seconds"],
                    round(max_score, 4),
                    "",
                    "",
                ]
            )
    return sheet_path


def generate_all_review_sheets(review_root: Path) -> list[Path]:
    sheets = []
    for sub in sorted(review_root.iterdir()):
        if sub.is_dir() and (sub / "events.json").exists():
            sheets.append(generate_review_sheet(sub))
    return sheets


def _read_sheet(sheet_path: Path) -> list[dict]:
    with open(sheet_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    lines = [f"Ground truth events (union of all TP + MISS-flagged negatives): {len(ground_truth)}", ""]
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
