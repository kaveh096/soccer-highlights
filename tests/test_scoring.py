import csv
import json
from pathlib import Path

from soccer_highlights.scoring import generate_all_review_sheets, generate_review_sheet, score_all


def _write_events(dir_path: Path, intervals: list[tuple[float, float]]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": end - start,
            "peaks": [{"time_seconds": (start + end) / 2, "score": 5.0}],
        }
        for start, end in intervals
    ]
    with open(dir_path / "events.json", "w", encoding="utf-8") as f:
        json.dump(events, f)


def _write_sheet_verdicts(sheet_path: Path, verdicts: list[str]) -> None:
    with open(sheet_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    for row, verdict in zip(rows, verdicts):
        row["verdict"] = verdict
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_generate_review_sheet_matches_event_count(tmp_path):
    strategy_dir = tmp_path / "crowd_loose"
    _write_events(strategy_dir, [(0.0, 10.0), (20.0, 30.0)])

    sheet_path = generate_review_sheet(strategy_dir)
    with open(sheet_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["clip_file"] == "clip_001.mp4"
    assert rows[0]["verdict"] == ""


def test_generate_all_review_sheets_skips_dirs_without_events(tmp_path):
    _write_events(tmp_path / "crowd_loose", [(0.0, 10.0)])
    (tmp_path / "not_a_strategy").mkdir()

    sheets = generate_all_review_sheets(tmp_path)

    assert len(sheets) == 1
    assert sheets[0].parent.name == "crowd_loose"


def test_score_all_computes_precision_recall_f1(tmp_path):
    # crowd_loose: 3 clips, 2 TP (both real events), 1 FP
    _write_events(tmp_path / "crowd_loose", [(0.0, 10.0), (100.0, 110.0), (200.0, 210.0)])
    generate_review_sheet(tmp_path / "crowd_loose")
    _write_sheet_verdicts(tmp_path / "crowd_loose" / "review_sheet.csv", ["TP", "TP", "FP"])

    # crowd_strict: 1 clip, 1 TP -- overlaps the same real event as crowd_loose's first TP
    _write_events(tmp_path / "crowd_strict", [(2.0, 8.0)])
    generate_review_sheet(tmp_path / "crowd_strict")
    _write_sheet_verdicts(tmp_path / "crowd_strict" / "review_sheet.csv", ["TP"])

    # negatives: one flagged MISS -- a third real event neither strategy found
    _write_events(tmp_path / "negatives", [(300.0, 310.0)])
    generate_review_sheet(tmp_path / "negatives")
    _write_sheet_verdicts(tmp_path / "negatives" / "review_sheet.csv", ["MISS"])

    scores, ground_truth = score_all(tmp_path)

    # 3 canonical ground-truth events: ~[0,10] (merged from both strategies), [100,110], [300,310]
    assert len(ground_truth) == 3

    by_name = {s.name: s for s in scores}
    loose = by_name["crowd_loose"]
    assert loose.true_positives == 2
    assert loose.false_positives == 1
    assert loose.precision == 2 / 3
    assert loose.recall == 2 / 3  # found 2 of the 3 ground-truth events, missed the MISS-flagged one

    strict = by_name["crowd_strict"]
    assert strict.true_positives == 1
    assert strict.false_positives == 0
    assert strict.precision == 1.0
    assert strict.recall == 1 / 3  # only overlaps the shared [0,10]-ish event


def test_score_all_unlabeled_clips_excluded_from_precision(tmp_path):
    _write_events(tmp_path / "crowd_loose", [(0.0, 10.0), (100.0, 110.0)])
    generate_review_sheet(tmp_path / "crowd_loose")
    _write_sheet_verdicts(tmp_path / "crowd_loose" / "review_sheet.csv", ["TP", ""])

    scores, _ = score_all(tmp_path)
    loose = scores[0]

    assert loose.total_clips == 2
    assert loose.labeled_clips == 1
    assert loose.true_positives == 1
    assert loose.false_positives == 0
    assert loose.precision == 1.0
