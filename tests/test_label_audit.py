import json

from soccer_highlights.label_audit import (
    AuditRow,
    JudgeVerdict,
    LabeledRow,
    _parse_describe_json,
    _parse_judge_json,
    is_flagged,
    parse_review_sheet_rows,
    run_audit,
    run_describe_only,
    sort_by_disagreement,
    write_report_csv,
)
from soccer_highlights.timeline import Interval


def test_parse_review_sheet_rows_builds_labeled_rows():
    csv_rows = [
        {"clip_file": "clip_001.mp4", "start_seconds": "10.0", "end_seconds": "20.0", "verdict": "TP", "notes": "nice shot"},
        {"clip_file": "clip_002.mp4", "start_seconds": "30.0", "end_seconds": "40.0", "verdict": "fp", "notes": ""},
    ]
    rows = parse_review_sheet_rows("strike_loose", csv_rows)

    assert len(rows) == 2
    assert rows[0] == LabeledRow(
        strategy="strike_loose", clip_file="clip_001.mp4",
        interval=Interval(start_seconds=10.0, end_seconds=20.0), verdict="TP", notes="nice shot",
    )
    assert rows[1].verdict == "FP"  # normalized to uppercase


def test_parse_review_sheet_rows_skips_unlabeled_rows():
    csv_rows = [{"clip_file": "clip_001.mp4", "start_seconds": "0", "end_seconds": "10", "verdict": "", "notes": ""}]
    assert parse_review_sheet_rows("negatives", csv_rows) == []


def test_parse_describe_json_well_formed():
    raw = json.dumps({"description": "Players take a shot on goal, ball saved by keeper."})
    assert _parse_describe_json(raw) == "Players take a shot on goal, ball saved by keeper."


def test_parse_describe_json_strips_code_fence():
    raw = "```json\n" + json.dumps({"description": "Routine passing, nothing happens."}) + "\n```"
    assert _parse_describe_json(raw) == "Routine passing, nothing happens."


def test_parse_describe_json_empty_description_raises():
    try:
        _parse_describe_json(json.dumps({"description": "   "}))
        assert False, "expected an exception"
    except ValueError:
        pass


def test_parse_judge_json_well_formed():
    raw = json.dumps({"agreement": "consistent", "distance_score": 0.1, "rationale": "both agree"})
    verdict = _parse_judge_json(raw)
    assert verdict.agreement == "consistent"
    assert verdict.distance_score == 0.1


def test_parse_judge_json_invalid_agreement_raises():
    try:
        _parse_judge_json(json.dumps({"agreement": "sort_of", "distance_score": 0.5}))
        assert False, "expected an exception"
    except ValueError:
        pass


def test_parse_judge_json_distance_out_of_range_raises():
    try:
        _parse_judge_json(json.dumps({"agreement": "consistent", "distance_score": 1.5}))
        assert False, "expected an exception"
    except ValueError:
        pass


def _row(strategy="strike_loose", clip="clip_001.mp4", verdict="TP") -> LabeledRow:
    return LabeledRow(strategy=strategy, clip_file=clip, interval=Interval(0.0, 10.0), verdict=verdict, notes="")


def test_is_flagged_none_judge_is_flagged():
    assert is_flagged(None, threshold=0.5) is True


def test_is_flagged_consistent_low_distance_not_flagged():
    verdict = JudgeVerdict(agreement="consistent", distance_score=0.1, rationale="")
    assert is_flagged(verdict, threshold=0.5) is False


def test_is_flagged_consistent_but_high_distance_is_flagged():
    verdict = JudgeVerdict(agreement="consistent", distance_score=0.6, rationale="")
    assert is_flagged(verdict, threshold=0.5) is True


def test_is_flagged_inconsistent_agreement_is_flagged_regardless_of_distance():
    verdict = JudgeVerdict(agreement="human_likely_wrong", distance_score=0.2, rationale="")
    assert is_flagged(verdict, threshold=0.5) is True


def test_sort_by_disagreement_orders_descending_and_puts_failures_first():
    rows = [
        AuditRow(row=_row(clip="low.mp4"), description="d", judge=JudgeVerdict("consistent", 0.1, "")),
        AuditRow(row=_row(clip="failed.mp4"), description=None, judge=None),
        AuditRow(row=_row(clip="high.mp4"), description="d", judge=JudgeVerdict("ai_likely_wrong", 0.9, "")),
    ]
    ordered = sort_by_disagreement(rows)
    clip_order = [ar.row.clip_file for ar in ordered]
    assert clip_order[0] in ("failed.mp4", "high.mp4")  # both are distance 1.0 vs 0.9, failed treated as 1.0
    assert clip_order[-1] == "low.mp4"


def test_run_audit_calls_describe_then_judge_and_caches(tmp_path):
    rows = [_row(clip="clip_001.mp4", verdict="TP"), _row(clip="clip_002.mp4", verdict="FP")]
    describe_calls = []
    judge_calls = []

    def fake_describe(row, chunks, cfg):
        describe_calls.append(row.clip_file)
        return f"description for {row.clip_file}"

    def fake_judge(row, description, cfg):
        judge_calls.append((row.clip_file, description))
        return JudgeVerdict(agreement="consistent", distance_score=0.0, rationale="ok")

    cache_path = tmp_path / "audit.json"
    results = run_audit(rows, chunks=[], gemini_cfg=None, vision_cfg=None, cache_path=cache_path, describe_fn=fake_describe, judge_fn=fake_judge)

    assert describe_calls == ["clip_001.mp4", "clip_002.mp4"]
    assert len(judge_calls) == 2
    assert all(ar.judge.agreement == "consistent" for ar in results)
    assert cache_path.exists()


def test_run_audit_retries_only_missing_judge_not_description(tmp_path):
    rows = [_row(clip="clip_001.mp4")]
    cache_path = tmp_path / "audit.json"
    cache_path.write_text(
        json.dumps([{"strategy": "strike_loose", "clip_file": "clip_001.mp4", "start_seconds": 0.0, "end_seconds": 10.0,
                      "verdict": "TP", "notes": "", "description": "already described", "judge": None}]),
        encoding="utf-8",
    )
    describe_calls = []
    judge_calls = []

    def fake_describe(row, chunks, cfg):
        describe_calls.append(row.clip_file)
        return "should not be called"

    def fake_judge(row, description, cfg):
        judge_calls.append(description)
        return JudgeVerdict(agreement="ambiguous", distance_score=0.5, rationale="")

    results = run_audit(rows, chunks=[], gemini_cfg=None, vision_cfg=None, cache_path=cache_path, describe_fn=fake_describe, judge_fn=fake_judge)

    assert describe_calls == []  # description was cached, not re-fetched
    assert judge_calls == ["already described"]  # judge was retried with the cached description
    assert results[0].judge.agreement == "ambiguous"


def test_write_report_csv_includes_blank_new_label_columns(tmp_path):
    audit_rows = [
        AuditRow(row=_row(), description="a description", judge=JudgeVerdict("consistent", 0.1, "matches")),
    ]
    out_path = tmp_path / "report.csv"
    write_report_csv(audit_rows, out_path)

    content = out_path.read_text(encoding="utf-8")
    assert "new_label" in content
    assert "new_notes" in content
    assert "a description" in content
    assert content.strip().endswith(",")  # last two columns (new_label, new_notes) are blank


def test_run_describe_only_calls_describe_for_every_row_no_judge(tmp_path):
    rows = [_row(clip="clip_001.mp4"), _row(clip="clip_002.mp4")]
    describe_calls = []

    def fake_describe(row, chunks, cfg):
        describe_calls.append(row.clip_file)
        return f"description for {row.clip_file}"

    cache_path = tmp_path / "descriptions.json"
    results = run_describe_only(rows, chunks=[], gemini_cfg=None, cache_path=cache_path, describe_fn=fake_describe)

    assert describe_calls == ["clip_001.mp4", "clip_002.mp4"]
    assert results == ["description for clip_001.mp4", "description for clip_002.mp4"]
    assert cache_path.exists()


def test_run_describe_only_resumes_and_retries_only_failed_entries(tmp_path):
    rows = [_row(clip="clip_001.mp4"), _row(clip="clip_002.mp4")]
    cache_path = tmp_path / "descriptions.json"
    cache_path.write_text(
        json.dumps(
            [
                {"strategy": "strike_loose", "clip_file": "clip_001.mp4", "start_seconds": 0.0, "end_seconds": 10.0, "description": "already done"},
                {"strategy": "strike_loose", "clip_file": "clip_002.mp4", "start_seconds": 0.0, "end_seconds": 10.0, "description": None},
            ]
        ),
        encoding="utf-8",
    )
    describe_calls = []

    def fake_describe(row, chunks, cfg):
        describe_calls.append(row.clip_file)
        return "freshly described"

    results = run_describe_only(rows, chunks=[], gemini_cfg=None, cache_path=cache_path, describe_fn=fake_describe)

    assert describe_calls == ["clip_002.mp4"]  # only the previously-failed one was retried
    assert results == ["already done", "freshly described"]


def test_run_describe_only_detects_cache_mismatch(tmp_path):
    rows = [_row(clip="clip_999.mp4")]
    cache_path = tmp_path / "descriptions.json"
    cache_path.write_text(
        json.dumps([{"strategy": "strike_loose", "clip_file": "clip_001.mp4", "start_seconds": 0.0, "end_seconds": 10.0, "description": "x"}]),
        encoding="utf-8",
    )
    try:
        run_describe_only(rows, chunks=[], gemini_cfg=None, cache_path=cache_path, describe_fn=lambda r, c, g: "y")
        assert False, "expected a cache mismatch error"
    except ValueError:
        pass
