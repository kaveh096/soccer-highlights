"""Reusable evaluation harness for comparing candidate-window classifiers
(different providers, prompts, or frame-density settings) against the
golden event set -- the vision-refinement counterpart to tuning.py's
audio-only sweep helper. Two pieces:

- `collect_verdicts`: run a classify_fn over every candidate interval,
  caching each verdict to disk as it lands (JSON, incremental) so an
  interrupted run -- these call a real paid API, one request per
  interval -- doesn't lose progress, and resumes automatically if
  `cache_path` already has entries for some of the intervals.
- `sweep_drop_threshold`: given a fixed set of already-collected verdicts,
  score the resulting kept-interval list against golden_events at several
  `drop_confidence_threshold` values, all offline/free -- same idea as
  tuning.py's audio min_score sweep, just one layer up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from soccer_highlights.discovery import Chunk
from soccer_highlights.golden import GoldenEvent, GoldenScore, score_intervals_against_golden
from soccer_highlights.timeline import Interval
from soccer_highlights.vision import VisionVerdict, decide_confirm

ClassifyFn = Callable[[Interval, list[Chunk]], "VisionVerdict | None"]


def _verdict_to_dict(verdict: VisionVerdict | None) -> dict | None:
    if verdict is None:
        return None
    return {
        "is_event": verdict.is_event,
        "confidence": verdict.confidence,
        "frame_index": verdict.frame_index,
        "rationale": verdict.rationale,
        "caption": verdict.caption,
    }


def _verdict_from_dict(data: dict | None) -> VisionVerdict | None:
    if data is None:
        return None
    return VisionVerdict(
        is_event=data["is_event"],
        confidence=data["confidence"],
        frame_index=data.get("frame_index"),
        rationale=data.get("rationale", ""),
        caption=data.get("caption", ""),
    )


def load_cached_verdicts(cache_path: Path) -> list[dict]:
    if not cache_path.exists():
        return []
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def collect_verdicts(
    intervals: list[Interval], chunks: list[Chunk], classify_fn: ClassifyFn, cache_path: Path
) -> list[VisionVerdict | None]:
    """Classify every interval via classify_fn, resuming from cache_path if
    it already has entries -- matched by position, so intervals must come
    from the same deterministic detect_intervals call across resumed
    runs (cache entries also carry start_seconds for a sanity check). A
    cached REAL verdict is reused as-is; a cached `None` (a failed,
    errored, or rate-limited attempt -- not a genuine "no verdict" from a
    successful call) is retried rather than treated as permanently
    resolved, so a partially-failed run (e.g. hit a rate limit partway
    through) can be resumed to actual completion."""
    cached = load_cached_verdicts(cache_path)
    entries: list[dict | None] = list(cached) + [None] * max(0, len(intervals) - len(cached))
    results: list[VisionVerdict | None] = []

    for i, interval in enumerate(intervals):
        entry = entries[i]
        if entry is not None:
            if abs(entry["start_seconds"] - interval.start_seconds) > 0.5:
                raise ValueError(
                    f"Cache mismatch at index {i}: cached start={entry['start_seconds']}, "
                    f"actual start={interval.start_seconds}. Delete {cache_path} and rerun."
                )
            if entry["verdict"] is not None:
                results.append(_verdict_from_dict(entry["verdict"]))
                continue

        verdict = classify_fn(interval, chunks)
        results.append(verdict)
        entries[i] = {
            "start_seconds": round(interval.start_seconds, 2),
            "end_seconds": round(interval.end_seconds, 2),
            "verdict": _verdict_to_dict(verdict),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(entries[: i + 1], f, indent=2)

    return results


@dataclass
class _ThresholdCfg:
    drop_confidence_threshold: float


@dataclass
class SweepPoint:
    threshold: float
    kept: int
    total: int
    score: GoldenScore


def sweep_drop_threshold(
    intervals: list[Interval],
    verdicts: list[VisionVerdict | None],
    golden_events: list[GoldenEvent],
    thresholds: list[float],
) -> list[SweepPoint]:
    """Score the kept-interval list at several drop_confidence_threshold
    values against golden_events, all from already-collected verdicts --
    no new classify_fn calls, same idea as tuning.py's audio min_score
    sweep, one layer up."""
    points = []
    for threshold in thresholds:
        cfg = _ThresholdCfg(drop_confidence_threshold=threshold)
        kept = [iv for iv, vd in zip(intervals, verdicts) if decide_confirm(iv, vd, cfg)]
        points.append(
            SweepPoint(threshold=threshold, kept=len(kept), total=len(intervals), score=score_intervals_against_golden(kept, golden_events))
        )
    return points


def format_sweep_table(points: list[SweepPoint]) -> str:
    lines = [f"{'threshold':>10}  {'kept':>8}  {'precision':>9}  {'recall':>7}  {'f1':>6}  {'FN':>3}"]
    for p in points:
        s = p.score
        precision = f"{s.precision:.3f}" if s.precision is not None else "n/a"
        recall = f"{s.recall:.3f}" if s.recall is not None else "n/a"
        f1 = f"{s.f1:.3f}" if s.f1 is not None else "n/a"
        lines.append(f"{p.threshold:>10.2f}  {p.kept:>4}/{p.total:<3}  {precision:>9}  {recall:>7}  {f1:>6}  {s.false_negatives:>3}")
    return "\n".join(lines)
