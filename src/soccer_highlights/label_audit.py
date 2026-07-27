"""Audits the existing human-labeled Round 2 dataset (output/review/*)
against a fresh, independent AI read of the same clips, instead of
tuning detection thresholds further against it.

Three straight rounds of prompt tuning (see vision.py/vision_gemini.py)
all failed to clearly beat audio alone against testdata/golden_events.json
-- which is itself derived from this same Round 2 labeled data via
golden.build_golden_events(). Before assuming the AI is the bottleneck,
this checks whether the labels are: for every already-labeled row (every
strategy sheet's TP/FP clips, every negative-space TN/FN gap), Gemini
watches the clip fresh (no verdict/notes shown to it) and writes a
free-text scene description -- deliberately NOT a yes/no+confidence
judgment, since three rounds of exactly that shape all regressed (see
README). Claude then judges, from the human's verdict+notes and Gemini's
description alone, whether they agree.

This is a data-quality audit, not a detector -- the output is a ranked
list of disagreements for a human to actually watch and decide on, not
an automated relabeling. Pure decision logic (row parsing, response
parsing, flagging/sorting) is unit-tested against injected data; the
ffmpeg clip extraction and both API calls are not (same split as
vision.py/vision_gemini.py) -- validate with `label-audit --limit N`
against real footage first.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from soccer_highlights.config import GeminiConfig, VisionConfig
from soccer_highlights.discovery import Chunk
from soccer_highlights.timeline import Interval
from soccer_highlights.vision_gemini import _call_gemini, _extract_peak_clip

_DESCRIBE_PROMPT = """You are analyzing a short video clip (with audio) from a Sunday recreational soccer game (16-player, half-court). The camera is fixed near one goal, facing the field center -- action at this end is clearly visible, the far end of the field is not. This clip is {duration:.1f} seconds long.

This clip is being reviewed as part of building an automatic highlight-reel generator so the players' friends and family can watch the best moments after the game. The goal is to identify genuinely exciting, highlight-worthy moments -- goals, saves, shots on target, contested action near a goal, celebrations -- versus mundane stretches: players standing around, warm-ups, practice drills during a stoppage, routine passing with no shot, camera setup.

Describe what actually happens in this clip in 2-4 sentences. Mention both what you see and what you hear (ball-strike sounds, cheering, shouting). Be concrete and factual, and explicitly call out anything highlight-worthy if present -- or say plainly that nothing highlight-worthy happens, and describe what IS happening instead. Do not invent player names, jersey numbers, or the exact score unless clearly legible or audible.

Respond with ONLY a single JSON object, no other text, no code fence:
{{"description": "2-4 sentences describing what happens in this clip"}}"""

_JUDGE_PROMPT = """You are auditing ground-truth labels for a soccer-highlight-detection dataset. A human reviewer watched a short video clip and recorded a verdict and (optionally) a note. Separately, an AI model watched the same clip and wrote a free-text description, without seeing the human's verdict or note.

Human verdict: {verdict}
(TP = a real highlight-worthy moment was confirmed here; FP = flagged by an audio detector but not actually highlight-worthy; TN = correctly nothing happened in this audio-uncovered gap; FN = the human found a real highlight-worthy moment audio missed entirely.)
Human's note (may be empty): {notes}

AI's description of the same clip: {description}

Judge whether the AI's description is CONSISTENT with the human's verdict/note -- do they agree about whether something highlight-worthy (a goal, save, shot on target, or similar) happened in this clip?
- "consistent": the AI's description and the human's verdict clearly agree.
- "human_likely_wrong": the AI's description describes something highlight-worthy that the human's verdict says didn't happen (or vice versa), and the AI's account seems credible enough to warrant a second look at the human label.
- "ai_likely_wrong": the human's verdict/note is specific and credible, and the AI's description seems to have missed or misdescribed what actually happened.
- "ambiguous": genuinely unclear either way (e.g. an empty human note and a non-committal description).

Respond with ONLY a single JSON object, no other text, no code fence:
{{"agreement": "consistent" | "human_likely_wrong" | "ai_likely_wrong" | "ambiguous", "distance_score": <0.0 (fully consistent) to 1.0 (flatly contradictory)>, "rationale": "one short sentence explaining the judgment"}}"""


@dataclass
class LabeledRow:
    strategy: str  # e.g. "strike_loose", or "negatives"
    clip_file: str
    interval: Interval
    verdict: str  # TP / FP / TN / FN
    notes: str


@dataclass
class JudgeVerdict:
    agreement: str
    distance_score: float
    rationale: str


@dataclass
class AuditRow:
    row: LabeledRow
    description: str | None
    judge: JudgeVerdict | None


# ---------------------------------------------------------------------------
# Pure logic -- unit tested, no I/O.
# ---------------------------------------------------------------------------

_VALID_AGREEMENTS = {"consistent", "human_likely_wrong", "ai_likely_wrong", "ambiguous"}


def parse_review_sheet_rows(strategy: str, csv_rows: list[dict]) -> list[LabeledRow]:
    """Turn already-loaded review_sheet.csv rows (csv.DictReader dicts)
    into LabeledRow objects. Skips rows with no verdict recorded yet
    (blank -- not every row in a sheet is necessarily labeled)."""
    rows: list[LabeledRow] = []
    for row in csv_rows:
        verdict = row["verdict"].strip().upper()
        if not verdict:
            continue
        rows.append(
            LabeledRow(
                strategy=strategy,
                clip_file=row["clip_file"],
                interval=Interval(start_seconds=float(row["start_seconds"]), end_seconds=float(row["end_seconds"])),
                verdict=verdict,
                notes=row.get("notes", "").strip(),
            )
        )
    return rows


def _parse_describe_json(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    data = json.loads(text)
    description = str(data["description"]).strip()
    if not description:
        raise ValueError("empty description")
    return description


def _parse_judge_json(raw_text: str) -> JudgeVerdict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    data = json.loads(text)
    agreement = str(data["agreement"]).strip()
    if agreement not in _VALID_AGREEMENTS:
        raise ValueError(f"unknown agreement value: {agreement!r}")
    distance_score = float(data["distance_score"])
    if not 0.0 <= distance_score <= 1.0:
        raise ValueError(f"distance_score out of [0, 1] range: {distance_score}")
    return JudgeVerdict(agreement=agreement, distance_score=distance_score, rationale=str(data.get("rationale", "")))


def is_flagged(judge: JudgeVerdict | None, threshold: float) -> bool:
    """A row is flagged for human review if we couldn't judge it at all
    (no signal -- err toward showing it, not hiding it), the judge
    thinks it's inconsistent, or its distance score clears the bar."""
    if judge is None:
        return True
    return judge.agreement != "consistent" or judge.distance_score >= threshold


def sort_by_disagreement(rows: list[AuditRow]) -> list[AuditRow]:
    """Most-disagreement-first; a failed judge call sorts as if maximally
    disagreeing (1.0), since it needs attention too."""
    return sorted(rows, key=lambda r: r.judge.distance_score if r.judge else 1.0, reverse=True)


# ---------------------------------------------------------------------------
# I/O -- CSV/ffmpeg/network. Not unit tested; validate with
# `label-audit --limit N` against real footage.
# ---------------------------------------------------------------------------


def load_review_rows(review_root: Path) -> list[LabeledRow]:
    """Every already-labeled row across every review_sheet.csv under
    review_root (strategy sheets + negatives)."""
    rows: list[LabeledRow] = []
    for sheet_path in sorted(review_root.glob("*/review_sheet.csv")):
        strategy = sheet_path.parent.name
        with open(sheet_path, encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        rows.extend(parse_review_sheet_rows(strategy, csv_rows))
    return rows


def generate_description(row: LabeledRow, chunks: list[Chunk], cfg: GeminiConfig) -> str | None:
    import os
    import tempfile

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set -- see README's Vision AI section for setup")

    try:
        with tempfile.TemporaryDirectory(prefix="soccer_hl_label_audit_") as tmp_dir_str:
            clip_path = Path(tmp_dir_str) / "clip.mp4"
            extracted = _extract_peak_clip(row.interval, chunks, cfg.clip_max_width, clip_path)
            if extracted is None:
                return None
            prompt = _DESCRIBE_PROMPT.format(duration=row.interval.end_seconds - row.interval.start_seconds)
            raw = _call_gemini(extracted, prompt, cfg, api_key)
            return _parse_describe_json(raw)
    except Exception as exc:
        print(f"WARNING: describe failed for {row.strategy}/{row.clip_file}: {exc}")
        return None


def judge_agreement(row: LabeledRow, description: str, cfg: VisionConfig) -> JudgeVerdict | None:
    import os

    import anthropic

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set -- see README's Vision AI section for setup")

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=cfg.request_timeout_seconds, max_retries=cfg.max_retries)
        prompt = _JUDGE_PROMPT.format(verdict=row.verdict, notes=row.notes or "(none)", description=description)
        response = client.messages.create(
            model=cfg.model, max_tokens=300, messages=[{"role": "user", "content": prompt}]
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
        return _parse_judge_json(raw)
    except Exception as exc:
        print(f"WARNING: judge failed for {row.strategy}/{row.clip_file}: {exc}")
        return None


DescribeFn = Callable[[LabeledRow, list[Chunk], GeminiConfig], "str | None"]
JudgeFn = Callable[[LabeledRow, str, VisionConfig], "JudgeVerdict | None"]


def _entry_from_result(row: LabeledRow, description: str | None, judge: JudgeVerdict | None) -> dict:
    return {
        "strategy": row.strategy,
        "clip_file": row.clip_file,
        "start_seconds": round(row.interval.start_seconds, 2),
        "end_seconds": round(row.interval.end_seconds, 2),
        "verdict": row.verdict,
        "notes": row.notes,
        "description": description,
        "judge": None
        if judge is None
        else {"agreement": judge.agreement, "distance_score": judge.distance_score, "rationale": judge.rationale},
    }


def run_audit(
    rows: list[LabeledRow],
    chunks: list[Chunk],
    gemini_cfg: GeminiConfig,
    vision_cfg: VisionConfig,
    cache_path: Path,
    describe_fn: DescribeFn | None = None,
    judge_fn: JudgeFn | None = None,
) -> list[AuditRow]:
    """Describe + judge every row, resuming from cache_path -- matched by
    position, so `rows` must come from the same load_review_rows() call
    across resumed runs. A cached description or judge verdict is reused
    as-is; a cached `null` for either is retried (same "retry only what
    actually failed" fix as vision_eval.collect_verdicts) -- so a run
    that got descriptions but failed midway through judging only re-pays
    for the missing judge calls, not the Gemini calls too."""
    describe_fn = describe_fn or generate_description
    judge_fn = judge_fn or judge_agreement

    cached: list[dict] = []
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
    entries: list[dict | None] = list(cached) + [None] * max(0, len(rows) - len(cached))

    results: list[AuditRow] = []
    for i, row in enumerate(rows):
        entry = entries[i]
        if entry is not None and (entry["strategy"] != row.strategy or entry["clip_file"] != row.clip_file):
            raise ValueError(
                f"Cache mismatch at index {i}: cached {entry['strategy']}/{entry['clip_file']}, "
                f"actual {row.strategy}/{row.clip_file}. Delete {cache_path} and rerun."
            )

        description = entry.get("description") if entry else None
        judge_dict = entry.get("judge") if entry else None
        judge = JudgeVerdict(**judge_dict) if judge_dict else None

        if description is None:
            description = describe_fn(row, chunks, gemini_cfg)
        if description is not None and judge is None:
            judge = judge_fn(row, description, vision_cfg)

        results.append(AuditRow(row=row, description=description, judge=judge))
        entries[i] = _entry_from_result(row, description, judge)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(entries[: i + 1], f, indent=2)

    return results


def run_describe_only(
    rows: list[LabeledRow],
    chunks: list[Chunk],
    gemini_cfg: GeminiConfig,
    cache_path: Path,
    describe_fn: DescribeFn | None = None,
) -> list[str | None]:
    """Generate a Gemini description for every row with no judge step --
    for fresh, not-yet-labeled candidates (e.g. a brand-new game's
    audio-flagged clips), not an audit of an existing label. Resumable
    the same way run_audit is: a cached description is reused, a cached
    `null` (failed attempt) is retried."""
    describe_fn = describe_fn or generate_description

    cached: list[dict] = []
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
    entries: list[dict | None] = list(cached) + [None] * max(0, len(rows) - len(cached))

    results: list[str | None] = []
    for i, row in enumerate(rows):
        entry = entries[i]
        if entry is not None and (entry["strategy"] != row.strategy or entry["clip_file"] != row.clip_file):
            raise ValueError(
                f"Cache mismatch at index {i}: cached {entry['strategy']}/{entry['clip_file']}, "
                f"actual {row.strategy}/{row.clip_file}. Delete {cache_path} and rerun."
            )

        description = entry.get("description") if entry else None
        if description is None:
            description = describe_fn(row, chunks, gemini_cfg)

        results.append(description)
        entries[i] = {
            "strategy": row.strategy,
            "clip_file": row.clip_file,
            "start_seconds": round(row.interval.start_seconds, 2),
            "end_seconds": round(row.interval.end_seconds, 2),
            "description": description,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(entries[: i + 1], f, indent=2)

    return results


def write_report_csv(audit_rows: list[AuditRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "strategy", "clip_file", "start_seconds", "end_seconds", "duration_seconds",
                "original_verdict", "original_notes", "gemini_description",
                "judge_agreement", "judge_distance_score", "judge_rationale",
                "new_label", "new_notes",
            ]
        )
        for ar in audit_rows:
            row = ar.row
            judge = ar.judge
            writer.writerow(
                [
                    row.strategy,
                    row.clip_file,
                    f"{row.interval.start_seconds:.2f}",
                    f"{row.interval.end_seconds:.2f}",
                    f"{row.interval.end_seconds - row.interval.start_seconds:.2f}",
                    row.verdict,
                    row.notes,
                    ar.description or "",
                    judge.agreement if judge else "",
                    f"{judge.distance_score:.2f}" if judge else "",
                    judge.rationale if judge else "",
                    "",  # new_label -- blank, for the user to fill in
                    "",  # new_notes -- blank, for the user to fill in
                ]
            )
