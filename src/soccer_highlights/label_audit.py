"""Audits the existing human-labeled Round 2 dataset (output/review/*)
against a fresh, independent AI read of the same clips, instead of
tuning detection thresholds further against it.

Three straight rounds of prompt tuning (see vision.py/vision_gemini.py)
all failed to clearly beat audio alone against testdata/golden_events.json
-- which is itself derived from this same Round 2 labeled data via
golden.build_golden_events(). Before assuming the AI is the bottleneck,
this checks whether the labels are: for every already-labeled row (every
strategy sheet's TP/FP clips, every negative-space TN/FN gap), Gemini
watches the clip fresh (no verdict/notes shown to it) and rates it on a
1-5 highlight-worthiness scale (see _DESCRIBE_PROMPT), plus a caption and
a free-text description -- not a yes/no+confidence judgment, since three
rounds of exactly that shape all regressed (see README). Claude then
judges, from the human's verdict+notes and Gemini's score+description,
whether they agree.

The 1-5 scale (2026-07-27 revision, after the first real label-audit run
on 2026-07-26 flagged ~21% of rows -- most disagreements traced back to
what counts as "highlight-worthy" being underspecified, not a perception
failure) replaces an earlier free-text-only version: over-specific game
details (exact player count, field size) were dropped since they vary
game to game, and stable context that doesn't vary (this is a Sunday
morning recreational game between Persian middle-aged men in California,
white vs. dark t-shirt teams, camera fixed next to one goal facing
center) was added instead. A goal always outranks a non-goal regardless
of skill (only a truly exceptional non-goal moment reaches tier 4); the
4-vs-5 boundary is purely outcome (goal or not) -- both deliberate,
confirmed choices, not defaults.

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

_DESCRIBE_PROMPT = """You are analyzing a short video clip (with audio) from a Sunday morning recreational soccer game -- a group of Persian middle-aged men playing a friendly match in California. One team wears white t-shirts, the other wears dark/colored t-shirts. The camera is fixed next to one goal, facing the center of the field -- action at this end is clearly visible, the far end usually is not. This clip is {duration:.1f} seconds long.

Rate how highlight-worthy this clip is on this exact 1-5 scale (pick the single number that fits best -- do not invent in-between values):
1 = No game action: a break, warm-up, or setup/idle time.
2 = Casual in-play action: passing, dribbling, no shot attempt.
3 = A shot on target that missed narrowly or was deflected by the keeper/defense; OR a likely goal (usually at the far end) that wasn't clearly visible from this camera. Even if it sounds exciting (cheering, shouting), it is NOT usable as a highlight clip if you can't clearly see it happen.
4 = A clear, unambiguous goal at this end of the field (any goal counts, even a simple/easy one); OR, if there was no goal, a moment of exceptional, highly skillful play (a dangerous dribble/passing sequence or acrobatic attempt) that was a serious scoring threat but didn't result in a goal (maybe won a corner).
5 = An exceptionally skillful, spectacular play that results in a clear goal at this end of the field -- the rare combination of both outcome AND skill.

A goal always outranks a non-goal, no matter how skillful the non-goal play was -- only a truly exceptional (not just "good") non-goal moment reaches tier 4. Between tiers 4 and 5, the deciding factor is simply whether the ball went in.

Also write:
- a short, factual one-line caption suitable for use in a filename, describing only what's visibly/audibly confirmable (e.g. "Goal from close range, white team celebrates"). Do not invent player names, jersey numbers, or the exact score unless clearly legible/audible.
- a 2-4 sentence description of what actually happens in the clip, mentioning both what you see and what you hear (ball-strike sounds, cheering, shouting).

Respond with ONLY a single JSON object, no other text, no code fence:
{{"score": 1-5, "caption": "short factual caption", "description": "2-4 sentences describing what happens in this clip", "rationale": "one short sentence explaining the score"}}"""

_JUDGE_PROMPT = """You are auditing ground-truth labels for a soccer-highlight-detection dataset. A human reviewer watched a short video clip and recorded a verdict and (optionally) a note. Separately, an AI model watched the same clip and rated it on a 1-5 highlight-worthiness scale and wrote a free-text description, without seeing the human's verdict or note.

Human verdict: {verdict}
(TP = a real highlight-worthy moment was confirmed here; FP = flagged by an audio detector but not actually highlight-worthy; TN = correctly nothing happened in this audio-uncovered gap; FN = the human found a real highlight-worthy moment audio missed entirely.)
Human's note (may be empty): {notes}

AI's highlight-worthiness score (1=no action, 2=casual play, 3=near-miss or an unclear/far-side goal not usable as a highlight, 4=clear goal or exceptional non-goal skill, 5=exceptional skillful goal): {score}
AI's description of the same clip: {description}

Judge whether the AI's assessment (score + description) is CONSISTENT with the human's verdict/note -- do they agree about whether something highlight-worthy (score 4 or 5) happened in this clip?
- "consistent": the AI's assessment and the human's verdict clearly agree.
- "human_likely_wrong": the AI's score/description describes something highlight-worthy that the human's verdict says didn't happen (or vice versa), and the AI's account seems credible enough to warrant a second look at the human label.
- "ai_likely_wrong": the human's verdict/note is specific and credible, and the AI's score/description seems to have missed or misdescribed what actually happened.
- "ambiguous": genuinely unclear either way (e.g. an empty human note and a middling AI score).

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
class DescribeResult:
    score: int  # 1-5 highlight-worthiness, see _DESCRIBE_PROMPT
    caption: str
    description: str
    rationale: str = ""


@dataclass
class JudgeVerdict:
    agreement: str
    distance_score: float
    rationale: str


@dataclass
class AuditRow:
    row: LabeledRow
    description: DescribeResult | None
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


def _parse_describe_json(raw_text: str) -> DescribeResult:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    data = json.loads(text)
    score = int(data["score"])
    if score not in (1, 2, 3, 4, 5):
        raise ValueError(f"score out of [1, 5] range: {score}")
    caption = str(data["caption"]).strip()
    if not caption:
        raise ValueError("empty caption")
    description = str(data["description"]).strip()
    if not description:
        raise ValueError("empty description")
    return DescribeResult(score=score, caption=caption, description=description, rationale=str(data.get("rationale", "")))


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


def generate_description(row: LabeledRow, chunks: list[Chunk], cfg: GeminiConfig) -> DescribeResult | None:
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


def judge_agreement(row: LabeledRow, description: DescribeResult, cfg: VisionConfig) -> JudgeVerdict | None:
    import os

    import anthropic

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set -- see README's Vision AI section for setup")

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=cfg.request_timeout_seconds, max_retries=cfg.max_retries)
        prompt = _JUDGE_PROMPT.format(
            verdict=row.verdict, notes=row.notes or "(none)", score=description.score, description=description.description
        )
        response = client.messages.create(
            model=cfg.model, max_tokens=300, messages=[{"role": "user", "content": prompt}]
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
        return _parse_judge_json(raw)
    except Exception as exc:
        print(f"WARNING: judge failed for {row.strategy}/{row.clip_file}: {exc}")
        return None


DescribeFn = Callable[[LabeledRow, list[Chunk], GeminiConfig], "DescribeResult | None"]
JudgeFn = Callable[[LabeledRow, DescribeResult, VisionConfig], "JudgeVerdict | None"]


def _describe_to_dict(description: DescribeResult | None) -> dict | None:
    if description is None:
        return None
    return {
        "score": description.score,
        "caption": description.caption,
        "description": description.description,
        "rationale": description.rationale,
    }


def _describe_from_dict(data: dict | None) -> DescribeResult | None:
    if data is None:
        return None
    return DescribeResult(
        score=data["score"], caption=data["caption"], description=data["description"], rationale=data.get("rationale", "")
    )


def _entry_from_result(row: LabeledRow, description: DescribeResult | None, judge: JudgeVerdict | None) -> dict:
    return {
        "strategy": row.strategy,
        "clip_file": row.clip_file,
        "start_seconds": round(row.interval.start_seconds, 2),
        "end_seconds": round(row.interval.end_seconds, 2),
        "verdict": row.verdict,
        "notes": row.notes,
        "describe": _describe_to_dict(description),
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

        description = _describe_from_dict(entry.get("describe")) if entry else None
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
) -> list[DescribeResult | None]:
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

    results: list[DescribeResult | None] = []
    for i, row in enumerate(rows):
        entry = entries[i]
        if entry is not None and (entry["strategy"] != row.strategy or entry["clip_file"] != row.clip_file):
            raise ValueError(
                f"Cache mismatch at index {i}: cached {entry['strategy']}/{entry['clip_file']}, "
                f"actual {row.strategy}/{row.clip_file}. Delete {cache_path} and rerun."
            )

        description = _describe_from_dict(entry.get("describe")) if entry else None
        if description is None:
            description = describe_fn(row, chunks, gemini_cfg)

        results.append(description)
        entries[i] = {
            "strategy": row.strategy,
            "clip_file": row.clip_file,
            "start_seconds": round(row.interval.start_seconds, 2),
            "end_seconds": round(row.interval.end_seconds, 2),
            "describe": _describe_to_dict(description),
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
                "original_verdict", "original_notes",
                "gemini_score", "gemini_caption", "gemini_description",
                "judge_agreement", "judge_distance_score", "judge_rationale",
                "new_label", "new_notes",
            ]
        )
        for ar in audit_rows:
            row = ar.row
            desc = ar.description
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
                    desc.score if desc else "",
                    desc.caption if desc else "",
                    desc.description if desc else "",
                    judge.agreement if judge else "",
                    f"{judge.distance_score:.2f}" if judge else "",
                    judge.rationale if judge else "",
                    "",  # new_label -- blank, for the user to fill in
                    "",  # new_notes -- blank, for the user to fill in
                ]
            )
