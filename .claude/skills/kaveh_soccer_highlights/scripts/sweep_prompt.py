"""Run one Gemini describe profile (prompt template x model x fps x
schema-enforcement) against every row of a review_sheet.csv, caching results
incrementally so a Gemini 503-congestion interruption never loses completed
work -- rerun the exact same command and only the failed rows retry.

This is the generalized, reusable version of the ad-hoc sweep script written
2026-07-31 to compare v1 vs v2 prompt wording x {flash, pro} x {5fps, 10fps}
against the Jul-26 game's 66 hand-labeled clips (see ../evals.md for that
sweep's actual results and what they mean). A full multi-profile sweep is just
this script invoked once per profile with a different --tag/--model/--fps/
--prompt -- deliberately not a single script with a hardcoded profile list, so
adding a new profile to compare needs zero code edits, just a new invocation.

Usage (run from repo root, with the project venv):

  ./.venv/Scripts/python.exe .claude/skills/kaveh_soccer_highlights/scripts/sweep_prompt.py \
      --candidates-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Tests/pre_label/candidates" \
      --out-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Tests/pre_label/sweep_results" \
      --tag v2_flash_10fps --prompt v2 --model gemini-flash-latest --fps 10 --schema

Re-running the identical command is always safe and resumes -- only rows with a
null `describe` entry in the cache get retried.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

from soccer_highlights.config import GeminiConfig
from soccer_highlights.label_audit import (
    _DESCRIBE_PROMPT,
    _DESCRIBE_PROMPT_V2,
    _DESCRIBE_RESPONSE_SCHEMA_V2,
    generate_description,
    parse_review_sheet_rows,
    run_describe_only,
)

PROMPTS = {"v1": _DESCRIBE_PROMPT, "v2": _DESCRIBE_PROMPT_V2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-dir", required=True, help="Directory containing review_sheet.csv and clip_NNN.mp4 files")
    parser.add_argument(
        "--review-sheet", default=None, help="Override path to the review sheet (default: <candidates-dir>/review_sheet.csv)"
    )
    parser.add_argument("--out-dir", required=True, help="Where this profile's describe_cache_<tag>.json is written")
    parser.add_argument("--tag", required=True, help="Label for this profile, e.g. 'v2_flash_10fps' -- used in the cache filename")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v2", help="Which prompt template to use")
    parser.add_argument("--model", default="gemini-flash-latest", help="Gemini model id, e.g. gemini-flash-latest or gemini-pro-latest")
    parser.add_argument("--fps", type=int, default=10, help="video_metadata fps override sent to Gemini")
    parser.add_argument(
        "--schema", action=argparse.BooleanOptionalAction, default=True,
        help="Enforce structured JSON output via response_schema (only meaningful with --prompt v2, which defines the schema)",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds (bump for gemini-pro-*, it's slower)")
    args = parser.parse_args()

    candidates_dir = Path(args.candidates_dir)
    review_sheet = Path(args.review_sheet) if args.review_sheet else candidates_dir / "review_sheet.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"describe_cache_{args.tag}.json"

    with open(review_sheet, encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))
    rows = parse_review_sheet_rows("candidates", csv_rows, candidates_dir)
    print(f"Loaded {len(rows)} rows from {review_sheet}")

    prompt_template = PROMPTS[args.prompt]
    response_schema = _DESCRIBE_RESPONSE_SCHEMA_V2 if (args.schema and args.prompt == "v2") else None
    gemini_cfg = dataclasses.replace(GeminiConfig(), model=args.model, request_timeout_seconds=args.timeout)

    def describe_fn(clip_path, duration_seconds, cfg):
        return generate_description(
            clip_path, duration_seconds, cfg, prompt_template=prompt_template, fps=args.fps, response_schema=response_schema
        )

    print(f"=== {args.tag} (prompt={args.prompt}, model={args.model}, fps={args.fps}, schema={response_schema is not None}) ===")
    print(f"cache: {cache_path}")
    results = run_describe_only(rows, gemini_cfg, cache_path, describe_fn=describe_fn)
    n_ok = sum(1 for r in results if r is not None)
    print(f"{args.tag}: {n_ok}/{len(results)} succeeded")
    if n_ok < len(results):
        print("Some rows failed (likely Gemini 503 congestion) -- just rerun this exact command to retry only those.")


if __name__ == "__main__":
    main()
