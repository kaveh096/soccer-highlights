"""Compare every describe_cache_<tag>.json produced by sweep_prompt.py (run
in the same --out-dir) against a review sheet's hand-labeled `verdict` column,
plus the sheet's own `gemini_score` as a zero-cost "baseline" profile (whatever
generated the sheet in the first place -- usually current production defaults,
no need to re-run that one through sweep_prompt.py).

Prints precision/recall/F1 (score>=4 vs verdict>=4) per profile, flags any
goal_this_end=true-but-score<4 self-contradiction (should always be zero if the
v2 schema is enforced correctly), optionally spot-checks specific clips across
every profile side by side, and writes a full per-clip comparison CSV.

Usage:

  ./.venv/Scripts/python.exe .claude/skills/kaveh_soccer_highlights/scripts/analyze_sweep.py \
      --candidates-dir "G:/.../Tests/pre_label/candidates" \
      --sweep-dir "G:/.../Tests/pre_label/sweep_results" \
      [--review-sheet path/to/review_sheet.csv] \
      [--spot-check clip_012,clip_034]

sweep-dir defaults to candidates-dir if omitted (sweep_prompt.py's --out-dir can
point at either).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def prf(pred_scores: list[int], verdicts: list[int]) -> tuple[float, float, float, int, int, int]:
    tp = sum(1 for p, v in zip(pred_scores, verdicts) if p >= 4 and v >= 4)
    fp = sum(1 for p, v in zip(pred_scores, verdicts) if p >= 4 and v < 4)
    fn = sum(1 for p, v in zip(pred_scores, verdicts) if p < 4 and v >= 4)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-dir", required=True)
    parser.add_argument("--review-sheet", default=None, help="Default: <candidates-dir>/review_sheet.csv")
    parser.add_argument("--sweep-dir", default=None, help="Where describe_cache_<tag>.json files live. Default: candidates-dir")
    parser.add_argument("--spot-check", default=None, help="Comma-separated clip_file stems (no .mp4) to print side-by-side across profiles")
    parser.add_argument("--out-csv", default=None, help="Default: <sweep-dir>/sweep_comparison.csv")
    args = parser.parse_args()

    candidates_dir = Path(args.candidates_dir)
    review_sheet = Path(args.review_sheet) if args.review_sheet else candidates_dir / "review_sheet.csv"
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else candidates_dir
    out_csv = Path(args.out_csv) if args.out_csv else sweep_dir / "sweep_comparison.csv"

    with open(review_sheet, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    unlabeled = [r["clip_file"] for r in rows if not r.get("verdict", "").strip()]
    if unlabeled:
        print(f"WARNING: {len(unlabeled)} row(s) have no verdict filled in -- excluding from scoring: {unlabeled[:5]}{'...' if len(unlabeled) > 5 else ''}")
    rows = [r for r in rows if r.get("verdict", "").strip()]
    if not rows:
        raise SystemExit(f"No labeled (verdict-filled) rows found in {review_sheet}")

    tags = sorted(p.stem.removeprefix("describe_cache_") for p in sweep_dir.glob("describe_cache_*.json"))
    if not tags:
        print(f"No describe_cache_*.json files found in {sweep_dir} -- only the sheet's own baseline gemini_score will be scored.")

    profile_data: dict[str, dict[str, dict]] = {}
    for tag in tags:
        with open(sweep_dir / f"describe_cache_{tag}.json", encoding="utf-8") as f:
            entries = json.load(f)
        profile_data[tag] = {e["clip_file"]: e["describe"] for e in entries if e.get("describe") is not None}
        missing = [r["clip_file"] for r in rows if r["clip_file"] not in profile_data[tag]]
        if missing:
            print(f"WARNING: {tag} is missing {len(missing)} row(s) (failed/not yet run) -- excluded from that profile's scoring")

    verdicts = [int(r["verdict"]) for r in rows]

    print("\n=== Precision/Recall/F1 (score>=4 vs verdict>=4) ===")
    if all(r.get("gemini_score", "").strip() for r in rows):
        baseline_scores = [int(r["gemini_score"]) for r in rows]
        p, r_, f1, tp, fp, fn = prf(baseline_scores, verdicts)
        print(f"{'baseline (review sheet gemini_score)':38s} P={p:.3f} R={r_:.3f} F1={f1:.3f}  TP={tp} FP={fp} FN={fn}")

    for tag in tags:
        present = [(int(profile_data[tag][r["clip_file"]]["score"]), int(r["verdict"])) for r in rows if r["clip_file"] in profile_data[tag]]
        if not present:
            continue
        pred_scores, v = zip(*present)
        p, r_, f1, tp, fp, fn = prf(list(pred_scores), list(v))
        print(f"{tag:38s} P={p:.3f} R={r_:.3f} F1={f1:.3f}  TP={tp} FP={fp} FN={fn}  (n={len(present)})")

    print("\n=== 5s awarded (any profile) ===")
    for tag in tags:
        fives = [cf for cf, d in profile_data[tag].items() if d["score"] == 5]
        print(f"{tag}: {len(fives)} -> {fives}")

    print("\n=== goal_this_end sanity check (goal_this_end=True but score<4 -- should be empty) ===")
    found_any = False
    for tag in tags:
        for cf, d in profile_data[tag].items():
            if d.get("goal_this_end") and d["score"] < 4:
                print(f"{tag}: {cf} goal_this_end=True but score={d['score']}")
                found_any = True
    if not found_any:
        print("(none -- schema-enforced consistency held for every row in every profile)")

    if args.spot_check:
        clips = [c.strip() for c in args.spot_check.split(",")]
        by_clip = {r["clip_file"].removesuffix(".mp4"): r for r in rows}
        print("\n=== Spot-check ===")
        for c in clips:
            row = by_clip.get(c)
            if row is None:
                print(f"{c}: not found in labeled rows")
                continue
            line = f"{c}: verdict={row['verdict']} baseline={row.get('gemini_score', '?')}"
            for tag in tags:
                d = profile_data[tag].get(c + ".mp4")
                if d is None:
                    line += f"  {tag}=?"
                    continue
                goal_flag = d.get("goal_this_end")
                gflag = "" if goal_flag is None else (" G" if goal_flag else " -")
                line += f"  {tag}={d['score']}{gflag}"
            print(line)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_file", "verdict", "notes", "baseline_score"] + [f"{t}_score" for t in tags] + [f"{t}_goal" for t in tags])
        for r in rows:
            row_out = [r["clip_file"], r["verdict"], r.get("notes", ""), r.get("gemini_score", "")]
            row_out += [profile_data[t].get(r["clip_file"], {}).get("score", "") for t in tags]
            row_out += [profile_data[t].get(r["clip_file"], {}).get("goal_this_end", "") for t in tags]
            w.writerow(row_out)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
