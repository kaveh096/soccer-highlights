# Eval and labeling: tuning detection and the Gemini scoring prompt

**Not part of the weekly flow.** This is the playbook for the occasional job of
building human-labeled ground truth for a game and using it to check (and maybe
improve) the audio-detection pass or the Gemini describe prompt. A normal week
never touches any of it; the weekly skill lives in
`.claude/skills/kaveh_soccer_highlights/` and deliberately does not reference
this file.

Contents:

1. [What this is and when to do it](#1-what-this-is-and-when-to-do-it)
2. [Labeling a game independently (the ground-truth step)](#2-labeling-a-game-independently)
3. [Audio-detection tuning (rarely needed)](#3-audio-detection-tuning-rarely-needed)
4. [Gemini scoring-prompt tuning](#4-gemini-scoring-prompt-tuning)
5. [Is the ground truth itself wrong? (`label-audit`)](#5-is-the-ground-truth-itself-wrong-label-audit)
6. [Settled conclusions and open gaps](#6-settled-conclusions-and-open-gaps)

All commands assume repo root (`C:\dev\soccer-highlights`) as cwd and the venv
(`./.venv/Scripts/python.exe`); video-touching ones also need the WinGet ffmpeg
bin dir on `PATH` (see the skill's `processing.md` Part 1). Game folders follow the
`Raw\` / `Review\` / `Share\` layout; for games before 2026-10-11 the review
folder is the old `Tests\pre_label\candidates\`, and every command takes it as an
explicit path so that works unchanged.

---

## 1. What this is and when to do it

Two independent tuning surfaces:

- **Audio detection** is settled and rarely worth touching.
- **The Gemini describe prompt** is the more likely thing to need revisiting as
  new games surface new edge cases. Most of this document is about it.

Both need the same raw material: a game's `review_sheet.csv` with a
human-written `verdict` for every row. The weekly flow leaves `verdict`/`notes`
blank on purpose (labeling every row is the time-consuming part and isn't needed
to post clips), so ground truth only exists for the games that were labeled in a
dedicated pass. That pass can happen well after the game, on the same sheet,
independently of posting. Don't assume a sheet with posted clips has verdicts in
it -- check before treating it as eval-ready.

Do an eval pass only when you have a reason: a suspected recall regression on a
new game's audio, a pattern of Gemini misses seen while picking clips, or a
suspicion that the labels themselves have drifted.

## 2. Labeling a game independently

The point is that the human verdict is formed **without** seeing what Gemini said,
so the two can be compared fairly.

1. Run the normal `pre-label` (+ `name-candidates`) for the game so the review
   folder holds the rendered clips and `review_sheet.csv` (columns:
   `clip_file/start_seconds/end_seconds/duration_seconds/max_peak_score/verdict/
   notes/source/...gemini_score/gemini_caption/gemini_description`). `verdict` and
   `notes` start blank.
2. **Work on a copy**, e.g. `review_sheet_copy.csv`. Watch every clip in the
   review folder and fill in `verdict` (the same 1-5 highlight-worthiness scale
   Gemini uses) and `notes`. Scan the **whole** sheet, not only the high scorers --
   the score is for sorting, not filtering, and recall can only be measured
   against rows you actually judged.
   - To stay independent, hide or don't read the `gemini_*` columns while
     labeling.
3. **Do not edit `review_sheet.csv` in Excel.** The file has no BOM (so
   `csv.DictReader(..., encoding="utf-8")` reads the header cleanly elsewhere),
   which means Excel's default save silently turns the Farsi `gemini_caption`
   column into literal `?` -- real byte-level loss, not a display issue. If you
   must use Excel: edit a copy only; use *Data -> From Text/CSV* with UTF-8
   selected rather than opening the file directly; and always source captions for
   posting from the untouched original (`telegram-post --review-sheet` only
   needs `clip_file`/`gemini_caption`). Tally CSVs are vulnerable too: opening a
   raw tally export in a spreadsheet re-saves long epoch timestamps with
   thousands separators (the Sep-27 `"1,790,529,761,521"` crash).
4. With the `source` column from a watch-tagged game, `mark` rows (no audio peak
   near the tap) against total real events is the **recall** number for audio
   detection that was never measurable before the watch; `both`/`audio` rows give
   the precision side.

## 3. Audio-detection tuning (rarely needed)

**Settled state** (README Rounds 3/4; don't re-litigate without new evidence):
`detection.strategy = "onset_flux"`, `threshold_sigma=2.5`, `min_score=0.55`. It
beat `rms_energy` and `combined` fusion by a wide margin on real ground truth and
has held up since. Only revisit if a **new** game's audio genuinely differs
enough that `onset_flux` starts missing real events -- a different crowd size or
noise floor, different camera/mic placement. Don't tune preemptively.

Tools:

- **`golden-score`** scores the current `--strategy`/config against a golden-events
  file with no rendering:
  ```bash
  ./.venv/Scripts/python.exe -m soccer_highlights.cli --source-dir "..." golden-score --golden-events path/to/golden_events.json
  ```
  `testdata/golden_events.json` was removed in the Drive reorg, so a golden set
  usually has to be rebuilt per game. `--vision` also runs the Claude
  vision-refinement pass side by side, but `vision.py` / `classify_confirm` is a
  **concluded experiment**, not the live scoring path (see section 6).
- **`config/strategies.yaml`**: named parameter variants layered on
  `config/default.yaml`. `batch-review` runs every named strategy at once and
  renders small comparison clips for human labeling.
- **Building a golden set for a new game**: `golden.py`'s `build_golden_events`
  derives ground-truth timestamps from a human-labeled `batch-review` round
  (TP/FN clips) -- how the original `golden_events.json` was built from the
  Round 2 dataset. A labeled `pre-label` sheet (section 2) is the other source.

## 4. Gemini scoring-prompt tuning

The repeatable process used 2026-07-31 to take the describe prompt from F1 0.364
to 0.545 against real hand-labeled data. It generalizes to a future game's data.

Prerequisite: a labeled sheet (section 2). The sweep is meaningless without
verdicts to compare against.

### Step 1 -- find real anomalies, don't guess at wording

Never revise the prompt off a hunch. Compare `verdict` against `gemini_score`
and look for patterns:

```python
import csv
with open("review_sheet_copy.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
# big gaps: |verdict - gemini_score| >= 2
# self-contradictions: does gemini_description's own wording match gemini_score?
#   (e.g. a description saying "no active movement" but score=2, not 1)
# category patterns: grep gemini_description for a keyword (e.g. "catches"/
#   "collects" for keeper saves) and compare verdict vs score across that group
```

The 2026-07-31 pass found three distinct failure modes, not one: (a)
clearly-described goals scoring below the "any goal = tier 4" floor, (b) a
description/score self-contradiction on a single clip, (c) a keeper-save wording
gap where the prompt covered deflections but not clean catches. Each got a
targeted fix; one blanket "make scores more generous" edit would have fixed none
of them correctly.

### Step 2 -- draft a revision, then get an independent second opinion

Write the revision addressing the specific anomalies. Before spending API budget
on a full sweep, get a **fresh** subagent (no shared context -- a fork inherits
your bias) to critique it. Give it, self-contained: the project goal and the
scale's already-settled design rules (goal always outranks skill, etc.); the
*exact* v1 prompt text; the anomaly evidence with concrete clip quotes; the draft
revision; and an explicit instruction to find gaps and overcorrections, and to
check whether each fix would actually flip the specific failing examples.

This caught two real issues: a circular tier-3 wording ("a shot the keeper had to
actually save" restates itself with no visual proxy) and a missing categorical
self-consistency mechanism (Step 3).

### Step 3 -- consider structured-output enforcement

If the revision depends on the stated reasoning matching the final score, a prose
instruction ("double-check your score against your description") is a soft nudge.
Calls are plain prompted JSON by default. A categorical field the model must
commit to *before* the score, plus real schema enforcement, makes inconsistency
structurally impossible:

```python
# label_audit.py pattern (see _DESCRIBE_RESPONSE_SCHEMA_V2):
schema = {
    "type": "OBJECT",
    "properties": {
        "goal_this_end": {"type": "BOOLEAN"},   # forces an explicit, checkable commitment
        "description": {"type": "STRING"},
        "rationale": {"type": "STRING"},
        "score": {"type": "INTEGER", "minimum": 1, "maximum": 5},
        "caption": {"type": "STRING"},
    },
    "required": ["goal_this_end", "description", "rationale", "score", "caption"],
    "property_ordering": ["goal_this_end", "description", "rationale", "score", "caption"],
}
# generate_description(..., response_schema=schema) turns this on via
# generationConfig.response_mime_type=application/json + response_schema.
```

`property_ordering` matters: putting `description`/`rationale` before `score`
forces the numeric answer to follow its own stated reasoning in generation
order. Validated 2026-07-31: **zero** `goal_this_end=true`-but-`score<4`
contradictions across 264 real calls once enforcement was on, versus
reproducible contradictions under plain prompted JSON.

Gemini schema quirks: enum constraints work more reliably as `STRING` than
`INTEGER`; `minimum`/`maximum` on `INTEGER` are fine. The REST calls in
`vision_gemini.py` use **snake_case** JSON keys throughout (`inline_data`,
`video_metadata`, `generation_config`, `response_mime_type`, `response_schema`);
both spellings work, stay consistent with the existing code.

### Step 4 -- run the sweep

One `tools/eval/sweep_prompt.py` invocation per profile, all writing into the
same `--out-dir`:

```bash
cd C:/dev/soccer-highlights
for profile in \
  "v2_flash_10fps v2 gemini-flash-latest 10" \
  "v2_pro_10fps   v2 gemini-pro-latest   10 --timeout 120" \
  ; do
  read tag prompt model fps extra <<< "$profile"
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe tools/eval/sweep_prompt.py \
    --review-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Review" \
    --review-sheet "G:/.../review_sheet_copy.csv" \
    --out-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Review/sweep_results" \
    --tag "$tag" --prompt "$prompt" --model "$model" --fps "$fps" $extra
done
```

(The loop is illustrative; separate invocations, several as background jobs,
work just as well and is what was done 2026-07-31.) `--review-sheet` must point
at a sheet with real `verdict` values. `sweep_results\` is the one eval-only
subfolder; it only exists on a tuning run.

Gemini 503 "high demand" congestion during a multi-profile sweep is not a bug:
rerun the identical command per profile; `run_describe_only`'s cache retries only
rows with a null `describe` entry. The 2026-07-31 sweep (4 profiles x 66 clips =
264 calls) needed 3 retry rounds (43-52/66 -> 60-65/66 -> 65-66/66 -> 66/66).

If a profile you want to compare is exactly the sheet's own production config
(the prompt/model/fps that generated `gemini_score`), don't spend a run on it --
`analyze_sweep.py` scores that as the free "baseline" straight from the sheet.

### Step 5 -- analyze

```bash
./.venv/Scripts/python.exe tools/eval/analyze_sweep.py \
  --review-dir "G:/.../<date>/Review" \
  --review-sheet "G:/.../review_sheet_copy.csv" \
  --sweep-dir "G:/.../<date>/Review/sweep_results" \
  --spot-check clip_012,clip_034
```

Prints P/R/F1 per profile (score>=4 vs verdict>=4), which profiles ever award a
5, the `goal_this_end` self-consistency check, and an optional side-by-side
spot-check of chosen clips across profiles. Writes a per-clip comparison CSV.
(`--candidates-dir` is the old spelling of `--review-dir` and still works; for a
pre-2026-10-11 game point it at `Tests\pre_label\candidates`.)

### Step 6 -- decide and adopt

If a profile clearly wins, update `generate_description`'s defaults
(`prompt_template`/`fps`/`response_schema` in `label_audit.py`, and
`GeminiConfig.model`'s default in `config.py` if switching models) and
**document the decision in the function's docstring**: before/after F1, what was
tried and rejected and why, and known-still-open gaps. That is the project's
working convention -- the reasoning survives in the code, not in a chat
transcript.

## 5. Is the ground truth itself wrong? (`label-audit`)

Separate from tuning Gemini to match human verdicts, `label-audit` questions
whether the human verdicts are right: Gemini describes each already-labeled clip
fresh (no verdict shown to it), Claude judges agreement with the original
verdict/notes, and disagreeing rows are flagged for a second human look. This
was the origin of the 1-5 scale's last major revision (a first run found ~21% of
disagreements traced to "highlight-worthy" being underspecified in the prompt,
not a Gemini perception failure). Use it when you suspect the labels are noisy or
the definition of highlight-worthy has drifted; not for improving raw
score-vs-verdict agreement (section 4).

```bash
SOCCER_HL__LABEL_AUDIT__REVIEW_ROOT="G:/My Drive/Photos and Movies/Sunday Soccer/<date>" \
  ./.venv/Scripts/python.exe -m soccer_highlights.cli label-audit [--limit N]
```

`review_root` has no CLI flag, only the env var or a `--config` override. Point
it at the game's `<date>` folder (the parent of `Review\`): `load_review_rows`
globs `<review_root>/*/review_sheet.csv`, which `Review\review_sheet.csv` matches
directly. For a game before 2026-10-11 point it at `Tests\pre_label` instead (the
parent of `candidates\`). Needs both `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` --
unlike the rest of the pipeline, the judge step uses Claude.

## 6. Settled conclusions and open gaps

**Reference sweep: 2026-07-31, Jul-26 game, 66 clips, 8 true highlights**

| profile | precision | recall | F1 |
|---|---|---|---|
| baseline (v1 prompt / flash / 5fps, prod at the time) | 0.667 | 0.250 | 0.364 |
| **v2_flash_10fps (adopted, current production default)** | 0.429 | 0.750 | **0.545** |
| v2_flash_5fps | 0.400 | 0.750 | 0.522 |
| v2_pro_5fps | 0.200 | 0.125 | 0.154 |
| v2_pro_10fps | 0.000 | 0.000 | 0.000 |

- **Production default**: `_DESCRIBE_PROMPT_V2` / `gemini-flash-latest` / fps=10 /
  schema-enforced JSON (`generate_description`'s defaults, set 2026-07-31).
- **Pro was a clear regression**, not an upgrade -- it didn't set
  `goal_this_end=true` on the same clearly visible goals flash caught. Validated;
  don't assume bigger = better, and don't switch to pro without new evidence.
- **Live vs concluded code paths**: `label_audit.py`'s `generate_description` is
  the live production describe path (used by `pre-label` and `label-audit`).
  `vision.py` / `vision_gemini.py`'s `classify_confirm` (Rounds 1-4,
  `vision-highlights`/`vision-compare`) is a concluded experiment: Gemini won
  that comparison (F1 0.432) but it was never wired into a render path and the
  1-5 describe/score approach superseded it.
- **Provider split**: Gemini native video for describe/score, Claude for the
  label-audit judge. GPT-4o / AWS Rekognition / Google Video Intelligence were
  considered and rejected (README).

**Still open -- don't assume fixed:**

- **Tier 5 has never been awarded by any profile tested**, including on the one
  real 5-worthy clip in the 66-row batch. Fixing goal recognition
  (`goal_this_end`) was necessary but not sufficient; the model still doesn't
  separate "skillful" from "easy" goals. n=1 is too thin to design a fix against;
  revisit when a new game gives more tier-5-worthy examples.
- **The keeper-save tier-3 wording fix flipped only ~1 of 12 real opportunities.**
  The "dive/jump/stretch/react" visual-proxy language mostly didn't discriminate a
  real save from a routine catch.
- **A third gap (2026-08-31, from Aug-30's quick-share picks, not a formal
  eval):** `r20_s3_819a_c017.mp4` was a real goal scored 3, one below the tier-4
  "any goal" floor -- the only miss of 11 near-field goals that game. Distinct
  from the two above: `goal_this_end` apparently not firing (or not lifting the
  score) on a correctly recorded goal. n=1; check that clip's
  `gemini_description`/`rationale` against the video the next time this game's
  sheet is labeled (it is currently unlabeled).

Don't keep re-tuning against the same 66-row Jul-26 dataset -- overfitting to one
game's clips is a real risk. Revisit these items once a new game's labeled data
exists.
