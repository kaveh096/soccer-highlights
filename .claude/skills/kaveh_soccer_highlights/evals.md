# Evals: tuning detection and the Gemini scoring prompt

Two independent tuning surfaces. **Audio detection is settled and rarely worth
touching.** The Gemini describe prompt is the more likely thing to need
revisiting as new games surface new edge cases -- most of this document is
about that.

---

## Part A: Audio-detection parameter tuning (rarely needed)

**Current settled state** (README Round 3/4, don't re-litigate without new
evidence): `detection.strategy = "onset_flux"`, `threshold_sigma=2.5`,
`min_score=0.55`. This beat `rms_energy` and `combined` fusion by a wide margin
on real ground truth and has held up since.

### Tools

- **`golden-score`** -- scores the current `--strategy`/config against
  `testdata/golden_events.json` (if it still exists for a given game -- the
  data reorg to Google Drive may mean this needs rebuilding per-game, see
  below), no rendering needed:
  ```bash
  ./.venv/Scripts/python.exe -m soccer_highlights.cli --source-dir "..." golden-score --golden-events path/to/golden_events.json
  ```
  Add `--vision` to also run the Phase 2 Claude vision-refinement pass
  side-by-side -- but see SKILL.md's note that this whole code path
  (`vision.py`, `classify_confirm`) is a concluded experiment, not the live
  scoring path.
- **`config/strategies.yaml`** -- named parameter variants, each a config
  override layered on `config/default.yaml`. `batch-review` runs every named
  strategy at once and renders small comparison clips for human labeling.
- **Building/refreshing a golden set for a new game**: `golden.py`'s
  `build_golden_events` derives ground-truth timestamps from a human-labeled
  `batch-review` round (TP/FN clips). This is how `testdata/golden_events.json`
  was originally built from the Round 2 labeled dataset.

### When this is actually worth revisiting

Only if a *new* game's audio characteristics genuinely differ enough that
`onset_flux` starts missing real events (recall regression) -- e.g.
meaningfully different crowd size/noise floor, a different camera/mic
placement. Don't tune this preemptively; the settled parameters have held
across multiple games already.

---

## Part B: Gemini scoring-prompt tuning methodology

This is the actual repeatable process used 2026-07-31 to take the describe
prompt from F1 0.364 to 0.545 against real hand-labeled data. It generalizes
directly to a future game's data or a future problem with the scale.

### Step 1 -- find real anomalies, don't guess at wording

Never revise the prompt off a hunch. Compare a labeled review sheet's
`verdict` column against its `gemini_score` column and look for patterns:
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
The 2026-07-31 pass found three genuinely distinct failure modes this way, not
one: (a) clearly-described goals scoring below the "any goal = tier 4" floor,
(b) a description/score self-contradiction on a single clip, (c) a keeper-save
wording gap where the prompt only covered deflections, not clean catches. Each
got a different, targeted fix -- a single blanket "make scores more generous"
edit would not have addressed any of them correctly.

### Step 2 -- draft a revision, then get an independent second opinion

Write the revised prompt addressing the specific anomalies found. Before
spending real API budget on a full sweep, get a fresh subagent (no shared
context -- a fork would inherit your own bias) to critique it independently.
Give it, self-contained:
- the project goal and the scale's already-settled design rules (goal always
  outranks skill, etc. -- see SKILL.md)
- the *exact* v1 prompt text
- the anomaly evidence with concrete clip quotes (not just a summary)
- the draft revision
- an explicit instruction to find gaps/overcorrections, not just validate,
  and to check whether each specific fix would actually flip the specific
  failing examples given, not just sound plausible

This caught two real issues in practice: a circular tier-3 wording ("a shot
the keeper had to actually save" restates itself without giving the model a
visual proxy) that risked not discriminating anything, and a missing
categorical self-consistency mechanism (see Step 3).

### Step 3 -- consider structured-output enforcement

If the revision depends on the model's stated reasoning matching its final
score (the description/score self-contradiction problem from Step 1), a
prose instruction like "double-check your score against your description" is
a soft nudge, not a guarantee -- `_call_gemini`'s calls are plain prompted
JSON by default, not schema-enforced. Adding a categorical field the model
must commit to *before* the score, plus real schema enforcement, makes
inconsistency structurally impossible instead of just discouraged:

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
`property_ordering` matters -- putting `description`/`rationale` before
`score` forces the model's numeric answer to follow its own stated reasoning
in generation order, not be picked independently. This was validated
2026-07-31: **zero** `goal_this_end=true`-but-`score<4` contradictions across
264 real API calls once schema enforcement was on, versus real, reproducible
description/score contradictions under plain prompted JSON.

Gemini schema quirks worth knowing: enum constraints work more reliably as
`STRING` type than `INTEGER` type in this schema dialect; `minimum`/`maximum`
on `INTEGER` work fine instead. This codebase's REST calls to Gemini
(`vision_gemini.py`) use **snake_case** JSON keys throughout
(`inline_data`, `video_metadata`, `generation_config`, `response_mime_type`,
`response_schema`) -- protobuf-JSON's snake_case field names, not the
camelCase aliases -- match that convention for consistency, both work but stay
consistent with the existing code.

### Step 4 -- run the sweep

One `sweep_prompt.py` invocation per profile you want to compare, all writing
into the same `--out-dir`:
```bash
cd C:/dev/soccer-highlights
for profile in \
  "v2_flash_10fps v2 gemini-flash-latest 10" \
  "v2_pro_10fps   v2 gemini-pro-latest   10 --timeout 120" \
  ; do
  read tag prompt model fps extra <<< "$profile"
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe .claude/skills/kaveh_soccer_highlights/scripts/sweep_prompt.py \
    --candidates-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Tests/pre_label/candidates" \
    --review-sheet "G:/.../review_sheet_copy.csv" \
    --out-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Tests/pre_label/sweep_results" \
    --tag "$tag" --prompt "$prompt" --model "$model" --fps "$fps" $extra
done
```
(the shell loop above is illustrative -- running each invocation as its own
Bash tool call, several in a background job each, works just as well and is
what was actually done 2026-07-31). **`--review-sheet` must point at a sheet
with real `verdict` values filled in** -- the sweep is meaningless without
ground truth to compare against.

Expect **Gemini 503 "high demand" congestion** during a multi-profile sweep --
not a bug. Just rerun the identical `sweep_prompt.py` command per profile;
`run_describe_only`'s caching only retries rows with a null `describe` entry.
The 2026-07-31 sweep (4 profiles x 66 clips = 264 calls) needed 3 retry rounds
before every profile hit 100% (43-52/66 -> 60-65/66 -> 65-66/66 -> 66/66).

**If a profile you're comparing is exactly your sheet's own current production
config** (same prompt/model/fps that generated `gemini_score` in the first
place), don't waste a sweep run on it -- `analyze_sweep.py` already scores
that as the "baseline" from the sheet directly, free.

### Step 5 -- analyze

```bash
./.venv/Scripts/python.exe .claude/skills/kaveh_soccer_highlights/scripts/analyze_sweep.py \
  --candidates-dir "G:/.../Tests/pre_label/candidates" \
  --review-sheet "G:/.../review_sheet_copy.csv" \
  --sweep-dir "G:/.../Tests/pre_label/sweep_results" \
  --spot-check clip_012,clip_034
```
Prints P/R/F1 per profile (score>=4 vs verdict>=4), which profiles ever award
a 5, the `goal_this_end` self-consistency sanity check, and an optional
side-by-side spot-check of specific clips across every profile. Writes a full
per-clip comparison CSV.

### Step 6 -- decide and adopt

If a profile clearly wins, update `generate_description`'s defaults
(`prompt_template`/`fps`/`response_schema` in `label_audit.py`, and
`GeminiConfig.model`'s default in `config.py` if switching models) --
**document the decision in the function's docstring** with the before/after
F1, what was tried and rejected and why, and any known-still-open gaps. This
is the project's actual working convention (see the current docstring in
`generate_description` for the exact style to match) and is *why* this skill
could be written at all -- the reasoning survives in the code, not just in a
chat transcript.

### Reference point: the 2026-07-31 sweep results (Jul-26 game, 66 clips, 8 true highlights)

| profile | precision | recall | F1 |
|---|---|---|---|
| baseline (v1 prompt / flash / 5fps, prod at the time) | 0.667 | 0.250 | 0.364 |
| **v2_flash_10fps (adopted)** | 0.429 | 0.750 | **0.545** |
| v2_flash_5fps | 0.400 | 0.750 | 0.522 |
| v2_pro_5fps | 0.200 | 0.125 | 0.154 |
| v2_pro_10fps | 0.000 | 0.000 | 0.000 |

**Pro model was a clear regression**, not an upgrade -- it didn't even set
`goal_this_end=true` on the same clearly-described, clearly-visible goals
flash caught correctly. A genuine, surprising, validated finding -- don't
assume "bigger model = better" without checking.

**Still open, don't assume fixed:**
- **Tier 5 has never been awarded by any profile tested**, including on the
  one real 5-worthy clip in the whole 66-clip batch. Fixing goal-recognition
  (the `goal_this_end` mechanism) was necessary but not sufficient -- the
  model still isn't discriminating "skillful" from "easy" among goals. Worth
  another look once a new game's data gives more tier-5-worthy examples to
  learn from (n=1 in this dataset is too thin to trust a targeted fix against).
- **The keeper-save tier-3 wording fix only flipped ~1 of 12 real
  opportunities correctly.** The "dive/jump/stretch/react" visual-proxy
  language mostly didn't discriminate a real save from a routine catch in
  practice, despite sounding reasonable in the prompt draft.

Don't keep re-tuning against this same 66-row Jul-26 dataset indefinitely --
overfitting to one game's specific clips is a real risk. Revisit these two
open items once a new game's labeled data is available.

**A third gap, spotted 2026-08-31 from Aug-30's quick-share pick (not a
formal eval pass, just Kaveh's own viewing):** `r20_s3_819a_c017.mp4` was a
real goal that Gemini scored 3, one point below the tier-4 "any goal" floor
-- out of 11 near-field goals in the game, this was the only miss (10 of 11
already scored >=4 by rank 15; the 11th was this rank-20 clip). Distinct from
the two gaps above (not a tier-5/skillful-goal miss, not a keeper-save
wording issue) -- this is `goal_this_end` apparently not firing (or firing
but not lifting the score) on a real, correctly-recorded goal. n=1, not
enough to design a targeted fix around yet, but worth checking this specific
clip's `gemini_description`/`rationale` against the video next time a formal
eval pass runs on this game's data (fill in `verdict`/`notes` per
processing.md's full-eval mode note first -- this game's sheet is currently
unlabeled, a quick-share run).

---

## A third, different eval: is the ground truth itself wrong?

Separate from tuning Gemini to match human verdicts, `label-audit` questions
whether the human verdicts are even right in the first place -- Gemini
describes each already-labeled clip fresh (no verdict shown to it), Claude
judges agreement with the original human verdict/notes, disagreeing rows get
flagged for a second human look. This was the actual origin of the 1-5 score
scale's most recent major revision (a first label-audit run found ~21% of
disagreements traced back to "highlight-worthy" being underspecified in the
prompt, not a Gemini perception failure). Reach for this when you suspect the
labels themselves are noisy or the definition of "highlight-worthy" has
drifted, not when you're just trying to improve raw score-vs-verdict
agreement (that's Part B above).
```bash
SOCCER_HL__LABEL_AUDIT__REVIEW_ROOT="G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Tests/pre_label" \
  ./.venv/Scripts/python.exe -m soccer_highlights.cli label-audit [--limit N]
```
(`review_root` has no dedicated CLI flag, only the `SOCCER_HL__LABEL_AUDIT__REVIEW_ROOT`
env var or a `--config` file override -- point it at the parent of `candidates/`,
since `load_review_rows` globs `<review_root>/*/review_sheet.csv` and `pre-label`'s
`candidates/review_sheet.csv` matches that shape directly. Requires both
`GEMINI_API_KEY` and `ANTHROPIC_API_KEY` -- unlike the rest of this pipeline,
this command's judge step uses Claude.)
