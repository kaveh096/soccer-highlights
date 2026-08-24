---
name: kaveh_soccer_highlights
description: Kaveh's end-to-end pipeline for turning Sunday-morning recreational soccer recordings into a shared Telegram highlight reel (audio detection -> Gemini scoring -> manual pick -> export -> post), plus the eval/tuning methodology for both audio-detection parameters and the Gemini scoring prompt. Use whenever Kaveh mentions Sunday soccer, highlight clips, the review sheet, Gemini scoring/describe, or posting highlights to Telegram.
---

# Kaveh's Sunday Soccer Highlights pipeline

This is Kaveh's personal, home-setup-specific playbook for `C:\dev\soccer-highlights`
(GitHub: `kaveh096/soccer-highlights`). It is not written to be portable to another
user/machine -- paths, hardware notes, and credentials setup are all specific to this
one setup. It exists so a fresh Claude Code session can pick up the recurring weekly
workflow cold, without re-deriving decisions already made and validated over several
prior sessions.

**Two sections:**
- **[processing.md](processing.md)** -- the actual weekly pipeline: one-time setup
  (already done, reference only) + the recurring flow from raw recording to posted
  Telegram clips, with exact commands and folder structure.
- **[evals.md](evals.md)** -- how to tune this system when something needs
  improving: audio-detection parameters (rarely needed, mostly settled) and the
  Gemini scoring prompt (more likely to need revisiting as new games surface new
  edge cases). Includes two promoted, reusable scripts in `scripts/`.

## The one-sentence pipeline

Raw multi-hour DJI recording -> audio peak detection finds candidate moments ->
Gemini watches each candidate and scores it 1-5 for highlight-worthiness (+ writes
an English description and a Farsi caption) -> Kaveh scans the whole ranked sheet
himself and picks the real favorites (score is for *sorting*, not an automatic
cutoff) -> the picks get re-encoded at share quality -> posted to a Telegram group
with the Farsi caption.

## Where the deep history lives (don't duplicate it here)

- **`README.md`** (repo root) -- the full technical design history: audio-detection
  Rounds 1-4, the Vision AI (Phase 2) section (Claude vs. Gemini provider comparison,
  label audit, pre-labeling, describe-prompt revision). This is the authoritative
  record of *why* things are the way they are. This skill documents *what to run
  today*; go to README when you need the historical rationale behind a settled
  decision.
- **Claude Code memory** (`project_phase2_vision.md`, `project_soccer_highlights_hardware.md`,
  `project_soccer_highlights_workflow.md`) -- cross-session context that predates
  this skill. Memory can go stale; this skill and the current repo state are more
  authoritative for anything operational.

## Settled decisions -- don't re-litigate without new evidence

- **Audio detection**: `onset_flux` strategy, tuned via `golden-score` against
  `testdata/golden_events.json` (Round 3/4 in README). Rarely worth touching.
- **Vision provider split**: Gemini native video for the describe/score pass
  (`generate_description` in `label_audit.py`), Claude for the label-audit judge
  step. Both in active use for different jobs. Don't re-propose GPT-4o/AWS
  Rekognition/Google Video Intelligence -- already considered and rejected (README).
- **Two separate Gemini/vision code paths exist -- know which one is live:**
  - `label_audit.py`'s `generate_description` / `_DESCRIBE_PROMPT_V2` -- **this is
    the live, production path.** Used by `pre-label` and `label-audit`. This is
    what the whole weekly workflow runs on.
  - `vision.py` / `vision_gemini.py`'s `classify_confirm` (Rounds 1-4 experiment,
    `vision-highlights`/`vision-compare` commands) -- **concluded, not part of the
    weekly workflow.** Gemini won that comparison (F1 0.432) but was never wired
    into a production render path; the whole describe/score approach superseded it
    with a different framing (1-5 score instead of binary confirm+caption). Don't
    edit this path expecting it to affect real output, and don't confuse its
    prompts/config with `_DESCRIBE_PROMPT_V2`.
- **The 1-5 score is for sorting/triage, not an automated filter.** Kaveh's actual
  workflow is scanning the whole ranked sheet, not trusting `score>=4` as a hard
  cutoff. Don't design features around a hard threshold without checking this is
  still true.
- **`_DESCRIBE_PROMPT_V2` / `gemini-flash-latest` / fps=10 / schema-enforced JSON**
  is the current production default (`generate_description`'s defaults, set
  2026-07-31) -- validated via a 4-profile sweep against 66 real hand-labeled
  clips, F1 0.364 -> 0.545. **Pro model (`gemini-pro-latest`) was a clear
  regression** (F1 as low as 0.000) -- it doesn't even perceive the same clear
  goals flash does. Don't switch to pro without new evidence.
- **Recall-first.** Throughout audio tuning and vision work, the standing priority
  is "don't miss real events" over "don't bombard with false positives." A change
  that only cuts precision at the cost of recall is not obviously a net win.
- **Two known, still-open Gemini scoring gaps** (see evals.md's Gemini-prompt
  history for detail) -- don't assume these are fixed:
  - Tier 5 (skillful goal) has never been awarded by any profile tested so far,
    including the one real 5-worthy clip in the Jul-26 batch.
  - The keeper-save tier-3 wording fix only worked ~1 in 12 times it should have.

## Home-setup specifics worth knowing up front

- **Hardware**: old, weak laptop (i5-4300U, 2 core/4 thread, no HW video decode).
  Software H.265/HEVC decode of the full 4K source runs ~16-18x slower than
  realtime. This is why the pipeline prefers the `.LRF` proxy (720p H.264,
  ~1/16th the size) for everything except the final share-quality export.
- **This laptop has been interrupted by sleep and by the terminal app closing
  overnight**, more than once, during long unattended jobs -- not just a one-off.
  Every long-running step in this pipeline is designed to be resumable
  (incremental, position-keyed caching that only retries what actually failed).
  Trust that design; don't add fresh incremental-save logic from scratch if an
  existing resumable command already covers the case.
- **Check for other running render/API jobs before starting a heavy one** --
  `Get-CimInstance Win32_Process | Where-Object Name -match 'ffmpeg|python'` (this
  has bitten a prior session: two heavy jobs competing for this weak CPU at once).
- **Kaveh sometimes runs concurrent Claude Code sessions** against this same repo.
  Check `git status`/`git log` at the start of a session for changes another
  session may have already made.
