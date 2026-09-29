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
  edge cases).

**`scripts/`** holds four promoted, reusable helpers. Know which is which:
- `sweep_prompt.py` / `analyze_sweep.py` -- **eval only** (evals.md Part B). Run
  one Gemini describe profile over a labeled sheet, then compare profiles by
  P/R/F1. Not part of a normal weekly run.
- `seg_render.py` -- **weekly workflow** (processing.md Step 4). Exports one
  share-quality clip in resumable segments, for clips too long to finish in a
  single run on this laptop. Reach for it whenever a pick is longer than ~15s.
- `timelapse_render.py` -- **optional, only on request** (processing.md Step 6).
  Speeds the whole game down to a short clip with the scoreboard/clock burned
  in. Not part of the normal weekly flow -- only run it if Kaveh asks.

## The one-sentence pipeline

Raw multi-hour DJI recording -> audio peak detection finds candidate moments ->
Gemini watches each candidate and scores it 1-5 for highlight-worthiness (+ writes
an English description and a Farsi caption) -> Kaveh scans the whole ranked sheet
himself and picks the real favorites (score is for *sorting*, not an automatic
cutoff) -> the picks get re-encoded at share quality, with a scoreboard burned into
the video (running score + time of day, or just the time on a game with no watch
marks) -> posted to a Telegram group with the Farsi caption.

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
- **The burned-in goal counter flips at the TAP, not at the goal or the audio
  peak** (Kaveh's explicit call, 2026-09-05 -- don't "fix" it to the peak). A
  broadcast score graphic also lags the goal; tap-anchoring behaves identically
  whether or not audio found a peak, and doubles as feedback on tap speed. Two
  consequences: peak-anchored goal clips are stretched to cover their own tap
  (capped 10s past the peak, `marks.score_flip_cap_seconds`) and so routinely
  exceed the ~15s single-run export limit, and the derived score is worth a
  glance before exporting, since one missed tap shifts every later clip's
  counter. That check is cheap rather than a gate: Kaveh doesn't need to
  remember the final score himself -- a dozen other players will answer over
  Telegram in seconds.
  - **This is the decision most likely to be revisited after 2026-09-06.**
    Kaveh is explicitly open to pivoting the flip to the audio peak if real
    tap timestamps turn out to stretch clips consistently. That pivot is a
    config/policy change in `marks.score_for_interval`, not a rewrite -- so
    gather the evidence (how many clips got extended, by how much) before
    proposing it.
- **The scoreboard is a Pillow-drawn PNG composited by ffmpeg, plus drawtext
  for the parts that change** (`soccer_highlights/scoreboard.py`, design signed
  off 2026-09-05). Chrome (panel, team pills, gold rule) is static and cached;
  the score and clock are `drawtext` on top, since the score flips mid-clip and
  the clock ticks per frame. Two traps live in that split: the score text draws
  **digits only** (the names are already in the PNG's pills), and the panel is
  **measured from the configured clock format** (sizing it for a stand-in string
  runs the clock off the end). Both bit during implementation and both now have
  tests. A game with no marks gets no board at all -- the clock falls back to
  its old standalone box, so audio-only games look exactly as they did.
- **Time-of-day burn-in happens in the final share encode (processing.md Step
  4), never as a separate pass and never at posting time.** `export-picks` /
  `export` / `seg_render.py` all bake the clock in while re-encoding the pick,
  so by the time Step 5's `telegram-post` runs, the clips already carry it --
  posting is a pure upload and must stay that way. On by default;
  `--no-burn-in-time` opts out per invocation. Review/pre-label clips
  deliberately stay clean. Don't add a re-encode step after export to overlay a
  timestamp: that would cost a second full generation loss on this laptop's
  slowest path for something the export encode already does for free.
- **Most weeks are "quick share," not "full eval" -- don't assume `verdict`/
  `notes` get filled in.** The common case is posting highlights soon after
  the game, picking clips straight off `gemini_score` + `gemini_description`
  with the `verdict`/`notes` columns left blank -- that labeling pass is only
  needed later, occasionally, when actually tuning the prompt (evals.md Part
  B). See processing.md Part 3's mode note. Don't treat an unlabeled
  `review_sheet.csv` as unfinished or broken, and don't assume a sheet with
  posted clips already has ground-truth verdicts in it.
- **`_DESCRIBE_PROMPT_V2` / `gemini-flash-latest` / fps=10 / schema-enforced JSON**
  is the current production default (`generate_description`'s defaults, set
  2026-07-31) -- validated via a 4-profile sweep against 66 real hand-labeled
  clips, F1 0.364 -> 0.545. **Pro model (`gemini-pro-latest`) was a clear
  regression** (F1 as low as 0.000) -- it doesn't even perceive the same clear
  goals flash does. Don't switch to pro without new evidence.
- **Live tagging (watch marks) is part of the weekly flow as of 2026-09-06.**
  Kaveh wears a Pixel Watch running **Tallies** with three counters --
  `white goal`, `black goal`, `moment` -- and taps during the game, because
  audio detection provably misses real goals in a small no-crowd recreational
  game. **Every game now needs its three tally CSV exports collected into
  `<date>\Raw\` -- ask for them, see processing.md Step 0b.** They cannot be
  recovered after the fact. The marks union with audio candidates (never
  replace them) and produce the `source` column that finally makes recall
  measurable. Settled and not worth re-litigating: phone-side capture
  (Tasker/MacroDroid/BT clicker) is ruled out (no offline queue -- marks get
  lost or wrongly timestamped), the camera position is per-game config rather
  than a watch button, and there is no dedicated sync button (the clap +
  `moment` tap IS the sync).
- **Review order trusts taps over Gemini when they actively disagree**
  (Kaveh, 2026-09-07, after reviewing Sep-06's real ranked sheet and finding
  Gemini goal false positives on clips he'd explicitly tapped `moment`, i.e.
  *not* a goal). `marks.review_tier`: a genuine goal tap
  (`white_goal`/`black_goal`) decides tier 1 via `is_near_cam_goal` (the
  per-game camera setup, see below) with Gemini's `goal_this_end` **not
  consulted at all**; a `moment` tap can **never** reach tier 1 regardless of
  what Gemini claims; an untapped clip (no mark at all) still falls back to
  `goal_this_end`, since there's nothing to value over Gemini there. This
  cuts both ways -- it also promotes a real tapped goal Gemini badly
  underscored (seen on Sep-06: a confirmed near-cam goal tap Gemini scored
  2/5, invisible under the old score-only ranking, surfaced near the top
  once the tap alone was trusted).
  - **`--near-cam-team-first-half {white,black}`** (`pre-label`) is the
    per-game camera input this needs -- which team's goal the camera sits
    behind in the FIRST half, swapping automatically at the auto-detected
    halftime boundary (`discovery.detect_halftime_seconds`, the single
    largest inter-chunk recording gap). Omit it and goal-tapped clips fall
    back to `goal_this_end`, same as before 2026-09-07 -- forgetting this
    input degrades gracefully, it doesn't break the run.
  - **Same-category fixed-window marks that are close enough to overlap now
    merge into one clip** (`marks._merge_same_category_fixed_windows`) --
    two taps for what's really one event (typically an accidental
    double-tap) used to render as two 90%-overlapping near-duplicate clips.
    Cross-category pairs never merge. The existing "N goal taps inside one
    clip -- only the first is drawn" warning is the signal a merge folded in
    a likely double-tap; nothing auto-corrects the score for it.
  - **The fixed-window fallback lookback is 15s, not 60s** (`MarksConfig.
    goal_lookback_seconds`/`moment_lookback_seconds`, revised 2026-09-07 from
    a full real game's worth of oversized 65s fallback clips -- evidence-based
    now, not the original provisional guess). Lookahead stays 5s.
  - **Open, not yet automated**: reliably telling "two real close-together
    goals" apart from "one goal, accidentally double-tapped" well enough to
    auto-correct the derived score (rather than merging the clip and leaving
    the score as-is for Kaveh to eyeball) needs labeled real examples to
    validate against -- deliberately deferred to the post-export Sep-06
    analysis pass below, not guessed at mid-pipeline.
  - **First real-game correction, for calibration**: Sep-06's tally showed
    black 11, but two black-goal taps 6.8s apart with no audio peak near
    either and near-identical Gemini descriptions turned out to be one goal,
    double-tapped -- the later tap removed directly from the source CSV
    (2026-09-08, Kaveh's explicit instruction; the earlier session had
    instead worked around it downstream, which he considers the wrong fix
    once a real duplicate is confirmed).
- **A decrement (Tallies' minus button) is NOT auto-resolved as an undo**
  (changed 2026-09-07 in `load_tallies_csv`, after Sep-06 showed a decrement
  landing 2.5 minutes after the press it was meant to cancel, with no
  guarantee nothing else in that category happened in between -- popping
  the most-recently-active press is only correct by luck, not by
  construction). Every positive press is now kept as a real Mark
  unconditionally; decrements are counted and warned about, not resolved.
  **This has a real consequence for the printed/derived score: it is now an
  upper bound, not the true final tally, whenever any decrements exist for
  that game.** Sep-06 bit this exact way -- Kaveh flagged 2 known mis-taps
  for identification "during candidate review," but never actually named
  which marks to exclude before export, so the posted clips showed black's
  inflated raw count (12) instead of the agreed 10. Getting a decremented
  game's displayed score right now requires an explicit follow-up: identify
  the specific mis-tapped mark(s) and either drop them from the source CSV
  or hand-correct `score_white`/`score_black`, before export -- don't let
  "I'll figure out which ones later" silently ship as the displayed number.
  **A console warning alone wasn't enough** (that's exactly what Sep-06 had,
  and it still got missed) -- as of 2026-09-08, `pre-label` also writes
  `candidates/DECREMENTS_PENDING.txt` (self-clearing on a rerun with
  nothing pending) whenever this applies, and `ingest-marks` prints the
  same `***`-bracketed summary before any render. Treat that file's
  existence as a hard stop: ask Kaveh which press(es) each decrement cancels
  before Step 3/4, every time, don't just read past the console line.
- **Taps outside their own clip are now reported, not just discoverable in
  hindsight** (`marks.marks_outside_their_clip`, added 2026-09-08). A tap
  that lands outside its owning candidate's rendered window (even after
  `extend_for_score_flip`) makes that clip's score flip at a midpoint
  instead of visibly at the tap -- easy to miss until someone notices a
  clip's score just... changed, with no visible reason. Both `ingest-marks`
  (pre-render estimate) and `pre-label` (against the actual final
  candidates) print this list now.
- **`telegram-post` always sends in chronological (game) order**, regardless
  of what order `--clips` lists them in (fixed 2026-09-08, after Sep-06
  posted in review-rank order and read confusingly out of game order).
  Sorting happens inside the command itself, by the sheet's
  `start_seconds` -- nothing to remember when picking clips off the ranked
  sheet, which is naturally NOT chronological order.
- **Step 5b (the raw-footage Drive-link announcement) is now a standard part
  of every week's posting, not an occasional extra** (settled 2026-09-08,
  was previously framed as optional). Ask Kaveh for the Drive share link if
  you don't have it yet; don't skip the step for not having it.
- **A game's watch-tagging quirks are worth a dedicated post-export pass, not
  fixed reactively mid-review.** Sep-06 was the first real game with Tallies
  data, and surfaced things worth analyzing properly once the picks are
  locked in and exported: how well the assumed press-lag windows actually
  held up, whether other double-taps are hiding in the data, and whether a
  real "was this one goal or two" auto-correction is worth building from
  labeled examples. Kaveh's own plan (2026-09-07): come back to this after
  export, not before.
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
  **Measured end-to-end export cost is worse than that: ~37x realtime**
  (2026-08-23, sources on the Drive path) -- ~20x is decode floor, the rest is
  `preset medium` encode. Budget ~2.2 hours for a 10-clip batch, and see
  processing.md Step 4 before planning any export work.
- **This laptop has been interrupted by sleep and by the terminal app closing
  overnight**, more than once, during long unattended jobs -- not just a one-off.
  Every long-running step in this pipeline is designed to be resumable
  (incremental, position-keyed caching that only retries what actually failed).
  Trust that design; don't add fresh incremental-save logic from scratch if an
  existing resumable command already covers the case.
- **A killed job can leave a plausible-looking but broken artifact.** Two
  instances found 2026-08-23, both now fixed but worth recognizing the pattern:
  a truncated video with no `moov` atom that a `size > 0` check called "done"
  (use `render.is_playable()`), and a JSON cache written as a growing prefix
  (`entries[:i+1]`) that discarded 30 already-cached rows when killed mid-loop.
  **When adding any resume/skip check, validate the artifact, don't just stat
  it** -- and write whole files, not prefixes.
- **Anything over ~15s of footage cannot be exported in one command run** at 37x
  realtime. Use `scripts/seg_render.py`, and render to a local dir rather than
  straight to the Google Drive path (large writes there stall and get killed).
- **Check for other running render/API jobs before starting a heavy one** --
  `Get-CimInstance Win32_Process | Where-Object Name -match 'ffmpeg|python'` (this
  has bitten a prior session: two heavy jobs competing for this weak CPU at once).
- **Kaveh sometimes runs concurrent Claude Code sessions** against this same repo.
  Check `git status`/`git log` at the start of a session for changes another
  session may have already made.
