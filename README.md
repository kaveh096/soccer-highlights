# Sunday Soccer Highlights Engine

Extracts "shots on target" highlights from long continuous soccer recordings
(2+ hours, 60-80GB) by detecting audio peaks (crowd reaction, ball-strike
transients) and slicing out the preceding context around each one. Built for
a DJI Osmo Action 4 fixed sideline setup filming 16-player Sunday half-court
games, but the pipeline is camera/config-agnostic beyond the file-naming
assumptions described below.

Current status: **Phase 1 (audio-only detection)**. See [specs.md](specs.md)
for the full original design doc, including planned Phase 2 vision-AI
refinement (Google Video Intelligence / Rekognition / GPT-4o) that isn't
built yet.

## How it works

1. **Discovery** (`discovery.py`) finds a recording session's chunk files
   (DJI splits long recordings into ~16GB pieces) and orders them into one
   continuous timeline.
2. **Detection** (`detection.py`) extracts audio and scans it for peaks
   above an adaptive local threshold (rolling median + MAD), using one of
   two strategies -- see [Detection strategies](#detection-strategies)
   below. Detection reads the small `.LRF` proxy file DJI writes alongside
   each `.MP4`, not the full-res 4K source -- same audio track, ~1/16th the
   size, so nothing here ever needs the full recording on local disk.
3. **Timeline** (`timeline.py`) turns each peak into a
   lookback/post-peak window, merges overlapping windows into highlight
   intervals, and (for `render`) maps intervals back onto per-chunk file
   ranges, including intervals that span a chunk boundary.
4. **Clipping** (`clipping.py`) slices the final highlight clips out of the
   full-resolution source via ffmpeg stream-copy (no re-encode) -- only
   used by `render`, which is the only command that touches the full-res
   files at all.
5. **Review rendering** (`render.py`) produces small resized/downsampled
   clips (from the `.LRF` again) for fast human review of a strategy's
   true/false-positive rate, without needing the full-res files or much
   disk space.
6. **Scoring** (`scoring.py`) turns human-labeled review sheets into
   precision/recall/F1 per strategy.

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) / ffprobe on `PATH`
- A DJI-style recording session: `DJI_<YYYYMMDDHHMMSS>_<seq>_D.MP4` files
  with matching `.LRF` proxies in the same directory (see
  [Limitations](#limitations) if your camera differs)

## Install

```
python -m venv .venv
.venv\Scripts\activate        # source .venv/Scripts/activate on git-bash
pip install -e .
```

## Quick start

```
# 1. Detect peaks only -- writes events.json + a debug plot, no video output.
#    Cheap and fast; use this to sanity-check thresholds before rendering anything.
python -m soccer_highlights.cli detect

# 2. Once you trust a threshold, render the actual highlight clips (full-res source).
python -m soccer_highlights.cli render

# 3. Or: compare several named strategies at once via small review clips.
python -m soccer_highlights.cli batch-review
python -m soccer_highlights.cli review-sheet
#   ... watch the clips, fill in each review_sheet.csv's verdict column ...
python -m soccer_highlights.cli score

# 4. Re-tuning after a scored round: archive the labeled review_root, re-run
#    batch-review with new config/strategies.yaml numbers, then generate
#    sheets pre-filled with a guess (from time-overlap with the prior
#    round's labels) to speed up re-labeling.
cp -r output/review output/review_round1
python -m soccer_highlights.cli batch-review
python -m soccer_highlights.cli review-sheet --prior-root output/review_round1
#   ... verify each guess against the clip, correct verdict where the guess is wrong ...
python -m soccer_highlights.cli score
```

## CLI reference

### Global flags

These come *before* the subcommand (argparse quirk: `--strategy detect`
fails, `--strategy rms_energy detect` works).

| Flag | Description |
|---|---|
| `--config PATH` | Config YAML to load instead of `config/default.yaml`. |
| `--source-dir DIR` | Override `input.source_dir` (the recording session's folder). |
| `--strategy {rms_energy,onset_flux,combined}` | Override `detection.strategy` for `detect`/`render`. Has no effect on `batch-review`, which always uses every strategy in `config/strategies.yaml`. |

Any config field can also be overridden via environment variable:
`SOCCER_HL__<SECTION>__<FIELD>=value`, e.g.
`SOCCER_HL__DETECTION__RMS_ENERGY__THRESHOLD_SIGMA=4.0`. Env overrides apply
after the YAML file, so they win.

### Subcommands

**`detect`** -- run peak detection only. Writes `metadata.events_path`
(default `output/events.json`) and `metadata.debug_plot_path` (default
`output/debug_audio.png`, one subplot per source chunk showing the raw
signal, adaptive threshold, and shaded highlight intervals). No video
touched. Use this to iterate on thresholds cheaply.

**`render`** -- runs detection, then slices highlight clips from the
full-resolution source files via ffmpeg stream-copy into `output.dir`
(default `output/`), one `highlight_NNN.mp4` per interval. If
`output.mode: concat`, also joins them into `highlight_reel.mp4` and
deletes the individual clips. This is the only command that reads the
full-res `.MP4` files -- everything else uses the `.LRF` proxy.

**`batch-review`** -- runs detection + review-clip rendering for every
strategy in `config/strategies.yaml`, plus:
- **Negative-space clips**: gaps not covered by *any* strategy's candidate
  intervals, chunked to `review.max_negative_clip_seconds` and written to
  `<review.output_root>/negatives/`, so you can specifically check for
  events every strategy missed instead of re-watching already-covered
  footage.
- **Whole-game skim**: every chunk's `.LRF` concatenated and
  resized/downsampled into one `full_game_skim.mp4`, for a single quick
  pass over the whole recording.

All review output is small (resized, downsampled, from the `.LRF`) and
lands under `review.output_root` (default `output/review/`) -- this
command never touches the full-res source.

**`review-sheet [--prior-root PATH]`** -- (re)generates `review_sheet.csv`
in every strategy folder (and `negatives/`) under `review.output_root`,
from that folder's `events.json` and its `clip_NNN.mp4` files. Columns:
`clip_file`, `start_seconds`, `end_seconds`, `duration_seconds`,
`max_peak_score`, `verdict` (blank), `notes` (blank). Safe to re-run -- it
overwrites the sheet, so don't fill in `verdict` until you're done
second-guessing thresholds, or re-generate into a fresh copy first.

With `--prior-root` pointing at a previous round's `review_root` (e.g. an
archived `output/review_round1`), each row also gets `guess` and
`guess_basis` columns: a `TP`/`FP` prediction (or `AMBIGUOUS`/blank) based
on time-overlap with that prior round's labels, plus which old clip(s) and
notes it came from. This never writes to `verdict` itself -- it's a
starting point to speed up re-labeling after a retune, not a substitute
for actually watching the new clips.

**`score`** -- reads every `review_sheet.csv` under `review.output_root`
and computes precision/recall/F1 per strategy. Fill in `verdict` before
running this:
- Strategy sheets: `TP` (real shot-on-target) or `FP` (anything else).
  Unlabeled rows (blank verdict) are excluded from precision, not counted
  as either.
- `negatives/review_sheet.csv`: `FN` if a real event fell in that
  uncovered gap, `TN` if correctly nothing was there.

Ground truth is defined as the union of every `TP`-labeled interval across
*all* strategies (merged where they overlap) plus every `FN`-labeled
negative gap. A strategy's recall is how much of that union its own `TP`
clips overlap. See [Limitations](#limitations) -- this is relative to what
the whole batch found, not an independent ground truth.

**`golden-score [--golden-events PATH] [--vision]`** -- scores the
*current* `--strategy`/config against a pre-built golden event set (exact
ground-truth timestamps, see [golden.py](#round-3-results)) instead of a
labeled batch-review round: audio-only, no clip rendering, no human
review, just `detect` + compare against known timestamps. Default path is
`testdata/golden_events.json`. Once a golden set exists for a recording,
this is the fast path for re-checking any tuning change -- see
[tuning.py](src/soccer_highlights/tuning.py).

With `--vision`, also runs the Phase 2 vision refinement pass (see
[Vision AI (Phase 2)](#vision-ai-phase-2)) over the audio-only intervals
and prints its score alongside the audio-only one, so the two are
directly comparable -- the go/no-go check for whether vision actually
helps. Requires `vision.api_key_env` (default `ANTHROPIC_API_KEY`) to be
set; calls the Claude API once per candidate window/gap, so unlike every
other `golden-score` use it isn't free or instant.

## Detection strategies

Two underlying algorithms (`detection.py`), each responding to a
different physical signal:

- **`rms_energy`**: rolling RMS-dB envelope vs. an adaptive local
  threshold (rolling median + `threshold_sigma` * MAD). Responds to
  sustained loudness swells -- crowd cheering, celebration -- which
  typically *lag* the actual shot/goal by 1-3 seconds.
- **`onset_flux`**: spectral flux (frame-to-frame magnitude increase)
  vs. the same adaptive-threshold shape. Responds to sharp instantaneous
  transients -- ball strikes, whistles, contact -- contemporaneous with
  the play.
- **`combined`**: fusion of the two. Keeps an `onset_flux` transient only
  if a corroborating `rms_energy` swell also fires nearby (within
  `detection.combined.window_before_seconds` before /
  `window_after_seconds` after). Added in round 2 once round-1 scoring
  showed neither signal's amplitude alone cleanly separates true from
  false positives (see [Round 1 results](#round-1-results)) -- this
  encodes "shot on target = impact sound + crowd reaction" directly
  instead of picking a stricter number on one signal.

Both algorithms additionally require the peak to exceed the threshold by
at least `min_score_dbfs` (rms_energy) / `min_score` (onset_flux). This
floor matters a lot: `threshold_sigma` alone is unreliable because both
signals are heavy-tailed relative to what a MAD-based estimator assumes --
flux especially, since it spikes on nearly every footstep/contact, not
just significant events. In practice `min_score`/`min_score_dbfs` did
almost all of the real discriminating work during tuning; don't expect
`threshold_sigma` to matter much on its own (see `config/strategies.yaml`
for the numbers that came out of an actual sweep against test footage,
not first guesses).

`config/strategies.yaml` defines six named presets. **`strike_loose` is the
recommended default** (also `config/default.yaml`'s out-of-the-box
strategy) -- see [Round 3 results](#round-3-results) for why the others
are kept mainly for comparison, not because they're competitive:

| Name | Algorithm | Sensitivity |
|---|---|---|
| `crowd_loose` | rms_energy | wide net on crowd reaction |
| `crowd_strict` | rms_energy | only strong crowd reactions |
| `strike_loose` | onset_flux | **recommended default** -- catches every "must-catch" golden event |
| `strike_strict` | onset_flux | fewer clips, trades one real event for better precision |
| `combo_loose` | combined | loose flux + wide (20s) corroboration window |
| `combo_strict` | combined | tighter flux + shorter (10s) corroboration window |

These are tuned against one test recording -- retune (`golden-score` once
you have a golden set, or `score` after a fresh labeled round) if results
look off on a different game/venue.

## Round 1 results

Scored against a full human pass over one ~43-minute test recording (every
clip in all four strategies labeled `TP`/`FP`, every negative-space clip
labeled `TN`/`FN`), at commit [`9ffd493`](../../commit/9ffd493) -- the
`config/strategies.yaml` state described in the table above:

| strategy | labeled/total | TP | FP | precision | recall | F1 |
|---|---|---|---|---|---|---|
| `crowd_loose` | 12/12 | 6 | 6 | 0.50 | 0.55 | 0.52 |
| `crowd_strict` | 4/4 | 2 | 2 | 0.50 | 0.18 | 0.27 |
| `strike_loose` | 16/16 | 8 | 8 | 0.50 | 0.73 | 0.59 |
| `strike_strict` | 7/7 | 4 | 3 | 0.57 | 0.36 | 0.44 |

Ground truth: 11 events (union of all TP-labeled clips + the one FN-labeled
negative-space gap). Reproduce with `soccer-highlights score` once the
review sheets are filled in.

Takeaway that shaped round 2: plotting each clip's `max_peak_score` against
its verdict showed no threshold that separates TP from FP cleanly on either
signal alone -- e.g. `crowd_loose`'s FP scores include values *higher* than
several of its own TPs, and `strike_loose`'s TP/FP score ranges overlap
almost entirely. `threshold_sigma`/`min_score` tuning had already been
pushed about as far as it usefully goes; the ceiling looked structural
(both signals alone can't tell "real cheer" from "loud crowd noise during a
break", or "ball strike" from "footstep"), not a missed threshold. That's
what motivated the `combined` fusion strategy below rather than a third
round of pure retuning.

## Round 3 results

Round 2's full human relabeling included exact timestamps for every real
event (both which second within a clip, and offsets for events that fell
in negative-space gaps). That's enough to build a reusable **golden event
set** -- `testdata/golden_events.json`, 13 ground-truth timestamps for the
round-2 test recording, built once by
[`golden.build_golden_events`](src/soccer_highlights/golden.py) from the
labeled sheets (a TP clip's strongest detected peak as its anchor time,
an FN's exact "N seconds in" note as its offset; anchors within 12s
collapsed to one event). From here on, any config change can be scored
against real ground truth in seconds (`golden-score` or
[`tuning.py`](src/soccer_highlights/tuning.py)'s cached-audio sweep
helper) instead of rendering clips and asking for another labeling pass.

This overturned both earlier rounds' assumptions:

- **A duration-blind precision/recall metric is gameable.** The very first
  sweep found configs with *perfect* golden-set precision/recall -- by
  merging almost the entire game into 2-5 multi-hundred-second blobs that
  trivially "contain" every event. `score_intervals_against_golden` now
  also reports `max_clip_duration_seconds`/`total_duration_seconds`; never
  trust an F1 number without checking those alongside it.
- **`onset_flux` alone, properly tuned, beats every `rms_energy` and
  `combined`-fusion variant tried**, once duration is accounted for --
  reversing round 2's fusion bet. A sweep of `min_score` in 0.01 steps
  (fixed `threshold_sigma=2.0`, the round-2 timeline unchanged) found
  `min_score=0.55` catches all 10 "must-catch" golden events (the only 3
  misses in the whole 0.50-0.55 range are events the reviewer's own notes
  marked "not very loud, ok to miss it") in 32 clips averaging 11.9s,
  covering 14.7% of the game runtime:

| strategy (golden-set scoring) | clips | precision | recall | F1 | mean dur | % of game |
|---|---|---|---|---|---|---|
| `strike_loose` (onset_flux, min_score=0.55) | 32 | 0.31 | 0.77 (**1.00 must-catch**) | 0.44 | 11.9s | 14.7% |
| `strike_strict` (onset_flux, min_score=0.60) | 26 | 0.35 | 0.69 (0.90 must-catch) | 0.46 | 11.8s | 11.8% |
| `combo_loose` (combined, best found) | 18 | 0.33 | 0.46 | 0.39 | 18.1s | 12.6% |
| `crowd_strict` (rms_energy, best found) | 18 | 0.28 | 0.38 | 0.32 | 13.3s | 9.2% |

"Recall" here is raw (all 13 golden events); "must-catch" recall excludes
the 3 events the reviewer explicitly flagged as acceptable to miss (quiet
goals on the far side of the field) -- see the [Limitations](#limitations)
note on what that distinction does and doesn't mean.

`config/default.yaml` and `strike_loose` now default to
`onset_flux`/`min_score=0.55`. `crowd_*`/`combo_*` were retuned too (best
F1 found in the same sweep) but are kept as clearly inferior secondary
options, not because they're worth using over `strike_loose` as-is.

## Round 4: plateau check

One more pass to confirm round 3 wasn't a local optimum, both at
`min_score=0.55`:

- `threshold_sigma` sweep (1.0-4.0): 2.0-2.5 tie on must-catch recall;
  2.5 edges out 2.0 on precision (0.32 vs 0.31) and clip count (31 vs 32)
  for free, so that's the new default. 3.0+ starts dropping a must-catch
  event -- confirms round 1's "sigma barely matters" finding still holds,
  now to a decimal place instead of a shrug.
- `min_interval_seconds` (3.0/5.0/8.0): zero effect -- every merged
  interval at this operating point already clears even the 8s floor.

No further gain available by turning these particular knobs -- this is
the tuning plateau for a pure single-signal audio approach on this
recording. Getting past it (e.g. reliably catching the 3 quiet, far-side
goals without also catching a lot more background noise) would need a
different kind of signal, not another threshold sweep -- see
[specs.md](specs.md)'s planned Phase 2 vision refinement.

## Vision AI (Phase 2)

Groundwork for the vision refinement specs.md's Phase 2 named, layered on
top of the audio pipeline rather than replacing it. Targets the two
confirmed, human-labeled audio failure modes from Round 3/4 directly
(see [Limitations](#limitations)): practice shots / crowd chatter during
breaks that sound like real action, and quiet far-side goals that never
clear an audio threshold. Both are visually obvious even when barely
audible, which audio-only detection structurally can't use.

Provider is **Claude's vision API** (`anthropic` package) -- not the
Google Video Intelligence / AWS Rekognition / GPT-4o options specs.md
named. Google's shot/object/label detection doesn't answer "shot on
target vs. practice shot" (that's not a label it produces); AWS
Rekognition has no built-in "kicking"/"goal celebration" action class and
would need a custom-trained model. Both also need async job/bucket
plumbing (upload to cloud storage, poll an operation) this repo doesn't
have. A direct vision-LLM prompt answers the actual semantic question in
one synchronous call per candidate window/gap -- fits the existing
"small script calls ffmpeg, does one thing" shape directly, and frames
are always pulled from the `.LRF` proxy (never the full-res source), same
reasoning as audio detection on this hardware.

[`vision.py`](src/soccer_highlights/vision.py) runs two passes, both
built on the same `Interval`/`GlobalPeak` primitives
[`timeline.py`](src/soccer_highlights/timeline.py) already provides:

- **Confirm**: for every audio-flagged interval, sample
  `vision.frames_per_window` frames from a `vision.peak_window_seconds`
  window CENTERED ON THE INTERVAL'S DETECTED PEAK (not spread evenly
  across the whole clip -- see the 2026-07-25 finding below for why that
  distinction mattered in practice) and ask whether they show real
  in-play action or a false-positive-shaped moment (practice shot,
  break-time chatter), plus a short factual caption. An interval is only
  *dropped* on a false-positive verdict at or above
  `vision.drop_confidence_threshold` -- recall-first, matching this
  project's standing tuning priority: an uncertain or errored vision call
  keeps the interval rather than risk discarding a real event.
- **Scan**: for every gap no audio strategy flagged (`invert_intervals`,
  same negative-space chunking `batch-review` already uses), sample
  frames across it and ask whether a real event is visible anyway. A new
  interval is only *added* on an event verdict at or above
  `vision.add_confidence_threshold`, windowed with the same
  `timeline.lookback_seconds`/`post_peak_seconds` a real audio peak
  would get.

Both passes return per-window verdicts (`VisionRunLog`) written to
`vision_events.json` for spot-checking against real clips -- the same
role `metadata.events_path` plays for audio.

`soccer-highlights vision-highlights` runs confirm-only (no scan pass --
see below) over a real audio-detect pass, renders survivors as
`"{seconds} - caption.mp4"` (fast `cfg.review` renders by default,
`--full-quality` for the real `cfg.export` 4K delivery encode once
you trust the surviving timestamps), and archives pruned candidates in a
`pruned/` subfolder for audit instead of deleting them.

Pure decision logic (`_evenly_spaced_offsets`, `_parse_verdict_json`,
`_peak_anchor_time`, `sanitize_caption_for_filename`, `decide_confirm`,
`decide_scan`, `refine_with_vision`'s merge step) is unit-tested against
injected/fake verdicts, no real API or ffmpeg calls in `pytest`, same
split as `audio.py`/`clipping.py`/`render.py`.

### Round 1 (vision): confirm+caption only marginally beats audio alone

Tested against the real round-2 recording and `testdata/golden_events.json`
on 2026-07-25, in two passes:

1. **First pass** (whole-clip evenly-spaced frame sampling, the original
   design): recall collapsed, 0.77 -> 0.23 (FN 3 -> 10). Root cause: a
   real shot/strike is a sub-second transient; 5 frames spread across a
   12-20s clip usually land just before/after the actual moment, so the
   model confidently (0.75-0.85) misread real events as practice shots.
   Fixed by anchoring frame sampling on the interval's detected peak time
   (`_peak_anchor_time`) instead of spreading across the whole clip.
2. **Second pass** (peak-anchored, loosened audio to `min_score=0.45`
   first to test whether that recovered any additional recall -- it
   didn't: FN stayed at 3, loosening past `min_score=0.55` just added
   more false positives, confirming Round 3/4's finding again rather than
   revealing anything new). Confirm+caption's actual effect, swept
   offline against the cached verdicts (no extra API calls, same idea as
   the audio `min_score` sweep):

   | `drop_confidence_threshold` | kept | precision | recall | F1 | FN |
   |---|---|---|---|---|---|
   | (no filtering / audio-only) | 40/40 | 0.250 | 0.769 | 0.377 | 3 |
   | 0.75 (current default) | 31/40 | 0.290 | 0.692 | 0.409 | 4 |
   | 0.70 | 28/40 | 0.286 | 0.615 | 0.390 | 5 |
   | 0.65 | 26/40 | 0.231 | 0.462 | 0.308 | 7 |
   | <=0.60 | 9/40 | 0.222 | 0.154 | 0.182 | 11 |

**Conclusion**: the current default (0.75) is the best value found, but
"best" means +0.03 F1 over not running vision at all, at the cost of one
more missed real event -- not the clear precision win Phase 2 set out to
find. Below 0.70 it gets worse than doing nothing. The model's confidence
score does not cleanly separate true from false positives in the 0.6-0.75
band from a handful of low-res stills; peak-anchoring fixed the
catastrophic recall bug but didn't turn this into a reliable classifier.

### Round 2 (vision): provider/frame-density comparison infrastructure

Built `vision_gemini.py` (Gemini native-video counterpart to Claude's
stills-based confirm pass -- same peak-anchored sample window via the
now-shared `vision._peak_sample_window`, same unmodified `_CONFIRM_PROMPT`
for a fair first comparison, but sends one short video clip via
`generateContent`'s `inline_data` instead of discrete stills) and
`vision_eval.py` (a reusable, tested harness: `collect_verdicts` caches
each classify call incrementally to `output/vision_compare/<tag>.json`
so an interrupted run against a real paid API doesn't lose progress, and
resumes automatically; `sweep_drop_threshold` scores the same cached
verdicts at several thresholds, offline/free, same idea as the audio
`min_score` sweep one layer up). Exposed via `soccer-highlights
vision-compare --provider {claude,gemini} --tag LABEL` -- pure
measurement, no rendering.

Two results against the same 40 candidates (`min_score=0.45`,
unmodified confirm prompt):

- **Claude, denser frame sampling** (10 frames instead of 5, same 4s peak
  window -- 2.5 FPS instead of 1.25 FPS): essentially no change at the
  best threshold (F1 0.409, FN=4, identical to the 5-frame result).
  Doubling frame density inside the same window doesn't move the needle
  -- rules out "just sample more frames" as an easy fix; the earlier
  confidence-calibration problem isn't simply frame starvation.
- **Gemini, native video, same prompt**: **blocked by the free-tier
  quota** (5 requests/minute, 20/day for `gemini-3.6-flash`) -- only
  23/40 candidates got a real verdict before hitting
  `RESOURCE_EXHAUSTED`, the rest came back `None` and fail-open (kept by
  default), so the naive sweep table over that run is not a valid
  comparison (it mostly reflects the fail-open default, not the model).
  The 23 real verdicts are still a useful *qualitative* signal, though:
  confidences are much more decisive than Claude's (0.80-0.95 both
  directions, vs. Claude's mushy 0.4-0.75 band), and several captions
  read as correct (e.g. `"Shot on goal scored into the net"` at 0.95
  confidence for a candidate that likely is a real event). Not enough to
  draw a real conclusion from -- billing needs enabling on the Gemini
  key's Cloud project before a fair, complete comparison is possible
  (this also resolves the free-tier data-use terms question raised
  earlier). Re-run with `vision-compare --provider gemini --tag
  gemini-existing-prompt` once that's done -- the cache already has the
  23 completed candidates and will resume from there.

## Configuration reference

`config/default.yaml`:

| Section | Field | Meaning |
|---|---|---|
| `input` | `source_dir` | Folder with one session's `DJI_*_D.MP4`/`.LRF` files. |
| | `use_lrf_for_detection` | Detect against the `.LRF` proxy instead of the full-res file. Leave `true` unless you have a reason not to. |
| `audio` | `sample_rate` | Resample target (Hz) for analysis. |
| | `mono` | Downmix to mono for analysis. |
| `detection` | `strategy` | `rms_energy`, `onset_flux`, or `combined`. Default `onset_flux` -- see [Round 3 results](#round-3-results). |
| `detection.rms_energy` | `window_seconds` / `hop_seconds` | RMS analysis window/hop. |
| | `baseline_window_seconds` | Rolling window for the adaptive median/MAD baseline. |
| | `threshold_sigma` | MAD multiplier for the adaptive threshold (see caveat above). |
| | `min_absolute_dbfs` | Hard floor -- a frame below this is never a peak, regardless of local threshold. |
| | `min_score_dbfs` | Minimum required excess above the adaptive threshold (dB) to count as an event. |
| `detection.onset_flux` | `window_seconds` / `hop_seconds` | STFT window/hop for spectral flux. |
| | `baseline_window_seconds`, `threshold_sigma` | Same shape as rms_energy. |
| | `min_score` | Minimum required excess above threshold, in flux units (no fixed physical scale -- check a debug plot before picking a number). |
| `detection.combined` | `window_before_seconds` | How many seconds an rms_energy swell may *precede* a flux transient and still corroborate it. |
| | `window_after_seconds` | How many seconds an rms_energy swell may *follow* a flux transient and still corroborate it (crowd reaction typically lags, so this is usually the larger of the two). |
| `timeline` | `lookback_seconds` | Context to keep before each peak. Default `6.0` (round 2 -- see [Round 1 results](#round-1-results); round 1 used `45.0`). |
| | `post_peak_seconds` | Context to keep after each peak. Default `5.0` (round 1: `8.0`). |
| | `min_gap_seconds` | Intervals closer than this merge into one. |
| | `min_interval_seconds` | Drop merged intervals shorter than this. |
| | `warmup_seconds` | Ignore peaks before this point in the whole session (camera handling/setup noise). |
| `output` | `mode` | `clips` (one file per highlight) or `concat` (single reel). `render` only. |
| | `dir` | Output directory. `render` only. |
| | `force_reencode_on_concat` | Re-encode instead of stream-copy when concatenating (needed only if clips span a codec/timestamp discontinuity). |
| `metadata` | `events_path`, `debug_plot_path` | Where `detect` writes its output. |
| `review` | `output_root` | Where `batch-review` writes everything. |
| | `max_width`, `fps`, `crf`, `preset`, `threads`, `audio_bitrate_kbps` | Review-clip encode settings. Round 2 defaults (`1280`px/`30`fps/`crf 23`/`veryfast`/4 threads/`128`kbps) assume adequate disk space and daytime supervised runs; drop back toward round 1's `640`/`15`/`30`/`ultrafast`/2/`64` for an unattended overnight batch on tight disk space -- see [Limitations](#limitations). |
| | `max_negative_clip_seconds`, `min_negative_clip_seconds` | Chunking bounds for negative-space clips. |

`config/strategies.yaml` defines named presets as partial overlays on top
of `default.yaml` -- see [Detection strategies](#detection-strategies).

## Limitations

- **Audio-only.** No visual confirmation yet (Phase 2 in `specs.md` is
  unbuilt). Audio proxies for "shot on target" are imperfect: a blocked
  or saved shot may have no crowd reaction and no sharp mic transient,
  and would be invisible to both strategies.
- **`score`'s recall is relative, not absolute** (batch-review workflow
  only). It's measured against the union of everything the batch found --
  not an independent ground truth. An event every strategy misses, in a
  negative-space gap you also don't catch by eye, will never show up as a
  false negative. `golden-score` (round 3+) doesn't have this problem for
  recordings with a golden event set, but the golden set itself is
  specific to the one recording it was built from -- a new game/venue
  needs its own labeled round before `golden-score` means anything for it.
- **Golden-set recall counts every event equally, including ones you said
  were fine to miss.** Round 2's reviewer explicitly flagged a few quiet,
  far-side goals as "not very loud, ok to miss it" in the notes -- the
  golden set has no way to encode that distinction, so raw recall looks
  worse than actual usefulness. See [Round 3 results](#round-3-results)'s
  "must-catch" recall for the number that excludes those.
- **A TP clip's exact event time is approximated**, not read from the
  label. `golden.build_golden_events` uses that clip's strongest detected
  peak as the anchor -- close enough for onset_flux (contemporaneous with
  the event) but an approximation for rms_energy (which can lag by a
  couple seconds), and it can't split "two events in one clip" (round 2
  had at least one such note) into two golden events.
- **DJI file-naming assumption.** `discovery.py` expects
  `DJI_<YYYYMMDDHHMMSS>_<seq>_D.MP4` (+ matching `.LRF`) and assumes
  chunks are recorded back-to-back continuously; it warns (doesn't
  correct) if a chunk's start timestamp implies a gap of more than ~2s
  from the previous chunk's expected end, which would indicate dropped
  frames between files.
- **Heavy-tailed audio signals.** Both detection strategies' MAD-based
  adaptive threshold underestimates how spiky real audio is (flux
  especially) -- `min_score`/`min_score_dbfs` compensate, but they were
  tuned against one ~43-minute test recording, not a full 2-hour game
  with different crowd/ambient noise. Re-tune per session if results
  look off.
- **Old-hardware encoding load.** Round 1's review-clip encoder defaulted
  to `ultrafast`/640px/15fps/2 threads after the development machine had
  two unexpected shutdowns during an unattended overnight batch under
  sustained encoding load (likely thermal/power related on an older
  laptop). Round 2 defaults are back up (`veryfast`/1280px/30fps/4
  threads) since clips are now much shorter (social-media-length, not
  45-135s) and the run is daytime/supervised rather than overnight -- drop
  back to the round-1 settings for another unattended overnight batch.
- **`combined` underperforms `onset_flux` alone on this recording.**
  Round 2's bet was that AND-fusion (impact sound + corroborating crowd
  swell) would beat either signal's amplitude alone; round 3's golden-set
  sweep showed the opposite -- fusion can only ever match or reduce
  `onset_flux`'s own recall (a flux event survives only if rms also
  fires), and here rms corroboration filtered out more real events than
  false positives. Kept in `strategies.yaml` for comparison, not as a
  live recommendation. It may still be worth revisiting on a recording
  with a noisier/more ambiguous flux signal, where rms corroboration has
  more false positives to actually cut.
- **ffmpeg concat requires matching codecs.** `clipping.concat_clips`
  stream-copies by default, which only works because all source chunks
  share one recording session's codec parameters. Mixing sessions/cameras
  would need `force_reencode_on_concat: true`.

## Tests

```
python -m pytest tests/
```

Detection, timeline, config, scoring, golden-set, and tuning logic are all
covered with synthetic data (no real video/audio required). ffmpeg-dependent
code (`audio.py`, `clipping.py`, `render.py`) isn't unit-tested -- validate
those against real footage with `detect`/`batch-review`/`golden-score`
before trusting a tuning round.
