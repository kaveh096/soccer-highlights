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

# 3. Or: compare several named strategies at once via small review clips
#    (see "Tuning workflow" below).
python -m soccer_highlights.cli batch-review
python -m soccer_highlights.cli review-sheet
#   ... watch the clips, fill in each review_sheet.csv's verdict column ...
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
| `--strategy {rms_energy,onset_flux}` | Override `detection.strategy` for `detect`/`render`. Has no effect on `batch-review`, which always uses every strategy in `config/strategies.yaml`. |

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

**`review-sheet`** -- (re)generates `review_sheet.csv` in every strategy
folder (and `negatives/`) under `review.output_root`, from that folder's
`events.json` and its `clip_NNN.mp4` files. Columns: `clip_file`,
`start_seconds`, `end_seconds`, `duration_seconds`, `max_peak_score`,
`verdict` (blank), `notes` (blank). Safe to re-run -- it overwrites the
sheet, so don't fill in `verdict` until you're done second-guessing
thresholds, or re-generate into a fresh copy first.

**`score`** -- reads every `review_sheet.csv` under `review.output_root`
and computes precision/recall/F1 per strategy. Fill in `verdict` before
running this:
- Strategy sheets: `TP` (real shot-on-target) or `FP` (anything else).
  Unlabeled rows (blank verdict) are excluded from precision, not counted
  as either.
- `negatives/review_sheet.csv`: `MISS` if a real event fell in that
  uncovered gap; leave blank otherwise.

Ground truth is defined as the union of every `TP`-labeled interval across
*all* strategies (merged where they overlap) plus every `MISS`-labeled
negative gap. A strategy's recall is how much of that union its own `TP`
clips overlap. See [Limitations](#limitations) -- this is relative to what
the whole batch found, not an independent ground truth.

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

`config/strategies.yaml` defines four named presets crossing each
algorithm with a loose/strict sensitivity:

| Name | Algorithm | Sensitivity |
|---|---|---|
| `crowd_loose` | rms_energy | wide net on crowd reaction |
| `crowd_strict` | rms_energy | only strong crowd reactions |
| `strike_loose` | onset_flux | wide net on sharp transients |
| `strike_strict` | onset_flux | only strong transients |

These are a starting point tuned against one test recording -- retune
after scoring a real game (see `score`'s output for where each strategy
currently lands).

## Configuration reference

`config/default.yaml`:

| Section | Field | Meaning |
|---|---|---|
| `input` | `source_dir` | Folder with one session's `DJI_*_D.MP4`/`.LRF` files. |
| | `use_lrf_for_detection` | Detect against the `.LRF` proxy instead of the full-res file. Leave `true` unless you have a reason not to. |
| `audio` | `sample_rate` | Resample target (Hz) for analysis. |
| | `mono` | Downmix to mono for analysis. |
| `detection` | `strategy` | `rms_energy` or `onset_flux`. |
| `detection.rms_energy` | `window_seconds` / `hop_seconds` | RMS analysis window/hop. |
| | `baseline_window_seconds` | Rolling window for the adaptive median/MAD baseline. |
| | `threshold_sigma` | MAD multiplier for the adaptive threshold (see caveat above). |
| | `min_absolute_dbfs` | Hard floor -- a frame below this is never a peak, regardless of local threshold. |
| | `min_score_dbfs` | Minimum required excess above the adaptive threshold (dB) to count as an event. |
| `detection.onset_flux` | `window_seconds` / `hop_seconds` | STFT window/hop for spectral flux. |
| | `baseline_window_seconds`, `threshold_sigma` | Same shape as rms_energy. |
| | `min_score` | Minimum required excess above threshold, in flux units (no fixed physical scale -- check a debug plot before picking a number). |
| `timeline` | `lookback_seconds` | Context to keep before each peak. |
| | `post_peak_seconds` | Context to keep after each peak. |
| | `min_gap_seconds` | Intervals closer than this merge into one. |
| | `min_interval_seconds` | Drop merged intervals shorter than this. |
| | `warmup_seconds` | Ignore peaks before this point in the whole session (camera handling/setup noise). |
| `output` | `mode` | `clips` (one file per highlight) or `concat` (single reel). `render` only. |
| | `dir` | Output directory. `render` only. |
| | `force_reencode_on_concat` | Re-encode instead of stream-copy when concatenating (needed only if clips span a codec/timestamp discontinuity). |
| `metadata` | `events_path`, `debug_plot_path` | Where `detect` writes its output. |
| `review` | `output_root` | Where `batch-review` writes everything. |
| | `max_width`, `fps`, `crf`, `preset`, `threads`, `audio_bitrate_kbps` | Review-clip encode settings. `preset`/`threads` default to `ultrafast`/2 deliberately -- see [Limitations](#limitations). |
| | `max_negative_clip_seconds`, `min_negative_clip_seconds` | Chunking bounds for negative-space clips. |

`config/strategies.yaml` defines named presets as partial overlays on top
of `default.yaml` -- see [Detection strategies](#detection-strategies).

## Limitations

- **Audio-only.** No visual confirmation yet (Phase 2 in `specs.md` is
  unbuilt). Audio proxies for "shot on target" are imperfect: a blocked
  or saved shot may have no crowd reaction and no sharp mic transient,
  and would be invisible to both strategies.
- **Recall is relative, not absolute.** `score`'s recall is measured
  against the union of everything the batch (all strategies + your own
  negative-space review) found -- not an independent ground truth. An
  event every strategy misses, in a negative-space gap you also don't
  catch by eye, will never show up as a false negative anywhere.
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
- **Old-hardware encoding load.** The review-clip encoder defaults to
  `ultrafast` preset with a 2-thread cap after the development machine
  had two unexpected shutdowns during an overnight batch under sustained
  encoding load (likely thermal/power related on an older laptop). If
  you're on capable hardware, `veryfast`/more threads will encode faster
  at similar output size.
- **ffmpeg concat requires matching codecs.** `clipping.concat_clips`
  stream-copies by default, which only works because all source chunks
  share one recording session's codec parameters. Mixing sessions/cameras
  would need `force_reencode_on_concat: true`.

## Tests

```
python -m pytest tests/
```

Detection, timeline, config, and scoring logic are covered with synthetic
data (no real video/audio required). ffmpeg-dependent code (`audio.py`,
`clipping.py`, `render.py`) isn't unit-tested -- validate those against
real footage with `detect`/`batch-review` before trusting a tuning round.
