"""CLI entry point.

Subcommands:
  detect        - run audio peak detection only; writes events.json + a
                  debug plot for visually tuning thresholds.
  render        - run detection, then slice (or concat) the actual
                  highlight clips from the full-resolution source files.
  batch-review  - run every named strategy from config/strategies.yaml,
                  rendering small resized/downsampled review clips (from
                  the .LRF proxy) per strategy, plus a whole-game skim and
                  negative-space clips, for fast true/false-positive review.
  review-sheet  - (re)generate a fillable review_sheet.csv per strategy
                  (+ negatives) from an existing batch-review output.
                  With --prior-root, also fills a guess/guess_basis column
                  per clip from time-overlap with a previous round's labels.
  score         - read back filled-in review_sheet.csv files and print
                  precision/recall/F1 per strategy.
  export        - detect, then re-encode (not stream-copy) each highlight
                  from the full-res source at export.* settings -- a
                  compressed, widely-compatible copy for sharing (unlike
                  render's lossless-but-huge stream-copy clips).
  export-picks  - re-encode a user-specified subset of an existing
                  review_sheet.csv's rows (by clip_file) at export.*
                  settings -- for exporting just the final hand-picked
                  top clips, not every detected candidate (see export
                  above for that). Reads start/end seconds straight from
                  the sheet, no re-detection.
  telegram-post - post a user-specified subset of already-exported clips
                  (by clip_file) to a Telegram group via sendVideo, using
                  the clip's Farsi gemini_caption from the review sheet
                  as the post caption. --dry-run validates without
                  posting. Tracks sent clips in a state file next to the
                  clips so a rerun doesn't double-post.
  telegram-message - post a one-off plain-text announcement to the same
                  group (no video), via sendMessage. Prefer --text-file
                  over --text for non-ASCII/Farsi text.
  golden-score  - score the current strategy/config against a pre-built
                  golden event set (golden.py), audio-only, no rendering
                  or human review needed. For re-checking tuning changes
                  once a golden set exists (see testdata/README.md).
                  With --vision, also runs the Phase 2 vision refinement
                  pass (vision.py) and prints its score alongside the
                  audio-only one.
  vision-highlights - detect, classify+caption each candidate with
                  peak-anchored frames, and render only the survivors as
                  clips named "{seconds} - caption.mp4". Fast
                  review-quality renders by default; --full-quality for
                  the real export.* (4K) delivery encode once you trust
                  the surviving timestamps. Pruned candidates get a clip
                  in a pruned/ subfolder for audit, not deleted outright.
                  NOTE (2026-07-25 real-footage test): this only
                  marginally beats audio alone on this recording (F1
                  0.377->0.409, one more missed event) -- see README's
                  Vision AI section before assuming it's a clear win.
                  Gemini video scored better in the same test (F1 0.432,
                  see vision-compare below) but isn't wired into this
                  command yet -- still Claude-only.
  vision-compare - classify every candidate interval via --provider
                  {claude,gemini}'s classify_confirm, caching verdicts
                  incrementally (resumable), then sweep
                  drop_confidence_threshold offline against the golden
                  set. Pure measurement, no rendering -- for comparing
                  providers/prompts/frame-density settings on the exact
                  same candidate set. See README's Vision AI section.
  label-audit   - audits the existing human-labeled Round 2 dataset
                  (output/review/*) instead of tuning detection further:
                  Gemini scores/describes every already-labeled clip (no
                  verdict shown to it, reading the same clip file the
                  human labeled -- no separate render), Claude judges
                  whether that agrees with the original human verdict/
                  notes. Copies flagged (disagreeing) rows' clips to
                  flagged_clips/ for human review, and writes a CSV with
                  every row -- original label, Gemini description, judge
                  verdict, and two blank columns for you to fill in a
                  revised label/notes. See label_audit.py and README's
                  Vision AI section.
  pre-label     - for a BRAND-NEW, not-yet-labeled recording: detect
                  candidates, render clips at the shared review spec
                  (cfg.review), generate a Gemini description for each
                  from that same file (no judge step -- there's no prior
                  human label yet), and write a fillable review_sheet.csv
                  (+events.json) in the same batch-review-compatible
                  shape, plus a bonus gemini_description column, for a
                  first labeling pass.
  ingest-marks  - report-only v1 (2026-08-31) of the live-tagging ingest
                  (see marks.py): parse a wall-clock marks CSV
                  (timestamp/category/sequence -- white_goal/black_goal/
                  moment, undo resolved by dropping the row it cancels),
                  map each mark onto the global recording timeline via
                  discovery.wallclock_to_global, and classify it as
                  snapped-to-an-existing-audio-peak, fixed-window fallback
                  (no nearby peak -- the actual recall gain), or landed in
                  an unrecorded gap between chunks. Runs audio detection to
                  get peaks to snap against, but does NOT render any clips
                  or write events.json yet -- this only proves the mapping
                  is correct against real data before it feeds pre-label's
                  candidate set. Also prints the watch's white/black tally
                  for the free score-checksum sanity check.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from soccer_highlights import clipping, label_audit, marks, render, telegram, vision, vision_eval, vision_gemini
from soccer_highlights.audio import extract_audio_samples
from soccer_highlights.config import Config, load_config, load_strategy_configs
from soccer_highlights.detection import analyze
from soccer_highlights.discovery import Chunk, discover_chunks
from soccer_highlights.golden import GoldenScore, load_golden_events, score_intervals_against_golden
from soccer_highlights.metadata import ChunkTrace, plot_debug, write_events_json
from soccer_highlights.scoring import format_score_report, generate_all_review_sheets, generate_review_sheet, score_all
from soccer_highlights.timeline import (
    GlobalPeak,
    Interval,
    invert_intervals,
    map_interval_to_chunks,
    merge_intervals,
    peaks_to_raw_intervals,
)
from soccer_highlights.tuning import detect_intervals


def _audio_source_path(chunk: Chunk, cfg: Config) -> Path:
    if cfg.input.use_lrf_for_detection and chunk.lrf_path is not None:
        return chunk.lrf_path
    return chunk.mp4_path


def _run_detection(cfg: Config, chunks: list[Chunk]) -> tuple[list[Interval], list[ChunkTrace]]:
    all_peaks: list[GlobalPeak] = []
    traces: list[ChunkTrace] = []

    for chunk in chunks:
        source_path = _audio_source_path(chunk, cfg)
        print(f"Analyzing chunk #{chunk.sequence}: {source_path.name}")
        samples = extract_audio_samples(source_path, cfg.audio.sample_rate, cfg.audio.mono)
        trace = analyze(samples, cfg.audio.sample_rate, cfg.detection.strategy, cfg.detection)
        print(f"  -> {len(trace.events)} raw peak(s) detected")

        for event in trace.events:
            all_peaks.append(GlobalPeak(time_seconds=chunk.global_start_seconds + event.time_seconds, score=event.score))
        traces.append(
            ChunkTrace(
                chunk=chunk,
                times_local=trace.times,
                values=trace.values,
                threshold=trace.threshold,
                value_label=trace.value_label,
            )
        )

    warmed_up_peaks = [p for p in all_peaks if p.time_seconds >= cfg.timeline.warmup_seconds]
    dropped = len(all_peaks) - len(warmed_up_peaks)
    if dropped:
        print(f"  (dropped {dropped} peak(s) within the {cfg.timeline.warmup_seconds}s warm-up period)")

    raw_intervals = peaks_to_raw_intervals(warmed_up_peaks, cfg.timeline.lookback_seconds, cfg.timeline.post_peak_seconds)
    merged = merge_intervals(raw_intervals, cfg.timeline.min_gap_seconds, cfg.timeline.min_interval_seconds)
    print(f"Merged into {len(merged)} highlight interval(s) from {len(warmed_up_peaks)} raw peak(s)")
    return merged, traces


def cmd_detect(cfg: Config) -> None:
    chunks = discover_chunks(cfg.input.source_dir)
    merged, traces = _run_detection(cfg, chunks)

    events_path = Path(cfg.metadata.events_path)
    write_events_json(merged, events_path)
    print(f"Wrote {len(merged)} event(s) to {events_path}")

    plot_path = Path(cfg.metadata.debug_plot_path)
    plot_debug(traces, merged, plot_path)
    print(f"Wrote debug plot to {plot_path}")


def cmd_render(cfg: Config) -> None:
    chunks = discover_chunks(cfg.input.source_dir)
    merged, _traces = _run_detection(cfg, chunks)

    events_path = Path(cfg.metadata.events_path)
    write_events_json(merged, events_path)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="soccer_hl_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        clip_paths: list[Path] = []
        for i, interval in enumerate(merged):
            slices = map_interval_to_chunks(interval, chunks)
            clip_path = out_dir / f"highlight_{i + 1:03d}.mp4"
            print(f"Rendering highlight {i + 1}/{len(merged)}: {clip_path.name}")
            clipping.build_highlight_clip(slices, clip_path, tmp_dir)
            clip_paths.append(clip_path)

        if cfg.output.mode == "concat" and clip_paths:
            reel_path = out_dir / "highlight_reel.mp4"
            print(f"Concatenating {len(clip_paths)} clip(s) into {reel_path.name}")
            clipping.concat_clips(clip_paths, reel_path, force_reencode=cfg.output.force_reencode_on_concat)
            for clip_path in clip_paths:
                if clip_path.exists():
                    clip_path.unlink()

    print("Done.")


def cmd_batch_review(base_cfg: Config) -> None:
    chunks = discover_chunks(base_cfg.input.source_dir)
    strategy_configs = load_strategy_configs(base_cfg)
    review_cfg = base_cfg.review

    output_root = Path(review_cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_strategy_intervals: list[Interval] = []
    with tempfile.TemporaryDirectory(prefix="soccer_hl_review_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        for name, cfg in strategy_configs.items():
            print(f"\n=== Strategy: {name} ===")
            merged, traces = _run_detection(cfg, chunks)
            all_strategy_intervals.extend(merged)

            strategy_dir = output_root / name
            strategy_dir.mkdir(parents=True, exist_ok=True)
            write_events_json(merged, strategy_dir / "events.json")
            plot_debug(traces, merged, strategy_dir / "debug_audio.png")

            for i, interval in enumerate(merged):
                slices = map_interval_to_chunks(interval, chunks)
                clip_path = strategy_dir / f"clip_{i + 1:03d}.mp4"
                print(f"  Rendering {clip_path.name} ({interval.end_seconds - interval.start_seconds:.1f}s)")
                render.render_review_clip(slices, clip_path, tmp_dir, review_cfg)

        print("\n=== Negative-space clips (no strategy fired) ===")
        union = merge_intervals(all_strategy_intervals, min_gap_seconds=0.0, min_interval_seconds=0.0)
        total_duration = sum(c.duration_seconds for c in chunks)
        negatives = invert_intervals(
            union, total_duration, review_cfg.min_negative_clip_seconds, review_cfg.max_negative_clip_seconds
        )
        negatives_dir = output_root / "negatives"
        negatives_dir.mkdir(parents=True, exist_ok=True)
        write_events_json(negatives, negatives_dir / "events.json")
        for i, interval in enumerate(negatives):
            slices = map_interval_to_chunks(interval, chunks)
            clip_path = negatives_dir / f"clip_{i + 1:03d}.mp4"
            print(f"  Rendering {clip_path.name} ({interval.end_seconds - interval.start_seconds:.1f}s)")
            render.render_review_clip(slices, clip_path, tmp_dir, review_cfg)

    print("\n=== Whole-game skim ===")
    skim_path = output_root / "full_game_skim.mp4"
    render.render_whole_game_skim(chunks, skim_path, review_cfg)
    print(f"Wrote {skim_path}")

    print("\nBatch review complete.")


def cmd_review_sheet(cfg: Config, prior_root: str | None) -> None:
    output_root = Path(cfg.review.output_root)
    sheets = generate_all_review_sheets(output_root, Path(prior_root) if prior_root else None)
    for sheet_path in sheets:
        print(f"Wrote {sheet_path}")


def cmd_score(cfg: Config) -> None:
    output_root = Path(cfg.review.output_root)
    scores, ground_truth = score_all(output_root)
    print(format_score_report(scores, ground_truth))


def cmd_export(cfg: Config, out_dir_override: str | None) -> None:
    """Detect, then re-encode (not stream-copy) each highlight interval
    from the full-res source at export.* settings -- a compressed,
    widely-compatible delivery copy for sharing, as opposed to render's
    lossless-but-huge stream-copy clips."""
    chunks = discover_chunks(cfg.input.source_dir)
    merged, _traces = _run_detection(cfg, chunks)

    export_cfg = cfg.export
    out_dir = Path(out_dir_override) if out_dir_override else Path(export_cfg.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="soccer_hl_export_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        skipped = 0
        for i, interval in enumerate(merged):
            clip_path = out_dir / f"highlight_{i + 1:03d}.mp4"
            if clip_path.exists() and clip_path.stat().st_size > 0:
                skipped += 1
                continue
            slices = map_interval_to_chunks(interval, chunks)
            print(f"Exporting {i + 1}/{len(merged)}: {clip_path.name} ({interval.end_seconds - interval.start_seconds:.1f}s)")
            render.render_export_clip(slices, clip_path, tmp_dir, export_cfg)

    if skipped:
        print(f"Skipped {skipped} already-exported clip(s) in {out_dir}")
    print(f"\nExported {len(merged)} clip(s) to {out_dir}")


def cmd_export_picks(
    cfg: Config, review_sheet_path: str, clip_files: list[str], out_dir_override: str | None, crf_override: int | None
) -> None:
    """Re-encode a user-specified subset of a review_sheet.csv's rows (by
    clip_file) from the full-res source at export.* settings -- for
    exporting just the final hand-picked top clips, not every detected
    candidate (see cmd_export for that). Reads start/end seconds straight
    from the sheet instead of re-running detection, so there's no risk of
    interval-index drift if detection config has changed since the sheet
    was generated."""
    with open(review_sheet_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_clip_file = {row["clip_file"]: row for row in rows}
    missing = [c for c in clip_files if c not in by_clip_file]
    if missing:
        raise SystemExit(f"clip_file(s) not found in {review_sheet_path}: {', '.join(missing)}")

    chunks = discover_chunks(cfg.input.source_dir)
    export_cfg = cfg.export
    if crf_override is not None:
        export_cfg.crf = crf_override
    out_dir = Path(out_dir_override) if out_dir_override else Path(export_cfg.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="soccer_hl_export_picks_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for i, clip_file in enumerate(clip_files):
            row = by_clip_file[clip_file]
            clip_path = out_dir / clip_file
            if render.is_playable(clip_path):
                print(f"Skipping {i + 1}/{len(clip_files)}: {clip_path.name} (already exported)")
                continue
            if clip_path.exists():
                # Present but unplayable == a killed render's truncated leftover.
                # Drop it rather than skipping it (see render.is_playable).
                print(f"Re-exporting {clip_path.name}: existing file is truncated/unplayable")
                clip_path.unlink()
            interval = Interval(start_seconds=float(row["start_seconds"]), end_seconds=float(row["end_seconds"]))
            slices = map_interval_to_chunks(interval, chunks)
            print(f"Exporting {i + 1}/{len(clip_files)}: {clip_path.name} ({interval.end_seconds - interval.start_seconds:.1f}s)")
            render.render_export_clip(slices, clip_path, tmp_dir, export_cfg)

    print(f"\nExported {len(clip_files)} clip(s) to {out_dir}")


def cmd_telegram_post(
    cfg: Config, review_sheet_path: str, clips_dir: str, clip_files: list[str], dry_run: bool
) -> None:
    """Post a user-specified subset of already-exported clips (by clip_file,
    same identifiers export-picks uses) to a Telegram group via sendVideo.
    Caption is the clip's Farsi gemini_caption from the review sheet, if
    present, else just the clip_file name.

    Tracks successfully-sent clips in <clips_dir>/.telegram_sent.json so a
    rerun after a partial failure doesn't double-post to the group -- unlike
    a redundant local render, a duplicate post is visible to everyone in the
    group and can't be quietly cleaned up."""
    with open(review_sheet_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_clip_file = {row["clip_file"]: row for row in rows}
    missing = [c for c in clip_files if c not in by_clip_file]
    if missing:
        raise SystemExit(f"clip_file(s) not found in {review_sheet_path}: {', '.join(missing)}")

    clips_dir_path = Path(clips_dir)
    missing_files = [c for c in clip_files if not (clips_dir_path / c).exists()]
    if missing_files:
        raise SystemExit(f"clip_file(s) not found in {clips_dir}: {', '.join(missing_files)}")

    sent_state_path = clips_dir_path / ".telegram_sent.json"
    sent_state: dict = json.loads(sent_state_path.read_text(encoding="utf-8")) if sent_state_path.exists() else {}

    if not dry_run:
        bot_info = telegram.get_me(cfg.telegram)
        print(f"Connected as @{bot_info.get('username', '?')}")

    for i, clip_file in enumerate(clip_files):
        video_path = clips_dir_path / clip_file
        caption = by_clip_file[clip_file].get("gemini_caption", "").strip() or clip_file

        if clip_file in sent_state:
            print(f"Skipping {i + 1}/{len(clip_files)}: {clip_file} (already sent {sent_state[clip_file]['sent_at']})")
            continue

        size_mb = video_path.stat().st_size / (1024 * 1024)
        if dry_run:
            print(f"[DRY RUN] {i + 1}/{len(clip_files)}: {clip_file} ({size_mb:.1f}MB) -- caption: {caption}")
            continue

        print(f"Sending {i + 1}/{len(clip_files)}: {clip_file} ({size_mb:.1f}MB)...")
        result = telegram.send_video(video_path, caption, cfg.telegram)
        sent_state[clip_file] = {
            "message_id": result["result"]["message_id"],
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        sent_state_path.write_text(json.dumps(sent_state, indent=2, ensure_ascii=False), encoding="utf-8")

    if not dry_run:
        print(f"\nDone. {len(sent_state)} clip(s) recorded as sent in {sent_state_path}")


def cmd_telegram_message(cfg: Config, text: str, text_file: str | None, dry_run: bool) -> None:
    """Post a one-off plain-text announcement to the same Telegram group
    telegram-post sends clips to -- e.g. a note about where to find raw
    footage. Prefer --text-file for anything non-ASCII (Farsi): passing RTL
    text as a raw CLI argument risks shell/console encoding mangling it
    before Python ever sees it, the same class of problem as Step 2's Excel
    mojibake warning."""
    if text_file:
        text = Path(text_file).read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("telegram-message needs non-empty text via --text or --text-file")

    if dry_run:
        bot_info = telegram.get_me(cfg.telegram)
        print(f"[DRY RUN] would send as @{bot_info.get('username', '?')}: {text}")
        return

    bot_info = telegram.get_me(cfg.telegram)
    print(f"Connected as @{bot_info.get('username', '?')}")
    result = telegram.send_message(text, cfg.telegram)
    print(f"Sent message_id={result['result']['message_id']}")


def _print_golden_score(score: GoldenScore, game_duration: float) -> None:
    pct = 100 * score.total_duration_seconds / game_duration if game_duration else 0.0
    mean_dur = f"{score.mean_clip_duration_seconds:.1f}s" if score.mean_clip_duration_seconds is not None else "n/a"
    print(f"clips={score.total_clips}  TP={score.true_positives}  FP={score.false_positives}  FN={score.false_negatives}")
    print(f"precision={score.precision}  recall={score.recall}  F1={score.f1}")
    print(f"mean_clip_duration={mean_dur}  max_clip_duration={score.max_clip_duration_seconds:.1f}s  "
          f"total_duration={score.total_duration_seconds:.1f}s ({pct:.1f}% of game)")


def cmd_golden_score(cfg: Config, golden_path: str, use_vision: bool) -> None:
    """Score the current config's detection.strategy against a pre-built
    golden event set (see golden.py / testdata/golden_events.json) --
    audio-only, no clip rendering or human review needed. Useful for
    quickly re-checking a tuning change against known ground truth.

    With --vision, also runs the Phase 2 vision refinement pass (see
    vision.py) over the audio-only intervals and prints its score
    alongside the audio-only one, so the delta is directly visible --
    this is the go/no-go check for whether vision actually helps, not
    just a different number to eyeball on its own."""
    chunks = discover_chunks(cfg.input.source_dir)
    samples_by_chunk = []
    for chunk in chunks:
        source_path = _audio_source_path(chunk, cfg)
        print(f"Decoding chunk #{chunk.sequence}: {source_path.name}")
        samples = extract_audio_samples(source_path, cfg.audio.sample_rate, cfg.audio.mono)
        samples_by_chunk.append((chunk, samples))

    golden_events = load_golden_events(Path(golden_path))
    intervals = detect_intervals(samples_by_chunk, cfg.audio.sample_rate, cfg.detection.strategy, cfg.detection, cfg.timeline)
    game_duration = sum(c.duration_seconds for c in chunks)
    audio_score = score_intervals_against_golden(intervals, golden_events)

    print(f"\nstrategy={cfg.detection.strategy}  golden_events={len(golden_events)}")
    print("--- audio-only ---")
    _print_golden_score(audio_score, game_duration)

    if not use_vision:
        return

    if not os.environ.get(cfg.vision.api_key_env):
        raise SystemExit(
            f"--vision requires {cfg.vision.api_key_env} to be set (see README's Vision AI section for setup)."
        )

    print("\nRunning vision refinement pass (calls the Claude API for every candidate window and gap)...")
    refined, vision_log = vision.refine_with_vision(intervals, chunks, game_duration, cfg.timeline, cfg.vision)
    vision_score = score_intervals_against_golden(refined, golden_events)

    print("\n--- vision-refined ---")
    _print_golden_score(vision_score, game_duration)

    vision_log_path = Path(cfg.metadata.events_path).parent / "vision_events.json"
    vision.save_vision_log(vision_log, vision_log_path)
    print(f"\nWrote vision verdict log to {vision_log_path}")


def cmd_vision_highlights(cfg: Config, full_quality: bool) -> None:
    """Detect, then classify+caption every candidate interval with
    peak-anchored vision frames (see vision.classify_confirm), and render
    only the survivors as clips named "{seconds} - caption.mp4". No
    negative-space scan pass here -- Round 3/4 already showed the audio
    net alone gets 1.0 must-catch recall on this recording; the actual
    gap vision needs to close is precision, not more recall, so this
    command spends its whole API budget on the confirm+caption pass over
    a (typically loosened) audio-only candidate set instead. Pruned
    candidates get a clip in a pruned/ subfolder for audit rather than
    being deleted outright.

    Measured against testdata/golden_events.json (2026-07-25, 40
    candidates from a loosened onset_flux pass): this only marginally
    beats not running vision at all (F1 0.377 audio-only -> 0.409 at the
    default drop_confidence_threshold=0.75, at the cost of one more
    missed real event). The model's confidence doesn't cleanly separate
    true from false positives from a handful of still frames -- don't
    assume this is a solved problem; see README's Vision AI section.

    Defaults to cheap review-quality (cfg.review) renders for both kept
    and pruned clips -- fast, low-res, safe to run alongside something
    else on this hardware. Pass --full-quality once you trust the
    surviving timestamps and want the real cfg.export (4K/CRF18)
    delivery encode instead; that's slow and CPU-heavy, so don't run it
    at the same time as another render job.

    The verdict log is written after EVERY interval (not just at the
    end) so an interrupted run -- e.g. killed to free up the CPU for
    something else -- still leaves a usable partial vision_events.json
    and whatever clips finished, instead of losing everything."""
    if not os.environ.get(cfg.vision.api_key_env):
        raise SystemExit(f"vision-highlights requires {cfg.vision.api_key_env} to be set.")

    chunks = discover_chunks(cfg.input.source_dir)
    merged, _traces = _run_detection(cfg, chunks)

    out_dir = Path(cfg.vision.highlights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pruned_dir = out_dir / "pruned"
    vision_log_path = out_dir / "vision_events.json"

    log = vision.VisionRunLog()
    kept_count = 0
    with tempfile.TemporaryDirectory(prefix="soccer_hl_vision_highlights_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for i, interval in enumerate(merged):
            print(f"\nClassifying candidate {i + 1}/{len(merged)}: [{interval.start_seconds:.1f}, {interval.end_seconds:.1f}]")
            verdict = vision.classify_confirm(interval, chunks, cfg.vision)
            keep = vision.decide_confirm(interval, verdict, cfg.vision)
            log.entries.append(vision.VisionLogEntry("confirm", interval.start_seconds, interval.end_seconds, verdict, keep))
            vision.save_vision_log(log, vision_log_path)  # incremental -- survives an interrupted run

            caption = verdict.caption if verdict and verdict.caption else "highlight"
            safe_caption = vision.sanitize_caption_for_filename(caption)
            slices = map_interval_to_chunks(interval, chunks)
            clip_name = f"{interval.start_seconds:.0f} - {safe_caption}.mp4"

            if keep:
                clip_path = out_dir / clip_name
                print(f"  KEEP  -> {clip_path.name}  ({verdict.rationale if verdict else 'no verdict, kept by default'})")
                if full_quality:
                    render.render_export_clip(slices, clip_path, tmp_dir, cfg.export)
                else:
                    render.render_review_clip(slices, clip_path, tmp_dir, cfg.review)
                kept_count += 1
            else:
                pruned_dir.mkdir(parents=True, exist_ok=True)
                clip_path = pruned_dir / clip_name
                print(f"  PRUNE -> {clip_path.name}  ({verdict.rationale if verdict else ''})")
                render.render_review_clip(slices, clip_path, tmp_dir, cfg.review)

    print(f"\nKept {kept_count}/{len(merged)} candidate(s) in {out_dir}. Wrote log to {vision_log_path}")


def cmd_vision_compare(cfg: Config, provider: str, tag: str, golden_path: str) -> None:
    """Classify every candidate interval via the chosen provider's
    classify_confirm, caching verdicts incrementally to
    output/vision_compare/<tag>.json (safe to interrupt and rerun -- only
    uncached intervals get re-classified), then sweep
    drop_confidence_threshold offline against the golden set. Pure
    measurement like golden-score -- no clip rendering, so this is the
    cheap way to compare providers/prompts/frame-density settings on the
    exact same candidate set."""
    if provider == "claude":
        if not os.environ.get(cfg.vision.api_key_env):
            raise SystemExit(f"vision-compare --provider claude requires {cfg.vision.api_key_env} to be set.")
        classify_fn = lambda interval, chunks: vision.classify_confirm(interval, chunks, cfg.vision)  # noqa: E731
    elif provider == "gemini":
        if not os.environ.get(cfg.gemini.api_key_env):
            raise SystemExit(f"vision-compare --provider gemini requires {cfg.gemini.api_key_env} to be set.")
        classify_fn = lambda interval, chunks: vision_gemini.classify_confirm(interval, chunks, cfg.gemini)  # noqa: E731
    else:
        raise SystemExit(f"Unknown provider: {provider!r}")

    chunks = discover_chunks(cfg.input.source_dir)
    merged, _traces = _run_detection(cfg, chunks)
    golden_events = load_golden_events(Path(golden_path))

    cache_path = Path("output/vision_compare") / f"{tag}.json"
    print(f"Classifying {len(merged)} candidate(s) via {provider} (cache: {cache_path})...")
    verdicts = vision_eval.collect_verdicts(merged, chunks, classify_fn, cache_path)

    thresholds = [1.01, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    points = vision_eval.sweep_drop_threshold(merged, verdicts, golden_events, thresholds)
    print(f"\n=== {tag} ({provider}) vs golden set ({len(golden_events)} events, {len(merged)} candidates) ===")
    print(vision_eval.format_sweep_table(points))


def cmd_label_audit(cfg: Config, limit: int | None) -> None:
    """Audit the existing human-labeled Round 2 dataset (output/review/*)
    against a fresh AI read of the same clips -- see label_audit.py's
    module docstring for why (three rounds of detection-prompt tuning all
    failed to beat audio alone against a golden set derived from these
    same labels, raising the question of whether the labels themselves
    hold up). Gemini describes each clip fresh (reading the same clip file
    the human labeled -- no separate re-render), Claude judges agreement
    with the original verdict/notes, flagged (disagreeing) rows are copied
    to flagged_clips/ for human review, and every row -- flagged or not --
    lands in the final CSV with blank new_label/new_notes columns to fill
    in."""
    if not os.environ.get(cfg.gemini.api_key_env):
        raise SystemExit(f"label-audit requires {cfg.gemini.api_key_env} to be set (for Gemini descriptions).")
    if not os.environ.get(cfg.vision.api_key_env):
        raise SystemExit(f"label-audit requires {cfg.vision.api_key_env} to be set (for the Claude judge).")

    rows = label_audit.load_review_rows(Path(cfg.label_audit.review_root))
    if limit is not None:
        rows = rows[:limit]
    print(f"Loaded {len(rows)} labeled row(s) from {cfg.label_audit.review_root}")

    out_dir = Path(cfg.label_audit.output_dir)
    cache_path = out_dir / "audit_cache.json"
    results = label_audit.run_audit(rows, cfg.gemini, cfg.vision, cache_path)

    flagged = [ar for ar in results if label_audit.is_flagged(ar.judge, cfg.label_audit.flag_distance_threshold)]
    flagged = label_audit.sort_by_disagreement(flagged)
    print(f"\n{len(flagged)}/{len(results)} row(s) flagged for review (threshold={cfg.label_audit.flag_distance_threshold})")

    clips_dir = out_dir / "flagged_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for i, ar in enumerate(flagged):
        row = ar.row
        score = ar.judge.distance_score if ar.judge else 1.0
        agreement = ar.judge.agreement if ar.judge else "no_verdict"
        clip_name = f"{i + 1:03d} - dist{score:.2f} - {agreement} - {row.strategy}_{Path(row.clip_file).stem}.mp4"
        clip_path = clips_dir / clip_name
        print(f"  Copying {clip_path.name}")
        shutil.copy2(row.clip_path, clip_path)

    report_path = out_dir / "label_audit_report.csv"
    label_audit.write_report_csv(results, report_path)
    print(f"\nWrote {len(results)}-row report to {report_path}")
    print(f"Wrote {len(flagged)} flagged clip(s) to {clips_dir}")


def cmd_pre_label(cfg: Config, out_dir: str, lrf_cache_dir: str | None = None, fps: int | None = None) -> None:
    """Detect candidates in a brand-new (not-yet-labeled) recording,
    render small/fast clips, generate a Gemini description for each (no
    judge step -- there's no prior human label yet to compare against),
    and produce a fillable review_sheet.csv + events.json in the same
    shape batch-review's per-strategy folders use (clip_file/start/end/
    duration/max_peak_score/verdict/notes, verdict+notes left blank),
    plus one bonus gemini_description column. Staying in that shape
    keeps this compatible with score/golden.build_golden_events later if
    a golden set ever gets built from it -- see label_audit.py for the
    describe-only pass this reuses.

    With --lrf-cache-dir: chunk discovery/duration-probing still reads
    source_dir's .MP4 files (a small, fast metadata-only ffprobe read),
    but every chunk's .lrf_path is redirected to a same-named file in
    lrf_cache_dir if one exists there -- so the actual heavy reads
    (audio decode, clip/frame extraction, all of which already prefer
    .LRF over the full-res source) hit a local copy instead of
    source_dir. Built for source_dir on an unreliable network/cloud
    drive (observed in practice: ffmpeg failing with 3221225794 /
    0xC0000006 STATUS_IN_PAGE_ERROR reading a freshly-uploaded Google
    Drive file) -- copy just the much-smaller .LRF proxies locally
    (e.g. via `robocopy source_dir lrf_cache_dir *.LRF /R:5 /W:15`,
    which has its own retry logic) rather than the full-res source."""
    if not os.environ.get(cfg.gemini.api_key_env):
        raise SystemExit(f"pre-label requires {cfg.gemini.api_key_env} to be set (for Gemini descriptions).")

    chunks = discover_chunks(cfg.input.source_dir)
    if lrf_cache_dir:
        cache_dir = Path(lrf_cache_dir)
        for chunk in chunks:
            if chunk.lrf_path is not None:
                local_lrf = cache_dir / chunk.lrf_path.name
                if local_lrf.exists():
                    chunk.lrf_path = local_lrf
                else:
                    print(f"WARNING: no local LRF cache for {chunk.lrf_path.name} in {cache_dir}, using {chunk.lrf_path}")
    merged, _traces = _run_detection(cfg, chunks)

    strategy_dir = Path(out_dir) / "candidates"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    write_events_json(merged, strategy_dir / "events.json")

    # Shared spec (cfg.review) -- the same clip a human labels here is the
    # exact file Gemini scores and, later, label-audit re-checks against.
    # No bespoke lower-res tier: benchmarked at ~0.33x realtime on this
    # laptop (2026-07-28), so a full candidate batch renders in minutes,
    # not the hours that would justify a cheaper/lossier one-off tier.
    with tempfile.TemporaryDirectory(prefix="soccer_hl_pre_label_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for i, interval in enumerate(merged, start=1):
            clip_path = strategy_dir / f"clip_{i:03d}.mp4"
            print(f"Rendering {i}/{len(merged)}: {clip_path.name}")
            slices = map_interval_to_chunks(interval, chunks)
            render.render_review_clip(slices, clip_path, tmp_dir, cfg.review)

    rows = [
        label_audit.LabeledRow(
            strategy="candidates",
            clip_file=f"clip_{i:03d}.mp4",
            clip_path=strategy_dir / f"clip_{i:03d}.mp4",
            interval=interval,
            verdict="",
            notes="",
        )
        for i, interval in enumerate(merged, start=1)
    ]
    cache_path = strategy_dir / "descriptions_cache.json"
    describe_fn = functools.partial(label_audit.generate_description, fps=fps) if fps is not None else None
    fps_note = f" at fps={fps} (override)" if fps is not None else ""
    print(f"\nGenerating {len(rows)} Gemini description(s){fps_note} (cache: {cache_path})...")
    descriptions = label_audit.run_describe_only(rows, cfg.gemini, cache_path, describe_fn=describe_fn)

    sheet_path = generate_review_sheet(strategy_dir)
    with open(sheet_path, encoding="utf-8") as f:
        sheet_rows = list(csv.DictReader(f))
    fieldnames = list(sheet_rows[0].keys()) + ["gemini_score", "gemini_caption", "gemini_description"] if sheet_rows else []
    for sheet_row, description in zip(sheet_rows, descriptions):
        sheet_row["gemini_score"] = description.score if description else ""
        sheet_row["gemini_caption"] = description.caption if description else ""
        sheet_row["gemini_description"] = description.description if description else ""
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sheet_rows)

    print(f"\nWrote {len(merged)} candidate clip(s) + {sheet_path} to {strategy_dir}")


_CANONICAL_CLIP_RE = re.compile(r"c(?:lip_)?(\d{3})")


def _canonical_clip_file(name: str) -> str:
    """Recover the canonical ``clip_NNN.mp4`` id from either a canonical name
    or a decorated one (``r01_s4_746a_c002.mp4``). Every decorated name keeps
    a ``c0NN`` token precisely so this is always possible -- that token is
    what makes the rename reversible and keeps the position-keyed describe
    cache, the review sheet, export-picks and telegram-post all resolving to
    one stable identity no matter which naming a given folder is currently in."""
    match = _CANONICAL_CLIP_RE.search(Path(name).stem)
    if match is None:
        raise ValueError(f"Cannot recover a canonical clip id from {name!r}")
    return f"clip_{match.group(1)}.mp4"


def _wall_clock_tag(global_seconds: float, chunks: list[Chunk]) -> str:
    """Time-of-day label (e.g. ``746a``) for a global-timeline offset.

    discover_chunks builds global_start_seconds by accumulating durations
    alone -- an explicit continuous-recording assumption -- so elapsed media
    time drifts from real time by however long the camera was stopped between
    chunks. On the 2026-08-23 game that was 41.5 min of unrecorded time across
    a 110.3 min window (68.8 min of footage), making late clips read ~4.5 min
    early. Re-anchoring through the owning chunk's own filename timestamp
    removes that drift entirely, so the label always matches what a human
    remembers about when something happened."""
    for chunk in chunks:
        if chunk.global_start_seconds <= global_seconds < chunk.global_start_seconds + chunk.duration_seconds:
            stamp = chunk.start_time + timedelta(seconds=global_seconds - chunk.global_start_seconds)
            break
    else:
        last = chunks[-1]
        stamp = last.start_time + timedelta(seconds=last.duration_seconds)
    return stamp.strftime("%I%M%p").lower().replace("am", "a").replace("pm", "p").lstrip("0")


def cmd_name_candidates(cfg: Config, candidates_dir: str, revert: bool = False) -> None:
    """Rename pre-label candidate clips to ``r01_s4_746a_c002.mp4`` -- review
    rank, Gemini score, wall-clock time, canonical id -- so the folder's
    default A-Z sort *is* the review order (best first, chronological within a
    score band) instead of burying the interesting clips among the 2s.

    Renames in place and rewrites ``clip_file`` in both review_sheet.csv and
    descriptions_cache.json in the same pass, so the cache's position-keyed
    consistency check still passes and no Gemini call gets re-paid for. Run
    this only once pre-label has fully finished: rank depends on the score, so
    renaming while describe retries are still filling in nulls would produce a
    ranking that's wrong the moment the next retry lands (hence the
    all-rows-scored guard below).

    ``--revert`` restores canonical ``clip_NNN.mp4`` names, which is what to
    run before re-running pre-label over an already-renamed folder -- pre-label
    re-renders to canonical names and would otherwise leave both namings side
    by side and trip the cache mismatch check."""
    candidates_path = Path(candidates_dir)
    sheet_path = candidates_path / "review_sheet.csv"
    cache_path = candidates_path / "descriptions_cache.json"
    if not sheet_path.exists():
        raise SystemExit(f"No review_sheet.csv in {candidates_path}")

    with open(sheet_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    if revert:
        new_names = {row["clip_file"]: _canonical_clip_file(row["clip_file"]) for row in rows}
    else:
        unscored = [row["clip_file"] for row in rows if not row.get("gemini_score")]
        if unscored:
            raise SystemExit(
                f"{len(unscored)} row(s) still have no gemini_score ({', '.join(unscored[:5])}"
                f"{'...' if len(unscored) > 5 else ''}). Rank is derived from the score, so finish "
                "the pre-label describe pass first (re-run it -- the cache only retries failures)."
            )
        chunks = discover_chunks(cfg.input.source_dir)
        ranked = sorted(rows, key=lambda r: (-int(r["gemini_score"]), float(r["start_seconds"])))
        new_names = {}
        for rank, row in enumerate(ranked, start=1):
            clock = _wall_clock_tag(float(row["start_seconds"]), chunks)
            canonical = _canonical_clip_file(row["clip_file"])
            new_names[row["clip_file"]] = (
                f"r{rank:02d}_s{int(row['gemini_score'])}_{clock}_c{canonical[5:8]}.mp4"
            )

    # Two-phase rename: a new name can collide with some *other* clip's current
    # name (re-running after scores changed reshuffles ranks), so park
    # everything under a temp name first rather than clobbering mid-pass.
    staged: list[tuple[Path, Path]] = []
    for old_name, new_name in new_names.items():
        if old_name == new_name:
            continue
        old_path = candidates_path / old_name
        if not old_path.exists():
            print(f"WARNING: {old_name} not found on disk, skipping rename")
            continue
        tmp_path = candidates_path / f".renaming_{new_name}"
        old_path.rename(tmp_path)
        staged.append((tmp_path, candidates_path / new_name))
    for tmp_path, final_path in staged:
        tmp_path.rename(final_path)

    for row in rows:
        row["clip_file"] = new_names.get(row["clip_file"], row["clip_file"])
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            entry["clip_file"] = new_names.get(entry["clip_file"], entry["clip_file"])
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    action = "Reverted" if revert else "Renamed"
    print(f"{action} {len(staged)} clip(s); updated {sheet_path.name}" + (" + descriptions_cache.json" if cache_path.exists() else ""))
    if not revert:
        for row in sorted(rows, key=lambda r: r["clip_file"])[:10]:
            print(f"  {row['clip_file']}")
        if len(rows) > 10:
            print(f"  ... {len(rows) - 10} more")


def cmd_ingest_marks(cfg: Config, marks_csv: str, out: str | None, final_score: str | None) -> None:
    chunks = discover_chunks(cfg.input.source_dir)
    raw_marks = marks.load_marks_csv(marks_csv)
    active_marks = marks.resolve_undos(raw_marks)
    undone = len(raw_marks) - len(active_marks)

    merged, _traces = _run_detection(cfg, chunks)
    audio_peaks = [p for interval in merged for p in interval.peaks]

    resolved = marks.resolve_marks(active_marks, chunks, audio_peaks, cfg.marks, cfg.timeline)

    n_peak = sum(1 for r in resolved if r.anchor == "audio_peak")
    n_fixed = sum(1 for r in resolved if r.anchor == "fixed_window")
    n_gap = sum(1 for r in resolved if r.anchor == "unrecorded_gap")
    print(f"{len(raw_marks)} row(s) in {marks_csv} -> {len(active_marks)} active mark(s) ({undone} undone)")
    print(f"  {n_peak} snapped to an existing audio peak (audio would likely have found these anyway)")
    print(f"  {n_fixed} had no nearby audio peak -- fixed-window fallback (this IS the recall gain)")
    if n_gap:
        print(f"  {n_gap} landed in an UNRECORDED GAP between chunks -- marked but not recorded, no clip possible")

    for r in resolved:
        local = r.mark.timestamp.astimezone(marks.RECORDING_TZ)
        if r.global_seconds is None:
            print(f"  seq={r.mark.sequence:>3} {r.mark.category:<10} {local:%H:%M:%S}  -> UNRECORDED GAP")
        else:
            print(
                f"  seq={r.mark.sequence:>3} {r.mark.category:<10} {local:%H:%M:%S}  -> global={r.global_seconds:8.1f}s "
                f"[{r.anchor:<12}] window=({r.interval.start_seconds:.1f}, {r.interval.end_seconds:.1f})"
            )

    white = sum(1 for r in active_marks if r.category == "white_goal")
    black = sum(1 for r in active_marks if r.category == "black_goal")
    print(f"\nWatch tally: white {white} - black {black}")
    if final_score:
        print(f"Confirm final score was {final_score} -- a mismatch means presses were missed or mis-tapped.")

    if out:
        out_path = Path(out)
        marks.write_ingest_report_csv(resolved, out_path)
        print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sunday Soccer Highlights Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to a config YAML file (default: config/default.yaml)")
    parser.add_argument("--source-dir", type=str, default=None, help="Override input.source_dir")
    parser.add_argument(
        "--strategy", type=str, default=None, choices=["rms_energy", "onset_flux", "combined"], help="Override detection.strategy"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("detect", help="Detect peaks only; writes events.json + a debug plot")
    subparsers.add_parser("render", help="Detect peaks and produce highlight clips/reel")
    subparsers.add_parser("batch-review", help="Run all named strategies, render small review clips + skim + negatives")
    review_sheet_parser = subparsers.add_parser(
        "review-sheet", help="(Re)generate fillable review_sheet.csv files from batch-review output"
    )
    review_sheet_parser.add_argument(
        "--prior-root",
        type=str,
        default=None,
        help="Path to a previous round's review_root (e.g. output/review_round1); "
        "when given, each new clip gets a guess/guess_basis column from time-overlap with those labels",
    )
    subparsers.add_parser("score", help="Compute precision/recall/F1 per strategy from filled-in review_sheet.csv files")
    export_parser = subparsers.add_parser(
        "export", help="Detect, then re-encode each highlight from the full-res source at export.* settings (for sharing)"
    )
    export_parser.add_argument(
        "--out-dir", type=str, default=None, help="Override export.dir (where the compressed highlight_NNN.mp4 files go)"
    )
    export_picks_parser = subparsers.add_parser(
        "export-picks",
        help="Re-encode a user-specified subset of an existing review_sheet.csv's rows (by clip_file) "
        "from the full-res source at export.* settings -- for exporting just the final hand-picked clips",
    )
    export_picks_parser.add_argument(
        "--review-sheet", required=True, help="Path to the review_sheet.csv containing the picked rows"
    )
    export_picks_parser.add_argument(
        "--clips", required=True, help="Comma-separated clip_file names to export, e.g. clip_055.mp4,clip_057.mp4"
    )
    export_picks_parser.add_argument(
        "--out-dir", type=str, default=None, help="Override export.dir (where the compressed clips go)"
    )
    export_picks_parser.add_argument(
        "--crf", type=int, default=None, help="Override export.crf (higher = smaller file/lower quality), e.g. for a Telegram-size-limited copy"
    )
    telegram_post_parser = subparsers.add_parser(
        "telegram-post",
        help="Post a user-specified subset of already-exported clips (by clip_file) to a Telegram group via sendVideo. "
        "Caption is the clip's Farsi gemini_caption from the review sheet. Tracks sent clips to avoid double-posting.",
    )
    telegram_post_parser.add_argument(
        "--review-sheet", required=True, help="Path to the review_sheet.csv containing the picked rows (for captions)"
    )
    telegram_post_parser.add_argument(
        "--clips-dir", required=True, help="Directory containing the already-exported clip files (e.g. export-picks' --out-dir)"
    )
    telegram_post_parser.add_argument(
        "--clips", required=True, help="Comma-separated clip_file names to post, e.g. clip_055.mp4,clip_057.mp4"
    )
    telegram_post_parser.add_argument(
        "--dry-run", action="store_true", help="Validate files/captions/credentials and print what would be sent, without posting"
    )
    telegram_message_parser = subparsers.add_parser(
        "telegram-message",
        help="Post a one-off plain-text announcement to the Telegram group (no video attachment) via sendMessage.",
    )
    telegram_message_parser.add_argument(
        "--text", default=None, help="Message text (prefer --text-file for non-ASCII/Farsi to avoid shell encoding issues)"
    )
    telegram_message_parser.add_argument(
        "--text-file", default=None, help="Path to a UTF-8 file containing the message text (takes priority over --text)"
    )
    telegram_message_parser.add_argument(
        "--dry-run", action="store_true", help="Validate credentials and print what would be sent, without posting"
    )
    golden_score_parser = subparsers.add_parser(
        "golden-score", help="Score the current --strategy/config against a pre-built golden event set (no rendering)"
    )
    golden_score_parser.add_argument(
        "--golden-events",
        type=str,
        default="testdata/golden_events.json",
        help="Path to a golden_events.json (default: testdata/golden_events.json)",
    )
    golden_score_parser.add_argument(
        "--vision",
        action="store_true",
        help="Also run the Phase 2 vision refinement pass and print its score alongside the audio-only one "
        "(calls the Claude API; requires vision.api_key_env, default ANTHROPIC_API_KEY, to be set)",
    )
    vision_highlights_parser = subparsers.add_parser(
        "vision-highlights",
        help="Detect, classify+caption each candidate with peak-anchored vision frames, "
        "render only the survivors as clips named '{seconds} - caption.mp4' "
        "(fast review-quality by default; --full-quality for the real 4K delivery encode)",
    )
    vision_highlights_parser.add_argument(
        "--full-quality",
        action="store_true",
        help="Render kept clips with cfg.export (4K/CRF18) instead of the fast cfg.review default. "
        "Slow and CPU-heavy -- don't run alongside another render job on this hardware.",
    )
    vision_compare_parser = subparsers.add_parser(
        "vision-compare",
        help="Classify every candidate via --provider {claude,gemini}, cache verdicts, sweep "
        "drop_confidence_threshold offline against the golden set -- pure measurement, no rendering",
    )
    vision_compare_parser.add_argument("--provider", choices=["claude", "gemini"], required=True)
    vision_compare_parser.add_argument(
        "--tag", required=True, help="Label for this run's verdict cache, e.g. 'claude-existing-prompt'"
    )
    vision_compare_parser.add_argument(
        "--golden-events", default="testdata/golden_events.json", help="Path to a golden_events.json"
    )
    label_audit_parser = subparsers.add_parser(
        "label-audit",
        help="Audit output/review/*'s existing human labels against a fresh Gemini description + Claude "
        "judge for each clip; render flagged (disagreeing) clips, write a full CSV report",
    )
    label_audit_parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N labeled rows (for a quick smoke test)"
    )
    pre_label_parser = subparsers.add_parser(
        "pre-label",
        help="Detect candidates in a brand-new recording, render small/fast clips, generate a Gemini "
        "description for each, and write a fillable review_sheet.csv (+events.json) for a first labeling pass",
    )
    pre_label_parser.add_argument(
        "--out-dir", required=True, help="Where to write events.json/clips/review_sheet.csv, e.g. a game's Tests folder"
    )
    pre_label_parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override Gemini describe-call fps sampling (default: generate_description's own default, currently 15 "
        "-- set 2026-08-30 by explicit request, not a re-swept value; 10 was the last sweep-validated default)",
    )
    pre_label_parser.add_argument(
        "--lrf-cache-dir",
        default=None,
        help="Redirect heavy .LRF reads to a local copy in this directory (source_dir's .MP4 files are still "
        "used for chunk discovery/duration, a small fast read) -- for an unreliable network/cloud source_dir",
    )

    name_candidates_parser = subparsers.add_parser(
        "name-candidates",
        help="Rename pre-label candidates to r<rank>_s<score>_<wallclock>_c<id>.mp4 so the folder's "
        "default sort is the review order; updates review_sheet.csv + descriptions_cache.json to match",
    )
    name_candidates_parser.add_argument(
        "--candidates-dir", required=True, help="The pre_label candidates/ folder to rename in place"
    )
    name_candidates_parser.add_argument(
        "--revert",
        action="store_true",
        help="Restore canonical clip_NNN.mp4 names (run this before re-running pre-label on a renamed folder)",
    )

    ingest_marks_parser = subparsers.add_parser(
        "ingest-marks",
        help="Report-only: map a wall-clock marks CSV onto the recording timeline, classify each mark "
        "as audio-peak-snapped/fixed-window/unrecorded-gap, print the white/black tally checksum",
    )
    ingest_marks_parser.add_argument(
        "--marks-csv", required=True, help="CSV with timestamp (ISO 8601 + UTC offset), category, sequence columns"
    )
    ingest_marks_parser.add_argument(
        "--out", default=None, help="Optional path to also write the resolved-marks report as a CSV"
    )
    ingest_marks_parser.add_argument(
        "--final-score",
        default=None,
        help="Known final score e.g. '6-4' (white-black), printed alongside the watch tally as a free correctness checksum",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.source_dir:
        cfg.input.source_dir = args.source_dir
    if args.strategy:
        cfg.detection.strategy = args.strategy

    if args.command == "detect":
        cmd_detect(cfg)
    elif args.command == "render":
        cmd_render(cfg)
    elif args.command == "batch-review":
        cmd_batch_review(cfg)
    elif args.command == "review-sheet":
        cmd_review_sheet(cfg, args.prior_root)
    elif args.command == "score":
        cmd_score(cfg)
    elif args.command == "export":
        cmd_export(cfg, args.out_dir)
    elif args.command == "export-picks":
        cmd_export_picks(cfg, args.review_sheet, [c.strip() for c in args.clips.split(",")], args.out_dir, args.crf)
    elif args.command == "telegram-post":
        cmd_telegram_post(
            cfg, args.review_sheet, args.clips_dir, [c.strip() for c in args.clips.split(",")], args.dry_run
        )
    elif args.command == "telegram-message":
        cmd_telegram_message(cfg, args.text, args.text_file, args.dry_run)
    elif args.command == "golden-score":
        cmd_golden_score(cfg, args.golden_events, args.vision)
    elif args.command == "vision-highlights":
        cmd_vision_highlights(cfg, args.full_quality)
    elif args.command == "vision-compare":
        cmd_vision_compare(cfg, args.provider, args.tag, args.golden_events)
    elif args.command == "label-audit":
        cmd_label_audit(cfg, args.limit)
    elif args.command == "pre-label":
        cmd_pre_label(cfg, args.out_dir, args.lrf_cache_dir, args.fps)
    elif args.command == "name-candidates":
        cmd_name_candidates(cfg, args.candidates_dir, args.revert)
    elif args.command == "ingest-marks":
        cmd_ingest_marks(cfg, args.marks_csv, args.out, args.final_score)


if __name__ == "__main__":
    main()
