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
  vision-compare - classify every candidate interval via --provider
                  {claude,gemini}'s classify_confirm, caching verdicts
                  incrementally (resumable), then sweep
                  drop_confidence_threshold offline against the golden
                  set. Pure measurement, no rendering -- for comparing
                  providers/prompts/frame-density settings on the exact
                  same candidate set. See README's Vision AI section.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from soccer_highlights import clipping, render, vision, vision_eval, vision_gemini
from soccer_highlights.audio import extract_audio_samples
from soccer_highlights.config import Config, load_config, load_strategy_configs
from soccer_highlights.detection import analyze
from soccer_highlights.discovery import Chunk, discover_chunks
from soccer_highlights.golden import GoldenScore, load_golden_events, score_intervals_against_golden
from soccer_highlights.metadata import ChunkTrace, plot_debug, write_events_json
from soccer_highlights.scoring import format_score_report, generate_all_review_sheets, score_all
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
        for i, interval in enumerate(merged):
            slices = map_interval_to_chunks(interval, chunks)
            clip_path = out_dir / f"highlight_{i + 1:03d}.mp4"
            print(f"Exporting {i + 1}/{len(merged)}: {clip_path.name} ({interval.end_seconds - interval.start_seconds:.1f}s)")
            render.render_export_clip(slices, clip_path, tmp_dir, export_cfg)

    print(f"\nExported {len(merged)} clip(s) to {out_dir}")


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
    elif args.command == "golden-score":
        cmd_golden_score(cfg, args.golden_events, args.vision)
    elif args.command == "vision-highlights":
        cmd_vision_highlights(cfg, args.full_quality)
    elif args.command == "vision-compare":
        cmd_vision_compare(cfg, args.provider, args.tag, args.golden_events)


if __name__ == "__main__":
    main()
