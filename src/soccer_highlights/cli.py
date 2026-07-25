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
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from soccer_highlights import clipping, render
from soccer_highlights.audio import extract_audio_samples
from soccer_highlights.config import Config, load_config, load_strategy_configs
from soccer_highlights.detection import analyze
from soccer_highlights.discovery import Chunk, discover_chunks
from soccer_highlights.golden import load_golden_events, score_intervals_against_golden
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


def cmd_golden_score(cfg: Config, golden_path: str) -> None:
    """Score the current config's detection.strategy against a pre-built
    golden event set (see golden.py / testdata/golden_events.json) --
    audio-only, no clip rendering or human review needed. Useful for
    quickly re-checking a tuning change against known ground truth."""
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
    score = score_intervals_against_golden(intervals, golden_events)

    pct = 100 * score.total_duration_seconds / game_duration if game_duration else 0.0
    mean_dur = f"{score.mean_clip_duration_seconds:.1f}s" if score.mean_clip_duration_seconds is not None else "n/a"
    print(f"\nstrategy={cfg.detection.strategy}  golden_events={len(golden_events)}")
    print(f"clips={score.total_clips}  TP={score.true_positives}  FP={score.false_positives}  FN={score.false_negatives}")
    print(f"precision={score.precision}  recall={score.recall}  F1={score.f1}")
    print(f"mean_clip_duration={mean_dur}  max_clip_duration={score.max_clip_duration_seconds:.1f}s  "
          f"total_duration={score.total_duration_seconds:.1f}s ({pct:.1f}% of game)")


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
        cmd_golden_score(cfg, args.golden_events)


if __name__ == "__main__":
    main()
