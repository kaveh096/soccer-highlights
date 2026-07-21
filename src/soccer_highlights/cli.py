"""CLI entry point.

Two subcommands:
  detect  - run audio peak detection only; writes events.json + a debug plot
            for visually tuning thresholds before trusting the pipeline.
  render  - run detection, then slice (or concat) the actual highlight clips
            from the full-resolution source files.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from soccer_highlights import clipping
from soccer_highlights.audio import extract_audio_samples
from soccer_highlights.config import Config, load_config
from soccer_highlights.detection import analyze
from soccer_highlights.discovery import Chunk, discover_chunks
from soccer_highlights.metadata import ChunkTrace, plot_debug, write_events_json
from soccer_highlights.timeline import GlobalPeak, Interval, map_interval_to_chunks, merge_intervals, peaks_to_raw_intervals


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

    raw_intervals = peaks_to_raw_intervals(all_peaks, cfg.timeline.lookback_seconds, cfg.timeline.post_peak_seconds)
    merged = merge_intervals(raw_intervals, cfg.timeline.min_gap_seconds, cfg.timeline.min_interval_seconds)
    print(f"Merged into {len(merged)} highlight interval(s) from {len(all_peaks)} raw peak(s)")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sunday Soccer Highlights Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to a config YAML file (default: config/default.yaml)")
    parser.add_argument("--source-dir", type=str, default=None, help="Override input.source_dir")
    parser.add_argument("--strategy", type=str, default=None, choices=["rms_energy", "onset_flux"], help="Override detection.strategy")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("detect", help="Detect peaks only; writes events.json + a debug plot")
    subparsers.add_parser("render", help="Detect peaks and produce highlight clips/reel")

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


if __name__ == "__main__":
    main()
