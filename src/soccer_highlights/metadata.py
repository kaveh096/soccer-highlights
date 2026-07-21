"""Event log (JSON) and debug plot output, for inspecting/tuning detection
before trusting it to produce final clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soccer_highlights.discovery import Chunk
from soccer_highlights.timeline import Interval, map_interval_to_chunks


def write_events_json(intervals: list[Interval], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "start_seconds": round(interval.start_seconds, 2),
            "end_seconds": round(interval.end_seconds, 2),
            "duration_seconds": round(interval.end_seconds - interval.start_seconds, 2),
            "peaks": [
                {"time_seconds": round(p.time_seconds, 2), "score": round(p.score, 4)}
                for p in interval.peaks
            ],
        }
        for interval in intervals
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@dataclass
class ChunkTrace:
    chunk: Chunk
    times_local: np.ndarray
    values: np.ndarray
    threshold: np.ndarray
    value_label: str


def plot_debug(traces: list[ChunkTrace], merged_intervals: list[Interval], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(traces), 1, figsize=(14, 3 * len(traces)), squeeze=False)

    for ax, trace in zip(axes[:, 0], traces):
        ax.plot(trace.times_local, trace.values, linewidth=0.6, label=trace.value_label)
        ax.plot(trace.times_local, trace.threshold, linewidth=0.6, linestyle="--", label="adaptive threshold")

        chunk_slices = [
            cs
            for interval in merged_intervals
            for cs in map_interval_to_chunks(interval, [trace.chunk])
        ]
        for cs in chunk_slices:
            ax.axvspan(cs.local_start_seconds, cs.local_end_seconds, color="orange", alpha=0.25)

        ax.set_title(f"Chunk #{trace.chunk.sequence} ({trace.chunk.mp4_path.name})")
        ax.set_xlabel("seconds into chunk")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
