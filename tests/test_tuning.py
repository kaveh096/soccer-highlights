import numpy as np

from soccer_highlights.config import DetectionConfig, RmsEnergyConfig, TimelineConfig
from soccer_highlights.discovery import Chunk
from soccer_highlights.tuning import detect_intervals


def _make_chunk(sequence: int, global_start_seconds: float, duration_seconds: float) -> Chunk:
    return Chunk(
        sequence=sequence,
        start_time=None,
        mp4_path=None,
        lrf_path=None,
        duration_seconds=duration_seconds,
        global_start_seconds=global_start_seconds,
    )


def _signal_with_bursts(sample_rate: int, duration_seconds: float, burst_times: list[float]) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    n = int(duration_seconds * sample_rate)
    samples = rng.normal(0.0, 0.01, n).astype(np.float32)
    burst_len = int(0.3 * sample_rate)
    for t in burst_times:
        start = int(t * sample_rate)
        samples[start : start + burst_len] += rng.normal(0.0, 0.5, min(burst_len, n - start)).astype(np.float32)
    return samples


def test_detect_intervals_reuses_cached_samples_across_two_chunks():
    sample_rate = 22050
    chunk0 = _make_chunk(1, 0.0, 60.0)
    chunk1 = _make_chunk(2, 60.0, 60.0)
    samples_by_chunk = [
        (chunk0, _signal_with_bursts(sample_rate, 60.0, [20.0])),
        (chunk1, _signal_with_bursts(sample_rate, 60.0, [10.0])),  # global time 70.0
    ]

    detection_cfg = DetectionConfig(
        strategy="rms_energy",
        rms_energy=RmsEnergyConfig(
            window_seconds=0.2, hop_seconds=0.05, baseline_window_seconds=20.0, threshold_sigma=3.0,
            min_absolute_dbfs=-60.0, min_score_dbfs=5.0,
        ),
    )
    timeline_cfg = TimelineConfig(lookback_seconds=5.0, post_peak_seconds=5.0, min_gap_seconds=2.0, min_interval_seconds=1.0, warmup_seconds=0.0)

    intervals = detect_intervals(samples_by_chunk, sample_rate, "rms_energy", detection_cfg, timeline_cfg)

    assert len(intervals) == 2
    starts = sorted(iv.start_seconds for iv in intervals)
    assert abs(starts[0] - 15.0) < 1.0
    assert abs(starts[1] - 65.0) < 1.0  # global offset applied correctly for chunk1's peak


def test_detect_intervals_respects_warmup():
    sample_rate = 22050
    chunk0 = _make_chunk(1, 0.0, 30.0)
    samples_by_chunk = [(chunk0, _signal_with_bursts(sample_rate, 30.0, [5.0, 20.0]))]

    detection_cfg = DetectionConfig(
        strategy="rms_energy",
        rms_energy=RmsEnergyConfig(
            window_seconds=0.2, hop_seconds=0.05, baseline_window_seconds=15.0, threshold_sigma=3.0,
            min_absolute_dbfs=-60.0, min_score_dbfs=5.0,
        ),
    )
    timeline_cfg = TimelineConfig(lookback_seconds=2.0, post_peak_seconds=2.0, min_gap_seconds=1.0, min_interval_seconds=1.0, warmup_seconds=10.0)

    intervals = detect_intervals(samples_by_chunk, sample_rate, "rms_energy", detection_cfg, timeline_cfg)

    assert len(intervals) == 1  # the burst at 5.0s is dropped by the 10s warm-up
    assert abs(intervals[0].start_seconds - 18.0) < 1.0
