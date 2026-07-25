import numpy as np

from soccer_highlights.config import CombinedConfig, OnsetFluxConfig, RmsEnergyConfig
from soccer_highlights.detection import PeakEvent, _confirm_by_proximity, analyze_combined, analyze_rms_energy


def _signal_with_bursts(sample_rate: int, duration_seconds: float, burst_times: list[float]) -> np.ndarray:
    """Quiet background noise with short loud bursts injected at burst_times."""
    rng = np.random.default_rng(seed=0)
    n = int(duration_seconds * sample_rate)
    samples = rng.normal(0.0, 0.01, n).astype(np.float32)
    burst_len = int(0.3 * sample_rate)
    for t in burst_times:
        start = int(t * sample_rate)
        samples[start : start + burst_len] += rng.normal(0.0, 0.5, min(burst_len, n - start)).astype(np.float32)
    return samples


def test_detects_burst_near_expected_time():
    sample_rate = 22050
    burst_times = [10.0, 40.0, 70.0]
    samples = _signal_with_bursts(sample_rate, duration_seconds=90.0, burst_times=burst_times)

    cfg = RmsEnergyConfig(
        window_seconds=0.2,
        hop_seconds=0.05,
        baseline_window_seconds=20.0,
        threshold_sigma=3.0,
        min_absolute_dbfs=-60.0,
        min_score_dbfs=5.0,
    )
    trace = analyze_rms_energy(samples, sample_rate, cfg)

    assert len(trace.events) == len(burst_times)
    detected_times = sorted(e.time_seconds for e in trace.events)
    for expected, actual in zip(burst_times, detected_times):
        assert abs(expected - actual) < 0.5


def test_silence_produces_no_events():
    sample_rate = 22050
    rng = np.random.default_rng(seed=1)
    samples = rng.normal(0.0, 0.01, int(30 * sample_rate)).astype(np.float32)

    cfg = RmsEnergyConfig(
        window_seconds=0.2,
        hop_seconds=0.05,
        baseline_window_seconds=10.0,
        threshold_sigma=3.0,
        min_absolute_dbfs=-60.0,
        min_score_dbfs=5.0,
    )
    trace = analyze_rms_energy(samples, sample_rate, cfg)

    assert trace.events == []


def test_confirm_by_proximity_keeps_only_corroborated_flux_events():
    flux_events = [
        PeakEvent(time_seconds=10.0, score=1.0),  # rms fires 3s after -> kept
        PeakEvent(time_seconds=40.0, score=1.0),  # no nearby rms at all -> dropped
        PeakEvent(time_seconds=70.0, score=1.0),  # rms fires 1s before -> kept
    ]
    rms_times = np.array([13.0, 69.0])

    confirmed = _confirm_by_proximity(flux_events, rms_times, window_before_seconds=2.0, window_after_seconds=5.0)

    assert sorted(e.time_seconds for e in confirmed) == [10.0, 70.0]


def test_confirm_by_proximity_empty_rms_drops_everything():
    flux_events = [PeakEvent(time_seconds=10.0, score=1.0)]
    confirmed = _confirm_by_proximity(flux_events, np.array([]), window_before_seconds=2.0, window_after_seconds=5.0)
    assert confirmed == []


def test_analyze_combined_keeps_bursts_that_raise_both_signals():
    sample_rate = 22050
    # Each burst raises both rms and flux together (a real "shot" has both
    # a transient and a sustained loudness rise) so all three survive
    # corroboration; the drop-if-unconfirmed behavior itself is covered by
    # test_confirm_by_proximity_keeps_only_corroborated_flux_events above.
    samples = _signal_with_bursts(sample_rate, duration_seconds=90.0, burst_times=[10.0, 40.0, 70.0])

    rms_cfg = RmsEnergyConfig(
        window_seconds=0.2, hop_seconds=0.05, baseline_window_seconds=20.0, threshold_sigma=3.0,
        min_absolute_dbfs=-60.0, min_score_dbfs=5.0,
    )
    flux_cfg = OnsetFluxConfig(
        window_seconds=0.05, hop_seconds=0.01, baseline_window_seconds=20.0, threshold_sigma=2.0, min_score=1.0,
    )
    combined_cfg = CombinedConfig(window_before_seconds=2.0, window_after_seconds=5.0)

    trace = analyze_combined(samples, sample_rate, rms_cfg, flux_cfg, combined_cfg)

    assert len(trace.events) == 3
    for expected, actual in zip([10.0, 40.0, 70.0], sorted(e.time_seconds for e in trace.events)):
        assert abs(expected - actual) < 1.0


def test_min_absolute_dbfs_floor_suppresses_quiet_local_maxima():
    sample_rate = 22050
    rng = np.random.default_rng(seed=2)
    samples = rng.normal(0.0, 0.0005, int(30 * sample_rate)).astype(np.float32)

    cfg = RmsEnergyConfig(
        window_seconds=0.2,
        hop_seconds=0.05,
        baseline_window_seconds=10.0,
        threshold_sigma=0.5,  # very permissive relative threshold
        min_absolute_dbfs=-20.0,  # but everything here is far below this floor
    )
    trace = analyze_rms_energy(samples, sample_rate, cfg)

    assert trace.events == []
