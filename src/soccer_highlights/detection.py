"""Pluggable audio peak-detection strategies.

Each strategy turns a mono sample array into a time/value/threshold trace
plus a list of ``PeakEvent`` (time + score), using an adaptive local
threshold: a rolling median baseline plus a multiple of the rolling MAD
(median absolute deviation, scaled to be a robust stand-in for standard
deviation). This keeps detection responsive to a game's own ambient noise
floor rather than requiring one fixed dB value re-tuned per session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from soccer_highlights.config import DetectionConfig, OnsetFluxConfig, RmsEnergyConfig

_MAD_TO_STD = 1.4826  # scale factor making MAD a consistent estimator of std for normal-ish data
_EPS = 1e-10


@dataclass
class PeakEvent:
    time_seconds: float
    score: float  # amount by which the value exceeded its local threshold


@dataclass
class DetectionTrace:
    times: np.ndarray
    values: np.ndarray
    threshold: np.ndarray
    value_label: str
    events: list[PeakEvent]


def _rolling_threshold(values: np.ndarray, frames_per_baseline_window: int, threshold_sigma: float) -> np.ndarray:
    size = max(3, frames_per_baseline_window | 1)  # odd size for median_filter
    baseline = median_filter(values, size=size, mode="nearest")
    residual = np.abs(values - baseline)
    mad = median_filter(residual, size=size, mode="nearest")
    return baseline + threshold_sigma * mad * _MAD_TO_STD


def _events_from_mask(times: np.ndarray, values: np.ndarray, threshold: np.ndarray, above: np.ndarray) -> list[PeakEvent]:
    """Collapse contiguous True runs in `above` into a single apex event each."""
    events: list[PeakEvent] = []
    in_run = False
    run_start = 0
    for i, flag in enumerate(above):
        if flag and not in_run:
            in_run = True
            run_start = i
        elif not flag and in_run:
            in_run = False
            _append_apex(events, times, values, threshold, run_start, i)
    if in_run:
        _append_apex(events, times, values, threshold, run_start, len(above))
    return events


def _append_apex(
    events: list[PeakEvent],
    times: np.ndarray,
    values: np.ndarray,
    threshold: np.ndarray,
    start: int,
    end: int,
) -> None:
    segment = values[start:end]
    apex_offset = int(np.argmax(segment))
    apex_index = start + apex_offset
    events.append(
        PeakEvent(
            time_seconds=float(times[apex_index]),
            score=float(values[apex_index] - threshold[apex_index]),
        )
    )


def analyze_rms_energy(samples: np.ndarray, sample_rate: int, cfg: RmsEnergyConfig) -> DetectionTrace:
    window = max(1, int(cfg.window_seconds * sample_rate))
    hop = max(1, int(cfg.hop_seconds * sample_rate))

    n_frames = max(0, (len(samples) - window) // hop + 1)
    if n_frames <= 0:
        empty = np.array([])
        return DetectionTrace(times=empty, values=empty, threshold=empty, value_label="RMS dBFS", events=[])

    frame_view = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, window),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(frame_view.astype(np.float64) ** 2, axis=1) + _EPS)
    dbfs = 20.0 * np.log10(rms + _EPS)
    times = (np.arange(n_frames) * hop) / sample_rate

    frames_per_baseline = int(cfg.baseline_window_seconds / cfg.hop_seconds)
    threshold = _rolling_threshold(dbfs, frames_per_baseline, cfg.threshold_sigma)

    above = (dbfs - threshold >= cfg.min_score_dbfs) & (dbfs > cfg.min_absolute_dbfs)
    events = _events_from_mask(times, dbfs, threshold, above)
    return DetectionTrace(times=times, values=dbfs, threshold=threshold, value_label="RMS dBFS", events=events)


def analyze_onset_flux(samples: np.ndarray, sample_rate: int, cfg: OnsetFluxConfig) -> DetectionTrace:
    nperseg = max(2, int(cfg.window_seconds * sample_rate))
    hop = max(1, int(cfg.hop_seconds * sample_rate))
    noverlap = max(0, nperseg - hop)

    _freqs, times, stft = signal.stft(samples, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(stft)  # shape: (freq_bins, n_frames)

    diff = np.diff(magnitude, axis=1, prepend=magnitude[:, :1])
    flux = np.sum(np.maximum(diff, 0.0), axis=0)

    frames_per_baseline = int(cfg.baseline_window_seconds / cfg.hop_seconds)
    threshold = _rolling_threshold(flux, frames_per_baseline, cfg.threshold_sigma)

    above = flux - threshold >= cfg.min_score
    events = _events_from_mask(times, flux, threshold, above)
    return DetectionTrace(times=times, values=flux, threshold=threshold, value_label="spectral flux", events=events)


def analyze(samples: np.ndarray, sample_rate: int, strategy: str, cfg: DetectionConfig) -> DetectionTrace:
    if strategy == "rms_energy":
        return analyze_rms_energy(samples, sample_rate, cfg.rms_energy)
    if strategy == "onset_flux":
        return analyze_onset_flux(samples, sample_rate, cfg.onset_flux)
    raise ValueError(f"Unknown detection strategy: {strategy!r}")
