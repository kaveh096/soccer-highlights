"""A hung ffmpeg must be killed, not waited on (2026-10-13: one segment hung
~41h and stalled the export batch)."""

from __future__ import annotations

import subprocess

import pytest

from soccer_highlights import render


def test_timeout_scales_with_duration_and_has_a_floor() -> None:
    assert render.encode_timeout_seconds(4.46) == 900.0
    assert render.encode_timeout_seconds(11.0) == 1320.0
    assert render.encode_timeout_seconds(30.0) == 3600.0


def test_run_ffmpeg_passes_timeout_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    render._run_ffmpeg(["-i", "x"], timeout=12.5)
    assert seen["timeout"] == 12.5
    assert seen["check"] is True


def test_run_ffmpeg_surfaces_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        render._run_ffmpeg(["-i", "x"], timeout=1.0)
