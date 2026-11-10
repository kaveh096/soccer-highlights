"""--silent maps to Telegram's disable_notification and is off by default."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from soccer_highlights import telegram
from soccer_highlights.config import TelegramConfig


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> TelegramConfig:
    cfg = TelegramConfig()
    monkeypatch.setenv(cfg.bot_token_env, "token")
    monkeypatch.setenv(cfg.chat_id_env, "-100123")
    return cfg


def test_send_video_silent_sets_disable_notification(
    env: TelegramConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sent: list[dict[str, str]] = []
    monkeypatch.setattr(telegram, "_post", lambda method, cfg, fields, ff, fp: sent.append(fields) or {})
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    telegram.send_video(clip, "cap", env)
    telegram.send_video(clip, "cap", env, silent=True)
    assert "disable_notification" not in sent[0]
    assert sent[1]["disable_notification"] == "true"


def test_send_message_silent_sets_disable_notification(env: TelegramConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict] = []
    monkeypatch.setattr(
        telegram, "_send_with_retry", lambda request, cfg, method: bodies.append(json.loads(request.data)) or {}
    )
    telegram.send_message("hi", env)
    telegram.send_message("hi", env, silent=True)
    assert "disable_notification" not in bodies[0]
    assert bodies[1]["disable_notification"] is True
