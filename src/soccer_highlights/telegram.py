"""Posts final, hand-picked export clips to a Telegram group via the Bot
API's sendVideo endpoint -- a direct HTTP call (stdlib urllib + a
hand-built multipart/form-data body), no SDK, matching this project's
existing pattern of calling providers' REST APIs directly (see
vision_gemini.py's _call_gemini for the same style against Gemini).

Bot-uploaded files are hard-capped at 50MB by Telegram itself, regardless
of upload method -- checked client-side (TelegramConfig.max_file_size_mb)
before attempting an upload, so an oversized file fails fast with a clear
message instead of a slow upload followed by a confusing HTTP error.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from soccer_highlights.config import TelegramConfig

_API_URL = "https://api.telegram.org/bot{token}/{method}"


class TelegramError(RuntimeError):
    pass


def _multipart_body(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8")
    )
    parts.append(file_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _post(method: str, cfg: TelegramConfig, fields: dict[str, str], file_field: str, file_path: Path) -> dict:
    token = os.environ.get(cfg.bot_token_env)
    if not token:
        raise TelegramError(f"{cfg.bot_token_env} is not set")
    url = _API_URL.format(token=token, method=method)
    body, content_type = _multipart_body(fields, file_field, file_path)
    request = urllib.request.Request(url, data=body, headers={"Content-Type": content_type}, method="POST")

    last_exc: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=cfg.request_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            if not data.get("ok"):
                raise TelegramError(f"Telegram API returned ok=false: {data}")
            return data
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_exc = TelegramError(f"HTTP {exc.code}: {body}")
            if 400 <= exc.code < 500:
                # a client error (bad request, chat not found, etc.) won't fix
                # itself on retry -- fail fast instead of re-uploading the
                # video 2 more times for nothing.
                break
            if attempt < cfg.max_retries:
                time.sleep(2**attempt)
        except (urllib.error.URLError, TelegramError) as exc:
            last_exc = exc
            if attempt < cfg.max_retries:
                time.sleep(2**attempt)
    raise TelegramError(f"sendVideo failed: {last_exc}") from last_exc


def get_me(cfg: TelegramConfig) -> dict:
    """Read-only connectivity/credential check -- confirms the bot token is
    valid and returns the bot's own identity, without sending anything."""
    token = os.environ.get(cfg.bot_token_env)
    if not token:
        raise TelegramError(f"{cfg.bot_token_env} is not set")
    url = _API_URL.format(token=token, method="getMe")
    with urllib.request.urlopen(url, timeout=cfg.request_timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not data.get("ok"):
        raise TelegramError(f"getMe returned ok=false: {data}")
    return data["result"]


def send_video(video_path: Path, caption: str, cfg: TelegramConfig) -> dict:
    chat_id = os.environ.get(cfg.chat_id_env)
    if not chat_id:
        raise TelegramError(f"{cfg.chat_id_env} is not set")

    size_mb = video_path.stat().st_size / (1024 * 1024)
    if size_mb > cfg.max_file_size_mb:
        raise TelegramError(
            f"{video_path.name} is {size_mb:.1f}MB, over Telegram's {cfg.max_file_size_mb:.0f}MB bot-upload limit"
        )

    fields = {"chat_id": chat_id, "caption": caption, "supports_streaming": "true"}
    return _post("sendVideo", cfg, fields, "video", video_path)
