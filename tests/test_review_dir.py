"""Folder-layout contract (2026-10-12): pre-label writes FLAT into --out-dir (the
game's Review\\ folder), and name-candidates takes --review-dir with the old
--candidates-dir spelling kept as an alias."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from soccer_highlights import cli


def test_review_dir_is_the_out_dir_itself(tmp_path: Path) -> None:
    out_dir = tmp_path / "2026-Oct-11" / "Review"
    assert not out_dir.exists()

    result = cli.review_dir_for(str(out_dir))

    assert result == out_dir
    assert out_dir.is_dir()
    # No nested candidates/ (or any other) subfolder is created.
    assert list(out_dir.iterdir()) == []


def test_review_dir_is_idempotent_and_keeps_existing_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "Review"
    out_dir.mkdir()
    (out_dir / "review_sheet.csv").write_text("clip_file\n", encoding="utf-8")

    assert cli.review_dir_for(out_dir) == out_dir
    assert (out_dir / "review_sheet.csv").read_text(encoding="utf-8") == "clip_file\n"


@pytest.mark.parametrize("flag", ["--review-dir", "--candidates-dir"])
def test_name_candidates_accepts_new_flag_and_deprecated_alias(
    flag: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_cmd(cfg: object, review_dir: str, revert: bool = False) -> None:
        seen["review_dir"] = review_dir
        seen["revert"] = revert

    monkeypatch.setattr(cli, "cmd_name_candidates", fake_cmd)
    monkeypatch.setattr(sys, "argv", ["soccer_highlights", "name-candidates", flag, str(tmp_path), "--revert"])

    cli.main()

    assert seen == {"review_dir": str(tmp_path), "revert": True}
