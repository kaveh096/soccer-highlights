"""Render small, resized/downsampled review clips -- plus a whole-game skim
and negative-space (uncovered) clips -- from the .LRF proxies only. Meant
for fast human true/false-positive review on modest hardware, not final
output quality."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from soccer_highlights.clipping import concat_clips
from soccer_highlights.config import ExportConfig, ReviewConfig
from soccer_highlights.discovery import Chunk, slice_start_epoch
from soccer_highlights.timeline import ChunkSlice


def _scale_filter(max_width: int) -> str:
    # -2 keeps height a multiple of 2 (required by libx264) while preserving
    # the source's aspect ratio, instead of forcing a fixed WxH that would
    # distort a 16:9 source.
    return f"scale={max_width}:-2"


def _escape_filter_value(value: str) -> str:
    """Quote a value (a font path, here) for use inside an ffmpeg filter-option
    string, where ':' separates options.

    Both halves are required on ffmpeg 8.1.2 (all three alternatives were tried
    against the installed build on 2026-08-31): single-quoting alone leaves the
    drive-letter colon acting as an option separator (`No option name near
    '/Windows/Fonts/arial.ttf...'`), and backslash-escaping alone fails the same
    way. Only quoted AND escaped parses. Backslashes are normalised to forward
    slashes first -- Windows accepts them, and a literal '\\' is itself the
    filtergraph escape character."""
    return "'" + value.replace("\\", "/").replace(":", "\\:") + "'"


def _drawtext_filter(cfg: ExportConfig, epoch_base: float) -> str:
    """A drawtext stage burning the real time of day into every frame.

    `%{pts\\:gmtime\\:EPOCH\\:FORMAT}` adds the frame's own presentation
    timestamp (which restarts at ~0 for each `-ss`-trimmed piece) to a Unix
    epoch, so the clock advances with the footage instead of showing the time
    the render happened. `gmtime`, not `localtime`, deliberately: the epoch it
    formats is a display value that already carries the recording's own local
    wall clock (see discovery.slice_start_epoch, which is where that subtlety
    is explained), so the burned-in time can't drift with the render machine's
    OS timezone setting.

    Font is passed as an explicit file path, never as a `font=<family>` name --
    this ffmpeg build has fontconfig compiled in but no fontconfig config file
    on Windows, so resolving a family name SEGFAULTS the whole process
    (reproduced 2026-08-31 with `font=Arial`)."""
    if ":" in cfg.burn_in_time_format:
        raise ValueError(
            f"burn_in_time_format must not contain ':' (got {cfg.burn_in_time_format!r}) -- "
            "drawtext's %{pts:gmtime:...} expansion splits its own arguments on colons with a "
            "hard 3-argument cap, so a colon here fails the render with "
            "'%{pts} requires at most 3 arguments'. Use '.' or '-' separators."
        )
    text = f"%{{pts\\:gmtime\\:{epoch_base:.3f}\\:{cfg.burn_in_time_format}}}"
    return (
        f"drawtext=fontfile={_escape_filter_value(cfg.burn_in_font_path)}"
        f":text='{text}'"
        f":x={cfg.burn_in_margin_px}:y={cfg.burn_in_margin_px}"
        f":fontsize={cfg.burn_in_font_size}:fontcolor=white"
        # Semi-opaque box: outdoor footage swings from dark grass to bright
        # sky, and plain white text disappears against the bright end.
        f":box=1:boxcolor=black@0.5:boxborderw={cfg.burn_in_margin_px // 3}"
    )


@dataclass
class ScoreOverlay:
    """A running goal counter to burn into one clip.

    `flip_seconds` is an offset into the WHOLE clip, not into whichever
    piece is currently being encoded -- `_encode_piece` rebases it. None
    means the score is static for the clip's whole length."""

    white: int  # score at the clip's start, i.e. before `flip_team` scores
    black: int
    flip_seconds: float | None = None
    flip_team: str = ""  # "white" / "black"

    @classmethod
    def from_sheet_row(cls, row: dict[str, str]) -> ScoreOverlay | None:
        """Build from a review/post sheet row, or None if it carries no score.

        Blank or absent columns are the normal case, not an error: a game
        recorded without the watch has no marks, so `pre-label` leaves these
        empty and no counter is drawn. Reading the score from the sheet
        rather than re-deriving it at export time is deliberate -- it means
        the numbers can be eyeballed, and corrected by hand, before a
        ~37x-realtime render commits them to a file."""
        if not row.get("score_white") or not row.get("score_black"):
            return None
        flip_raw = (row.get("score_flip_seconds") or "").strip()
        return cls(
            white=int(row["score_white"]),
            black=int(row["score_black"]),
            flip_seconds=float(flip_raw) if flip_raw else None,
            flip_team=(row.get("score_flip_team") or "").strip(),
        )

    def text_before(self, cfg: ExportConfig) -> str:
        return f"{cfg.score_home_label} {self.white} - {self.black} {cfg.score_away_label}"

    def text_after(self, cfg: ExportConfig) -> str:
        white = self.white + (1 if self.flip_team == "white" else 0)
        black = self.black + (1 if self.flip_team == "black" else 0)
        return f"{cfg.score_home_label} {white} - {black} {cfg.score_away_label}"


def _drawtext_score_stage(cfg: ExportConfig, text: str, enable: str | None = None) -> str:
    """One static-text drawtext stage for the score, sitting under the clock.

    Unlike the clock this draws a fixed string, so none of the
    `%{pts:gmtime:...}` argument-parsing restrictions apply -- but ':' is
    still a filter-option separator, which is why the score is rendered with
    a ' - ' separator and never a colon."""
    y = cfg.burn_in_margin_px * 2 + cfg.burn_in_font_size
    stage = (
        f"drawtext=fontfile={_escape_filter_value(cfg.burn_in_font_path)}"
        f":text={_escape_filter_value(text)}"
        f":x={cfg.burn_in_margin_px}:y={y}"
        f":fontsize={cfg.burn_in_font_size}:fontcolor=white"
        f":box=1:boxcolor=black@0.5:boxborderw={cfg.burn_in_margin_px // 3}"
    )
    if enable is not None:
        stage += f":enable={_escape_filter_value(enable)}"
    return stage


def _score_filters(cfg: ExportConfig, score: ScoreOverlay, piece_start: float, piece_duration: float) -> list[str]:
    """Score stages for ONE encoded piece, with the flip rebased onto that
    piece's own timeline (every `-ss`-trimmed piece restarts t at 0).

    A clip can be split two different ways -- across a chunk boundary
    (render_export_clip) and into resumable segments (seg_render.py) -- and
    in both cases a piece can sit entirely before or entirely after the
    flip. Getting this wrong puts the score change in the wrong segment,
    which is the same trap the time-of-day clock hit."""
    if score.flip_seconds is None or not score.flip_team:
        return [_drawtext_score_stage(cfg, score.text_before(cfg))]

    local_flip = score.flip_seconds - piece_start
    if local_flip <= 0:
        return [_drawtext_score_stage(cfg, score.text_after(cfg))]
    if local_flip >= piece_duration:
        return [_drawtext_score_stage(cfg, score.text_before(cfg))]
    return [
        _drawtext_score_stage(cfg, score.text_before(cfg), enable=f"lt(t,{local_flip:.3f})"),
        _drawtext_score_stage(cfg, score.text_after(cfg), enable=f"gte(t,{local_flip:.3f})"),
    ]


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def is_playable(path: Path) -> bool:
    """True only if ffprobe can read a duration out of `path`.

    Existence plus a non-zero size is NOT enough to call a render finished: a
    process killed mid-encode leaves a large file with no `moov` atom, which
    ffprobe rejects as "moov atom not found" and no player will open. This bit
    hard on 2026-08-23 -- an export killed partway left a 9.5MB corpse that
    cmd_export_picks' resume check happily treated as done, and it would have
    been posted to Telegram if it hadn't been spot-checked. Any resume/skip
    decision about an existing media file should go through this, not through
    `stat().st_size > 0`."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() not in ("", "N/A")


def _encode_piece(
    source_path: Path,
    start: float,
    duration: float,
    out_path: Path,
    cfg: ReviewConfig | ExportConfig,
    epoch_base: float | None = None,
    score: ScoreOverlay | None = None,
    piece_start: float = 0.0,
) -> None:
    """Encode one piece. `epoch_base` (Unix epoch of the piece's first frame)
    burns a wall-clock overlay on, and `score` burns a running goal counter
    on -- export path only; review clips never pass either, so they stay
    clean.

    `piece_start` is this piece's offset into the WHOLE clip, needed to
    rebase the score flip onto this piece's own timeline. It is 0.0 for a
    single-piece clip, and the accumulated offset for a chunk-spanning or
    segmented one."""
    audio_args = ["-ac", "1"] if cfg.mono_audio else []
    video_filter = f"{_scale_filter(cfg.max_width)},fps={cfg.fps}"
    if epoch_base is not None and isinstance(cfg, ExportConfig) and cfg.burn_in_time:
        video_filter += f",{_drawtext_filter(cfg, epoch_base)}"
    if score is not None and isinstance(cfg, ExportConfig) and cfg.burn_in_score:
        for stage in _score_filters(cfg, score, piece_start, duration):
            video_filter += f",{stage}"
    _run_ffmpeg(
        [
            "-ss", f"{start:.3f}",
            "-i", str(source_path),
            "-t", f"{duration:.3f}",
            "-vf", video_filter,
            "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf), "-threads", str(cfg.threads),
            "-c:a", "aac", "-strict", "-2", "-b:a", f"{cfg.audio_bitrate_kbps}k", *audio_args,
            str(out_path),
        ]
    )


def render_review_clip(slices: list[ChunkSlice], out_path: Path, tmp_dir: Path, cfg: ReviewConfig) -> None:
    """Render a (possibly chunk-boundary-spanning) interval as one small
    review clip, sourced from each chunk's .LRF proxy."""
    if len(slices) == 1:
        cs = slices[0]
        source_path = cs.chunk.lrf_path or cs.chunk.mp4_path
        _encode_piece(source_path, cs.local_start_seconds, cs.local_end_seconds - cs.local_start_seconds, out_path, cfg)
        return

    part_paths: list[Path] = []
    for i, cs in enumerate(slices):
        source_path = cs.chunk.lrf_path or cs.chunk.mp4_path
        part_path = tmp_dir / f"{out_path.stem}_part{i}.mp4"
        _encode_piece(source_path, cs.local_start_seconds, cs.local_end_seconds - cs.local_start_seconds, part_path, cfg)
        part_paths.append(part_path)

    concat_clips(part_paths, out_path, force_reencode=False)
    for part_path in part_paths:
        part_path.unlink(missing_ok=True)


def render_export_clip(
    slices: list[ChunkSlice],
    out_path: Path,
    tmp_dir: Path,
    cfg: ExportConfig,
    score: ScoreOverlay | None = None,
) -> None:
    """Render a (possibly chunk-boundary-spanning) interval as one
    full-resolution, re-encoded delivery clip, always sourced from the
    full-res .MP4 (never the .LRF proxy) -- unlike render_review_clip,
    this is meant for sharing, not just fast true/false-positive review.

    Each piece gets its own wall-clock epoch, derived from its owning chunk's
    filename timestamp. For an interval spanning a chunk boundary that is the
    correct behavior, not an approximation: the camera really was stopped
    between those chunks, so the burned-in clock jumps by the length of the
    gap rather than pretending the recording was continuous."""
    if len(slices) == 1:
        cs = slices[0]
        epoch_base = slice_start_epoch(cs.chunk, cs.local_start_seconds)
        _encode_piece(
            cs.chunk.mp4_path,
            cs.local_start_seconds,
            cs.local_end_seconds - cs.local_start_seconds,
            out_path,
            cfg,
            epoch_base,
            score,
        )
        return

    part_paths: list[Path] = []
    piece_start = 0.0
    for i, cs in enumerate(slices):
        part_path = tmp_dir / f"{out_path.stem}_part{i}.mp4"
        epoch_base = slice_start_epoch(cs.chunk, cs.local_start_seconds)
        duration = cs.local_end_seconds - cs.local_start_seconds
        _encode_piece(
            cs.chunk.mp4_path,
            cs.local_start_seconds,
            duration,
            part_path,
            cfg,
            epoch_base,
            score,
            piece_start,
        )
        piece_start += duration
        part_paths.append(part_path)

    concat_clips(part_paths, out_path, force_reencode=False)
    for part_path in part_paths:
        part_path.unlink(missing_ok=True)


def render_whole_game_skim(chunks: list[Chunk], out_path: Path, cfg: ReviewConfig) -> None:
    """Concatenate every chunk's .LRF proxy and resize/downsample the whole
    thing into one small file for a quick full skim."""
    inputs: list[str] = []
    filter_inputs: list[str] = []
    for i, chunk in enumerate(chunks):
        source_path = chunk.lrf_path or chunk.mp4_path
        inputs += ["-i", str(source_path)]
        filter_inputs.append(f"[{i}:v:0][{i}:a:0]")
    concat_filter = "".join(filter_inputs) + f"concat=n={len(chunks)}:v=1:a=1[v][a]"
    filter_complex = f"{concat_filter};[v]{_scale_filter(cfg.max_width)},fps={cfg.fps}[vout]"

    _run_ffmpeg(
        [
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[a]",
            "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf), "-threads", str(cfg.threads),
            "-c:a", "aac", "-strict", "-2", "-b:a", f"{cfg.audio_bitrate_kbps}k", "-ac", "1",
            str(out_path),
        ]
    )
