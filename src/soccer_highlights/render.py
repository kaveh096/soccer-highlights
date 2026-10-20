"""Render small, resized/downsampled review clips -- plus a whole-game skim
and negative-space (uncovered) clips -- from the .LRF proxies only. Meant
for fast human true/false-positive review on modest hardware, not final
output quality."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from soccer_highlights import scoreboard
from soccer_highlights.clipping import concat_clips
from soccer_highlights.config import ExportConfig, ReviewConfig
from soccer_highlights.discovery import Chunk, slice_start_epoch
from soccer_highlights.picks import SlowMo
from soccer_highlights.timeline import ChunkSlice, Interval, map_interval_to_chunks

# Crop window as frame fractions (x0, y0, w, h) -- see soccer_highlights.crop.
Rect = tuple[float, float, float, float]


def _scale_filter(max_width: int) -> str:
    # -2 keeps height a multiple of 2 (required by libx264) while preserving
    # the source's aspect ratio, instead of forcing a fixed WxH that would
    # distort a 16:9 source.
    return f"scale={max_width}:-2"


def _crop_filter(rect: Rect) -> str:
    """ffmpeg `crop` for a window given as frame fractions (2026-10-18).

    Expressed in iw/ih rather than pixels so it is independent of the source
    resolution, and snapped to EVEN pixels with round(): yuv420 chroma is
    half-resolution, and a 3840*0.666667 style product is not an integer
    (2/3 of 2160 is 1439.99... in doubles, which truncation would turn into
    1439). The six-decimal fractions round to exactly 2560x1440 at
    (0|640|1280, 360) on a 3840x2160 source. Applied FIRST in the chain,
    before scale, so the crop is cut from full-res pixels instead of from an
    already-downscaled frame."""
    x0, y0, w, h = rect
    return (
        f"crop=w=round(iw*{w:.6f}/2)*2:h=round(ih*{h:.6f}/2)*2"
        f":x=round(iw*{x0:.6f}/2)*2:y=round(ih*{y0:.6f}/2)*2"
    )


def _slomo_pts_filter(speed: float) -> str:
    """Stretch timestamps by 1/speed. Run on the FULL-RATE decoded stream
    (before `fps`), so a 119.88fps source at speed 0.25 becomes ~29.97fps of
    real frames -- true 4x slow-mo, no frame blending or minterpolate.
    `-STARTPTS` pins the first frame to t=0 whatever the seek left behind."""
    return f"setpts=(PTS-STARTPTS)/{speed:g}"


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


def _check_time_format(cfg: ExportConfig, fmt: str | None = None, name: str = "burn_in_time_format") -> None:
    """drawtext's `%{pts:gmtime:...}` expansion splits its own arguments on
    colons with a hard 3-argument cap, so a colon anywhere in the time format
    fails the render outright. Caught here with a clear message rather than
    surfacing as ffmpeg's cryptic complaint partway through a batch."""
    fmt = cfg.burn_in_time_format if fmt is None else fmt
    if ":" in fmt:
        raise ValueError(
            f"{name} must not contain ':' (got {fmt!r}) -- "
            "drawtext's %{pts:gmtime:...} expansion splits its own arguments on colons with a "
            "hard 3-argument cap, so a colon here fails the render with "
            "'%{pts} requires at most 3 arguments'. Use '.' or '-' separators."
        )


def _drawtext_filter(cfg: ExportConfig, epoch_base: float) -> str:
    """A standalone clock, used when there is no scoreboard to put it in --
    i.e. a game with no watch marks, which is exactly today's behavior.

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
    _check_time_format(cfg)
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

    def rebased(self, t0: float, t1: float) -> ScoreOverlay:
        """The same scoreboard for the sub-clip [t0, t1] of this clip (a `trim=N`
        export, 2026-10-19), whose own t=0 is `t0`.

        The overlay is "score at clip start + one flip at an offset", so cutting
        the clip moves both: a flip inside the window becomes `flip - t0`; a flip
        at/before t0 already happened, so the cut STARTS on the post-flip score
        (and has no flip); a flip at/after t1 is not on screen, so the start score
        stays and nothing flips (only reachable through trim_window's cap, which
        warns). Boundaries match `_score_filters`, which treats a flip at or
        before a piece's start as already done and at/after its end as not yet."""
        if self.flip_seconds is None or not self.flip_team:
            return ScoreOverlay(self.white, self.black)
        if self.flip_seconds <= t0:
            white, black = (int(x) for x in self.digits_after.split(" - "))
            return ScoreOverlay(white, black)
        if self.flip_seconds >= t1:
            return ScoreOverlay(self.white, self.black)
        return ScoreOverlay(self.white, self.black, self.flip_seconds - t0, self.flip_team)

    # Digits only: the team names live in the scoreboard PNG's pills, so
    # drawing them again here would print the labels on top of themselves.
    @property
    def digits_before(self) -> str:
        return f"{self.white} - {self.black}"

    @property
    def digits_after(self) -> str:
        white = self.white + (1 if self.flip_team == "white" else 0)
        black = self.black + (1 if self.flip_team == "black" else 0)
        return f"{white} - {black}"


def _drawtext_score_stage(cfg: ExportConfig, text: str, layout, enable: str | None = None) -> str:
    """One drawtext stage for the score, centred in the scoreboard's reserved
    slot.

    Centring is done with drawtext's own `text_w` variable rather than by
    measuring the string in Python, so a two-digit score grows symmetrically
    about the slot's centre instead of drifting into the BLACK pill.

    Unlike the clock this draws a fixed string, so none of the
    `%{pts:gmtime:...}` argument-parsing restrictions apply -- but ':' is
    still a filter-option separator, which is why the score is rendered with
    a ' - ' separator and never a colon."""
    stage = (
        f"drawtext=fontfile={_escape_filter_value(cfg.burn_in_font_path_bold)}"
        f":text={_escape_filter_value(text)}"
        f":x={layout.score_center_abs}-text_w/2"
        f":y={layout.center_y_abs}-text_h/2"
        f":fontsize={layout.font_score}:fontcolor={scoreboard.SCORE_COLOR}"
    )
    if enable is not None:
        stage += f":enable={_escape_filter_value(enable)}"
    return stage


def _drawtext_clock_stage(cfg: ExportConfig, epoch_base: float, layout) -> str:
    """The clock, sitting in the scoreboard's tail past the gold rule.

    Uses `scoreboard_time_format` (time only) rather than the standalone
    clock's dated format -- see that config field for why."""
    _check_time_format(cfg, cfg.scoreboard_time_format, "scoreboard_time_format")
    text = f"%{{pts\\:gmtime\\:{epoch_base:.3f}\\:{cfg.scoreboard_time_format}}}"
    return (
        f"drawtext=fontfile={_escape_filter_value(cfg.burn_in_font_path)}"
        f":text='{text}'"
        f":x={layout.clock_x_abs}"
        f":y={layout.center_y_abs}-text_h/2"
        f":fontsize={layout.font_clock}:fontcolor={scoreboard.CLOCK_COLOR}"
    )


def _score_filters(
    cfg: ExportConfig, score: ScoreOverlay, piece_start: float, piece_duration: float, layout
) -> list[str]:
    """Score stages for ONE encoded piece, with the flip rebased onto that
    piece's own timeline (every `-ss`-trimmed piece restarts t at 0).

    A clip can be split two different ways -- across a chunk boundary
    (render_export_clip) and into resumable segments (seg_render.py) -- and
    in both cases a piece can sit entirely before or entirely after the
    flip. Getting this wrong puts the score change in the wrong segment,
    which is the same trap the time-of-day clock hit."""
    if score.flip_seconds is None or not score.flip_team:
        return [_drawtext_score_stage(cfg, score.digits_before, layout)]

    local_flip = score.flip_seconds - piece_start
    if local_flip <= 0:
        return [_drawtext_score_stage(cfg, score.digits_after, layout)]
    if local_flip >= piece_duration:
        return [_drawtext_score_stage(cfg, score.digits_before, layout)]
    return [
        _drawtext_score_stage(cfg, score.digits_before, layout, enable=f"lt(t,{local_flip:.3f})"),
        _drawtext_score_stage(cfg, score.digits_after, layout, enable=f"gte(t,{local_flip:.3f})"),
    ]


def _run_ffmpeg(args: list[str], timeout: float | None = None) -> None:
    """Run ffmpeg to completion. `timeout` (seconds) kills a hung encode and
    raises subprocess.TimeoutExpired instead of waiting forever -- see
    `encode_timeout_seconds`."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, timeout=timeout)


def encode_timeout_seconds(duration: float) -> float:
    """Generous wall-clock cap for encoding `duration` seconds of footage.

    2026-10-13: one segment's ffmpeg hung for ~41 hours (0 bytes out, 3 cores
    spinning, a demux "Invalid argument" in stderr) and stalled the whole
    export batch; the same segment re-rendered in 131s. Real encodes here
    run ~40x realtime, i.e. up to ~43s of wall clock per footage second at
    the slowest seen (851s for a 20s clip), so 120s per footage second with
    a 15 min floor is ~3x slower than anything observed and still turns a
    hang into a retry within the hour."""
    return max(900.0, duration * 120.0)


@dataclass(frozen=True)
class AudioParams:
    sample_rate: int
    channel_layout: str


@lru_cache(maxsize=None)
def probe_audio(path: Path) -> AudioParams | None:
    """Sample rate and channel layout of `path`'s first audio stream, or None
    if it has none. The slow-mo replay is muted, but it must still carry an
    audio track of the SAME rate and layout as the live part: mp4 players
    (Telegram's included) and the concat demuxer's stream-copy join both break
    or drop the audio when a file switches from a stereo/48k track to none."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,channel_layout", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(result.stdout or "{}").get("streams", [])
    if not streams:
        return None
    stream = streams[0]
    layout = stream.get("channel_layout") or {1: "mono", 2: "stereo"}.get(int(stream.get("channels", 2)), "stereo")
    return AudioParams(sample_rate=int(stream.get("sample_rate", 48000)), channel_layout=layout)


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
    crop: Rect | None = None,
    slomo_speed: float | None = None,
) -> None:
    """Encode one piece. `epoch_base` (Unix epoch of the piece's first frame)
    burns a wall-clock overlay on, and `score` burns a running goal counter
    on -- export path only; review clips never pass either, so they stay
    clean.

    `piece_start` is this piece's offset into the WHOLE clip, needed to
    rebase the score flip onto this piece's own timeline. It is 0.0 for a
    single-piece clip, and the accumulated offset for a chunk-spanning or
    segmented one.

    Two overlay shapes, and which one applies is decided by whether there IS
    a score: with one, the full scoreboard is composited and the clock lives
    inside it; without one (a game recorded with no watch marks) the clock
    falls back to the standalone box it has always been drawn in. That
    fallback is what keeps an audio-only game looking exactly as it does
    today -- a scoreboard with empty team pills and no numbers would be
    worse than no scoreboard.

    Two compounding bugs here, both only surfacing on a real full-length
    export (2026-09-08 -- every prior scoreboard test was on hand-made/
    short data):

    1. The chrome PNG input needs `-loop 1`. Without it, ffmpeg treats the
       single-frame PNG as a 1-frame stream that hits EOF almost
       immediately, and `overlay`'s eof-recovery path against a much
       longer main stream is catastrophically slow.
    2. `-t` MUST be an OUTPUT option (after -map, as it is now), not placed
       between the two `-i` flags. ffmpeg's input-option parsing associates
       an option before an `-i` with THAT upcoming input -- so `-t` sitting
       between `-i source` and `-i chrome_path` was silently capping the
       chrome PNG's own duration, not the export's. With `-loop 1` added
       but `-t` left in that position, the PNG (now correctly looping)
       still stopped at `-t` seconds, `overlay`'s eof_action=repeat then
       held its last frame while the un-capped MAIN video kept decoding to
       the end of the source chunk -- silently exporting several MINUTES
       of footage instead of the intended clip. Caught because a "14.96s"
       export came back as 368s.

    Both had to be fixed together: `-loop 1` alone (bug 1 only) produced a
    real duration bug that just happened to look like "still slow, still
    bloated" on a partial download watched mid-run. Verified via isolated
    ffmpeg CLI tests, forcing `-t` after `-map` to confirm exact output
    duration, before trusting this against real export data again.

    `crop` (frame fractions, 2026-10-18) is cut from the full-res source first,
    ahead of scale/fps and of every overlay, so the scoreboard and clock are
    drawn on the OUTPUT frame at exactly the size and position they have
    without a crop. None (or a `full` pick) adds nothing to the filter string.

    `slomo_speed` turns the piece into a muted slow-motion replay: `duration`
    is then SOURCE seconds and the output runs `duration / slomo_speed`. The
    order of the chain is the whole trick -- crop, scale, overlays, THEN
    setpts, THEN fps:
      * setpts must precede fps, or fps would drop 3 of every 4 frames before
        the stretch and leave a choppy 30fps-sourced slow-mo;
      * the scoreboard and the `%{pts:gmtime:...}` clock are drawn BEFORE the
        stretch, while frame pts is still true footage time, so the clock reads
        the real wall-clock of each frame (ticking once per 1/speed output
        seconds) and the score flip (`piece_start` = the replay's offset into
        the clip) lands on the right footage frame with no pts maths. Drawing
        after setpts would make the clock run `1/speed` times too fast.
    Audio is a silent track (anullsrc) in the live part's rate/layout: the
    source audio would be wrong for slow-mo, and Telegram/concat need a track
    present across the whole file."""
    audio_args = ["-ac", "1"] if cfg.mono_audio else []
    replay = slomo_speed is not None
    head = ",".join(([_crop_filter(crop)] if crop is not None else []) + [_scale_filter(cfg.max_width)])
    # Live: fps closes the pre-overlay chain. Replay: fps moves to the very end
    # of the whole chain, after the pts stretch.
    pre_chain = head if replay else f"{head},fps={cfg.fps}"
    post_chain = f",{_slomo_pts_filter(slomo_speed)},fps={cfg.fps}" if replay else ""
    out_duration = duration / slomo_speed if replay else duration
    silent_audio: list[str] = []
    if replay:
        audio = probe_audio(source_path)
        if audio is not None:
            silent_audio = [
                "-f", "lavfi",
                "-i", f"anullsrc=channel_layout={audio.channel_layout}:sample_rate={audio.sample_rate}",
            ]
    encode_args = [
        "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf), "-threads", str(cfg.threads),
        "-c:a", "aac", "-strict", "-2", "-b:a", f"{cfg.audio_bitrate_kbps}k", *audio_args,
    ]
    use_board = score is not None and isinstance(cfg, ExportConfig) and cfg.burn_in_score

    if use_board:
        chrome_path, layout = scoreboard.chrome_png(cfg, cfg.max_width)
        stages = list(_score_filters(cfg, score, piece_start, duration, layout))
        if epoch_base is not None and cfg.burn_in_time:
            stages.append(_drawtext_clock_stage(cfg, epoch_base, layout))
        # The chrome is a second input, so this needs filter_complex rather
        # than -vf -- which also means the audio has to be mapped explicitly,
        # since -vf's implicit stream selection no longer applies.
        filter_complex = (
            f"[0:v]{pre_chain}[v];"
            f"[v][1:v]overlay=x={layout.inset}:y={layout.inset}[bg];"
            f"[bg]{','.join(stages)}{post_chain}[vout]"
        )
        # Replay: the silent track is input 2 (after source and chrome).
        audio_maps = (["-map", "2:a"] if silent_audio else []) if replay else ["-map", "0:a?"]
        badge_inputs: list[str] = []
        if replay and cfg.replay_badge:
            # REPLAY badge under the board, composited AFTER the pts stretch and
            # fps so `t` is output time: the dot's blink period is then what the
            # viewer sees, not 1/speed of it. Its own inputs come after the
            # silent track (if any) so the existing map indexes don't move.
            badge_png, dot_png, badge = scoreboard.replay_badge_pngs(cfg, cfg.max_width, slomo_speed)
            first = 2 + (1 if silent_audio else 0)
            badge_inputs = ["-loop", "1", "-i", str(badge_png), "-loop", "1", "-i", str(dot_png)]
            blink = f"lt(mod(t*{cfg.replay_badge_flash_hz:g},1),0.6)"
            filter_complex = (
                f"[0:v]{pre_chain}[v];"
                f"[v][1:v]overlay=x={layout.inset}:y={layout.inset}[bg];"
                f"[bg]{','.join(stages)}{post_chain}[sv];"
                f"[sv][{first}:v]overlay=x={badge.x}:y={badge.y}[sb];"
                f"[sb][{first + 1}:v]overlay=x={badge.x}:y={badge.y}:enable='{blink}'[vout]"
            )
        _run_ffmpeg(
            [
                "-ss", f"{start:.3f}",
                "-i", str(source_path),
                "-loop", "1",
                "-i", str(chrome_path),
                *silent_audio,
                *badge_inputs,
                "-filter_complex", filter_complex,
                "-map", "[vout]", *audio_maps,
                "-t", f"{out_duration:.3f}",
                *encode_args,
                str(out_path),
            ],
            timeout=encode_timeout_seconds(out_duration),
        )
        return

    video_filter = pre_chain
    if epoch_base is not None and isinstance(cfg, ExportConfig) and cfg.burn_in_time:
        video_filter += f",{_drawtext_filter(cfg, epoch_base)}"
    video_filter += post_chain
    # The replay needs explicit maps (its silent track is a second input); the
    # live path keeps its implicit stream selection, byte-for-byte as before.
    replay_maps = ["-map", "0:v:0", *(["-map", "1:a"] if silent_audio else [])] if replay else []
    _run_ffmpeg(
        [
            "-ss", f"{start:.3f}",
            "-i", str(source_path),
            *silent_audio,
            *replay_maps,
            "-t", f"{out_duration:.3f}",
            "-vf", video_filter,
            *encode_args,
            str(out_path),
        ],
        timeout=encode_timeout_seconds(out_duration),
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


@dataclass(frozen=True)
class Replay:
    """A slow-motion replay mapped onto source chunks. `slices` is normally one
    piece; it is several only if the window straddles a recording boundary."""

    slomo: SlowMo
    slices: list[ChunkSlice]


@dataclass(frozen=True)
class ReplayPiece:
    """Everything `_encode_piece` needs to render one source piece of a replay."""

    source_path: Path
    start: float  # local seconds into the source chunk
    duration: float  # SOURCE seconds
    epoch_base: float
    clip_offset: float  # offset of this piece's first frame into the whole clip (true time)
    speed: float

    @property
    def output_seconds(self) -> float:
        return self.duration / self.speed


def build_replay(clip: Interval, slomo: SlowMo, chunks: list[Chunk]) -> Replay:
    """Map a clip-relative slow-mo window onto the source chunks.

    Clip-relative t=0 is the sheet row's start_seconds, which is both the
    review clip's first frame and the export's first frame (review clips and
    exports are cut from the same Interval via map_interval_to_chunks), so
    what is seen in the review player at t is what the replay window means."""
    window = Interval(
        start_seconds=clip.start_seconds + slomo.start_seconds,
        end_seconds=clip.start_seconds + slomo.end_seconds,
    )
    slices = map_interval_to_chunks(window, chunks)
    if not slices:
        raise ValueError(f"slow-mo window {slomo.start_seconds:g}-{slomo.end_seconds:g}s maps onto no source footage")
    return Replay(slomo=slomo, slices=slices)


def replay_pieces(replay: Replay) -> list[ReplayPiece]:
    pieces: list[ReplayPiece] = []
    offset = replay.slomo.start_seconds
    for cs in replay.slices:
        duration = cs.local_end_seconds - cs.local_start_seconds
        pieces.append(
            ReplayPiece(
                source_path=cs.chunk.mp4_path,
                start=cs.local_start_seconds,
                duration=duration,
                epoch_base=slice_start_epoch(cs.chunk, cs.local_start_seconds),
                clip_offset=offset,
                speed=replay.slomo.speed,
            )
        )
        offset += duration
    return pieces


def encode_replay_piece(
    piece: ReplayPiece, out_path: Path, cfg: ExportConfig, score: ScoreOverlay | None = None, crop: Rect | None = None
) -> None:
    _encode_piece(
        piece.source_path, piece.start, piece.duration, out_path, cfg,
        piece.epoch_base, score, piece.clip_offset, crop, piece.speed,
    )


def render_export_clip(
    slices: list[ChunkSlice],
    out_path: Path,
    tmp_dir: Path,
    cfg: ExportConfig,
    score: ScoreOverlay | None = None,
    crop: Rect | None = None,
    replay: Replay | None = None,
    replay_score: ScoreOverlay | None = None,
) -> None:
    """Render a (possibly chunk-boundary-spanning) interval as one
    full-resolution, re-encoded delivery clip, always sourced from the
    full-res .MP4 (never the .LRF proxy) -- unlike render_review_clip,
    this is meant for sharing, not just fast true/false-positive review.

    Each piece gets its own wall-clock epoch, derived from its owning chunk's
    filename timestamp. For an interval spanning a chunk boundary that is the
    correct behavior, not an approximation: the camera really was stopped
    between those chunks, so the burned-in clock jumps by the length of the
    gap rather than pretending the recording was continuous.

    `crop` applies to every piece, live and replay alike. `replay` appends a
    slow-mo replay after the live-speed clip (2026-10-18). It is encoded as
    its own piece(s) with the SAME encoder settings, size, fps and audio
    layout as the live part, so the join is the same stream-copy concat that
    already joins chunk-spanning pieces -- no third, quality-losing re-encode
    of the whole clip.

    `replay_score` (2026-10-19): for a trimmed pick `score` is rebased onto the
    trimmed window, but the replay is timed on the ORIGINAL clip timeline
    (replay_pieces' clip_offset), so it needs the un-rebased overlay to show the
    score as of that moment. Defaults to `score`."""
    if len(slices) == 1 and replay is None:
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
            crop=crop,
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
            crop,
        )
        piece_start += duration
        part_paths.append(part_path)

    if replay is not None:
        for i, piece in enumerate(replay_pieces(replay)):
            part_path = tmp_dir / f"{out_path.stem}_replay{i}.mp4"
            encode_replay_piece(piece, part_path, cfg, replay_score if replay_score is not None else score, crop)
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
