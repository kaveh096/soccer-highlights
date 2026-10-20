"""Draw the scoreboard "bug" that gets burned into shared clips.

Split of responsibilities, and the reason this module exists at all:

  * The CHROME -- dark panel, the two team pills, the gold rule -- never
    changes during a clip, so it is drawn once here with Pillow and handed to
    ffmpeg as a PNG to composite with `overlay`. Drawing rounded pills and a
    ringed dark pill in ffmpeg's own filters would be miserable; Pillow makes
    it trivial.
  * The SCORE and CLOCK do change (the score flips mid-clip when a goal goes
    in, the clock ticks every frame), so they stay `drawtext` and are only
    positioned by the anchors this module computes.

Geometry was designed against a real 2560x1440 export and signed off on
2026-09-05, then scaled: every dimension below is expressed for a 2560px-wide
frame and multiplied by `frame_width / 2560`, so a smaller export gets a
proportionally smaller board rather than a giant one.

The score slot is deliberately a FIXED width sized for the widest realistic
score, not sized to the score being drawn. Without that, "10 - 9" is wider
than "2 - 1" and shoves the BLACK pill -- except the pills are baked into a
static PNG and cannot move, so it would simply collide.
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from soccer_highlights.config import ExportConfig

# Reference frame the geometry was designed against.
_REF_WIDTH = 2560
# Widest score the slot must hold without the pills moving. Two digits a side
# covers anything this game will produce (Aug 30 finished 6-6).
_WIDEST_SCORE = "10 - 10"

_W_FILL = (247, 249, 252, 255)
_B_FILL = (22, 25, 31, 255)
_INK = (14, 16, 20, 255)
_PANEL = (11, 13, 17, 224)
_MUTED = (203, 210, 220, 255)
_ACCENT = (233, 186, 48, 255)

# Text colours as ffmpeg colour strings, so drawtext matches the PNG.
SCORE_COLOR = "0xF7F9FC"
CLOCK_COLOR = "0xCBD2DC"


@dataclass(frozen=True)
class ScoreboardLayout:
    """Pixel geometry for one frame width. `x_score_center` and `x_clock` are
    what the drawtext stages anchor to."""

    width: int
    height: int
    inset: int  # margin from the frame's top-left corner
    x_score_center: int
    x_clock: int
    y_center: int
    font_team: int
    font_score: int
    font_clock: int

    @property
    def score_center_abs(self) -> int:
        return self.inset + self.x_score_center

    @property
    def clock_x_abs(self) -> int:
        return self.inset + self.x_clock

    @property
    def center_y_abs(self) -> int:
        return self.inset + self.y_center


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def sample_clock_text(cfg: ExportConfig) -> str:
    """A worst-case rendering of the configured clock format, for measuring.

    Deliberately a wide instant -- a two-digit day, a two-digit 12-hour hour,
    and PM -- so the panel is sized for the widest string the format can
    produce rather than for whatever time the first clip happens to be at."""
    return datetime(2026, 12, 30, 22, 46, 36).strftime(cfg.scoreboard_time_format)


def pill_label(label: str) -> str:
    """Team pills are set in caps by design. Applied here rather than expected
    of the config, so `score_home_label` can stay human ("White") -- and, more
    importantly, so measuring and drawing can never disagree about the case,
    which would size the pill for one string and draw another."""
    return label.upper()


def compute_layout(cfg: ExportConfig, frame_width: int) -> ScoreboardLayout:
    """Measure the board for `frame_width`. Everything is derived from the
    measured text, so a longer team label widens its pill instead of
    overflowing it."""
    k = (frame_width / _REF_WIDTH) * cfg.scoreboard_scale

    def s(v: float) -> int:
        return max(1, round(v * k))

    font_team, font_score = s(42), s(56)
    # The clock is bumped beyond the shared scale: at phone size it was the
    # element that needed the most help (2026-09-05).
    font_clock = s(34) + s(6)
    pill_h, pad, gap, inset = s(60), s(22), s(20), s(24)

    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    fteam = _font(cfg.scoreboard_font_narrow, font_team)
    fscore = _font(cfg.burn_in_font_path_bold, font_score)
    fclock = _font(cfg.burn_in_font_path, font_clock)

    white_w = _text_width(probe, pill_label(cfg.score_home_label), fteam) + s(18) * 2
    black_w = _text_width(probe, pill_label(cfg.score_away_label), fteam) + s(18) * 2
    slot_w = _text_width(probe, _WIDEST_SCORE, fscore)
    # Measure the clock through the ACTUAL format, not a stand-in string:
    # sizing the panel for "07.46.36 AM" while drawing "Sep 06 07.46.36 AM"
    # runs the clock straight off the end of the panel (caught in the first
    # real render, 2026-09-05).
    clock_w = _text_width(probe, sample_clock_text(cfg), fclock)

    x_white = pad
    x_slot = x_white + white_w + gap
    x_black = x_slot + slot_w + gap
    x_rule = x_black + black_w + gap
    x_clock = x_rule + s(20)
    width = x_clock + clock_w + pad
    height = pill_h + s(24) * 2

    return ScoreboardLayout(
        width=width,
        height=height,
        inset=inset,
        x_score_center=round(x_slot + slot_w / 2),
        x_clock=x_clock,
        y_center=round(height / 2),
        font_team=font_team,
        font_score=font_score,
        font_clock=font_clock,
    )


def _pill(draw, x, y, w, h, label, font, *, dark: bool) -> None:
    """A team's name on its own colour. The dark pill needs the white ring:
    without it, "BLACK" on the near-black panel is invisible (pass 1 of the
    design, 2026-09-05)."""
    draw.rounded_rectangle(
        [x, y, x + w, y + h],
        radius=h // 2,
        fill=_B_FILL if dark else _W_FILL,
        outline=(255, 255, 255, 235) if dark else None,
        width=3 if dark else 0,
    )
    draw.text((x + w / 2, y + h / 2), label, font=font, fill=_W_FILL if dark else _INK, anchor="mm")


def render_chrome(cfg: ExportConfig, layout: ScoreboardLayout) -> Image.Image:
    """The static half of the board: panel, both team pills, the gold rule."""
    k = layout.height / (60 + 24 * 2)  # recover the scale from the measured height

    def s(v: float) -> int:
        return max(1, round(v * k))

    im = Image.new("RGBA", (layout.width, layout.height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, layout.width - 1, layout.height - 1], radius=s(16), fill=_PANEL)

    fteam = _font(cfg.scoreboard_font_narrow, layout.font_team)
    pill_h, pad, gap = s(60), s(22), s(20)
    white_w = _text_width(d, pill_label(cfg.score_home_label), fteam) + s(18) * 2
    black_w = _text_width(d, pill_label(cfg.score_away_label), fteam) + s(18) * 2

    x = pad
    _pill(d, x, s(24), white_w, pill_h, pill_label(cfg.score_home_label), fteam, dark=False)
    x_black = layout.x_clock - s(20) - gap - black_w
    _pill(d, x_black, s(24), black_w, pill_h, pill_label(cfg.score_away_label), fteam, dark=True)
    x_rule = layout.x_clock - s(20)
    d.line([x_rule, s(24) + s(2), x_rule, s(24) + pill_h - s(2)], fill=_ACCENT, width=3)
    return im


@dataclass(frozen=True)
class ReplayBadgeLayout:
    """The REPLAY badge hangs directly under the board, left edges aligned.
    `x`/`y` are absolute frame coordinates of its top-left corner."""

    width: int
    height: int
    x: int
    y: int
    dot_x: int  # relative to the badge's own top-left
    dot_d: int
    font_label: int
    font_speed: int
    x_label: int
    x_rule: int
    x_speed: int


def speed_label(speed: float) -> str:
    """`0.25x`, `0.5x`, `1x` -- the multiplication sign is a real glyph in both
    shipped Arial files, unlike the unicode arrows tried first."""
    return f"{speed:g}×"


def compute_badge_layout(cfg: ExportConfig, board: ScoreboardLayout, speed: float) -> ReplayBadgeLayout:
    k = board.height / (60 + 24 * 2)  # same recovered scale as render_chrome

    def s(v: float) -> int:
        return max(1, round(v * k))

    font_label, font_speed = s(34), s(30)
    pad, gap = s(22), s(14)
    dot_d = s(16)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    flabel = _font(cfg.scoreboard_font_narrow, font_label)
    fspeed = _font(cfg.burn_in_font_path, font_speed)
    label_w = _text_width(probe, "REPLAY", flabel)
    speed_w = _text_width(probe, speed_label(speed), fspeed)
    x_label = pad + dot_d + gap
    x_rule = x_label + label_w + gap
    x_speed = x_rule + gap
    return ReplayBadgeLayout(
        width=x_speed + speed_w + pad,
        height=s(60),
        x=board.inset,
        y=board.inset + board.height + s(10),
        dot_x=pad,
        dot_d=dot_d,
        font_label=font_label,
        font_speed=font_speed,
        x_label=x_label,
        x_rule=x_rule,
        x_speed=x_speed,
    )


def render_replay_badge(cfg: ExportConfig, badge: ReplayBadgeLayout, speed: float) -> Image.Image:
    """The static half of the badge: the same dark panel as the board, the word
    REPLAY, a gold rule, the speed. The dot is a separate same-size image
    (`render_replay_dot`) so ffmpeg can blink it without blinking the text."""
    k = badge.height / 60
    im = Image.new("RGBA", (badge.width, badge.height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, badge.width - 1, badge.height - 1], radius=max(1, round(14 * k)), fill=_PANEL)
    cy = badge.height / 2
    d.text((badge.x_label, cy), "REPLAY", font=_font(cfg.scoreboard_font_narrow, badge.font_label), fill=_W_FILL, anchor="lm")
    rule_h = round(badge.height * 0.46)
    d.line([badge.x_rule, cy - rule_h / 2, badge.x_rule, cy + rule_h / 2], fill=_ACCENT, width=3)
    d.text((badge.x_speed, cy), speed_label(speed), font=_font(cfg.burn_in_font_path, badge.font_speed), fill=_MUTED, anchor="lm")
    return im


def render_replay_dot(badge: ReplayBadgeLayout) -> Image.Image:
    im = Image.new("RGBA", (badge.width, badge.height), (0, 0, 0, 0))
    cy = badge.height / 2
    ImageDraw.Draw(im).ellipse(
        [badge.dot_x, cy - badge.dot_d / 2, badge.dot_x + badge.dot_d, cy + badge.dot_d / 2], fill=_ACCENT
    )
    return im


def replay_badge_pngs(cfg: ExportConfig, frame_width: int, speed: float) -> tuple[Path, Path, ReplayBadgeLayout]:
    """(badge_png, dot_png, layout), cached in the system temp dir by a hash of
    everything that affects the pixels -- same scheme as `chrome_png`."""
    board = compute_layout(cfg, frame_width)
    badge = compute_badge_layout(cfg, board, speed)
    key = "|".join(
        str(x)
        for x in (
            "badge-v1",
            frame_width,
            cfg.scoreboard_scale,
            speed_label(speed),
            cfg.burn_in_font_path,
            cfg.scoreboard_font_narrow,
        )
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    tmp = Path(tempfile.gettempdir())
    badge_path = tmp / f"soccer_hl_badge_{digest}.png"
    dot_path = tmp / f"soccer_hl_badge_dot_{digest}.png"
    if not badge_path.exists():
        render_replay_badge(cfg, badge, speed).save(badge_path)
    if not dot_path.exists():
        render_replay_dot(badge).save(dot_path)
    return badge_path, dot_path, badge


def chrome_png(cfg: ExportConfig, frame_width: int) -> tuple[Path, ScoreboardLayout]:
    """Path to the chrome PNG for this config, drawing it if it isn't cached.

    Cached by a hash of everything that affects the pixels, in the system temp
    dir: the render path calls this once per encoded piece, and a segmented
    export (seg_render.py) is a fresh process per invocation, so recomputing
    it every time would be wasteful for something that never varies within a
    game."""
    layout = compute_layout(cfg, frame_width)
    key = "|".join(
        str(x)
        for x in (
            frame_width,
            cfg.scoreboard_scale,
            cfg.scoreboard_time_format,
            cfg.score_home_label,
            cfg.score_away_label,
            cfg.burn_in_font_path,
            cfg.burn_in_font_path_bold,
            cfg.scoreboard_font_narrow,
        )
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"soccer_hl_scoreboard_{digest}.png"
    if not path.exists():
        render_chrome(cfg, layout).save(path)
    return path, layout
