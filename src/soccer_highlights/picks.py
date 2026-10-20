"""Structured pick lines for export (2026-10-18): which clips to export, with an
optional crop override, an optional slow-mo replay and (2026-10-19) an optional
trim to a tight window around the goal.

One pick per line (blank lines and `#` comments ignored; `;` also separates
picks so the same text works inline as `--picks "r03; r05 crop=left"`). Tokens
are separated by whitespace and/or commas; the rank and the keys are
case-insensitive (`R01 trim=31, slomo` == `r01 trim=31 slomo`):

    <rank> [crop=left|center|right|full] [trim=N]
           [slomo | slomo=A-B | slomo=A-B@S | slomo@S]

    r03
    r05 crop=left
    r07 slomo
    r07 slomo=6.5-9
    r09 slomo=4-7@0.5 crop=full
    r04 trim=58

`<rank>` is the `rN` prefix of the rank-named clip_file `name-candidates`
produces (`r03_s5_746a_c032.mp4`); a full clip_file name (or its stem) is also
accepted, which is what the older `--clips` flag passes through. Window times
(`A-B`) and `trim=N` are seconds on the review clip's own timeline: row
`start_seconds` is t=0 of the review clip AND of the untrimmed export, so what
Kaveh sees in the review player is what the number means.

`trim=N` (2026-10-19): audio-peak merging on a small court yields 40-110s
candidates; Kaveh supplies ONE number, the second the goal/moment happens, and
`trim_window` turns it into the exported sub-interval. The slow-mo window stays
on the ORIGINAL review-clip timeline whether or not the pick is trimmed.

Pure module: no ffmpeg, no filesystem. Everything is validated up front and
ALL bad lines are reported together -- a ~37x-realtime export batch must not
find out about a typo in pick 9 after rendering picks 1-8.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from soccer_highlights.config import TrimConfig
from soccer_highlights.crop import REGIONS, CropConfig, crop_rect

FULL = "full"
CROP_CHOICES = (*REGIONS, FULL)

# 0.25x of the 119.88fps source at the 30fps export rate is true 4x slow-mo
# from real frames, no interpolation (decision 2026-10-18).
DEFAULT_SLOMO_SPEED = 0.25
# Default replay window around slomo_center_s: more lead than tail, because the
# centre is the audio peak / tap-derived estimate of the strike and the run-up
# is the part worth seeing slowly.
SLOMO_LEAD_SECONDS = 2.0
SLOMO_TAIL_SECONDS = 1.0
# Slower than this is a multi-minute encode for a few seconds of footage; above
# 1.0 would not be slow motion at all.
MIN_SLOMO_SPEED = 0.05
MAX_SLOMO_SPEED = 1.0
# Shorter windows round to a handful of frames.
MIN_SLOMO_WINDOW_SECONDS = 0.25

_NUM = r"\d+(?:\.\d+)?"
_SLOMO_RE = re.compile(rf"^slomo(?:=(?P<a>{_NUM})-(?P<b>{_NUM}))?(?:@(?P<s>{_NUM}))?$")
_TRIM_RE = re.compile(rf"^trim=(?P<n>{_NUM})$")
_RANK_RE = re.compile(r"^r(\d+)$", re.IGNORECASE)
# Leading rank(s) of a clip_file: `r03_...`, and the merged `r30_31_s2_...`
# that Step 4b post sheets produce (both ranks address the merged clip).
_CLIP_RANKS_RE = re.compile(r"^r(\d+)((?:_\d+)*)_")


class PickError(ValueError):
    """One or more bad pick lines. `errors` has one message per problem."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass(frozen=True)
class SlowMo:
    """A slow-motion replay of clip-relative seconds [start, end] at `speed`."""

    start_seconds: float
    end_seconds: float
    speed: float = DEFAULT_SLOMO_SPEED

    @property
    def source_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def output_seconds(self) -> float:
        return self.source_seconds / self.speed


@dataclass(frozen=True)
class PickSpec:
    """One parsed pick line, before it is matched against a sheet."""

    ref: str
    crop: str | None = None  # None = the sheet's action_region
    slomo: bool = False
    window: tuple[float, float] | None = None  # None = default window around slomo_center_s
    speed: float | None = None  # None = DEFAULT_SLOMO_SPEED
    trim: float | None = None  # second (review-clip timeline) of the goal/moment; None = whole clip


@dataclass(frozen=True)
class TrimWindow:
    """Exported sub-interval [start_seconds, end_seconds] of a trimmed pick, in
    seconds on the review clip's own timeline, plus a loud `warning` when the
    score flip could not be kept inside it."""

    start_seconds: float
    end_seconds: float
    warning: str | None = None


@dataclass(frozen=True)
class ResolvedPick:
    """A pick matched to its sheet row, with every default filled in."""

    clip_file: str
    row: dict[str, str]
    region: str  # left | center | right | full
    slomo: SlowMo | None
    trim: TrimWindow | None = None  # None = export the row's whole [start, end]

    def export_seconds(self) -> tuple[float, float]:
        """[start, end] in session seconds of the LIVE part actually exported:
        the row's own interval, or its trimmed sub-window. (The slow-mo replay
        is mapped from the full row interval -- see render.build_replay.)"""
        start = float(self.row["start_seconds"])
        if self.trim is None:
            return start, float(self.row["end_seconds"])
        return start + self.trim.start_seconds, start + self.trim.end_seconds

    def describe(self) -> str:
        parts = [f"crop={self.region}"]
        if self.trim is not None:
            parts.append(f"trim={self.trim.start_seconds:g}-{self.trim.end_seconds:g}")
        if self.slomo is not None:
            s = self.slomo
            parts.append(f"slomo={s.start_seconds:g}-{s.end_seconds:g}@{s.speed:g}")
        return " ".join(parts)


def _parse_tokens(tokens: list[str]) -> PickSpec:
    ref = tokens[0]
    if "=" in ref or ref.lower().startswith(("slomo", "crop", "trim")):
        raise ValueError(f"the line must start with a rank (e.g. r03), got {ref!r}")
    if _RANK_RE.match(ref):
        ref = ref.lower()

    crop: str | None = None
    slomo = False
    window: tuple[float, float] | None = None
    speed: float | None = None
    trim: float | None = None
    for token in tokens[1:]:
        token = token.lower()
        if token.startswith("crop="):
            value = token[len("crop=") :].lower()
            if crop is not None:
                raise ValueError("crop given twice")
            if value not in CROP_CHOICES:
                raise ValueError(f"bad crop {value!r} (expected {'|'.join(CROP_CHOICES)})")
            crop = value
        elif token.startswith("slomo"):
            if slomo:
                raise ValueError("slomo given twice")
            match = _SLOMO_RE.match(token)
            if match is None:
                raise ValueError(f"bad slomo token {token!r} (expected slomo, slomo=A-B, slomo=A-B@S or slomo@S)")
            slomo = True
            if match["a"] is not None:
                window = (float(match["a"]), float(match["b"]))
                if window[0] >= window[1]:
                    raise ValueError(f"slomo window {token[len('slomo='):]!r} must have A < B")
            if match["s"] is not None:
                speed = float(match["s"])
                if not MIN_SLOMO_SPEED <= speed <= MAX_SLOMO_SPEED:
                    raise ValueError(f"slomo speed {speed:g} must be between {MIN_SLOMO_SPEED:g} and {MAX_SLOMO_SPEED:g}")
        elif token.startswith("trim"):
            if trim is not None:
                raise ValueError("trim given twice")
            match = _TRIM_RE.match(token)
            if match is None:
                raise ValueError(f"bad trim token {token!r} (expected trim=N, N = seconds on the review clip, e.g. trim=31)")
            trim = float(match["n"])
            if trim <= 0:
                raise ValueError(f"trim={trim:g} must be greater than 0")
        else:
            raise ValueError(f"unknown token {token!r}")
    return PickSpec(ref=ref, crop=crop, slomo=slomo, window=window, speed=speed, trim=trim)


def _parse_lines(text: str) -> tuple[list[PickSpec], list[str]]:
    specs: list[PickSpec] = []
    errors: list[str] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        body = raw.split("#", 1)[0]
        items = [item.strip() for item in body.split(";")]
        items = [item for item in items if item]
        for item_no, item in enumerate(items, start=1):
            label = f"line {line_no}" + (f" item {item_no}" if len(items) > 1 else "")
            try:
                specs.append(_parse_tokens(item.replace(",", " ").split()))
            except ValueError as exc:
                errors.append(f"{label}: {item!r}: {exc}")
    return specs, errors


def parse_picks(text: str) -> list[PickSpec]:
    """Parse pick lines. Raises PickError listing EVERY bad line, not the first."""
    specs, errors = _parse_lines(text)
    if errors:
        raise PickError(errors)
    return specs


def default_slomo_window(center_seconds: float, clip_seconds: float) -> tuple[float, float] | None:
    """[center - 2s, center + 1s] clamped to the clip, or None if that leaves
    less than MIN_SLOMO_WINDOW_SECONDS (centre outside the clip)."""
    start = max(0.0, center_seconds - SLOMO_LEAD_SECONDS)
    end = min(clip_seconds, center_seconds + SLOMO_TAIL_SECONDS)
    if end - start < MIN_SLOMO_WINDOW_SECONDS:
        return None
    return (start, end)


def trim_window(trim_s: float, duration: float, flip_s: float | None, cfg: TrimConfig) -> TrimWindow:
    """Exported window [t0, t1] on the review-clip timeline for `trim=trim_s`.

    Base window is [N - lead, N + tail]. The burned-in counter flips at the TAP
    (project decision 2026-09-05, deliberately NOT moved to the goal), and the
    tap usually lands after the goal -- so if the sheet's `flip_s` is past t1 the
    window is stretched to flip + tap_tail, otherwise the clip would end before
    the number changes. The stretch is capped (`max_tap_extension_seconds` after
    N): a tap that late is a slow press or a wrong N, and following it would
    rebuild the long clip trim exists to avoid. In that case t1 = N + cap and
    `warning` says loudly that the flip falls outside the clip (the scoreboard
    then shows the pre-flip score throughout, see ScoreOverlay.rebased). A flip
    BEFORE t0 needs no handling here: the rebased overlay starts on the
    post-flip score. Everything is clamped to [0, duration]."""
    if not 0 < trim_s < duration:
        raise ValueError(f"trim={trim_s:g} must be inside the {duration:.1f}s clip (0 < N < duration)")
    t0 = trim_s - cfg.lead_seconds
    t1 = trim_s + cfg.tail_seconds
    if flip_s is not None and flip_s > t1:
        if flip_s - trim_s <= cfg.max_tap_extension_seconds:
            t1 = flip_s + cfg.tap_tail_seconds
        else:
            t1 = trim_s + cfg.max_tap_extension_seconds
    t0 = max(0.0, t0)
    t1 = min(duration, t1)
    warning = None
    if flip_s is not None and flip_s >= t1:
        warning = (
            f"score flip at {flip_s:g}s is {flip_s - trim_s:g}s after trim={trim_s:g} and falls OUTSIDE the "
            f"exported window {t0:g}-{t1:g}s -- the scoreboard will not change in this clip "
            f"(slow tap or wrong trim? check the score by hand)"
        )
    return TrimWindow(t0, t1, warning)


def row_flip_seconds(row: dict[str, str]) -> float | None:
    """The sheet's score_flip_seconds, or None when blank/absent/unparseable
    (no score change inside the review clip)."""
    raw = (row.get("score_flip_seconds") or "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def clip_ranks(clip_file: str) -> list[int]:
    """Review ranks a clip_file answers to: `r03_s5_...` -> [3], merged
    `r30_31_s2_...` -> [30, 31], anything else (clip_055.mp4, gs01_...) -> []."""
    match = _CLIP_RANKS_RE.match(clip_file)
    if match is None:
        return []
    return [int(match[1])] + [int(n) for n in match[2].split("_") if n]


def match_rows(ref: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Rows addressed by `ref`: a rank (`r03`, `r3`), a full clip_file, or a
    clip_file stem."""
    rank = _RANK_RE.match(ref)
    if rank is not None:
        wanted = int(rank[1])
        return [r for r in rows if wanted in clip_ranks(r["clip_file"])]
    return [r for r in rows if r["clip_file"] == ref or r["clip_file"] == f"{ref}.mp4"]


def row_region(row: dict[str, str]) -> str:
    """The sheet's action_region, or `full` when blank/absent/unrecognised
    (an older sheet has no such column; that must mean no crop, not an error)."""
    value = (row.get("action_region") or "").strip().lower()
    return value if value in REGIONS else FULL


def row_slomo_center(row: dict[str, str]) -> float | None:
    raw = (row.get("slomo_center_s") or "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def resolve_picks(
    specs: list[PickSpec],
    rows: list[dict[str, str]],
    use_sheet_crop: bool = True,
    trim_cfg: TrimConfig | None = None,
) -> list[ResolvedPick]:
    """Match specs to sheet rows and fill defaults; raises PickError listing
    every unknown/ambiguous rank, duplicate clip, bad slow-mo window and
    out-of-range trim.

    `use_sheet_crop=False` ignores the sheet's action_region (explicit `crop=`
    on a pick still applies). `trim_cfg` sets the `trim=N` window (defaults)."""
    trim_cfg = trim_cfg if trim_cfg is not None else TrimConfig()
    resolved: list[ResolvedPick] = []
    errors: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        matches = match_rows(spec.ref, rows)
        if not matches:
            errors.append(f"{spec.ref!r}: no clip in the sheet matches")
            continue
        if len(matches) > 1:
            names = ", ".join(r["clip_file"] for r in matches)
            errors.append(f"{spec.ref!r}: ambiguous, matches {names}")
            continue
        row = matches[0]
        clip_file = row["clip_file"]
        if clip_file in seen:
            errors.append(f"{spec.ref!r}: {clip_file} is picked more than once")
            continue
        seen.add(clip_file)

        region = spec.crop if spec.crop is not None else (row_region(row) if use_sheet_crop else FULL)
        clip_seconds = float(row["end_seconds"]) - float(row["start_seconds"])
        trim: TrimWindow | None = None
        if spec.trim is not None:
            if not 0 < spec.trim < clip_seconds:
                errors.append(f"{spec.ref!r}: trim={spec.trim:g} is outside the {clip_seconds:.1f}s clip (needs 0 < N < {clip_seconds:.1f})")
                continue
            trim = trim_window(spec.trim, clip_seconds, row_flip_seconds(row), trim_cfg)
        slomo: SlowMo | None = None
        if spec.slomo:
            speed = spec.speed if spec.speed is not None else DEFAULT_SLOMO_SPEED
            if spec.window is not None:
                start, end = spec.window
                if end > clip_seconds + 1e-6:
                    errors.append(
                        f"{spec.ref!r}: slomo window {start:g}-{end:g} is outside the {clip_seconds:.1f}s clip"
                    )
                    continue
                if end - start < MIN_SLOMO_WINDOW_SECONDS:
                    errors.append(f"{spec.ref!r}: slomo window {start:g}-{end:g} is shorter than {MIN_SLOMO_WINDOW_SECONDS:g}s")
                    continue
            elif spec.trim is not None:
                # Trimmed pick: the replay is of the goal Kaveh named, not of the
                # sheet's (possibly different) dot. Still on the review timeline.
                default = default_slomo_window(spec.trim, clip_seconds)
                if default is None:
                    errors.append(f"{spec.ref!r}: the {clip_seconds:.1f}s clip is too short for a slow-mo replay")
                    continue
                start, end = default
            else:
                center = row_slomo_center(row)
                if center is None:
                    errors.append(
                        f"{spec.ref!r}: slomo needs slomo_center_s but {clip_file} has none -- give a window (slomo=A-B)"
                    )
                    continue
                default = default_slomo_window(center, clip_seconds)
                if default is None:
                    errors.append(
                        f"{spec.ref!r}: slomo_center_s={center:g} is outside the {clip_seconds:.1f}s clip -- give a window (slomo=A-B)"
                    )
                    continue
                start, end = default
            slomo = SlowMo(start, min(end, clip_seconds), speed)
        resolved.append(ResolvedPick(clip_file=clip_file, row=row, region=region, slomo=slomo, trim=trim))
    if errors:
        raise PickError(errors)
    return resolved


def rect_for_region(region: str, cfg: CropConfig) -> tuple[float, float, float, float] | None:
    """Crop rect (frame fractions) for a region, or None for `full`."""
    return None if region == FULL else crop_rect(region, cfg)


def load_picks(
    text: str, rows: list[dict[str, str]], use_sheet_crop: bool = True, trim_cfg: TrimConfig | None = None
) -> list[ResolvedPick]:
    """Parse AND resolve in one pass, so a syntax error on one line and an
    unknown rank on another are reported together."""
    specs, errors = _parse_lines(text)
    resolved: list[ResolvedPick] = []
    try:
        resolved = resolve_picks(specs, rows, use_sheet_crop, trim_cfg)
    except PickError as exc:
        errors.extend(exc.errors)
    if errors:
        raise PickError(errors)
    return resolved
