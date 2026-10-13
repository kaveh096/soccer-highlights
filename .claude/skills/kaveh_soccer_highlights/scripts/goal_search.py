"""Search a stretch of raw footage for goals that neither audio detection nor
a watch tap's clip window captured, by scanning it with Gemini in overlapping
windows that carry a burned-in global-second counter.

Weekly workflow, only when the tap-vs-Gemini check says so (processing.md
Step 1c -> Step 1d). Not part of a normal run: it exists for the case where a
genuine goal tap's clip scored low and describes something else (midfield
play, a stoppage, a keeper swapping gloves) and the goal is not visible in any
neighbouring candidate either -- i.e. the tap says "a goal happened", but no
rendered clip contains it.

Why this exists (2026-10-04): Kaveh tapped the first goals of that game ~22s
late (swapping goalkeeper gloves). Both the 15s fixed-window fallback and the
audio-peak-anchored window started after the ball was already in the net, so
the tapped clips scored 1-2 and described nothing goal-like. A scratch version
of this script found the missing first black goal at global second ~607
(confirmed by Kaveh watching the review clip; the tap was at 629.9).

How it works:
  * Cuts [--start-seconds, --end-seconds] (GLOBAL seconds, the same axis as
    the review sheet's start_seconds) out of the `.LRF` proxy -- never the 4K
    MP4 -- into --window-seconds windows overlapping by --overlap-seconds,
    ~640px wide at 10fps, with `T=<global second>` burned into every frame so
    Gemini's timestamps map straight back to the global timeline. Windows
    never straddle a chunk boundary (each chunk's slice is windowed on its
    own).
  * Sends each window to Gemini (cfg.gemini.model, gemini-flash-latest --
    pro regressed, see SKILL.md) with a purpose-built "list every
    ball-in-goal moment" prompt and schema, NOT `_DESCRIBE_PROMPT_V2`. The
    --camera-note text (which goal is near, who attacks it this half) is
    injected into the prompt; the camera position is per-game and not
    derivable from footage. Transport is `vision_gemini._call_gemini`, the
    same inline-base64 upload the production describe path uses.
  * --far-crop is the second pass: a full-resolution crop of the middle/far
    pitch (--crop, default 880:330:400:230 on the 1280x720 proxy) aimed at
    far-end goals and kickoff restarts. Gemini's far-end recall is poor even
    so -- treat a far-end miss as "not found", not "no goal".
  * Consolidates every window JSON in --out-dir (both passes) that overlaps
    the range, clusters goal claims across windows, attaches corroborators
    (celebration, keeper fetching the ball, kickoff restart afterwards) and
    prints a ranked candidate table. "probable" needs two independent
    windows/passes agreeing or a corroborator; everything else is
    "possible". Restarts with no goal claim before them are listed too
    (a goal Gemini didn't see may precede them). Nothing here CONFIRMS a goal
    -- only Kaveh watching the review clip does.
  * --render-top N cuts review clips (~12s before to ~10s after, 1280px,
    30fps, crf 23, veryfast, stereo AAC) of the top N candidates as
    `gsNN_<hhmmss>_g<globalsecond>.mp4` into <out-dir>/review, then copies
    them to --review-dir (the game's flat `Review\\` folder on Drive, next to
    review_sheet.csv -- the `gs` prefix keeps them apart from the `rNN_` clips).
    Rendering stays local first for the same Drive-write-stall reason as
    seg_render.py.

Cost (2026-10-04): ~17 Gemini flash calls for ~12 minutes of footage (full
pass + far-crop pass, retries included), ~20-90s per call. The window encodes
dominate wall time and depend on load: a 65s window encoded in 16s on an idle
laptop (2026-10-05), but 105s windows took 2.5-5 min each on Oct-04 with
other jobs running. Don't run this next to an export or a Drive copy.

Resumable like the rest of the pipeline: a window whose JSON already exists
and parses with the expected keys is skipped; a window's .mp4 is reused only
if `render.is_playable`. Every file is written whole (tmp + rename), never as
a growing prefix. Rerun the identical command after an interruption.

Request timeout: --timeout-seconds (default 420) overrides
cfg.gemini.request_timeout_seconds for this run only. The 60s default timed
out ~1/3 of fps=10 video uploads on 2026-10-04 ("The read operation timed
out").

Usage (from repo root, with the project venv and ffmpeg on PATH -- see
processing.md Part 1):

  ./.venv/Scripts/python.exe .claude/skills/kaveh_soccer_highlights/scripts/goal_search.py \\
      --source-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw" \\
      --start-seconds 0 --end-seconds 660 \\
      --camera-note "camera behind white's goal in the first half, black attacks the near goal" \\
      --out-dir "C:/local/scratch/goal_search_<date>" \\
      [--far-crop] [--render-top 4 --review-dir "G:/.../<date>/Review"]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from soccer_highlights import render  # noqa: E402
from soccer_highlights.config import load_config  # noqa: E402
from soccer_highlights.discovery import Chunk, discover_chunks  # noqa: E402
from soccer_highlights.vision_gemini import _call_gemini  # noqa: E402

SCAN_FPS = 10  # encoded frame rate of each window
GEMINI_FPS = 5  # Gemini's sampling rate over it (what the 2026-10-04 scan used)
SCAN_WIDTH = 640
RETRY_SLEEP_SECONDS = 20
GOAL_TYPES = {"goal", "possible_goal"}
CLUSTER_SECONDS = 8.0  # goal claims this close across windows are one candidate
CORROBORATE_BEFORE = 3.0  # celebration/keeper-fetch window around a goal claim
CORROBORATE_AFTER = 25.0
RESTART_AFTER = 150.0  # a kickoff restart this long after a claim still counts for it
REVIEW_BEFORE = 12.0
REVIEW_AFTER = 10.0

_COMMON_RULES = """Your job: find EVERY goal in this clip, at EITHER goal. A goal = the ball fully crosses the goal line into the net. Goals in this game are often scored quietly with little celebration. Also list strong goal indicators and near-misses:
- "goal": you see the ball go into the net.
- "possible_goal": likely a goal but not clearly visible (e.g. far end, obscured).
- "celebration": players celebrate (arms raised, hugging, running away cheering) or audible shouts like "goal"/"gol"/"yes".
- "keeper_retrieves_ball_from_net": a keeper or player picks the ball out of the inside of a net.
- "kickoff_restart": play restarts with both teams in their own halves and the ball at the center circle/center spot (this happens after every goal; it is a strong goal signal, so report it and try to identify which goal preceded it).
- "near_miss": shot hits the post/crossbar, is saved, or goes just wide.
Be honest: do NOT invent goals. A shot that hits the post or crossbar is a near_miss, not a goal. If there are none, return an empty events list. For each event give the T= counter value you read on screen at that moment (integer), the seconds since the start of this clip, which end (near/far/unclear), which team scored or shot (black/white/unclear), your confidence 0-1, and a concrete description of what you saw and heard (who shot from where, where the ball went, what happened next).
Also briefly describe the camera view and summarize the flow of play in the clip."""

_TEAMS_NOTE = """Game-specific camera/team note from the person who filmed it: {camera_note}
Shirt colors are mixed: some players wear red, maroon, pink or orange shirts, and goalkeepers may wear other colors, so shirt colors do not perfectly identify teams -- say "unclear" when you cannot tell; use the direction of attack (which goal the ball went into) as the main evidence."""

_FULL_PROMPT = (
    """You are watching a {dur:.0f}-second video clip, WITH AUDIO, from a recreational Sunday soccer game (about 9 players a side, no crowd, quiet). A white counter in the top-left corner shows the footage time in seconds ("T=..."); use it for timestamps.

Camera setup: a fixed wide-angle camera on a tripod stands right beside one goal, looking down the length of the pitch. The NEAR goal is the large goal frame very close to the camera, at one side of the picture. The FAR goal is at the opposite end of the pitch: a tiny goal frame near the horizon, hard to see. The game may not have started at the beginning of the clip (warm-ups) -- only report events during real match play.
"""
    + _TEAMS_NOTE
    + "\n\n"
    + _COMMON_RULES
)

_FAR_PROMPT = (
    """You are watching a {dur:.0f}-second video clip, WITH AUDIO, from a recreational Sunday soccer game (about 9 players a side, no crowd, quiet). A white counter in the top-left corner shows the footage time in seconds ("T=..."); use it for timestamps.

Camera setup: a fixed camera stands beside the NEAR goal, which is OFF-SCREEN in this cropped view. This clip is a CROP showing the middle and FAR part of the pitch at full resolution. The FAR goal is the small goal frame near the horizon/tree line (distant, small; watch it closely). The halfway line / center of the pitch is also in this crop.
FOCUS: goals scored at the FAR goal are the main thing we are looking for -- they are small and easy to miss. Watch for the ball ending up in the far net, the far keeper fetching the ball from his net, players near the far goal raising arms / jogging back, and especially a kickoff restart (everyone back in their own half, ball placed at the center spot) which follows every goal. Also report goals at the near (off-screen) goal if they can be inferred (e.g. a kickoff restart after play went off-screen toward the camera).
"""
    + _TEAMS_NOTE
    + "\n\n"
    + _COMMON_RULES
)

_EVENT_TYPES = ["goal", "possible_goal", "celebration", "keeper_retrieves_ball_from_net", "kickoff_restart", "near_miss"]
_EVENT_FIELDS = ["overlay_T", "clip_seconds", "event_type", "end", "team", "confidence", "description"]
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "camera_view": {"type": "STRING"},
        "summary": {"type": "STRING"},
        "events": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "overlay_T": {"type": "INTEGER"},
                    "clip_seconds": {"type": "NUMBER"},
                    "event_type": {"type": "STRING", "enum": _EVENT_TYPES},
                    "end": {"type": "STRING", "enum": ["near", "far", "unclear"]},
                    "team": {"type": "STRING", "enum": ["black", "white", "unclear"]},
                    "confidence": {"type": "NUMBER"},
                    "description": {"type": "STRING"},
                },
                "required": _EVENT_FIELDS,
                "propertyOrdering": _EVENT_FIELDS,
            },
        },
        "goals_seen_count": {"type": "INTEGER"},
    },
    "required": ["camera_view", "summary", "events", "goals_seen_count"],
    "propertyOrdering": ["camera_view", "summary", "events", "goals_seen_count"],
}


@dataclass
class Window:
    prefix: str  # "win" (full frame) or "far" (far crop)
    chunk: Chunk
    global_start: float
    duration: float

    @property
    def stem(self) -> str:
        return f"{self.prefix}_g{int(round(self.global_start)):05d}"

    @property
    def local_start(self) -> float:
        return self.global_start - self.chunk.global_start_seconds


@dataclass
class Event:
    t: float
    event_type: str
    end: str
    team: str
    confidence: float
    description: str
    source: str  # window stem it came from


@dataclass
class Candidate:
    t: float
    kind: str  # "goal_claim" or "restart_only"
    claims: list[Event] = field(default_factory=list)
    corroborators: list[Event] = field(default_factory=list)

    @property
    def sources(self) -> set[str]:
        return {e.source for e in self.claims}

    @property
    def passes(self) -> set[str]:
        return {s.split("_")[0] for s in self.sources}

    @property
    def confidence(self) -> float:
        events = self.claims or self.corroborators
        return max(e.confidence for e in events)

    @property
    def status(self) -> str:
        if self.kind == "restart_only":
            return "restart-only"
        if len(self.sources) >= 2 or self.corroborators:
            return "probable"
        return "possible"

    def majority(self, attr: str) -> str:
        values = [getattr(e, attr) for e in self.claims or self.corroborators]
        known = [v for v in values if v != "unclear"]
        return statistics.mode(known) if known else "unclear"


def _write_whole(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_valid_json(path: Path) -> dict | None:
    """The window's result, or None if missing/unparseable/incomplete -- a
    killed run must never leave a JSON the skip check would trust."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {"events", "summary", "_window_start", "_window_dur"}
    if not isinstance(data, dict) or not required <= data.keys() or not isinstance(data["events"], list):
        return None
    return data


def _plan_windows(
    chunks: list[Chunk], start: float, end: float, window: float, overlap: float, prefix: str
) -> list[Window]:
    """Overlapping windows over [start, end], windowed per chunk so none
    straddles a recording boundary. The last window of each chunk slice is
    shortened rather than run past the slice."""
    step = window - overlap
    windows: list[Window] = []
    for chunk in chunks:
        lo = max(start, chunk.global_start_seconds)
        hi = min(end, chunk.global_start_seconds + chunk.duration_seconds)
        if hi - lo < 1.0:
            continue
        if chunk.lrf_path is None:
            raise SystemExit(f"{chunk.mp4_path.name} has no matching .LRF proxy -- this script never reads the 4K MP4")
        s = lo
        while True:
            dur = min(window, hi - s)
            windows.append(Window(prefix=prefix, chunk=chunk, global_start=s, duration=dur))
            if s + dur >= hi - 1e-6:
                break
            s += step
    return windows


def _encode_window(w: Window, path: Path, crop: str | None, font_path: str) -> None:
    if render.is_playable(path):
        return
    path.unlink(missing_ok=True)
    geometry = f"crop={crop}" if crop else f"scale={SCAN_WIDTH}:-2"
    vf = (
        f"fps={SCAN_FPS},{geometry},"
        f"drawtext=fontfile={render._escape_filter_value(font_path)}:"
        f"text='T=%{{eif\\:t+{w.global_start:.3f}\\:d}}':"
        f"x=8:y=8:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6"
    )
    tmp = path.with_name(path.stem + ".part.mp4")
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-threads", "2",
            "-ss", f"{w.local_start:.3f}", "-i", str(w.chunk.lrf_path), "-t", f"{w.duration:.3f}",
            "-map", "0:v:0", "-map", "0:a:0",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-threads", "2",
            "-c:a", "aac", "-b:a", "64k", "-ac", "1",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(path)


def _scan_window(w: Window, mp4: Path, prompt: str, cfg, api_key: str, retries: int) -> tuple[dict | None, int]:
    """(result, calls_made). Failures are printed and retried; the API key is
    never printed."""
    calls = 0
    generation_config = {"response_mime_type": "application/json", "response_schema": SCHEMA}
    for attempt in range(1, retries + 1):
        calls += 1
        t0 = time.time()
        try:
            raw = _call_gemini(mp4, prompt, cfg, api_key, video_metadata={"fps": GEMINI_FPS}, generation_config=generation_config)
            data = json.loads(raw)
            data["_window_start"] = round(w.global_start, 3)
            data["_window_dur"] = round(w.duration, 3)
            print(f"  gemini {w.stem} ok in {time.time() - t0:.0f}s, events={len(data.get('events', []))}", flush=True)
            return data, calls
        except Exception as exc:  # noqa: BLE001 -- timeouts, 503s, bad JSON: all just retried
            print(f"  gemini {w.stem} attempt {attempt}/{retries} failed: {str(exc)[:300]}", flush=True)
            if attempt < retries:
                time.sleep(RETRY_SLEEP_SECONDS)
    return None, calls


def _collect_events(out_dir: Path, start: float, end: float) -> list[Event]:
    """Every event from every valid window JSON in out_dir (both passes)
    overlapping [start, end]. Uses the burned-in T when it is plausible for
    that window, else falls back to window start + clip_seconds."""
    events: list[Event] = []
    for path in sorted(list(out_dir.glob("win_g*.json")) + list(out_dir.glob("far_g*.json"))):
        data = _load_valid_json(path)
        if data is None:
            continue
        ws, wd = float(data["_window_start"]), float(data["_window_dur"])
        if ws + wd < start or ws > end:
            continue
        for e in data["events"]:
            t = float(e.get("overlay_T", -1))
            if not (ws - 2 <= t <= ws + wd + 2):
                t = ws + float(e.get("clip_seconds", 0.0))
            events.append(
                Event(
                    t=t,
                    event_type=e.get("event_type", ""),
                    end=e.get("end", "unclear"),
                    team=e.get("team", "unclear"),
                    confidence=float(e.get("confidence", 0.0)),
                    description=e.get("description", ""),
                    source=path.stem,
                )
            )
    return events


def _consolidate(events: list[Event]) -> list[Candidate]:
    claims = sorted((e for e in events if e.event_type in GOAL_TYPES), key=lambda e: e.t)
    candidates: list[Candidate] = []
    for e in claims:
        if candidates and candidates[-1].kind == "goal_claim" and e.t - candidates[-1].claims[-1].t <= CLUSTER_SECONDS:
            candidates[-1].claims.append(e)
        else:
            candidates.append(Candidate(t=e.t, kind="goal_claim", claims=[e]))
    for c in candidates:
        c.t = statistics.median(e.t for e in c.claims)

    used_restarts: set[int] = set()
    for c in candidates:
        for i, e in enumerate(events):
            if e.event_type in ("celebration", "keeper_retrieves_ball_from_net"):
                if c.t - CORROBORATE_BEFORE <= e.t <= c.t + CORROBORATE_AFTER:
                    c.corroborators.append(e)
            elif e.event_type == "kickoff_restart" and c.t < e.t <= c.t + RESTART_AFTER:
                c.corroborators.append(e)
                used_restarts.add(i)

    # Restarts no goal claim accounts for: a goal Gemini didn't see (typically
    # far end) may precede them. Cluster them the same way.
    orphans = sorted(
        (e for i, e in enumerate(events) if e.event_type == "kickoff_restart" and i not in used_restarts),
        key=lambda e: e.t,
    )
    for e in orphans:
        if candidates and candidates[-1].kind == "restart_only" and e.t - candidates[-1].t <= CLUSTER_SECONDS * 2:
            candidates[-1].corroborators.append(e)
        else:
            candidates.append(Candidate(t=e.t, kind="restart_only", corroborators=[e]))

    order = {"probable": 0, "possible": 1, "restart-only": 2}
    candidates.sort(key=lambda c: (order[c.status], -len(c.sources), -c.confidence, c.t))
    return candidates


def _wallclock(t: float, chunks: list[Chunk]) -> str:
    for chunk in chunks:
        if chunk.global_start_seconds <= t < chunk.global_start_seconds + chunk.duration_seconds:
            return (chunk.start_time + timedelta(seconds=t - chunk.global_start_seconds)).strftime("%H:%M:%S")
    return "--:--:--"


def _rationale(c: Candidate) -> str:
    parts: list[str] = []
    if c.claims:
        parts.append(
            f"{len(c.claims)} goal claim(s) from {len(c.sources)} window(s) [{','.join(sorted(c.sources))}]"
        )
    corr = sorted(c.corroborators, key=lambda e: e.t)
    if corr:
        parts.append("; ".join(f"{e.event_type}@{e.t:.0f}" for e in corr))
    best = max(c.claims or c.corroborators, key=lambda e: e.confidence)
    parts.append(f'"{best.description}"')
    if c.kind == "restart_only":
        parts.insert(0, "restart with no goal claim before it -- look for a goal in the ~2 min before")
    return " | ".join(parts)


def _render_review(c: Candidate, rank: int, chunks: list[Chunk], local_dir: Path, review_dir: Path | None) -> Path | None:
    chunk = next(
        (ch for ch in chunks if ch.global_start_seconds <= c.t < ch.global_start_seconds + ch.duration_seconds), None
    )
    if chunk is None or chunk.lrf_path is None:
        print(f"  cannot render candidate at {c.t:.0f}: no chunk/.LRF covers it")
        return None
    local_start = max(0.0, c.t - REVIEW_BEFORE - chunk.global_start_seconds)
    local_end = min(chunk.duration_seconds, c.t + REVIEW_AFTER - chunk.global_start_seconds)
    name = f"gs{rank:02d}_{_wallclock(c.t, chunks).replace(':', '')}_g{int(round(c.t))}.mp4"
    out = local_dir / name
    if not render.is_playable(out):
        out.unlink(missing_ok=True)
        tmp = out.with_name(out.stem + ".part.mp4")
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-threads", "2",
                "-ss", f"{local_start:.3f}", "-i", str(chunk.lrf_path), "-t", f"{local_end - local_start:.3f}",
                "-map", "0:v:0", "-map", "0:a:0", "-vf", "scale=1280:-2,fps=30",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2",
                "-c:a", "aac", "-ac", "2", "-b:a", "128k", "-movflags", "+faststart",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(out)
    if review_dir is not None:
        review_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, review_dir / name)
    print(f"  review clip {name} ({c.t - REVIEW_BEFORE:.0f}-{c.t + REVIEW_AFTER:.0f}s global)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True, help="Raw/ folder or card dir with DJI_*_D.MP4 + .LRF pairs")
    parser.add_argument("--start-seconds", type=float, required=True, help="GLOBAL second to start scanning at")
    parser.add_argument("--end-seconds", type=float, required=True, help="GLOBAL second to stop scanning at")
    parser.add_argument("--out-dir", required=True, help="Local scratch dir for windows/JSON -- not Google Drive")
    parser.add_argument(
        "--camera-note",
        required=True,
        help="Free text injected into the prompt: which goal the camera is behind and who attacks the near goal "
        "in this stretch, e.g. \"camera behind white's goal in the first half, black attacks the near goal\"",
    )
    parser.add_argument("--window-seconds", type=float, default=105.0, help="Window length (default 105)")
    parser.add_argument("--overlap-seconds", type=float, default=15.0, help="Overlap between windows (default 15)")
    parser.add_argument(
        "--timeout-seconds", type=float, default=420.0, help="Gemini request timeout for this run (default 420)"
    )
    parser.add_argument("--retries", type=int, default=3, help="Gemini attempts per window (default 3)")
    parser.add_argument("--far-crop", action="store_true", help="Second pass: full-res crop of the middle/far pitch")
    parser.add_argument(
        "--crop", default="880:330:400:230", help="ffmpeg crop W:H:X:Y for --far-crop on the 1280x720 proxy"
    )
    parser.add_argument("--render-top", type=int, default=0, help="Cut review clips for the top N candidates")
    parser.add_argument(
        "--review-dir", help="Where review clips are copied after rendering locally (the game's flat Review folder)"
    )
    args = parser.parse_args()

    if args.overlap_seconds >= args.window_seconds:
        raise SystemExit("--overlap-seconds must be smaller than --window-seconds")
    if args.end_seconds <= args.start_seconds:
        raise SystemExit("--end-seconds must be after --start-seconds")

    cfg = load_config()
    cfg.gemini.request_timeout_seconds = args.timeout_seconds
    if "pro" in cfg.gemini.model:
        print(f"WARNING: gemini model is {cfg.gemini.model} -- pro regressed against flash on this footage (SKILL.md)")
    api_key = os.environ.get(cfg.gemini.api_key_env)
    if not api_key:
        raise SystemExit(f"{cfg.gemini.api_key_env} is not set (it normally comes from the repo's .env)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = discover_chunks(args.source_dir)
    prefix = "far" if args.far_crop else "win"
    template = _FAR_PROMPT if args.far_crop else _FULL_PROMPT
    windows = _plan_windows(
        chunks, args.start_seconds, args.end_seconds, args.window_seconds, args.overlap_seconds, prefix
    )
    print(f"{len(windows)} {prefix} window(s) over {args.start_seconds:.0f}-{args.end_seconds:.0f}s, model {cfg.gemini.model}")

    calls = 0
    failed: list[str] = []
    for w in windows:
        result_path = out_dir / f"{w.stem}.json"
        if _load_valid_json(result_path) is not None:
            print(f"skip {w.stem} (already scanned)")
            continue
        mp4 = out_dir / f"{w.stem}.mp4"
        t0 = time.time()
        _encode_window(w, mp4, args.crop if args.far_crop else None, cfg.export.burn_in_font_path)
        print(f"encoded {w.stem} ({w.duration:.0f}s) in {time.time() - t0:.0f}s, {mp4.stat().st_size / 1e6:.1f} MB", flush=True)
        prompt = template.format(dur=w.duration, camera_note=args.camera_note)
        data, n = _scan_window(w, mp4, prompt, cfg.gemini, api_key, args.retries)
        calls += n
        if data is None:
            failed.append(w.stem)
            continue
        _write_whole(result_path, json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Gemini calls this run: {calls}")
    if failed:
        print(f"FAILED windows (rerun the same command to retry): {', '.join(failed)}")

    events = _collect_events(out_dir, args.start_seconds, args.end_seconds)
    candidates = _consolidate(events)
    summary = [
        {
            "rank": i,
            "global_seconds": round(c.t, 1),
            "wallclock": _wallclock(c.t, chunks),
            "status": c.status,
            "confidence": round(c.confidence, 2),
            "end": c.majority("end"),
            "team": c.majority("team"),
            "rationale": _rationale(c),
        }
        for i, c in enumerate(candidates, start=1)
    ]
    _write_whole(out_dir / "candidates.json", json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{len(candidates)} candidate(s) -- UNCONFIRMED until Kaveh watches the review clip:")
    print(f"{'#':>2}  {'global_s':>8}  {'wall':>8}  {'status':<12} {'conf':>4}  {'end':<7} {'team':<7} rationale")
    for row in summary:
        print(
            f"{row['rank']:>2}  {row['global_seconds']:>8.1f}  {row['wallclock']:>8}  {row['status']:<12} "
            f"{row['confidence']:>4.2f}  {row['end']:<7} {row['team']:<7} {row['rationale']}"
        )

    if args.render_top > 0:
        local_dir = out_dir / "review"
        local_dir.mkdir(exist_ok=True)
        review_dir = Path(args.review_dir) if args.review_dir else None
        for rank, c in enumerate(candidates[: args.render_top], start=1):
            _render_review(c, rank, chunks, local_dir, review_dir)


if __name__ == "__main__":
    main()
