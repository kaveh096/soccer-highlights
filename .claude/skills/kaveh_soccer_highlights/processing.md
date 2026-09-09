# Processing: raw recording -> posted Telegram highlights

All commands below assume the repo root (`C:\dev\soccer-highlights`) as cwd and the
project venv: `./.venv/Scripts/python.exe` (Bash/Git Bash) or `.\.venv\Scripts\python.exe`
(PowerShell). The package is installed editable (`pip install -e .`), so
`from soccer_highlights import ...` and `python -m soccer_highlights.cli ...` both work
from anywhere once the venv is active.

---

## Part 1: One-time setup (already done -- reference/troubleshooting only)

Skip this section for a normal weekly run. Come back to it if: ffmpeg breaks, a new
Telegram bot/group is needed, or `.env` gets lost/needs rotating.

### ffmpeg

System ffmpeg must be a **modern** build -- an old ~2015 build was the actual
laptop-speed bottleneck for local rendering, not the CPU. Installed via:
```
winget install --id Gyan.FFmpeg
```
Current install path:
```
C:\Users\Kaveh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin
```

**Gotcha**: the Claude Code Bash tool's shell session caches `$PATH` at session
start and does **not** pick up Windows PATH changes made mid-session or in a prior
session. If any command below fails with `FileNotFoundError: [WinError 2]` on
`ffprobe`/`ffmpeg`, check `echo $PATH | tr ':' '\n' | grep ffmpeg`. Fix by
prepending the bin dir in the **same** command that runs the pipeline (a separate
`export` doesn't persist across Bash tool calls):
```bash
export PATH="/c/Users/Kaveh/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin:$PATH"
```
Prepend this to every `soccer_highlights.cli` invocation below that touches video
(everything except pure-analysis steps).

### Telegram bot + group

Already done for the "Soccer for Fun" group -- bot `@sunday_soccer_highlights_bot`.
Reference procedure if it ever needs redoing (new bot, new group, token rotation):

1. **Create the bot**: message **@BotFather** on Telegram -> `/newbot` -> name +
   unique username ending in `bot`. It replies with an HTTP API token.
2. **Add the bot to the group**: use the group's actual "Add Member"/"Add People"
   UI and search for the bot by username. **Typing the bot's `@username` in a chat
   message is NOT the same as adding it as a member** -- if the bot was never
   formally added, mentioning it in text does nothing (confirmed the hard way).
3. **Find the group's chat ID** -- this is the fiddly part:
   - Telegram bots default to **privacy mode ON**: a bot that's a group member
     still only *receives* messages that either start with `/` or `@`-mention it
     by username. A plain "hi" is invisible to it even as a member.
   - Send a message in the group that satisfies one of those (e.g.
     `@sunday_soccer_highlights_bot test`, or any `/command`).
   - Then call (read-only, safe): `https://api.telegram.org/bot<TOKEN>/getUpdates`
     (or the diagnostic snippet below). Look for a `"chat":{"id": -100xxxxxxxxxx,
     "type":"supergroup"|"group", "title": "..."}` entry -- **group/supergroup IDs
     are negative numbers.** If nothing shows up, also check
     `getWebhookInfo` -- if a webhook URL is set, `getUpdates` returns nothing at
     all (the two are mutually exclusive); `pending_update_count` there is
     Telegram's own authoritative count of what it's tried to deliver.
   ```python
   import soccer_highlights  # triggers .env loading -- import this first
   import os, json, urllib.request
   token = os.environ.get('TELEGRAM_BOT_TOKEN')
   with urllib.request.urlopen(f'https://api.telegram.org/bot{token}/getUpdates', timeout=30) as resp:
       print(json.dumps(json.load(resp), indent=2, ensure_ascii=False))
   ```
4. **Store both as env vars in `.env`** (repo root, gitignored):
   ```
   TELEGRAM_BOT_TOKEN=<token>
   TELEGRAM_CHAT_ID=<the negative group id>
   GEMINI_API_KEY=<key>
   ANTHROPIC_API_KEY=<key>
   ```
   `.env` auto-loads for every entry point (`soccer_highlights/__init__.py` calls
   `load_dotenv` at import time) -- no need to also set Windows user env vars,
   though the Gemini/Anthropic keys happen to be set both ways historically.

**Security note (this actually happened once):** never paste a real bot token
into a chat message. If it happens, treat it as compromised immediately --
@BotFather -> `/mybots` -> select the bot -> API Token -> **Revoke current
token**, generate a new one, update `.env`. Don't wait to see if it's actually
been misused first.

### Python env / dependencies

```bash
cd C:/dev/soccer-highlights
./.venv/Scripts/python.exe -m pip install -e .
```
Key deps: `numpy`, `scipy`, `pyyaml`, `matplotlib`, `anthropic`, `python-dotenv`.
`GEMINI_API_KEY`/`ANTHROPIC_API_KEY`/`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` all
come from `.env` (see above) -- these are dedicated, project-scoped API keys for
cost-tracking purposes, not shared personal keys.

---

## Part 2: Data location and folder structure

All game footage lives on Google Drive, per game date, **not** in the repo
(`repo/output` and `repo/testdata` were removed once this reorg happened):

```
G:\My Drive\Photos and Movies\Sunday Soccer\<date>\
  Raw\                              DJI_*_D.MP4 + .LRF  (capital "Raw")
    tally_white_goal.csv            the three Tallies watch exports, one per
    tally_black_goal.csv            counter -- live-tagged marks for this game
    tally_moment.csv                (see Step 0b; absent if the watch wasn't used)
  Tests\
    pre_label\
      candidates\
        clip_001.mp4 ... clip_NNN.mp4   review-quality renders (1280px/30fps/
                                         crf23/veryfast/stereo, from .LRF) --
                                         the SAME file a human watches, Gemini
                                         scores, and label-audit later re-checks
        review_sheet.csv                clip_file/start/end/duration/max_peak_score/
                                         verdict/notes/source/gemini_score/
                                         gemini_caption/gemini_description --
                                         verdict/notes start blank, fill them in
                                         during Part 3 Step 2. `source` is only
                                         meaningful on a watch-tagged game (Step 0b)
        events.json                     raw detected intervals
        descriptions_cache.json         resumable Gemini describe-call cache
  Sharable\
    clip_NNN.mp4 ...                  final share-quality exports (2560px, CRF
                                       override for Telegram, from full-res .MP4)
    .telegram_sent.json               tracks which clip_files have been posted,
                                       so a rerun never double-posts
```

Every command below needs an explicit `--source-dir` (global CLI flag, points at
`Raw\`) and/or `SOCCER_HL__...` env var override -- `config/default.yaml`'s
defaults still point at old repo-relative paths that no longer exist.

---

## Part 3: The recurring weekly workflow

**Two modes -- most weeks are "quick share," not "full eval."** Steps 0/1/1b
(detect + render + Gemini describe/score, rename) are always run the same
way. Steps 2/3 (deciding the final picks) branch:

- **Quick share (the common case)**: Kaveh watches clips soon after the game
  to post highlights fast. He picks `clip_file`s to export/post directly off
  the ranked sheet -- `gemini_score` plus a quick skim of `gemini_description`
  is enough to decide. **`verdict`/`notes` are left blank**, deliberately --
  filling them out is the time-consuming part and isn't needed just to post
  clips. Skip straight to Step 3/4/5 below.
- **Full eval (occasional, later)**: only when actually tuning the Gemini
  prompt or re-checking detection recall (evals.md Part B) does the
  `verdict`/`notes` column need to be filled in for every row -- that's what
  turns a game's sheet into labeled ground truth for a P/R/F1 sweep. This can
  happen well after the game, on the same `review_sheet.csv`, as a separate
  pass -- it doesn't have to happen before clips get posted, and most weeks
  it doesn't happen at all.

Don't assume a given week's `review_sheet.csv` has verdicts filled in just
because clips already got posted -- check before treating a sheet as eval-
ready ground truth for evals.md Part B.

### Step 0 -- get footage onto Drive

Copy the DJI card's `DJI_*_D.MP4` + `.LRF` files into a new
`G:\My Drive\Photos and Movies\Sunday Soccer\<date>\Raw\` folder. If Drive sync is
flaky reading freshly-uploaded `.LRF` files (seen as ffmpeg `0xC0000006
STATUS_IN_PAGE_ERROR`), use `--lrf-cache-dir` in Step 1 to redirect heavy reads to
a local copy:
```bash
robocopy "G:\...\<date>\Raw" "C:\local\lrf\<date>" *.LRF /R:5 /W:15
```

### Step 0b -- ASK KAVEH FOR THE WATCH TALLY EXPORTS

**Do this every game, before Step 1, without waiting to be asked.** If Kaveh
mentions a new game or points at a new `Raw\` folder, the first thing to check
is whether the watch tallies came with it -- the exports are easy to forget
(they're a separate manual step on the watch, done after the game is over and
the phone is back in hand) and there is **no way to recover them later**. A
game processed without them silently loses the recall measurement for good.

Say something like: *"Did you export the Tallies counters for this game? They
need to be in `<date>\Raw\`."*

The ritual, for reference:
1. On the watch, Tallies exports **one CSV per counter**, so there are
   **three** files: white goal, black goal, moment.
2. They land in Google Drive under whatever name Tallies gives them (a
   free-form string like `soccer tally test - white goal - Sep 2.csv`).
3. **Rename them to the stable convention** and put them in the game's `Raw\`
   folder next to the DJI files: `tally_white_goal.csv`,
   `tally_black_goal.csv`, `tally_moment.csv`. Nothing parses the filename --
   the category is passed explicitly on the command line -- but a stable name
   is what makes the Step 0c/Step 1 commands copy-pasteable between games.

Putting them in `Raw\` is safe: `discover_chunks` globs `DJI_*_D.MP4` only and
ignores everything else in the folder.

**If the tallies are missing**, say so plainly and carry on -- everything below
degrades to audio-only exactly as before. Don't silently skip the `--tally-csv`
flags and let a `source`-less sheet imply the watch simply found nothing.

### Step 0c -- measure the clap-sync offset (`ingest-marks`)

Run this **before** Step 1, because Step 1 wants the resulting offset. It is
report-only: it renders nothing and costs no API calls, just an audio-detection
pass.

```bash
export PATH="/c/Users/Kaveh/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin:$PATH"
cd C:/dev/soccer-highlights
./.venv/Scripts/python.exe -m soccer_highlights.cli \
  --source-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw" \
  ingest-marks \
  --tally-csv "white_goal=G:/.../<date>/Raw/tally_white_goal.csv" \
  --tally-csv "black_goal=G:/.../<date>/Raw/tally_black_goal.csv" \
  --tally-csv "moment=G:/.../<date>/Raw/tally_moment.csv" \
  --final-score "<white>-<black>"
```

What to read off the output:

- **The clap-sync residual.** Kaveh claps in front of the camera and taps
  `moment` at the same instant at kickoff, so the two are simultaneous by
  construction and any difference is watch-vs-camera clock skew. Since he
  synced the camera RTC to his phone (2026-09-02) and the watch runs off the
  same phone clock, **expect ~0**. DJI filename timestamps are only
  1-second-resolution, so anything within ~2s is noise, not skew -- the command
  says so itself and only suggests a correction past that. If it does suggest
  one, pass it as `--clock-offset-seconds` to **both** this command and Step 1.
- **"no audio peak within Ns of the first moment mark"** means either the clap
  wasn't detected or the skew is bigger than the search window -- check whether
  the camera RTC sync actually took before trusting any mark for that game.
- **The white/black tally vs. the real final score.** Ask Kaveh for the score;
  a mismatch means presses were missed or mis-tapped, and it's the only free
  check on the watch data that exists. **Always ask** -- he always knows it.
- **Marks in an unrecorded gap.** These were marked while the camera was
  stopped between chunks. They can't be rendered at all; report them as
  "marked but not recorded" rather than letting them vanish.
- **Taps that land outside their own clip** (added 2026-09-08, after Sep-06
  posting surfaced a clip whose score never visibly flipped). `ingest-marks`
  now runs the same union+extend logic pre-label will, purely as a report --
  no render, no API calls -- and lists any tap still outside its clip's
  window even after `extend_for_score_flip`. That clip's score falls back to
  a midpoint flip instead of the tap, which is silent unless you're looking
  for it. A handful is normal on a real game (slow presses past the cap);
  a lot of them is worth checking `score_flip_cap_seconds` or the game's
  press-lag assumptions before running Step 1.

### Step 1 -- detect candidates + Gemini describe (`pre-label`)

```bash
export PATH="/c/Users/Kaveh/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin:$PATH"
cd C:/dev/soccer-highlights
./.venv/Scripts/python.exe -m soccer_highlights.cli \
  --source-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw" \
  pre-label \
  --out-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Tests/pre_label" \
  [--lrf-cache-dir "C:/local/lrf/<date>"] \
  [--tally-csv "white_goal=G:/.../<date>/Raw/tally_white_goal.csv"] \
  [--tally-csv "black_goal=G:/.../<date>/Raw/tally_black_goal.csv"] \
  [--tally-csv "moment=G:/.../<date>/Raw/tally_moment.csv"] \
  [--clock-offset-seconds <from Step 0c, usually omit>] \
  [--near-cam-team-first-half {white,black}]
```

**`--near-cam-team-first-half`** (added 2026-09-07): which team's goal the
camera sits behind in the FIRST half -- ask Kaveh every game, it's not
derivable from the footage. Drives review order's near-field-goal tier
ahead of Gemini's `goal_this_end` for any genuine goal tap (see SKILL.md's
settled-decisions list). Halftime is auto-detected from the recording gap;
omitting the flag just falls back to `goal_this_end`, not a broken run.

**Pass the same `--tally-csv` flags here that Step 0c used.** They union the
watch marks with the audio candidates, so a goal the watch caught but audio
missed becomes a real rendered, Gemini-scored clip in the same sheet as
everything else, and the sheet gains a **`source` column**:

| `source` | meaning |
|---|---|
| `audio` | an audio candidate no mark corroborated |
| `both` | an audio candidate a mark landed on -- audio found it too |
| `mark` | **no audio peak near the mark: an event audio detection missed** |

`mark` rows are the whole point of the watch. Counting them against the total
real events is the recall number this project has never been able to measure.
Provenance follows the mark's *anchor*, not interval overlap -- see
`marks.union_with_audio`, which documents why crediting audio for a merely
overlapping candidate would erase a real miss.

Omitting the flags is a deliberate, tested no-op: no marks means behavior
identical to the audio-only pipeline, so forgetting the watch degrades
gracefully instead of breaking the week's run.

Requires `GEMINI_API_KEY`. What it does, in order: audio peak detection
(`onset_flux`) -> render each candidate at review quality from the `.LRF` -> a
Gemini describe call per clip (`_DESCRIBE_PROMPT_V2`, `gemini-flash-latest`,
fps=10, schema-enforced JSON -- current production default, see SKILL.md's
settled-decisions list) -> writes `review_sheet.csv` with `gemini_score` (1-5),
`gemini_caption` (Farsi), `gemini_description` (English) filled in, `verdict`/
`notes` left blank for you.

**Expect this to take a while but not forever**: LRF-based review renders run
~0.33x realtime on this laptop -- a full game's worth of candidates (60-70 clips)
renders in well under 15 minutes. The describe calls are the slower part.

**Gemini 503 "high demand" congestion is common and not a bug.** `pre-label`
re-runs detection+render every time (**confirmed 2026-09-06: there is no
skip-if-exists check on the render loop at all** -- re-running always
redoes the full render pass, ~35-40 min on this laptop for a full game),
but the describe step IS resumable (`descriptions_cache.json`, only retries
`null` entries) -- a re-run costs a full render redo but pays for Gemini
calls only once per clip. Expect needing 2-3 retry rounds during a sustained
congestion window (seen 37->23->6->1->0 failures across rounds in practice,
and separately 27->9->3->0 on Aug-30). If a JSON cache read mid-write looks
truncated (fewer entries than expected), that's a transient race with the
writer, not real data loss -- re-check a moment later. A single Gemini
`429 RESOURCE_EXHAUSTED` (prepayment credits depleted, not congestion) needs
Kaveh to add credits at ai.studio/projects before any retry will help --
don't burn a render pass retrying a billing error. For a lone straggler
clip after credits are fixed, it's cheaper to call
`label_audit.generate_description` directly on the already-rendered clip
and patch `descriptions_cache.json` + `review_sheet.csv` by hand than to
redo a full render pass for one clip (done this way 2026-09-06).

**Checking `descriptions_cache.json` success count: use the nested `describe`
field, not list-entry truthiness.** Each entry is always a dict (`{strategy,
clip_file, start_seconds, end_seconds, describe}`), so `v is not None` over the
list is always true and silently overcounts successes -- this cost a whole
monitoring cycle of wrong progress reports on Aug-30 before being caught. Count
real successes with:
```bash
python -c "import json; d=json.load(open(r'<out_dir>/candidates/descriptions_cache.json', encoding='utf-8')); missing=[v['clip_file'] for v in d if v.get('describe') is None]; print(len(d)-len(missing), '/', len(d), 'missing:', missing)"
```

**If a `pre-label` run launched via the Bash tool's `run_in_background` gets
killed with empty output before finishing (seen repeatedly on Aug-30, killed
within seconds to a few minutes, with zero stdout captured even right before
the kill) -- this is the tool's own background-task tracking, not the OS, not
Drive I/O, and not the laptop sleeping.** Confirmed by: no reboot
(`Get-CimInstance Win32_OperatingSystem | select LastBootUpTime`), no orphaned
process left behind, and it recurred even with `PYTHONUNBUFFERED=1` and zero
other Bash/PowerShell calls interleaved. Workaround: launch fully detached from
PowerShell instead, which escapes the tool's tracking entirely --
```powershell
$env:PATH = "C:\Users\Kaveh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin;" + $env:PATH
$env:PYTHONUNBUFFERED = "1"
$argList = @(
  "-u", "-m", "soccer_highlights.cli",
  "--source-dir", '"G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw"',
  "pre-label",
  "--out-dir", '"G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Tests/pre_label"',
  "--fps", "15",
  "--lrf-cache-dir", '"C:/local/lrf/<date>"'
)
Start-Process -FilePath "C:\dev\soccer-highlights\.venv\Scripts\python.exe" -ArgumentList $argList `
  -WorkingDirectory "C:\dev\soccer-highlights" `
  -RedirectStandardOutput "C:\local\lrf\pre_label_out.log" -RedirectStandardError "C:\local\lrf\pre_label_err.log" `
  -WindowStyle Hidden -PassThru
```
**Each path-bearing argument must be its own array element wrapped in embedded
double-quotes** (`'"G:/My Drive/..."'`) -- `Start-Process -ArgumentList` joins
array elements with spaces without preserving quoting otherwise, so an
unquoted path with spaces silently splits into multiple argv tokens and
argparse fails with a confusing `invalid choice` error pointing at a path
fragment. Poll progress with `Get-Process -Id <pid> -ErrorAction
SilentlyContinue` (gone = exited) and by tailing the redirected log files --
don't wrap this launch in the Bash tool's `run_in_background` at all, since
that's the thing being worked around. Use a fresh pair of `-Redirect*` log
filenames per retry round (`_out2.log`, `_out3.log`, ...) so you can tell
rounds apart.

Separately: on a same-day freshly-uploaded game, reading `.LRF`/`.MP4`
directly off the Google Drive path during detection can be extremely slow
(one Aug-30 attempt sat for 20+ min with zero output before being killed,
`ffprobe` metadata reads were fast (~1.7s) but the ffmpeg audio-decode child
hadn't even started yet) -- always do the `robocopy ... *.LRF` local-cache
step (already documented at Step 0/Step 1's `--lrf-cache-dir`) proactively for
a same-day game, not just when you actually see the `STATUS_IN_PAGE_ERROR`
symptom. `robocopy` exits with code 1 on a normal successful copy (its
convention for "files copied", not failure) -- don't read that as an error.

**On a watch-tagged game, `pre-label` also prints the score it derived from
the marks and fills in four `score_*` columns.** Two things to do with that:

- **Glance at the printed score against the real final score before
  exporting.** One missed tap silently shifts every later clip's counter. This
  is a cheap sanity check, not a blocker -- Kaveh doesn't have to remember the
  final score himself, since a dozen other players will confirm it over
  Telegram in seconds. If it disagrees, fix the `score_white`/`score_black`
  columns by hand (that is exactly why they live in the sheet) or blank them
  to skip the counter for that game.
- **Goal clips get longer.** A clip whose tap landed after the audio peak is
  stretched to cover it (capped at `marks.score_flip_cap_seconds`, 10s past
  the peak), so a 14s goal clip becomes up to ~21s -- which puts it over the
  ~15s single-run export limit. Expect `seg_render.py` to be the normal path
  for goal clips now, not the exception.
- **A decrement (Tallies' minus button) is no longer auto-resolved** (see
  SKILL.md's settled-decisions list) -- every positive press is counted as a
  real goal by default, so if any presses were corrected with a minus, the
  printed/derived score is the UPPER bound, not the true final score, until
  the specific mis-tapped press(es) are identified and removed or the
  `score_white`/`score_black` columns hand-corrected. **This bit Sep-06**:
  Kaveh flagged 2 known mis-taps for later identification, but the
  correction was never actually applied to the sheet before export, so the
  posted clips showed an inflated black score (12, not the agreed 10). Don't
  let "I'll figure out which ones later" silently ship as the displayed
  score -- either get the specific marks to exclude before export, or note
  clearly in the post that the running counter is provisional.
- **The score shown in a clip is "as of that clip's start," not the game's
  final tally** -- if the picked clips don't include one covering the very
  last goal of the game, no posted clip will ever show the true final
  score, even with perfectly correct data. Not a bug, just how the flip
  design works; mention the real final score separately (e.g. in the Step
  5b raw-footage message) if the picks don't happen to end on the last goal.
- **A tap that lands outside its own clip** (see Step 0c) makes that clip's
  score flip at a midpoint instead of visibly at the tap -- `pre-label`
  prints the same list Step 0c does, worth a glance here too since it now
  reflects the actual rendered candidates, not a pre-render estimate.

### Step 1b -- rename candidates into review order (`name-candidates`)

Run this **only once `pre-label` has fully finished** (every row scored -- the
command refuses otherwise, since rank is derived from the score and would go
stale the moment another describe retry lands).

```bash
./.venv/Scripts/python.exe -m soccer_highlights.cli \
  --source-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw" \
  name-candidates \
  --candidates-dir "G:/.../Tests/pre_label/candidates"
```

Renames `clip_002.mp4` -> `r01_s4_746a_c002.mp4`: **r**eview rank, gemini
**s**core, wall-clock time, and the canonical **c**lip id. The point is that the
folder's default A-Z sort becomes the review order (best first, chronological
within a score band) instead of burying the good clips among the 2s.

**On a watch-tagged sheet, rank is tiered, not just score-sorted** (revised
2026-09-07, `marks.review_tier`): tier 1 is a genuine goal tap
(`white_goal`/`black_goal`) confirmed near-cam by `--near-cam-team-first-half`,
or (no tap at all) `goal_this_end`; tier 2 is a `moment` tap Gemini still
rates >=3; tier 3 is any other clip scoring >=4; tier 4 is any other `moment`
tap; tier 5 is everything else by score. A `moment` tap never reaches tier 1
even if Gemini claims a goal -- see SKILL.md's settled-decisions list for the
full "taps beat Gemini" rationale. A sheet without `mark_category`/
`goal_this_end` (an older or audio-only game) falls back to the original
plain score-desc sort, unchanged.

- The `c0NN` token is what keeps every rename reversible and keeps the review
  sheet, the describe cache, `export-picks` and `telegram-post` all resolving to
  one identity. Don't strip it.
- It rewrites `clip_file` in `review_sheet.csv` **and** `descriptions_cache.json`
  in the same pass, so the cache's position-keyed check still passes and no
  Gemini call gets re-paid for.
- The time is **wall-clock, not elapsed footage time** -- `discover_chunks` sums
  durations only, so media time drifts from real time by however long the camera
  was stopped (41.5 min across a 110.3 min window on Aug-23). The tag re-anchors
  through each chunk's own filename timestamp so it matches what you remember.
- `--revert` restores canonical names and round-trips byte-identically. **Run it
  before re-running `pre-label` on an already-renamed folder** -- `pre-label`
  re-renders to canonical names and would otherwise leave both namings side by
  side and trip the cache mismatch check.
- Renaming after posting would break `.telegram_sent.json`'s double-post
  protection (it keys on `clip_file`). Order is always rename -> export -> post.

### Step 2 (optional, full-eval runs only) -- fill in verdict/notes, but DO NOT edit `review_sheet.csv` in Excel

**Skip this step for a normal quick-share run** -- see the mode note at the
top of Part 3. Only do this when the goal is building/refreshing labeled
ground truth for evals.md Part B.

Watch the clips in `Tests\pre_label\candidates\`, fill in each row's `verdict`
(same 1-5 highlight-worthiness scale Gemini uses) and `notes` columns. Use
`gemini_score` to triage which clips to watch first, but **scan the whole
sheet** -- the score is for sorting, not an automatic filter (see SKILL.md).

**Critical gotcha, already happened once**: `review_sheet.csv` has no BOM
(deliberately, so `csv.DictReader(..., encoding="utf-8")` reads the header
correctly elsewhere in the codebase) -- but that also means **Excel's default
save silently mangles the Farsi `gemini_caption` column into literal `?`
characters**, not just a display issue, real byte-level data loss on save. If you
edit in Excel:
- Work on a **copy** (e.g. `review_sheet_copy.csv`), never the original.
- When you later need captions for posting (Step 4), **always source them from
  the original, untouched `review_sheet.csv`**, not whatever copy you edited --
  `telegram-post`'s `--review-sheet` flag only needs `clip_file`/`gemini_caption`
  columns, so pointing it at the pristine original is always safe and correct
  even if verdicts live in a separate edited copy.
- If Excel mojibake becomes a real recurring annoyance: use Excel's *Data -> From
  Text/CSV* import with UTF-8 explicitly selected, rather than opening the file
  directly.

### Step 3 -- decide the final picks

No command for this -- read the ranked sheet, watch clips as needed, decide which
`clip_file`s are the real highlights. Typically the score>=4 clips plus any
score-3 clips whose `notes`/`gemini_description` make them worth including (a
near-miss, a funny moment, etc.) -- gemini_score is a triage aid, not the
final word. **On a quick-share run this is the actual decision step** (no
`verdict` column to lean on) -- Kaveh watches the higher-ranked clips
directly and picks off the ranked order + `gemini_description`/caption text,
same judgment call as Step 2 would use, just not written back to the sheet.

### Step 4 -- export share-quality clips (`export-picks`)

```bash
export PATH="/c/Users/Kaveh/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin:$PATH"
cd C:/dev/soccer-highlights
./.venv/Scripts/python.exe -m soccer_highlights.cli \
  --source-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw" \
  export-picks \
  --review-sheet "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Tests/pre_label/candidates/review_sheet.csv" \
  --clips clip_0NN.mp4,clip_0MM.mp4,... \
  --crf 30 \
  --out-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Sharable"
```
Re-encodes just the picked clips (by `clip_file`, reading start/end straight from
the sheet -- no re-detection, no index-drift risk) from the **full-res 4K `.MP4`**
(never the `.LRF`) at `export.*` settings, with `--crf` overriding the archival
default (18) **for this invocation only** -- it doesn't touch `config/default.yaml`
or persist anywhere; the plain `export` command (every detected candidate, not just
picks) still uses CRF 18 unless you pass the flag there too.

**The wall-clock time of day is burned into every exported clip** (top-left,
e.g. `Aug 30 07.46.36 AM`), added 2026-08-31. It's on by default; pass
`--no-burn-in-time` to turn it off for one invocation. Only exports get it --
`pre-label` review clips stay clean.

- The clock is anchored per source chunk's own filename timestamp, not by
  adding elapsed seconds to a session start, so it doesn't inherit the
  continuous-recording drift that makes media time run early (same
  re-anchoring `name-candidates` does -- the burned-in time and the clip's
  `746a` name tag agree, which is a free sanity check on any exported clip).
- A clip spanning a chunk boundary correctly **jumps** the clock across the
  camera-stop gap rather than interpolating through it.
- **Measured cost: none.** Paired local re-encodes came out at 80.3s (on) vs
  81.1s (off) -- the overlay is below noise, and output size moved -0.04%, so
  it doesn't change the CRF-vs-50MB calculus below. (Full export runs on this
  laptop vary ±30% run to run from Drive I/O, so don't try to read the
  overlay's cost out of an end-to-end export timing.)
**A scoreboard is burned in** on any game that has watch marks (design signed
off 2026-09-05): a dark bar top-left with a WHITE pill, the score, a BLACK
pill, and the clock past a gold rule. It reads the `score_*` columns straight
from the sheet -- so it is only as right as those columns, and Step 1's score
check is what stands between a wrong tally and a wrong score in the group.

- **No marks means no scoreboard.** With blank score columns the clock falls
  back to the plain box it has always been drawn in, so an audio-only game
  looks exactly as it did before. A scoreboard with empty pills and no
  numbers would be worse than none.
- **How it is built** (`soccer_highlights/scoreboard.py`): the chrome --
  panel, pills, gold rule -- is drawn once with Pillow and composited by
  ffmpeg `overlay`; the score and clock stay `drawtext` on top, because both
  change (the score flips, the clock ticks). Geometry is authored for a
  2560px frame and scaled by the actual output width. The chrome PNG is
  cached in the temp dir, keyed by everything that affects its pixels.
- **The board is ~37% of frame width.** If that ever needs trimming, the
  clock is the cheapest thing to pull back out of the bar.
- Inside the board the clock is time-only (`scoreboard_time_format`); the
  standalone fallback clock keeps its date. The panel is measured from the
  configured format, so changing the format resizes the panel to match --
  don't hardcode a width.

- **The score flips at the TAP, not at the goal.** Deliberate: a broadcast
  score graphic also updates a beat late, it behaves identically whether or
  not audio found a peak, and it doubles as feedback on how fast the tap was.
- If a press was too slow to cover within the cap, the flip falls back to the
  clip's midpoint -- deterministic, and dependent on neither audio nor press
  timing, which are the two things that failed in that case.
- Both `export-picks` and `seg_render.py` read the same columns, so a long
  clip rendered in segments gets the identical overlay. The flip is rebased
  per segment/per chunk-piece; if you ever see the score change in the wrong
  place, that arithmetic is the first suspect.
- Tunables live in `ExportConfig` (`burn_in_time`, `burn_in_time_format`,
  `burn_in_font_path`, `burn_in_font_size`, `burn_in_margin_px`; plus
  `burn_in_score`, `scoreboard_scale`, `scoreboard_time_format`,
  `score_home_label`/`score_away_label`, and the bold/narrow font paths the
  board draws with). Two hard
  constraints, both verified against ffmpeg 8.1.2 and both documented in the
  code: the time format **must not contain a `:`** (drawtext's
  `%{pts:gmtime:...}` splits its own arguments on colons and dies with
  `%{pts} requires at most 3 arguments`), and the font must be given as a
  **file path, never a `font=<family>` name** (fontconfig has no config file
  on this Windows box, so a family name segfaults ffmpeg outright).

**Pick CRF per clip, not once for the batch.** Telegram's Bot API `sendVideo` has
a hard **50MB** per-file limit, and that limit is *per file* -- so a single global
CRF either busts the cap on the longest clip or throws away quality on all the
short ones. Measured on the Aug-23 game: output size varies **~5x with scene
motion at a fixed CRF** (1.6-3.5 MB/s at CRF 18 across that game's clips). What
that worked out to in practice, and a reasonable starting point for a new game:

| clip duration | CRF | resulting size |
|---|---|---|
| <= 14s | 18 | 23-47MB |
| ~18s | 22 | 23-26MB |
| 40-45s | 24-30 | 25-35MB |

**Re-measure instead of extrapolating.** The usual "+6 CRF halves the bitrate"
rule of thumb was badly wrong on this footage: on a static wide-shot clip the
real factor was ~1.25x per *single* CRF step (9.8MB at CRF 30 -> 88.7MB at CRF 20
on the same clip -- a 9x swing over 10 steps). Render one clip, read the actual
size, then solve for the CRF the rest need.

**Speed: budget ~37x realtime**, not the ~10x figure in `ExportConfig`'s older
comment (that benchmark was not run against sources on the Drive path).
Re-measured 2026-08-23: 5s of footage took 188s end to end. Decode alone is a
~20x realtime floor; `preset medium` adds ~88s per 5s, `preset veryfast` only
~36s. A 10-clip / ~208s batch is therefore **~2.2 hours**. Don't run it alongside
another heavy render/encode job.

**Render to a LOCAL out-dir, then copy to `Sharable\`.** Writing a large file
straight to the Google Drive path is what stalled and killed the first Aug-23
attempts -- Drive FS warnings in the Windows event log lined up exactly with the
deaths (`Get-WinEvent -FilterHashtable @{LogName='System'; ID=42,107,1}`).

**For any clip over ~15s of footage, use `scripts/seg_render.py` instead.** At
37x realtime, ~15s is all that fits in a 10-minute command timeout, and long
unattended runs on this laptop get killed often enough that a 25-minute
single-shot render is a coin flip. `seg_render.py` splits one clip into
segments, renders only the missing ones (one per invocation by default), and
stream-copy concats when they're all present -- rerun the identical command
until it prints `DONE`:

```bash
./.venv/Scripts/python.exe .claude/skills/kaveh_soccer_highlights/scripts/seg_render.py \
  --source-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw" \
  --review-sheet "G:/.../post_sheet.csv" \
  --clip "r30_31_s2_821a_c025_c026.mp4" --crf 24 \
  --out-dir "C:/local/scratch/export" [--seg-seconds 11.0] [--max-segments 1]
```

**Always verify an export before trusting it.** A killed render leaves a large
file with no `moov` atom that no player will open. `export-picks` and
`seg_render.py` both now check with `render.is_playable()` (an `ffprobe`
duration read) rather than `size > 0`, and delete-and-re-render anything
truncated -- but if you produce clips any other way, check them yourself:
`ffprobe -v error -show_entries format=duration -of csv=p=0 clip.mp4`.

### Step 4b -- merging two adjacent candidates into one clip

When two candidates are really one moment split in half, don't hand-edit
`review_sheet.csv` (it's the ground truth for F1 work). Build a small derived
**`post_sheet.csv`** next to it with the same `clip_file`/`start_seconds`/
`end_seconds`/`gemini_caption` columns, one row per final pick, and give the
merged row the earlier clip's start and the later clip's end -- the gap between
them gets filled in automatically since the export just cuts one interval. Join
the two Farsi captions for the merged row. Then point **both** `export-picks`
and `telegram-post` at `post_sheet.csv`; both only need those columns, and the
real `review_sheet.csv` stays pristine.

### Step 5 -- post to Telegram (`telegram-post`)

**Always dry-run first:**
```bash
cd C:/dev/soccer-highlights
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m soccer_highlights.cli \
  telegram-post \
  --review-sheet "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Tests/pre_label/candidates/review_sheet.csv" \
  --clips-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Sharable" \
  --clips clip_0NN.mp4,clip_0MM.mp4,... \
  --dry-run
```
Confirms: files exist and are under the 50MB limit, captions resolve (and are
real Farsi, not `?` mojibake -- see Step 2's warning), bot credentials work
(`getMe` check). `PYTHONIOENCODING=utf-8` avoids a `UnicodeEncodeError` crash when
printing Farsi captions to a cp1252 Windows console -- cosmetic only, doesn't
affect what actually gets sent, but without it you can't read the dry-run output.

**Posted in chronological (game) order automatically, not the order `--clips`
lists them in** (fixed 2026-09-08, after Sep-06 was posted in rank/score
order and read confusingly out of game order) -- `telegram-post` sorts by
the sheet's `start_seconds` internally regardless of how you pass `--clips`.
Nothing to remember here; just pass whichever clip_files you picked, in any
order.

**Clips arrive here already carrying the burned-in time of day** -- that
happens in Step 4's export encode, not here. `telegram-post` is a pure upload
and should stay one: it never re-encodes, so there's nothing to overlay at this
stage. If a posted clip is missing the timestamp, the fix is upstream (re-export
it), not a post-processing pass.

**Then the real send**, same command minus `--dry-run`. Caption is each clip's
`gemini_caption` (Farsi) from the review sheet. Successfully-sent clips are
recorded in `<clips-dir>/.telegram_sent.json`; **a rerun skips anything already
in that file**, so it's always safe to re-run after a partial failure without
double-posting to the group.

**Upload timeouts**: `request_timeout_seconds` covers the whole upload, not just
the connect, so it has to fit the largest file the 50MB cap allows on a home
upstream link. The old 120s default died mid-upload on a 45MB clip (2026-08-23)
right after succeeding on a 24MB one; the default is now 900s. If a future
connection is slower still, raise it for one run without editing config:
`SOCCER_HL__TELEGRAM__REQUEST_TIMEOUT_SECONDS=1800`. Posting in batches of ~3
clips also keeps any single failure cheap, and the sent-file makes resuming free.

`--review-sheet` here only needs `clip_file`/`gemini_caption` -- point it at the
**original** `review_sheet.csv`, not an Excel-edited copy (Step 2's warning).

### Step 5b -- announce where the raw footage lives (`telegram-message`)

**Standard, every game (settled 2026-09-08 -- was "optional/occasional"
before, now always do this).** Kaveh uploads the full, unedited game
footage to a Google Drive folder and wants a short Farsi text message
posted to the group pointing players at it, separate from the per-clip
highlight posts. Ask him for the Drive folder's share link if you don't
have it yet -- don't skip this step for not having it, ask. Use
`telegram-message` (added 2026-08-31), not `telegram-post` -- this is a
one-off plain-text `sendMessage`, no video attachment:

```bash
cd C:/dev/soccer-highlights
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m soccer_highlights.cli \
  telegram-message \
  --text-file "<path to a UTF-8 .txt with the Farsi message>" \
  --dry-run
```
then the same command minus `--dry-run` to actually send. **Always write the
Farsi text to a UTF-8 file first and pass `--text-file`, not `--text` with
the message inline on the command line** -- passing RTL text as a raw shell
argument risks the same class of console/shell-encoding mangling as Step 2's
Excel mojibake warning, and there's no `.telegram_sent.json`-style guard
against a duplicate send here, so get the dry-run output right before
sending for real. The Aug-30 game's message (a template to adapt, not a
fixed script):

```
سلام به همه بچه‌ها! ویدیوهای کامل و خام بازی این هفته رو می‌تونید توی این پوشه گوگل درایو ببینید:
<Google Drive folder link>
```

---

## Known gotchas, quick index

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError` on ffprobe/ffmpeg | Bash tool's cached `$PATH` predates ffmpeg reinstall | Prepend the WinGet bin dir in the *same* command |
| Farsi captions show as `?` | Excel silently re-saved the CSV in a non-UTF-8 codepage | Source captions from the untouched original sheet |
| `UnicodeEncodeError` printing captions | Windows console is cp1252, not UTF-8 | `PYTHONIOENCODING=utf-8` on the command |
| Gemini `503 high demand` | Provider-side congestion, not a bug | Re-run the same command; resumable caching only retries failures |
| `telegram-post` "chat not found" | Bot never actually received a group message (privacy mode) | Send a `/command` or `@mention` in the group, then re-check `getUpdates` |
| Background job silently dies, no error | Laptop sleep, or the terminal app closing overnight | Check `Get-CimInstance Win32_OperatingSystem \| select LastBootUpTime`; design for resumability, not a one-shot fix |
| Two heavy jobs compete for CPU | Another render/API job still running | `Get-CimInstance Win32_Process \| Where-Object Name -match 'ffmpeg\|python'` before starting a heavy step |
| Render dies partway, leaves a big file | Killed mid-encode -- no `moov` atom written | Never trust `size > 0`; `render.is_playable()` / `ffprobe` it. `export-picks` + `seg_render.py` now auto-delete and re-render these |
| Long render killed repeatedly, no progress | Writing straight to the Google Drive path stalls (Drive FS events in the Windows event log) | Render to a local dir, copy to `Sharable\` after |
| Clip too long to render in one run | ~37x realtime means >15s of footage blows a 10-min timeout | `scripts/seg_render.py`, rerun until `DONE` |
| Exported clip over 50MB | One global CRF can't fit both long and short clips; size varies ~5x with motion | Per-clip CRF (Step 4 table); re-measure, don't extrapolate the +6-CRF rule |
| `telegram-post` times out mid-upload | `request_timeout_seconds` covers the whole upload | Default is now 900s; raise via `SOCCER_HL__TELEGRAM__REQUEST_TIMEOUT_SECONDS`, post in small batches |
| `pre-label` background run killed with zero output | The Bash tool's own `run_in_background` tracking, not the OS/Drive/sleep | Launch detached via PowerShell `Start-Process` instead (Step 1); poll `Get-Process -Id` + the redirected log file |
| `descriptions_cache.json` progress looks higher than it is | Each entry is always a dict, so `v is not None` over the list is always true | Check the nested field: `v.get('describe') is None` per entry |
| Export dies with `%{pts} requires at most 3 arguments` | A `:` in `export.burn_in_time_format` -- drawtext splits its own args on colons | Use `.`/`-` separators in the time format |
| ffmpeg segfaults during an export | `font=<family>` needs fontconfig, which has no config file on this Windows box | Always give `burn_in_font_path` an explicit font FILE path |
| Burned-in clock reads hours off (e.g. 8:46 AM shows as 3:46 PM) | Feeding a true UTC epoch to drawtext's `gmtime`, which then renders UTC | `slice_start_epoch` reinterprets the naive local DJI timestamp as UTC on purpose -- don't "fix" it into a real tz conversion |
| Watch tallies missing for a game, found out too late | The exports are a separate manual step on the watch and can't be recovered afterwards | Ask at Step 0b, every game, before Step 1 -- never after |
| `source` column absent or all `audio` | `--tally-csv` flags weren't passed to `pre-label` | Re-run Step 1 with them; don't read a `source`-less sheet as "the watch found nothing" |
| Clap sync reports no peak near the moment mark | Clap not detected, or skew exceeds the search window | Check the camera RTC sync actually took before trusting that game's marks |
| Marks reported "marked but not recorded" | Tapped while the camera was stopped between chunks | Nothing to render -- report them, don't let them silently vanish |
| `pre-label` detection hangs with zero output on a same-day game | Reading `.LRF`/`.MP4` straight off Drive for a freshly-uploaded game can be very slow | `robocopy` the `.LRF`s local first (`--lrf-cache-dir`) proactively, don't wait for the `STATUS_IN_PAGE_ERROR` symptom |
| A detached `pre-label` process looks dead (`Get-Process -Id <launcher_pid>` shows 0% CPU or nothing) but the log is still growing | The venv `python.exe` launcher PID is a stub; the real work runs in a CHILD process with a different PID | Check `Get-CimInstance Win32_Process \| Where-Object {$_.Name -eq 'python.exe'}` for the real worker, or just compare the log/cache file's `LastWriteTime` to now -- recent write means alive, full stop |
| A PowerShell `Where-Object {$_.CommandLine -match ...}` filter via the Bash tool's nested `-Command` string returns empty even though the process is alive | Backslash-escaping `\$_` inside the Bash tool's double-quoted `-Command` string breaks the filter silently | Don't escape `$_`; if it still misbehaves, skip the filter and just check `LastWriteTime` |
| Full 5-chunk audio decode gets OOM-killed | This laptop has only 8GB RAM, and other concurrent work (browser, other Claude sessions, other scripts) eats into the ~4GB typically free | Don't re-run `_run_detection`/`extract_audio_samples` just to recompute categories/provenance -- reconstruct from `events.json`'s cached peaks instead (no decode), cross-checking against the sheet's real `source` column before trusting it |
| MP4s won't fit on Drive/local disk for a same-day game | Full-res chunks are ~17GB each | Point `--source-dir` straight at the memory card (e.g. `D:/DCIM/DJI_001`) instead of copying to Drive -- copy just the small `.LRF` proxies there too (`robocopy ... *.LRF`) so `discover_chunks` finds matching pairs; keep `--out-dir`/tally CSV paths on Drive as usual. Card reads are local, not subject to the Google-Drive-path slowness this skill otherwise warns about |
