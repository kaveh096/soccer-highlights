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
  Tests\
    pre_label\
      candidates\
        clip_001.mp4 ... clip_NNN.mp4   review-quality renders (1280px/30fps/
                                         crf23/veryfast/stereo, from .LRF) --
                                         the SAME file a human watches, Gemini
                                         scores, and label-audit later re-checks
        review_sheet.csv                clip_file/start/end/duration/max_peak_score/
                                         verdict/notes/gemini_score/gemini_caption/
                                         gemini_description -- verdict/notes start
                                         blank, fill them in during Part 3 Step 2
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

### Step 0 -- get footage onto Drive

Copy the DJI card's `DJI_*_D.MP4` + `.LRF` files into a new
`G:\My Drive\Photos and Movies\Sunday Soccer\<date>\Raw\` folder. If Drive sync is
flaky reading freshly-uploaded `.LRF` files (seen as ffmpeg `0xC0000006
STATUS_IN_PAGE_ERROR`), use `--lrf-cache-dir` in Step 1 to redirect heavy reads to
a local copy:
```bash
robocopy "G:\...\<date>\Raw" "C:\local\lrf\<date>" *.LRF /R:5 /W:15
```

### Step 1 -- detect candidates + Gemini describe (`pre-label`)

```bash
export PATH="/c/Users/Kaveh/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin:$PATH"
cd C:/dev/soccer-highlights
./.venv/Scripts/python.exe -m soccer_highlights.cli \
  --source-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Raw" \
  pre-label \
  --out-dir "G:/My Drive/Photos and Movies/Sunday Soccer/<date>/Tests/pre_label" \
  [--lrf-cache-dir "C:/local/lrf/<date>"]
```
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
re-runs detection+render every time (not cache-aware for that part), but the
describe step IS resumable (`descriptions_cache.json`, only retries `null`
entries). If a run finishes with failures, just re-run the exact same command --
it'll skip everything already rendered/described and only retry what failed.
Expect needing 2-3 retry rounds during a sustained congestion window (seen
37->23->6->1->0 failures across rounds in practice). If a JSON cache read
mid-write looks truncated (fewer entries than expected), that's a transient race
with the writer, not real data loss -- re-check a moment later.

### Step 2 -- fill in verdict/notes, but DO NOT edit `review_sheet.csv` in Excel

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
score-3 clips whose `notes` make them worth including (a near-miss, a funny
moment, etc.) -- gemini_score is a triage aid, not the final word.

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
(never the `.LRF`) at `export.*` settings, with `--crf 30` overriding the archival
default (18) **for this invocation only** -- it doesn't touch `config/default.yaml`
or persist anywhere; the plain `export` command (every detected candidate, not just
picks) still uses CRF 18 unless you pass the flag there too. Why 30 here: Telegram's Bot API `sendVideo` has a hard **50MB**
per-file limit, and CRF 18/2560px clips ran 65MB+ even for a 17s clip -- CRF 30
was validated (2026-07-31, Jul-26 game) to bring even a 49.5s clip down to ~19MB,
comfortable margin. **This ratio is specific to that game's clip-length
distribution and this laptop's ffmpeg build** -- for a new game, it's worth
spot-checking the single longest picked clip's output size before trusting CRF 30
blindly on the full batch (export the worst case first, in the background, check
`ls -la`, then batch the rest).

This is **decode-bound and slow** (full 4K/10-bit HEVC source) -- budget several
minutes per clip (a 49.5s clip took ~7 minutes on this laptop), run in the
background (`run_in_background: true`), don't run alongside another heavy
render/encode job.

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

**Then the real send**, same command minus `--dry-run`. Caption is each clip's
`gemini_caption` (Farsi) from the review sheet. Successfully-sent clips are
recorded in `<clips-dir>/.telegram_sent.json`; **a rerun skips anything already
in that file**, so it's always safe to re-run after a partial failure without
double-posting to the group.

`--review-sheet` here only needs `clip_file`/`gemini_caption` -- point it at the
**original** `review_sheet.csv`, not an Excel-edited copy (Step 2's warning).

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
