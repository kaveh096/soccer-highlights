# 2026-10-18 -- Oct-18 game + center-cam / action-crop / slow-mo features

Documentation only (Kaveh: no review/approval needed). Kaveh = PM, Claude = TL.

## Game facts

- Source: `D:\DCIM\DJI_001` (card), 4K 3840x2160 @ 119.88fps HEVC `.MP4` + 1280x720 @ 29.97fps `.LRF`.
- Chunks: 0001 (3.7s stub, 09:51:24), 0002-0004 continuous 09:51:48 -> ~10:33:45, halftime gap 779s,
  0005-0006 10:46:44 -> ~11:16:08. Battery died ~11:16; game ran to ~11:28.
- Final score 6-3 black. **Camera at center field** (side of a blacktop court), not behind a goal.
- Tallies (originals untouched in the date folder; stable-name copies in `Raw\`):
  - white: 2 decrements, each immediately after a press (09:53:40 -> minus 19s later; 10:29:33 -> minus 6s
    later). Kaveh confirmed both are undos -> those 4 rows dropped from `Raw\tally_white_goal.csv`. White = 3.
  - black: 6 clean presses.
  - moment: 2 presses (10:08:31, 10:18:12). **No kickoff clap/sync tap this game** -> clock skew unmeasured;
    8/13 taps snapped to an audio peak, which is reasonable indirect evidence the RTC sync held.
- Unrecorded (battery): black 11:27:07, white 11:27:34. Recorded score ends at **white 2 - black 5**,
  so the last clip's scoreboard reads 2-5, not 3-6.

## Features (decisions)

### 1. `pre-label --camera-position {end,center}` (default `end`)
- `center`: Gemini describe prompt's `goal_this_end` wording becomes "goal clearly visible at either end"
  (JSON key unchanged so sheet/code are untouched); every genuine goal tap is tier 1; `infer-camera` skipped.
- Cache entries record the prompt variant so an end-cam cache is never reused for a center-cam run.

### 2. `pre-label --no-sync-clap`
- Did not exist. Without it the first `moment` tap is treated as the kickoff clap and loses its review boost.

### 3. Action region (`action_region`: left|center|right) + crop at export
- Source: an extra field on the existing Gemini describe call (both camera modes), ordered LAST in the
  schema (after score/caption) so it can't steer the score. No extra API calls.
- Rejected alternatives: ball tracking (ball ~10-20px at 4K, blur/occlusion, no GPU; wrong crop loses the
  goal = recall regression), auto-pan (maybe v2), zoom only on slow-mo replay (Kaveh: no), separate
  single-frame Claude call before render (single frame often lands after the shot; overlay-before-Gemini
  would contaminate scoring).
- Crop: 2560x1440 window (1.5x) of the 3840x2160 source = today's share width, no upscale. Horizontal
  position by region, vertical position fixed/configurable (action band sits mid-frame in this setup).
  `full` = no crop. Applies to the whole exported clip and its slow-mo replay.

### 4. Slow-mo center dot + slow-mo replay
- `slomo_center_s` (clip-relative seconds) = audio peak if one exists, else tap - 3s.
- Shown as ONE small colored dot in a corner of the review clip for ~20 frames (Kaveh rejected two dots
  for peak vs tap -- confusing).
- Export: replay appended after the live-speed clip, default window [center-2s, center+1s] at 0.25x
  (120fps source -> true 4x slow-mo at 30fps, no interpolation) = 12s, muted, same crop, scoreboard on.

- 2026-10-19 (review pass 2): REPLAY badge added under the board for the replay only (dark panel, flashing
  gold dot, "REPLAY", speed; `ExportConfig.replay_badge`). R01's replay re-set to `slomo=27.5-30.5`: the
  default [31-2, 31+1] started after the strike (28.0) because Kaveh's mark lags the action by ~3s.
  Four replay clips (r01, r08, r09, r14) re-exported;

### 5. Review overlay pass (new ordering)
- Render clean review clips -> Gemini describe (+region) -> cheap re-encode of the 1280px review clips
  burning in the crop box outline + slow-mo dot. Gemini never sees overlays. ~10-15 min for ~55 clips.

### 6. Structured pick prompt (in the skill; printed by Claude when candidates are ready)
One line per pick, rank first, optional key=value tokens:

    r03
    r05 crop=left
    r07 slomo
    r07 slomo=6.5-9
    r09 slomo=4-7@0.5 crop=full

- `crop=left|center|right|full` overrides the sheet's `action_region`.
- `slomo` = default window around the dot; `slomo=A-B` = seconds on the review clip's own timeline;
  `@S` = speed (default 0.25).

## Execution order
1. Tally cleanup (done). `ingest-marks` run (done; output above).
2. Build 1, 2, 3 (Gemini field + sheet columns), 4 (center calc + dot), 5 -- needed before `pre-label`.
   In parallel (worktree): export-side crop + slow-mo replay + pick-line parser.
3. Smoke-test on 2-3 Oct-18 clips, then full `pre-label` from the card with
   `--camera-position center --no-sync-clap` + the three `--tally-csv`.
4. Overlay pass + `name-candidates`; Claude prints the pick format. Drive copy (robocopy) during review.
5. Export picks (crop + slow-mo), post (chronological), announcement message.
6. Update skill docs. Commit only when Kaveh asks.
