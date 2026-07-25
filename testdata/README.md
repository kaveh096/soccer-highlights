# Golden event set

`golden_events.json` is the ground-truth list of real "interesting moment"
timestamps for the round-2 test recording (`D:/DCIM/DJI_001`, the same
~43-minute session used throughout tuning). Generated once by
`soccer_highlights.golden.build_golden_events()` from the fully
human-labeled round-2 `output/review/*/review_sheet.csv` sheets (not
tracked in git -- see `.gitignore`):

- Every `TP`-labeled clip contributes its strategy's strongest detected
  peak time as an anchor for that real event.
- Every negative-space `FN` row contributes `start_seconds` + the exact
  offset given in `notes` (e.g. "38 second mark was a good shot...").
- Anchors within 12s of each other are treated as the same real-world
  event and collapsed to their mean time (several strategies often catch
  the same event independently, each with a slightly different peak time).

13 distinct events resulted. This lets later tuning rounds score any
candidate detection config against real ground truth without rendering
clips or asking for another human labeling pass -- see
`soccer_highlights.golden.score_intervals_against_golden`.

**Known gap:** round 1 flagged an event around ~175.5s (negatives
clip_002, "55 sec in an interesting moment") that round 2's corresponding,
wider negative-space clip (117.5-211.37s) labeled `TN` with no note. Round
2 is treated as authoritative (more recent, exact timestamps), so this
point is *not* in the golden set -- flagged here in case it was an
oversight rather than a reconsideration.
