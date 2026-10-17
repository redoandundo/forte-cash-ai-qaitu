# Preliminary video review

## Dataset decision

- 42 label rows map to 40 unique SHA-256 sources.
- 21 unique sources are positive and 19 are negative.
- `Х-4.mov` and `Х-47.mov` are byte-identical. Both annotations are retained as candidate events on one source: 57 and 68 seconds.
- `Х-5.mov` and `Х-48.mov` are byte-identical negatives and become one source.
- Source SHA-256 is the split group, so duplicate leakage is impossible in the saved split.
- All annotated timestamps are inside their source durations; there are no blocking label conflicts.

## Visual interpretation of Х-4/Х-47

At 57 seconds a person stands behind or close to the cashier work area while cash is visible. At 68 seconds the person is seated opposite the cashier while cash handling continues. The source labels do not identify an official violation type, so both positions remain candidate binary events and no multiclass claim is made.

## Current limitations

The dataset has no official violation taxonomy, bounding boxes, or interval boundaries. A single time point does not prove that nearby unlabelled frames are negative. The first iteration therefore trains binary clip classification and measures weak temporal localization only.
