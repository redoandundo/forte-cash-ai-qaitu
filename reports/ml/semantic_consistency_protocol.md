# Spatial consistency follow-up

2026-10-18. Follow-up to the measured sensitivity of the first DINO/MIL
candidate. This explicitly reuses development evidence; it is not a new blind
test. The source/scenario split plan and original Excel binary labels stay fixed.

Recipe: `configs/ml/spatial_temporal_mil_consistent.yaml`. The existing 80-epoch
head is supervised on both clean and 5%-translated training windows. Both views
retain the same timestamps and point anchors. Their binary objectives are
averaged, with a weight-1 penalty on variance of per-window sigmoid predictions.
The encoder remains frozen. No calibration/outer labels enter the objective.

Augmented features have separate cache identities including the transform name
and code hash. Clean features keep their previous identity. Shape, finite-value,
video hash and timeline validation apply to every view. The one-view default
retains the previous objective and gradient, verified by a regression test.

First assess clean development calibration on fold 1, training on folds 2–4.
If the candidate is worth a full audit, use the same five group-safe outer folds.
Keep clean per-fold thresholds fixed during stress tests. Include an opposite
translation not used in training when testing sensitivity. Do not interpret
invariance of an always-positive model as successful classification.

The existing development bundle remains the fallback until the full recognition,
temporal, compatibility and runtime checks justify a replacement. Any observed
gain on these repeatedly used videos still needs external validation. Event type
classification remains dependent on confirmed type annotations, absent in Excel.
