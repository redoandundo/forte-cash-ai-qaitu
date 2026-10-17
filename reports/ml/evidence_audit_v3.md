# Evidence audit for model v3

Date: 2026-10-17

## What the reported score means

The current `macro F1 = 0.750` is an apparent result on 8 validation sources. The model did not train on these sources, but the final threshold was selected on the same eight scores. It is useful for a hackathon baseline and is not an unbiased production estimate.

## Uncertainty and fragility

| Check | Result |
|---|---:|
| Apparent macro F1 | 0.750 |
| Fixed-threshold macro F1 bootstrap 95% interval | 0.365–1.000 |
| Leave-one-out threshold macro F1 | 0.619 |
| Leave-one-out precision / recall | 0.600 / 0.750 |
| Leave-one-out threshold range | 0.376713–0.377998 |
| Full score span | 0.016490 |
| Smallest distance from threshold | 0.000303 |

The bootstrap interval is wide because only four positive and four negative sources are present. The narrow score span and small margins mean small domain changes can flip decisions. The leave-one-out calculation audits threshold stability only; five-fold scenario-safe retraining remains required.

## Decision

- Keep v3 as the integration baseline.
- Do not describe `score` as probability or use v3 for autonomous sanctions.
- Do not claim a production quality ceiling from eight sources.
- Complete five-fold scenario-group OOF evaluation before promoting another model.

Warnings: `threshold_selected_on_evaluation_set, small_evaluation_set, narrow_score_range, decision_boundary_fragile, macro_f1_interval_crosses_0_5`.
