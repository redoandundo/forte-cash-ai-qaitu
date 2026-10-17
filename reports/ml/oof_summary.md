> **Diagnostic only.** Independently trained fold scores are not on a common calibrated scale. Use `nested_oof_summary.md` for the promotion decision.

# Raw pooled OOF diagnostic

Date: 2026-10-17

## What the reported score means

The current `macro F1 = 0.495` is an apparent result on 40 validation sources. The model did not train on these sources, but the final threshold was selected on the same eight scores. It is useful for a hackathon baseline and is not an unbiased production estimate.

## Uncertainty and fragility

| Check | Result |
|---|---:|
| Apparent macro F1 | 0.495 |
| Fixed-threshold macro F1 bootstrap 95% interval | 0.342–0.646 |
| Leave-one-out threshold macro F1 | 0.248 |
| Leave-one-out precision / recall | 0.286 / 0.286 |
| Leave-one-out threshold range | 0.381959–0.425563 |
| Full score span | 0.088477 |
| Smallest distance from threshold | 0.003200 |

The bootstrap interval is wide because only four positive and four negative sources are present. The narrow score span and small margins mean small domain changes can flip decisions. The leave-one-out calculation audits threshold stability only; five-fold scenario-safe retraining remains required.

## Decision

- Keep v3 as the integration baseline.
- Do not describe `score` as probability or use v3 for autonomous sanctions.
- Do not claim a production quality ceiling from eight sources.
- Complete five-fold scenario-group OOF evaluation before promoting another model.

Warnings: `narrow_score_range, decision_boundary_fragile, macro_f1_interval_crosses_0_5`.

## Fold results at the pooled threshold

| Fold | Sources | Macro F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0 | 8 | 0.333 | 0.500 | 1.000 |
| 1 | 8 | 0.564 | 1.000 | 0.250 |
| 2 | 8 | 0.273 | 0.000 | 0.000 |
| 3 | 8 | 0.467 | 0.500 | 0.750 |
| 4 | 8 | 0.333 | 0.500 | 1.000 |
