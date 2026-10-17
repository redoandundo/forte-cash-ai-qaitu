# Nested OOF report

Date: 2026-10-17

Each outer fold was evaluated once with a threshold selected on a different inner calibration fold. Scenario groups never cross train, calibration, and outer evaluation.

## Result

| Metric | Value |
|---|---:|
| Sources | 40 |
| Macro F1 | 0.495 |
| Macro F1 bootstrap 95% interval | 0.342–0.646 |
| Precision | 0.533 |
| Recall | 0.381 |
| False positives/minute | 0.556 |
| Timestamp MAE | 5.29 s |
| Within ±5 seconds | 0.667 |

## Fold stability

| Outer fold | Calibration threshold | Macro F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0 | 0.283769 | 0.467 | 0.500 | 0.250 |
| 1 | 0.263461 | 0.619 | 0.600 | 0.750 |
| 2 | 0.329948 | 0.619 | 0.750 | 0.600 |
| 3 | 0.355219 | 0.467 | 0.500 | 0.250 |
| 4 | 0.314609 | 0.200 | 0.000 | 0.000 |

## Interpretation

This is the current honest generalization estimate. The five thresholds differ because each fold has a separately trained model. A deployable final threshold still needs calibration for the single final model.
