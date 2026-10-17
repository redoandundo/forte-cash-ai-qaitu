# Nested OOF report

Date: 2026-10-17

Each outer fold was evaluated once with a threshold selected on a different inner calibration fold. Scenario groups never cross train, calibration, and outer evaluation.

## Result

| Metric | Value |
|---|---:|
| Sources | 40 |
| Macro F1 | 0.540 |
| Macro F1 bootstrap 95% interval | 0.386–0.693 |
| Precision | 0.560 |
| Recall | 0.667 |
| False positives/minute | 0.874 |
| Timestamp MAE | 8.48 s |
| Within ±5 seconds | 0.524 |

## Fold stability

| Outer fold | Calibration threshold | Macro F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0 | 0.475744 | 0.619 | 0.667 | 0.500 |
| 1 | 0.368725 | 0.619 | 0.600 | 0.750 |
| 2 | 0.320051 | 0.385 | 0.625 | 1.000 |
| 3 | 0.443033 | 0.467 | 0.500 | 0.750 |
| 4 | 0.443115 | 0.365 | 0.333 | 0.250 |

## Interpretation

This is the current honest generalization estimate. The five thresholds differ because each fold has a separately trained model. A deployable final threshold still needs calibration for the single final model.
