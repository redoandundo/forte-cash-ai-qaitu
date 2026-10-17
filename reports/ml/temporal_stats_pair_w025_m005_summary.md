# Nested OOF report

Date: 2026-10-17

Each outer fold was evaluated once with a threshold selected on a different inner calibration fold. Scenario groups never cross train, calibration, and outer evaluation.

## Result

| Metric | Value |
|---|---:|
| Sources | 40 |
| Macro F1 | 0.450 |
| Macro F1 bootstrap 95% interval | 0.293–0.600 |
| Precision | 0.474 |
| Recall | 0.429 |
| False positives/minute | 0.794 |
| Timestamp MAE | 6.86 s |
| Within ±5 seconds | 0.524 |

## Fold stability

| Outer fold | Calibration threshold | Macro F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0 | 0.447508 | 0.365 | 0.333 | 0.250 |
| 1 | 0.389525 | 0.500 | 0.500 | 0.500 |
| 2 | 0.376495 | 0.385 | 0.625 | 1.000 |
| 3 | 0.540917 | 0.365 | 0.333 | 0.250 |
| 4 | 0.407699 | 0.273 | 0.000 | 0.000 |

## Interpretation

This is the current honest generalization estimate. The five thresholds differ because each fold has a separately trained model. A deployable final threshold still needs calibration for the single final model.
