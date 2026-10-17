# Nested OOF report

Date: 2026-10-17

Each outer fold was evaluated once with a threshold selected on a different inner calibration fold. Scenario groups never cross train, calibration, and outer evaluation.

## Result

| Metric | Value |
|---|---:|
| Sources | 40 |
| Macro F1 | 0.479 |
| Macro F1 bootstrap 95% interval | 0.324–0.627 |
| Precision | 0.519 |
| Recall | 0.667 |
| False positives/minute | 1.033 |
| Timestamp MAE | 5.81 s |
| Within ±5 seconds | 0.619 |

## Fold stability

| Outer fold | Calibration threshold | Macro F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0 | 0.385138 | 0.564 | 1.000 | 0.250 |
| 1 | 0.379779 | 0.564 | 0.571 | 1.000 |
| 2 | 0.414906 | 0.385 | 0.625 | 1.000 |
| 3 | 0.438372 | 0.365 | 0.400 | 0.500 |
| 4 | 0.470206 | 0.200 | 0.333 | 0.500 |

## Interpretation

This is the current honest generalization estimate. The five thresholds differ because each fold has a separately trained model. A deployable final threshold still needs calibration for the single final model.
