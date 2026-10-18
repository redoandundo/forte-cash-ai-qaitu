# Scorecard: detection, classification and robustness

Date: 2026-10-18. Branch: feature/ml-model.

## Decision

Keep `baseline-audited-v4-development`. No new weights are promoted or exported.
Recognition quality did not improve in this round. The previous immutable ZIP
and its checksum remain valid. This work adds measured robustness, calibration
diagnostics and two controlled training runs. It does not establish readiness
for maximum jury points or production.

## Detection: controlled experiments

The candidate unfreezes the final two MobileNet blocks at epoch 3. Compared with
`weighted_objective.yaml`, only the unfreezing schedule changes. A second run
changes only epochs from 6 to 30. BatchNorm statistics stay frozen, AMP and
gradient accumulation remain enabled. Neither run changes inference settings.

Same frozen split: folds 2/3/4 training, fold 1 calibration (3 positives and
4 negatives); fold 0 was not scored for candidate selection. Six-epoch control
results come from the preceding recorded experiment. These are calibration
results, **not independent test metrics**.

| Run | Macro F1 | Recall | FP events/min | Event recall | Peak MAE s |
|---|---:|---:|---:|---:|---:|
| Frozen weighted control, 6 epochs | .300 | 1.000 | 2.534 | .667 | 5.333 |
| Unfreeze two blocks, 6 epochs | .300 | 1.000 | 2.534 | .667 | 5.333 |
| Same candidate, 30 epochs | .300 | 1.000 | 2.534 | .667 | 7.333 |

All seven calibration videos are positive in every run: confusion [[0,4],[0,3]].
Recall 1.0 alone therefore does not demonstrate useful discrimination.
The longer run worsens localization. Neither warrants another OOF search or
replacing the demo model. Training loss reduction is not a quality gain.

Calibration now reports class support, recall resolution, required hits and
all-positive/small-support warnings. With three positives, recall changes in
steps of 1/3, so recall >= .75 requires all three hits. These warnings do not
modify the threshold, scoring rules or public verdict.

## Robustness: paired full-video OOF stress evaluation

40 sources, five existing scenario/SHA/source-safe outer folds. Each source is
evaluated using its outer-fold checkpoint, never the final all-development
checkpoint. Threshold is calibrated once per fold on clean, separate calibration
videos and stays fixed for every perturbation. There are 200 source-condition
evaluations. Clean aggregate metrics reproduce the previous baseline.

| Condition | Macro F1 | Recall | FP events/min | Verdict changes / 40 |
|---|---:|---:|---:|---:|
| Clean | .458 | .619 | 1.033 | 0 |
| Brightness x0.70 | .486 | .762 | 1.112 | 4 |
| Brightness x1.30 | .486 | .762 | 1.112 | 4 |
| JPEG quality 35 | .405 | .857 | 1.350 | 9 |
| Rotation +7 degrees | .472 | .524 | .874 | 6 |

Exact values and per-fold metrics: [scorecard_metrics.json](scorecard_metrics.json).
Correct-to-wrong transitions: dark 1, bright 1, JPEG 4, rotation 3.
On fold 3, rotation reduces recall from .50 to .00. JPEG increases recall while
increasing false alarms; it is not an improvement to apply at inference.

Perturbations are deterministic, identical across frames of a clip, RGB-safe and
do not mutate cached frames. JPEG is applied to sampled resized frames, not
encoded video. Rotation is a synthetic stress proxy, not a real new viewpoint.
No evidence of robustness to a genuinely different camera is available.
Repeated development data and small folds limit all conclusions.

## Classification: explicit data prerequisite

No confirmed event-type taxonomy or multiclass labels are present. No new class
names, legal conclusions or multiclass scores were invented. The release stays
binary: no_violation / possible_violation. Criterion 3 remains unfulfilled for
specific event types.

Next data task, owned jointly with the teamlead/domain reviewer:

1. Approve a small set of visually distinguishable event types and definitions.
2. Review all 21 positive sources locally: confirmed event intervals, uncertainty,
   type when supportable, and whether the event list is complete.
3. Resolve the ambiguous duplicate timestamp before claiming two events.
4. Include negative examples of similar movements in the same scenes; do not
   assign negative labels to unreviewed portions of positive videos.
5. Collect permitted new-camera examples and keep entire scenarios separate.
6. Freeze revised folds before training; reserve a new external test set.

Detailed reviews and labels stay private. Type training is blocked on confirmed
labels; uncertain examples must not be relabeled from model predictions.

## Reproduction

Run from the ML checkout with the existing environment. Set QAITU_DATASET_MANIFEST
and QAITU_VIDEO_ROOT to local permitted data. Use new output folders on reruns.

```powershell
$env:PYTHONPATH = 'src'
$py = '.venv-ml/Scripts/python.exe'
$plan = 'artifacts/ml/repair-20261017/frozen_plan.json'
& $py -m fortebank_audit.ml.robustness --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan $plan --checkpoints artifacts/ml/repair-20261017/baseline --output artifacts/ml/scorecard-repro-stress
& $py -m fortebank_audit.ml.experiments --stage development --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan $plan --config configs/ml/weighted_finetune.yaml --output artifacts/ml/scorecard-repro-finetune
& $py -m fortebank_audit.ml.experiments --stage development --manifest "$env:QAITU_DATASET_MANIFEST" --data-root "$env:QAITU_VIDEO_ROOT" --plan $plan --config configs/ml/weighted_finetune_30.yaml --output artifacts/ml/scorecard-repro-finetune30
& $py -m pytest tests/ml -q
```

Checkpoints and frozen plan are private local artifacts; baseline reproduction
commands are in [ML handoff](../../docs/handoffs/ml-to-inference.md).
Detailed stress traces stay under ignored artifacts/ml/scorecard-20261018.
API, frontend, inference package and shared contracts were not changed.

## Executed checks

`python -m pytest tests/ml -q`: **61 passed in 18.78 s**. This includes nine
new tests covering deterministic/nonmutating perturbations, temporal consistency,
RGB preservation, matching source sets, unchanged thresholds, and calibration
warnings. Existing preprocessing, split, training, export and bundle tests pass.
`git diff --check` passed. No new ONNX export, API smoke or temporal shift was
performed in this round because the candidate was rejected and release scoring
was not changed; previous v4 checks remain described in the release record.
