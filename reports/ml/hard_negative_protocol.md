# Hard-negative follow-up, 2026-10-18

Single predefined experiment: retain the consistency recipe and add weight-1
mean softplus on the top quarter of logits in each labelled-normal training
video/view (ceil, minimum one). All normal windows retain the existing loss.
Positive-video context stays unlabelled. No outer-test false positive is fed to
the model evaluating that source. No new type/interval labels are invented.

Use the same frozen five scenario-group folds and independent per-fold
calibration (macro F1 subject to recall >= .75). Run all five folds and report
all results, including regressions. No search over weights or epochs. Cached
frozen encoder features are unchanged. Default weight zero preserves existing
training and serving behavior. Training-only change does not alter inference.

Compare against consistency candidate [TN, FP; FN, TP] = [6, 13; 5, 16],
macro F1 .52, event recall 13/22. Apply existing promotion gates unchanged.
These data have already informed development: this is not independent validation.
Keep current bundles intact unless quality and release checks pass.

Reproduction from an environment with the semantic dependencies and pinned
weights installed (`PYTHONPATH=artifacts/ml/semantic-vendor;src` on Windows):

```powershell
python scripts/ml/run_mil_experiment.py --manifest $env:QAITU_DATASET_MANIFEST --plan $env:QAITU_FROZEN_PLAN --config configs/ml/spatial_temporal_mil_hard_negative.yaml --cache artifacts/ml/cache --output artifacts/ml/hard-negative-reproduction
```

The original run used the equivalent local runner adapted from the previous
consistency experiment. The reusable runner refuses an existing output directory.

In parallel, produce a private review packet for every source, prioritising the
13 false-positive and 5 false-negative videos; include Excel points, model peaks,
full score timelines and same-scenario counterparts. Reviewer decisions must
remain pending until explicitly confirmed. Extra normal footage and independent
new-camera validation require new user-provided data.
