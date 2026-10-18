# Spatial-temporal MIL experiment

Recipe frozen before five-fold scoring, 2026-10-18. This is a development
experiment, not a production replacement or an independent external test.

The supplied archive contains 42 videos but only 40 distinct video byte streams,
21 positive and 19 negative, in 27 scenario groups. The Excel workbook has one
sheet (`labels`), 42 rows and three columns: filename, timestamp, binary verdict.
It contains no event taxonomy. Existing duplicate-aware scenario folds are kept.

The candidate retains local spatial information (DINOv2 class token and a 2x2
grid of patch features) and the order of eight frames over six seconds. Its
trainable head compares the first and last state, average state and adjacent
frame changes. The public encoder is frozen; only the small temporal head learns
from project videos. Dependency and encoder revisions are pinned separately.

Training uses every window of negative videos, a video-peak objective and the
annotated points of positive videos. Unlabelled positive-video windows are not
assigned negative labels. Matched positive/negative training scenarios contribute
a ranking loss; no calibration or held-out source enters this loss. Frozen
encoder features may be cached by video hash, encoder hash and preprocessing
fingerprint. No labels enter feature extraction.

Config: `configs/ml/spatial_temporal_mil.yaml`, 80 fixed epochs. Development used
training folds 2–4 and calibration fold 1. The five-fold audit trains on three
folds, calibrates on the next fold and scores the remaining fold, keeping whole
scenarios and duplicates together. Threshold selection, smoothing, peak matching
and promotion gates are unchanged. Report full-video precision, recall, macro
F1, confusion matrix, false alarms/minute and temporal matching. Do not tune the
recipe from the outer-fold results and call those results a fresh test.

One additional development-only ablation, `spatial_temporal_mil_20.yaml`, reduces
the fixed schedule to 20 epochs. Motivation: the 80-epoch training loss approaches
zero while the development calibration threshold collapses to 0.000626 and AUROC
is 0.5. All other settings stay fixed. This is a separately recorded candidate;
the 80-epoch outer scores are not used to select its hyperparameters. Any later
OOF results still belong to the repeatedly used development dataset.

The score is binary evidence for possible violation, not an event type or a
calibrated probability. Proposed visual action descriptions and contact sheets
stay in ignored local artifacts. Types require reviewed definitions and labelled
examples in both positive and negative videos before training and evaluating a
type classifier. Seeing a bag, an open cabinet or a person at a door alone does
not establish a violated rule or an access authorization.

References: [official DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md),
[pinned encoder](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056),
[video-level multiple-instance supervision](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html).
