# Next learning stage

## Temporal anomaly learning

Learn from complete-video snippet bags and localize sparse abnormal events.

**Exit criteria:** MIL beats v3 on OOF AUROC or timestamp MAE without violating latency and false-positive gates.

## Ranked experiments

1. **Weakly supervised temporal MIL ranking** (`H2-mil-ranking`)
   - Mechanism: Represent every complete video as a bag of snippets, rank anomalous snippets above normal snippets, and add temporal smoothness and sparsity losses.
   - Success: OOF AUROC +0.05 or timestamp MAE -20% without macro F1 regression; stable top snippets on contrastive pairs.
   - Evidence links: 3; estimate: 2.0 GPU h

2. **Human-review active learning and embedding memory** (`H4-active-learning`)
   - Mechanism: Store reviewed hard negatives, false negatives, embeddings, observable actions, and reviewer decisions; sample low-margin and novel clips for review.
   - Success: Every cycle adds >=10 reviewed high-value clips and reduces repeated error clusters on the next OOF run.
   - Evidence links: 3; estimate: 0.2 GPU h

3. **Fuse video score with people, zone, door, and table-state signals** (`H5-structured-fusion`)
   - Mechanism: Add interpretable trajectories and state changes as a small late-fusion feature vector rather than asking one classifier to infer every rule.
   - Success: At least one rule-specific detector improves its contrastive-pair F1 by 0.10 and produces reviewable evidence.
   - Evidence links: 1; estimate: 3.0 GPU h

4. **Self-supervised pretraining on local unlabelled windows** (`H3-domain-ssl`)
   - Mechanism: Pretrain a compact video encoder with masked reconstruction or contrastive objectives on all local footage, then fine-tune the anomaly head.
   - Success: OOF AUROC +0.03 with <=2x inference latency and no external video upload.
   - Evidence links: 1; estimate: 8.0 GPU h

## Unresolved error curriculum

- `Х-1` — **false_negative**: Positive source is missed; annotation confidence is low.
  Next action: Review the label and event interval before retraining.
- `Х-15` — **false_negative**: Positive source is missed; annotation confidence is high.
  Next action: Add the event window to the false-negative curriculum and inspect motion/ROI evidence.
- `Х-17` — **false_negative**: Positive source is missed; annotation confidence is medium.
  Next action: Review the label and event interval before retraining.
- `Х-18` — **false_negative**: Positive source is missed; annotation confidence is medium.
  Next action: Review the label and event interval before retraining.
- `Х-2` — **false_negative**: Positive source is missed; annotation confidence is low.
  Next action: Review the label and event interval before retraining.
- `Х-22` — **false_negative**: Door/exit action is subtle and the source label itself has low semantic confidence.
  Next action: Review event interval and door-state evidence before using it as a strong positive.
- `Х-4/Х-47` — **false_negative**: Positive source is missed; annotation confidence is medium.
  Next action: Review the label and event interval before retraining.
- `Х-44` — **false_negative**: Positive source is missed; annotation confidence is medium.
  Next action: Review the label and event interval before retraining.
- `Х-12` — **false_positive**: High-confidence normal source is ranked as a violation inside a contrastive scenario pair.
  Next action: Mine the peak window as a hard negative and add pairwise positive-over-negative ranking loss.
- `Х-14` — **false_positive**: High-confidence normal source is ranked as a violation.
  Next action: Mine the peak window as a hard negative and inspect scene-length/background shortcuts.
- `Х-21` — **false_positive**: High-confidence normal source is ranked as a violation inside a contrastive scenario pair.
  Next action: Mine the peak window as a hard negative and add pairwise positive-over-negative ranking loss.
- `Х-23` — **false_positive**: High-confidence normal source is ranked as a violation inside a contrastive scenario pair.
  Next action: Mine the peak window as a hard negative and add pairwise positive-over-negative ranking loss.
- `Х-25` — **false_positive**: High-confidence normal source is ranked as a violation inside a contrastive scenario pair.
  Next action: Mine the peak window as a hard negative and add pairwise positive-over-negative ranking loss.
- `Х-27` — **false_positive**: High-confidence normal source is ranked as a violation inside a contrastive scenario pair.
  Next action: Mine the peak window as a hard negative and add pairwise positive-over-negative ranking loss.
- `Х-29` — **false_positive**: High-confidence normal source is ranked as a violation inside a contrastive scenario pair.
  Next action: Mine the peak window as a hard negative and add pairwise positive-over-negative ranking loss.
- `Х-31` — **false_positive**: High-confidence normal source is ranked as a violation inside a contrastive scenario pair.
  Next action: Mine the peak window as a hard negative and add pairwise positive-over-negative ranking loss.
- `Х-35` — **false_positive**: Long normal video contains motion visually similar to the paired zone-crossing event Х-34.
  Next action: Mine windows around 27 seconds as explicit hard negatives and compare zone trajectories.
- `Х-35` — **false_positive**: High-confidence normal source is ranked as a violation inside a contrastive scenario pair.
  Next action: Mine the peak window as a hard negative and add pairwise positive-over-negative ranking loss.
- `Х-37` — **false_positive**: High-confidence normal source is ranked as a violation.
  Next action: Mine the peak window as a hard negative and inspect scene-length/background shortcuts.
- `Х-39` — **false_positive**: High-confidence normal source is ranked as a violation inside a contrastive scenario pair.
  Next action: Mine the peak window as a hard negative and add pairwise positive-over-negative ranking loss.
- `Х-13` — **localization_error**: Peak is 40.0 seconds from the annotated event.
  Next action: Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.
- `Х-2` — **localization_error**: Peak is 7.0 seconds from the annotated event.
  Next action: Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.
- `Х-26` — **localization_error**: Peak is 6.0 seconds from the annotated event.
  Next action: Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.
- `Х-28` — **localization_error**: Peak is 6.0 seconds from the annotated event.
  Next action: Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.
- `Х-30` — **localization_error**: Peak is 10.0 seconds from the annotated event.
  Next action: Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.
- `Х-34` — **localization_error**: Peak is 10.0 seconds from the annotated event.
  Next action: Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.
- `Х-38` — **localization_error**: Peak is 18.0 seconds from the annotated event.
  Next action: Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.
- `Х-4/Х-47` — **localization_error**: Peak is 30.0 seconds from the annotated event.
  Next action: Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.
- `Х-40` — **localization_error**: Peak is 14.0 seconds from the annotated event.
  Next action: Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.
- `Х-44` — **localization_error**: Frame-average encoder responds to scene context but does not isolate the small personal-object action.
  Next action: Add hand/table ROI features only after object identity is reviewed.
- `Х-44` — **localization_error**: Peak is 13.0 seconds from the annotated event.
  Next action: Use full-video snippet bags with temporal sparsity/smoothness and review the event interval.

## Promotion rule

A candidate replaces the current model only when reward is positive and every guardrail passes: five scenario-safe outer folds, no threshold-selection leakage, bootstrap macro-F1 lower bound ≥ 0.50, no scenario leakage, ONNX parity, macro F1 and false-positive non-regression, and recall ≥ 0.75.
