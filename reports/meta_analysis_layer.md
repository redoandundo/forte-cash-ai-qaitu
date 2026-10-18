# Contextual review and improvement layer

Date: 2026-10-17

## What is deployed locally

The original ONNX classifier remains the first layer. A new deterministic
second layer now:

1. summarizes the shape of window scores and temporal coverage;
2. abstains when activation is close to threshold or spread across most of a video;
3. stores run metadata, hashes, decisions, and human reviews in local SQLite;
4. compares new summaries only with human-reviewed examples from the same model version;
5. deduplicates context by video SHA-256;
6. turns reviewed error types into a ranked offline improvement plan.

It does not upload video or evidence, infer intent, change the first-layer
probability, retrain weights online, or automatically promote a model.

## First live cycle

The local store at `data/raw/meta/experience.sqlite3` contains two reviewed
videos. The database is ignored by Git.

- `X-1.mov`: first layer returned possible violation with score 0.5549 and peak
  at 6 seconds. Positive-window fraction was 1.0 and event coverage 0.925. The
  second layer abstained because activation was diffuse. Review confirmed the
  positive label but recorded a localization error against the supplied event
  near 22 seconds.
- `Х-14.mov`: first layer returned possible violation with score 0.5455.
  Positive-window fraction was 1.0 and coverage 0.996. The second layer
  abstained because the result was near threshold and diffuse. Review confirmed
  that the video is negative and recorded a false positive.

The generated plan therefore ranks:

1. mine reviewed hard-negative windows;
2. train or recalibrate temporal localization.

The initial training gate requires at least 20 unique reviewed videos, including
at least five confirmed positives and five confirmed negatives. Current state is
2/20, so the database correctly refuses to declare itself ready for challenger
training.

## Meaning of self-improvement

This implementation improves the system through an auditable loop:

```text
inference -> trace summary -> review -> reusable evaluation evidence
          -> ranked change -> offline challenger -> frozen eval -> promotion decision
```

It does not mean that the running model modifies its own weights. Online
self-training on unverified predictions would reinforce its current false
positives. Parameter count is an architecture choice for a future challenger;
it is not a reward and is not automatically increased. The reward is measured
behavior: fewer false positives, preserved recall, better localization,
calibration, and acceptable latency.

## Verification

- `pytest -q`: 23 passed;
- SQLite foreign keys and constrained review outcomes;
- video content represented by SHA-256 and basename, not an absolute path;
- repeated runs of the same video count once in contextual memory;
- automatic weight update is always disabled;
- a human-reviewed dataset gate is required before offline training.

## Next gate

Collect reviews for distinct sources, prioritizing false positives and uncertain
near-threshold cases. Once the gate is met, export reviewed clips into a new
training split, train a challenger outside the production process, and compare
it with the current champion on a frozen source-level holdout. Only a measured
improvement can change weights or architecture.
