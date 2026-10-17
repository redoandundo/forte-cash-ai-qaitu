# Output analysis and product impact

Date: 2026-10-17

## Why the output exists

The system reduces the amount of video a reviewer must inspect. Its useful unit is a reviewable candidate: a clickable timecode, local visual context, an honest model score, and a clear next action. The current binary model cannot establish a specific employee action or legal violation.

## Defects found

| Previous output | Risk | Correction |
|---|---|---|
| Score rendered as a percentage | Looks like a calibrated probability, although it is only a ranking score | Render the raw score and state its meaning |
| `Проверка пройдена` for `no_violation` | Claims more than the model measured | Say that signs were not found in this recording |
| Generic “unusual action” | Can be mistaken for a validated class | Use an open set observable event type |
| One isolated JPEG | Weak temporal context | Return up to three frames around the peak |
| Cards without a player | Reviewer must search the recording manually | Add a local player, timeline markers, and seek actions |
| No rule state | Encourages an invented legal explanation | Show `Правило не сопоставлено` until a verified mapping exists |
| No explicit human action | Output stops at detection | Tell the reviewer to inspect the fragment |

## Resulting decision path

```text
MOV recording
  → model score for temporal windows
  → grouped candidate and peak time
  → before / peak / after evidence frames
  → clickable timeline and event card
  → human reviews full context
  → rule or decision is added only from verified evidence
```

## Expected impact

- Faster review: a card and marker open the exact candidate instead of requiring a full manual search.
- Lower interpretation risk: score is no longer presented as probability or proof.
- Better explainability: three frames show immediate temporal context.
- Contract safety: required API fields and statuses remain compatible.
- Honest fallback: missing event class or rule is shown as unknown rather than guessed.

## Current limits

- The model is binary and has no validated multiclass action taxonomy.
- The score is not calibrated as probability.
- The present nested OOF baseline remains modest: macro F1 `0.540`, recall `0.667`, timestamp MAE `8.48 s`.
- A negative result only says that no evaluated window crossed the threshold.
- Browser playback still depends on support for the codec inside the MOV container; evidence cards remain available when playback fails.

The next improvement with the highest product value is a reviewed event taxonomy and source safe validation. A rule catalog should be connected only after mappings to observable event types are verified.
