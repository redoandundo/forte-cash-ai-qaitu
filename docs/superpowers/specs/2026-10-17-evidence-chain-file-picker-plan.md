# Evidence chain and file picker implementation plan

Date: 2026-10-17
Design: `2026-10-17-evidence-chain-file-picker-design.md`
Status: completed

## Step 1: Runtime evidence-chain contract

- Extend the internal runtime result with one structured explanation per event.
- Populate thesis, observations, reasoning, operational risk and an explicit
  unmatched-rule state from existing temporal event data.
- Remove human-approval wording from the public event description and summary.
- Add focused runtime tests before changing the frontend.

## Step 2: Internal progress endpoint

- Add typed Pydantic models for the evidence chain and rule reference.
- Store the structured details with the completed job.
- Return them only from the authenticated internal progress endpoint so the
  jury API schema remains unchanged.
- Extend API integration tests for successful and failed jobs.

## Step 3: Explicit file picker

- Replace the label-like visual control with a real button.
- Wire button activation to the hidden `.mov` input.
- Preserve empty-dropzone click and drag-and-drop without double dialogs.
- Keep the existing validation path shared by every selection method.

## Step 4: Evidence-chain UI

- Render labelled thesis, observations, reasoning, risk and rule sections in
  every event card.
- Remove manual-review calls to action from the primary result.
- Render `Нормативное основание не установлено` until a verified local catalog
  supplies an exact match.
- Preserve timecode navigation, JPEG evidence and failure semantics.

## Step 5: Verification and handoff

- Run API/inference tests, the full Python suite, JavaScript syntax check,
  dependency check and `git diff --check`.
- Browser-test button selection, drag-and-drop, keyboard activation, result
  rendering and timecode seeking on a local MOV.
- Confirm uploads/evidence are deleted and commit implementation separately
  from this plan.
