# Evidence chain and explicit file picker design

Date: 2026-10-17
Status: implemented and browser-verified
Owner: teamlead
Target branch: `feature/teamlead-ml-inference-integration`

## Goal

Make the result explain its conclusion as a structured argument and make local
file selection obvious without removing drag-and-drop.

The service still returns the required binary API verdict, but the interface
does not present a model output as a proven legal fact. It separates what is
visible, what inference was made, why that inference follows, and what risk or
verified rule may apply.

## Chosen result structure

Each detected episode is rendered as one evidence chain with five distinct
parts:

1. **Тезис — что происходит.** A concise hypothesis about the observable event,
   for example: `Вероятно, наличные размещены вне контролируемой зоны`.
2. **Наблюдаемые действия.** Two or more concrete visual facts tied to timecodes
   and JPEG evidence. These statements describe pixels and motion only; they do
   not infer intent, identity, ownership, or a completed banking operation.
3. **Логическое обоснование.** A deterministic explanation of how the listed
   observations support the event hypothesis. It must not introduce facts that
   are absent from the observations.
4. **Почему это важно.** The operational risk created by the suspicious action,
   such as loss of cash control, incomplete traceability, access outside the
   controlled zone, or an increased reconciliation risk. Risk language is not
   an assertion that loss or misconduct actually occurred.
5. **Предполагаемое правило.** A verified local catalog entry containing the
   document name, point/section, source URL, applicability condition and a
   cautious statement such as `может соответствовать`. If no verified rule
   matches the observable event, the UI says `Нормативное основание не
   установлено` and never invents a citation.

The old labels `Требует проверки сотрудником`, `ручная проверка` and
`окончательное решение принимает проверяющий` are removed from the primary
result. A compact product disclaimer remains: the output is an automatic model
assessment of visible signs, not a final legal or disciplinary finding.

## Confidence wording

The current detector score is a ranking score, not a calibrated probability.
Therefore the UI must not display fabricated percentages such as `87%`.

Until calibration is measured, the interface uses only:

- `модельный сигнал выше порога`;
- `предполагаемое событие`;
- `обнаружены признаки возможного нарушения`.

Calibrated `низкая / средняя / высокая вероятность` bands may be added only
after a documented calibration experiment defines their thresholds.

## Data contract boundary

The jury API remains unchanged: `verdict`, `type`, `timecode_sec`,
`description`, and `evidence_frames` keep their required meanings.

The structured evidence chain is assembled locally from typed internal fields.
For the first implementation slice, the deterministic public description may
contain the thesis, observations, reasoning and risk. A legal citation is added
only when a verified rule catalog entry exists. The frontend must not parse
free-form prose to recover these fields.

The existing mock predictor can demonstrate layout and data flow, but it cannot
claim a verified event category or law. Mock mode is visibly labelled and uses
generic observation/risk copy. Real category-to-rule mapping remains disabled
until the external model bundle and verified rule catalog are supplied.

## Explicit file selection

The upload component supports two equal paths:

1. `Выбрать файл` is a real `<button type="button">`. Activating it by mouse,
   touch, Enter or Space opens the native file chooser through the hidden
   `.mov` input.
2. Dragging a file onto the dropzone continues to work.

Clicking empty dropzone space may also open the chooser, but the explicit
button is the primary accessible action. The button must not trigger two file
dialogs through nested label/click behaviour. Existing extension and 100 MB
validation is shared by both paths.

## Alternatives considered

- A free-form AI explanation was rejected because it can invent observations,
  intent or legal citations.
- A verdict-only card was rejected because it does not satisfy explainability,
  classification and rule-grounding criteria.
- A human-approval call to action was rejected for the primary flow because the
  product must produce its own automatic result. The legal disclaimer remains
  separate from the operational verdict.

## Verification

- Browser test: click `Выбрать файл`, select a `.mov`, and reach the workspace.
- Browser test: drag the same `.mov` and reach the same workspace.
- Keyboard test: focus the button and open the chooser with Enter/Space.
- Result test: the episode shows thesis, observations, reasoning and risk in
  separate labelled sections.
- Copy test: the public result contains no `требует проверки сотрудником`,
  `ручная проверка` or equivalent approval instruction.
- Safety test: no probability percentage is shown for an uncalibrated score.
- Rule test: an unmatched event has no fabricated document or article number.
- Regression: public API schema and all existing Python tests remain valid.

## Definition of done

- File selection is visibly available as a normal button and drag-and-drop.
- The model output is an automatic verdict with a transparent evidence chain.
- Observable facts, inference, risk and legal hypothesis are visually and
  semantically separated.
- Every legal reference comes from a verified local catalog; otherwise the
  interface explicitly reports that no normative basis was established.
- The system never turns a technical failure into `no_violation`.
