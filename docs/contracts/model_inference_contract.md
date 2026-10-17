# Контракт ML-модели и inference pipeline

Версия контракта: 1

## Назначение

Этот контракт отделяет обучение модели от production inference. ML-инженер может менять внутреннюю архитектуру, если экспорт сохраняет согласованный manifest и семантику результата.

## Artifact bundle

Обязательные файлы handoff:

```text
model.onnx или model.ts
model_manifest.json
preprocessing.json
thresholds.json
labels.json
metrics.json
SHA256SUMS.txt
docs/handoffs/ml-to-inference.md
```

Вес модели хранится вне обычного Git. Остальные небольшие текстовые artifacts могут храниться в `artifacts/manifests/<model-version>/`.

## Вход модели

Manifest обязан определить:

- имя input;
- layout, например `NCTHW`;
- полную shape или разрешённые dynamic axes;
- dtype;
- RGB/BGR;
- число кадров;
- resize и crop;
- normalization;
- window и stride;
- стратегию temporal sampling.

Pipeline не угадывает отсутствующие значения для реальной модели.

## Выход модели

Первый milestone использует один score вероятности признаков возможного нарушения.

```text
0.0 <= violation_score <= 1.0
```

Manifest обязан определить:

- имя output;
- является ли output probability или logit;
- positive label;
- рекомендованный threshold;
- правила postprocessing до threshold.

## Predictor protocol

```python
class Predictor(Protocol):
    @property
    def model_version(self) -> str: ...

    def predict_batch(
        self,
        windows: Sequence[ClipWindow],
    ) -> list[WindowPrediction]: ...
```

`WindowPrediction` содержит:

- `window_index`;
- `start_sec`;
- `end_sec`;
- `center_sec`;
- `score`.

## Валидация до запуска

Inference обязан проверить:

1. SHA-256 model artifact.
2. Schema manifest.
3. Существование связанных config-файлов.
4. Поддерживаемый runtime.
5. Input/output names.
6. Shape, dtype, layout и color.
7. Доступный ONNX Runtime provider.

Ошибка проверки является технической ошибкой и не может давать verdict `no_violation`.

## Parity

ML handoff содержит сравнение исходной и экспортированной модели на одинаковых tensors. Допуск и наблюдаемая максимальная разница фиксируются в handoff.

## Совместимость

- Добавление необязательного поля является backward-compatible.
- Изменение layout, output semantics или обязательного поля является breaking change.
- Breaking change повышает `contract_version` и требует review тимлида и inference-инженера.
