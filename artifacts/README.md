# Model artifacts

В этой директории хранятся только небольшие текстовые manifests и связанные configs. Большие model weights не коммитятся обычным Git.

Рекомендуемая структура:

```text
artifacts/
└── manifests/
    └── <model-version>/
        ├── model_manifest.json
        ├── preprocessing.json
        ├── thresholds.json
        ├── labels.json
        ├── metrics.json
        └── SHA256SUMS.txt
```

Местоположение фактического веса задаётся локально через `MODEL_DIR` или согласованное приватное artifact storage.
