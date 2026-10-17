# ADR-0002: Формат production-модели

Статус: accepted

Дата: 2026-10-17

## Контекст

Production pipeline не должен зависеть от training framework и внутреннего Python-класса модели.

## Решение

Предпочтительный формат — ONNX с фиксированным manifest. ONNX Runtime использует CUDA provider при наличии и CPU fallback. TorchScript разрешён только как документированный fallback, если надёжный ONNX export невозможен.

R3D-18 является обязательным baseline. X3D-XS может заменить его только после сравнимого эксперимента на том же split. RT-DETR, ByteTrack и SmolVLM не входят в обязательный первый production path.

## Последствия

- Inference реализует общий Predictor protocol.
- Все preprocessing-параметры передаются явно.
- Artifact проверяется по SHA-256.
- PyTorch/ONNX parity является обязательной частью ML handoff.
