# Experimental Qwen sequence adapter

This directory contains a QLoRA adapter for `Qwen/Qwen3-VL-2B-Instruct`.
The base model is Apache-2.0 licensed and is downloaded separately by the pinned
installer; this repository does not contain the 4.25 GB base snapshot.

**Not release-ready.** This adapter did not pass the configured quality gate.
Its whole-video classification is advisory only, never an automatic replacement
for ONNX time localization or the semantic verifier. It does not learn identity,
legal status, regulations or per-frame explanations. Explicit experimental
opt-in is required to load it through the runtime.

Training used human-provided video labels, grouped by source and related scenario.
All development videos had previously been inspected. The small diagnostic
holdout is not an independent external benchmark. Private data, captions, images
and detailed per-source outputs are not distributed here.

See `manifest.json` for hashes, exact base revision, sampling, prompt and measured
aggregate metrics. `adapter_model.safetensors` contains only trained LoRA weights;
load it on the matching base model with PEFT, never as a standalone model.

License for this adapter: Apache-2.0; see LICENSE. Based on Qwen3-VL by the Qwen
team. Upstream: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct .
