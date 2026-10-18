# Local ML / Qwen / API integration

Date: 2026-10-18. Local branch: `codex/scorecard-semantic`. No push or public server.

## Provenance and changes

Base: teamlead integration `a364740`. ML experiments imported from `ea88857`
in commit `3767c46`. Qwen inference integrated from `6933204`, preserving
the teamlead's distinct JPEG selection and first-layer progress callbacks.
Inference changes: `4b41ac1`; API presentation: `059ff2d`. Subsequent merges
`68b3a20` / `277c387` include teamlead automatic rules and browser sessions through
`0334c25`. No public service was started or changed by this work.

- Restored the missing semantic pipeline and five-role temporal context.
- Kept strict schema, timestamp, signal and model checksum validation.
- Made model/processor initialization atomic and malformed enum output repairable.
- Included both generation attempts in timing; added a 90-second generation budget.
  This is cooperative between decoding steps, not a hard CUDA/process deadline.
  Budget exhaustion is a technical failure, never a normal-video verdict.
- Retained candidates when a normal verdict contradicts an unattended-items signal.
- Exposed stable event-type labels, observations, and canonical rule citations in
  the public description. The latest teamlead RuleAwareVerifier automatically
  matches fully supported observations to reviewed NBK rules. Those matches remain
  hypotheses requiring human review, never established legal violations. Our
  optional rule-candidate prompt remains off by default.
- Added an isolated experimental NF4 configuration for a 4 GB GPU, pinned snapshot
  download/checksum utility, paired evaluation utility, and localhost launcher.

## Evidence and limitations

The full suite passed 255 tests after integration of automatic rules, plus
15 browser-unit tests. After the final upstream browser-session update, all
96 API/integration tests and 15 browser-unit tests passed again. One existing
Starlette/httpx deprecation warning remains. The accompanying aggregate report
records the real-model check separately from mocked contract tests.

The actual FastAPI TestClient probe (tree content `68b3a20`) uploaded a complete
development video. Baseline ONNX returned one event in 1.171 seconds; the same
video through ONNX + Qwen returned `done`, one retained event, and three decoded
JPEGs in 101.480 seconds (startup separately 27.074 seconds). Invalid API key
returned 401. Candidate timing came from ONNX, not an annotation. The automatic
rule wrapper was enabled; the optional rule-candidate prompt was disabled.
This does not establish rule accuracy. The later browser-session-only update
was covered by the 96 API/integration and 15 browser-unit tests above.

Aggregate results: `reports/inference/scorecard_semantic_integration.json`.
Private response/evidence artifacts stay outside Git. A single successful full
video does not supersede the failed broader latency experiment below.

The pinned 4,255,140,312-byte Qwen snapshot was downloaded locally and its SHA-256
matched the manifest. On RTX 3050 Ti Laptop 4 GB, NF4 ran a five-frame component
probe in 107.77 seconds including one JSON repair, using 2.106 GiB peak allocated
CUDA memory. Context was selected using an annotation: this is ONLY a runtime
probe, not a detection or classification evaluation. Its answer missed relevant
activity and contained contradictory signals, motivating the retention guard.
After adding the generation budget, a second component probe with the optional
reviewed-rule prompt completed in 60.885 seconds, with 2.484 GiB peak allocated
CUDA memory. This also used annotation-selected context and is not an accuracy test.

A paired eight-video development evaluation with reviewed rules enabled was
aborted during the first video after its semantic stage exceeded 300 seconds.
No complete paired accuracy metrics exist for this run. NF4 is NOT promoted;
the manifest remains experimental/unvalidated. Full precision on an 8 GB GPU
was reported by the inference teammate, not reproduced on this machine.

All 40 unique project videos have already been used for development. They cannot
be presented as an independent test. Binary labels and point timestamps do not
establish event-category accuracy. Qwen only sees ONNX candidates and cannot
recover events missed by the first layer. Existing hard-negative experiments
increased false positives and were not promoted. No maximum-score claim is made.

## Local reproduction

Use a dedicated Python environment. The machine's original ML environment was
not modified: experimental packages are in ignored `artifacts/qwen-vendor`.
For the tested Windows driver, install Torch 2.6.0 / torchvision 0.21.0 from the
official CUDA 11.8 index, then `requirements-qwen-nf4.txt` and API dependencies.
Full precision uses `requirements-vlm.txt` and the original verifier manifest.

Download only public model assets:

```powershell
python scripts/inference/download_qwen_snapshot.py --output models/pretrained/qwen3-vl-2b-instruct
```

For this local workspace, the launcher accepts explicit interpreter and bundle
paths; set `FORTEBANK_API_KEY` privately first. The following requires a configured
environment and is an experimental local check, not a validated contest release:

```powershell
./scripts/run_semantic_local.ps1 -Python '<environment>/Scripts/python.exe' `
  -Bundle '../ml-repair/artifacts/ml/bundles/baseline-audited-v4' `
  -QwenModel 'models/pretrained/qwen3-vl-2b-instruct' `
  -QwenManifest 'configs/inference/qwen_nf4_manifest.json' -AllowDevelopmentModel
```

Add `-ReviewedRules` only for evaluation of the optional rule prompt. The launcher
binds to `127.0.0.1`; it does not create a tunnel or publish anything. Weights,
video-level outputs, frames, labels, and dependency caches remain ignored.

## Checking incoming Qwen updates

```powershell
python scripts/inference/check_qwen_update.py --fetch `
  --model models/pretrained/qwen3-vl-2b-instruct `
  --output artifacts/qwen-update-check.json
```

This reads the three remote branch heads and checks local weights against the
manifest at the exact fetched inference commit. It never merges code, overwrites
weights, or mutates a running model. A checksum mismatch or unsupported layout
blocks validation. Training should publish a completed immutable snapshot with
its manifest; do not overwrite a checkpoint in use. Identity validation is not
quality validation. Re-run contract tests and paired evaluation for a new snapshot.
At the latest check Qwen remained `6933204`; local weights matched its SHA-256.

## Scorecard next gate

Detection (25) needs a genuinely held-out, scenario-separated comparison with
acceptable false alarms; localization/evidence (15) has timestamp and JPEG checks
but still needs interval annotations. Classification (15) needs reviewed event
labels. Rules (15) now have a canonical optional integration but require measured
applicability. Robustness (10), product (10), and technical implementation (10)
have integration coverage; real full-video Qwen latency on the target GPU remains
a release blocker. Run the paired evaluator on the intended 8 GB+ deployment GPU
before enabling semantic inference in the public demo.
