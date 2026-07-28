# Verification reports

Generated artifacts belong under `reports/generated/` and must include the
exact command, config path, checkpoint schema, manifest fingerprints, and
environment limitations. Historical PNGs at the project root are exploratory
and do not establish benchmark claims.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\train.py --config configs\train.yaml
.venv\Scripts\python.exe scripts\evaluate.py --checkpoint checkpoints\best.pth --config configs\train.yaml --manifest manifests\v2\fold_0_test.jsonl --max-items 500 --output reports\generated\full_best_evaluation.json
.venv\Scripts\python.exe scripts\export.py --checkpoint checkpoints\best.pth --output checkpoints\full_best_export
.venv\Scripts\python.exe scripts\benchmark.py --checkpoint checkpoints\best.pth --device cuda --iterations 500 --warmup 50 --output reports\generated\full_best_benchmark.json
.venv\Scripts\python.exe scripts\verify.py --config configs\train.yaml --output reports\generated\full_verify.json
```

Training/evaluation/export require PyTorch. STOI, PESQ, ONNX, and ONNX Runtime
are optional; a report must record when they are unavailable.

Current local evidence:

- `full_training_report.md`: full 30,000-step training and evidence summary.
- `full_best_evaluation.json`: deterministic 500-item held-out evaluation with
  SI-SDR, SNR, STOI, PESQ, and the complete test-manifest fingerprint.
- `full_best_benchmark.json`: 500-iteration RTX 5060 Ti latency/memory run.
- `full_verify.json`: 42 tests pass, compileall pass, requirement-aware,
  hash-bound full-artifact
  audit pass,
  and manifest leakage
  audit pass.
- `full_best_export.ts` and `full_best_export.onnx`: stateful exports from the
  selected validation-best checkpoint.
- `verify.json`: earlier smoke gate retained for provenance.
- `production_smoke_evaluation.json`: four held-out test mixtures; one-step
  untrained model metrics only.
- `production_smoke_benchmark.json`: RTX 5060 Ti per-hop latency, realtime
  factor, parameter count, and peak CUDA memory.
- `compute_smoke_evaluation.json` and `compute_smoke_benchmark.json`: same
  artifacts after a bounded 10-step full-width optimizer run.
