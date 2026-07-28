# CPU Production Contract

Target: Windows desktop, CPU-only, one 10 ms hop per stream.

## Gate

- `streaming_end_to_end.p95_ms < 10`
- `onnx_core.p95_ms < 10`
- `device == cpu`
- benchmark iterations `>= 5000`
- single-core affinity and one Torch thread recorded in the receipt
- CPU soak pass: finite output, reset isolation, invalid-input rejection, bounded RSS
- export parity and runtime structure pass
- mono 16 kHz WAV CLI smoke preserves sample count and binds output hash
- bundle hashes match the selected checkpoint/export/config

## Current Evidence

- selected checkpoint: `checkpoints/production/best.pth`
- CPU benchmark: 5,000 hops, streaming p95 `3.962 ms`, ONNX p95 `2.148 ms`
- two-hour soak: `7,200 s`, `2,292,681` hops, zero failures, bounded RSS
- production verifier: pass
- deployable bundle: `artifacts/cpu_bundle`

Run the full benchmark:

```powershell
.venv\Scripts\python.exe scripts\benchmark_cpu.py --checkpoint checkpoints\production\best.pth --onnx checkpoints\production\export.onnx --iterations 5000 --warmup 500 --threshold-ms 10 --affinity-cpu 0 --output reports\generated\cpu_benchmark.json
```

Run the production soak:

```powershell
.venv\Scripts\python.exe scripts\soak_cpu.py --checkpoint checkpoints\production\best.pth --seconds 7200 --output reports\generated\cpu_soak.json
```

Verify all production evidence:

```powershell
.venv\Scripts\python.exe scripts\verify.py --production-readiness --output reports\generated\production_readiness_verify.json
```

Missing dependencies or failed gates remain `blocked`/`fail`; they are never converted to pass.
