# Full MobileDeepFilterNet Training Report

## Run contract

- Config: `configs/train.yaml`
- Audio: 16 kHz, 320-sample FFT, 160-sample hop, 4-second mixtures
- Schedule: 150 epochs, 200 optimizer steps per epoch, 30,000 total steps
- Model: 72,039 parameters; causal MobileOne encoder, stateful GRU, complex
  three-tap deep filter
- Runtime: PyTorch 2.13.0+cu130, NVIDIA GeForce RTX 5060 Ti
- Checkpoint schema: 2
- Selected checkpoint SHA-256:
  `0537635e2d5023e956bd0dc7a4d01987f389244dd902249dcf105f451c38d000`
- Approximate summed epoch time: 8,050.36 seconds (2 h 14 min 10 s)

The run resumed from step 1,500 after duplicate processes were removed. Resume
positioning was corrected to preserve the exact global step and OneCycle
scheduler budget. The final summary records epoch 149 and global step 30,000.

## Model selection

- Selected checkpoint: `checkpoints/best.pth`
- Selected epoch: 86 (checkpoint stores next epoch as 87)
- Selected global step: 17,400
- Best validation SI-SDR improvement: 3.3001 dB
- Last checkpoint: `checkpoints/step_30000.pth`
- Last validation SI-SDR improvement: 2.8513 dB
- Last validation SNR improvement: -0.1880 dB
- Last-10 mean validation SI-SDR improvement: 2.8980 dB
- Last-10 mean train loss: -2.8719

Training-time STOI/PESQ were disabled to keep the 150-epoch run bounded. Both
packages were installed and used in the held-out evaluation.

## Held-out evaluation

Command:

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --checkpoint checkpoints\best.pth --config configs\train.yaml --manifest manifests\v2\fold_0_test.jsonl --max-items 500 --output reports\generated\full_best_evaluation.json
```

The test manifest contains 42,269 mixtures. Metrics use the deterministic first
500 items, not the entire manifest. Test fingerprint:
`8e8e0958a96b28b0492cbc0e17be808c138e355fd65b103fabba04050e8b1cf3`.
Speaker overlap is zero for train/validation/test.

| Output | SI-SDR (dB) | SI-SDRi (dB) | SNR (dB) | SNRi (dB) | STOI | PESQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Noisy | 3.8297 | - | 4.9755 | - | 0.7798 | 1.3464 |
| Mask only | 3.4121 | -0.4176 | 0.0035 | -4.9720 | 0.7728 | 1.4107 |
| Mask + deep filter | 7.9926 | 4.1629 | -0.0078 | -4.9833 | 0.7790 | 1.5656 |

Evidence is mixed. The deep filter materially improves SI-SDR and PESQ over
noisy input. STOI is effectively unchanged. Scale-dependent SNR degrades by
about 4.98 dB, while mask-only SI-SDR also degrades. This indicates an output
gain/calibration failure mode; the checkpoint is not yet a strong production
quality result despite causal correctness and real-time throughput.

## Export and performance

- TorchScript: `checkpoints/full_best_export.ts`
- ONNX: `checkpoints/full_best_export.onnx`
- Export metadata: `checkpoints/full_best_export.json`
- Export parity evidence: `reports/generated/full_export_parity.json`
- Input contract: 11-frame encoder context plus explicit GRU hidden state;
  outputs contain only the newest mask/taps and updated hidden state
- Twenty-hop recurrent eager `forward_streaming` versus TorchScript maximum
  absolute error: `0`
- Twenty-hop recurrent eager `forward_streaming` versus ONNX maximum absolute
  errors: mask `1.02e-7`, taps `1.19e-6`, hidden state `2.38e-7`
- Export parity tolerance: `2e-4`; parity report binds checkpoint,
  TorchScript, and ONNX SHA-256 hashes
- Neural core: mean 1.3349 ms, p95 1.8845 ms, real-time capacity 7.4914x
- End-to-end streaming: mean 2.0927 ms, p95 2.7240 ms, real-time capacity
  4.7785x
- Peak CUDA allocated/reserved: 41.60/56.00 MiB

The benchmark covers 500 timed 10 ms hops after 50 warm-up hops on the local
GPU. It is machine-specific and does not establish CPU or mobile latency.

## Verification

`reports/generated/full_verify.json` records:

- compileall exit 0
- 42/42 unit tests pass
- manifest leakage audit pass
- full-artifact audit pass
- full-training contract checks pass: 30,000 steps, CUDA/GPU metadata,
  `configs/train.yaml`, test fingerprint, 500 samples across all three outputs,
  required metrics, and 500-iteration CUDA benchmark
- final checkpoint schema validates and model weights load strictly; all
  reported metric and latency summaries are finite
- eager/streaming/TorchScript parity pass
- eager/streaming/ONNX Runtime parity pass

Known warnings: the export path uses legacy TorchScript-based ONNX tracing, and
PyTorch reports tracer/deprecation warnings. The exported stateful artifacts
load successfully and match `forward_streaming` within the errors reported
above. Evaluation, export, and benchmark metadata all bind to the selected
checkpoint SHA-256.

## Next improvement goal

Prioritize output gain calibration before architectural expansion. Add a
scale-aware waveform or log-magnitude term, report gain-invariant and
scale-dependent metrics together, and run a controlled ablation against the
current checkpoint using multiple seeds. Acceptance should require positive
held-out SI-SDRi and SNRi without STOI regression, while retaining streaming
p95 below the 10 ms hop deadline.
