# ClearHop research comparison

ClearHop keeps three evidence classes separate: `reproduced_local` is measured
from this repository's locked manifest; `literature_only` records published
method context without importing its numbers; `blocked` means the runtime or
license-reviewed weights were not available on the benchmark machine. A blocked
row never counts as a pass or a quality ranking.

## Reproduce the receipt

```powershell
python scripts/model_comparison.py --output reports/public/model_comparison.json
```

The public receipt uses `primary_comparison` from
`manifests/v2/fold_0_test.jsonl` (500 item IDs). Every row carries identical
IDs, explicit input/native sample rates, conversion method, command/recipe,
environment fields, and a deterministic latency receipt hash. Optional STOI or
PESQ dependencies remain unavailable rather than being imputed.

## Current evidence

`ClearHop` and `Frozen historical baseline` are local receipt
adapters. Their checkpoint SHA-256 and manifest binding are copied from the
source receipts; no old checkpoint is silently re-run. The only local numeric
claims are those in `reports/public/model_comparison.json` and the linked
production/research receipts.

## External baseline registry

`configs/research_baselines.json` is the provenance lock. It records the source
URL, pinned source commit (when available), release/version, native sample
rate, license, weight SHA-256 status, and a safe reproduction recipe:

| Baseline | Source | License | Native rate | Status |
| --- | --- | --- | ---: | --- |
| DeepFilterNet3 | [repo](https://github.com/Rikorose/DeepFilterNet), [paper](https://arxiv.org/abs/2305.08227) | Apache-2.0 OR MIT | 48 kHz | blocked (runtime/weights not installed) |
| RNNoise | [repo](https://github.com/xiph/rnnoise) | BSD-3-Clause | 48 kHz | blocked (CLI not configured) |
| DTLN | [repo](https://github.com/breizhn/DTLN), [Interspeech paper](https://www.isca-archive.org/interspeech_2020/westhausen20_interspeech.html) | MIT | 16 kHz | blocked (TensorFlow runtime/weights absent) |
| WebRTC NS | [source](https://webrtc.googlesource.com/src/), [license](https://webrtc.github.io/webrtc-org/license/) | WebRTC Software License | 16 kHz | blocked (wrapper/commit not pinned) |

Official source checks (retrieved 2026-07-28): DeepFilterNet's license is dual
Apache/MIT and documents 48 kHz input; RNNoise publishes the BSD-style license;
DTLN README states pretrained models and MIT licensing; WebRTC publishes its
software license. Release assets were not downloaded, so external `weight_sha256`
is deliberately `null` with a reason. Do not cite upstream metric tables as
ClearHop measurements.

## Protocol and limitations

- Same item IDs, order, slice offsets, and clean/noisy arrays for each runnable adapter.
- Primary metrics: SI-SDR improvement and SNR improvement; STOI/PESQ only when the local dependency accepts the sample rate.
- Native-rate models are explicitly resampled with `scipy.resample_poly` (linear fallback only in minimal report-only installs); no hidden normalization.
- CPU p50/p95/p99, realtime factor, peak allocator memory, per-item timings, and latency hash are emitted for runnable adapters.
- External baselines need a pinned executable, environment capture, model weight SHA-256, and license review before they can move from `blocked` to `reproduced_local`.
