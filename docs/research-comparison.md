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
environment fields, and a deterministic latency receipt hash. Unavailable
optional metrics remain unavailable rather than being imputed.

The comparison receipt also embeds SHA-256 bindings for
`configs/research_baselines.json` and the tracked manifest, plus a fingerprint
of the ordered 500 item IDs. `scripts/verify_public_research.py` recomputes all
three from repository files and fails closed on missing or drifted inputs. The
research-readiness receipt separately carries a canonical self-hash and one
canonical hash per embedded evidence object, so changing a numeric result
without regeneration fails verification.

## Current evidence

`ClearHop` and `Frozen historical baseline` are local receipt adapters. Their
checkpoint SHA-256 and manifest binding are copied from source receipts; no old
checkpoint is silently re-run. DeepFilterNet3 was independently executed on
the same frozen mixtures. The only local numeric claims are those in
`reports/public/model_comparison.json` and its hash-bound source receipts.

| Model | Evidence | SI-SDRi | SNRi | STOI | PESQ |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepFilterNet3 | `reproduced_local`, 500 items | 6.390235 dB | 4.362339 dB | 0.856658 | 2.018783 |

The DeepFilterNet3 run used source commit
`d375b2d8309e0935d165700c91da9de862a99c31`, archive SHA-256
`49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2`,
and checkpoint SHA-256
`23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003`.
The receipt reports CPU latency p50/p95/p99 of `174.9839/218.6288/248.4983 ms`,
RTF p95 `0.054657`, and RSS p95 `602.3008 MB`. Latency covers 48 kHz model
inference; explicit 16→48→16 kHz resampling is excluded. See
[`reports/public/deepfilternet3_reproduction.json`](../reports/public/deepfilternet3_reproduction.json).

## External baseline registry

`configs/research_baselines.json` is the provenance lock. It records the source
URL, pinned source commit (when available), release/version, native sample
rate, license, weight SHA-256 status, and a safe reproduction recipe:

| Baseline | Source | License | Native rate | Status |
| --- | --- | --- | ---: | --- |
| DeepFilterNet3 | [repo](https://github.com/Rikorose/DeepFilterNet), [paper](https://arxiv.org/abs/2305.08227) | Apache-2.0 OR MIT | 48 kHz | `reproduced_local` (pinned, hash-bound receipt) |
| RNNoise | [repo](https://github.com/xiph/rnnoise) | BSD-3-Clause | 48 kHz | `blocked` (digest/source/toolchain recipe verified; no metrics) |
| DTLN | [repo](https://github.com/breizhn/DTLN), [Interspeech paper](https://www.isca-archive.org/interspeech_2020/westhausen20_interspeech.html) | MIT | 16 kHz | blocked (TensorFlow runtime/weights absent) |
| WebRTC NS | [source](https://webrtc.googlesource.com/src/), [license](https://webrtc.github.io/webrtc-org/license/) | WebRTC Software License | 16 kHz | blocked (wrapper/commit not pinned) |

Official source checks (retrieved 2026-07-28): DeepFilterNet's license is dual
Apache/MIT and documents 48 kHz input; RNNoise publishes the BSD-style license;
DTLN README states pretrained models and MIT licensing; WebRTC publishes its
software license. DeepFilterNet3 weights were downloaded outside Git and are
identified by SHA-256; they are not redistributed. Other unavailable external
weights retain `null` hashes with explicit reasons. Do not cite upstream metric
tables as ClearHop measurements.

## External coverage

The audited coverage tier is `one_plus_recipe`: DeepFilterNet3 is the single
reproduced external baseline; RNNoise supplies one pinned, hash-verified blocker
recipe. That recipe uses an immutable Ubuntu Linux/amd64 digest, a commit archive
with verified SHA-256, exact direct package versions checked against `apt-cache`
before install, and hash-bound Dockerfile/toolchain/setup files. Any unavailable
version blocks rather than silently upgrading. DTLN and WebRTC NS remain
blocked. This is eligible public evidence but not complete external coverage
and not a `10/10` research claim.

## Protocol and limitations

- Same item IDs, order, slice offsets, and clean/noisy arrays for each runnable adapter.
- Primary metrics: SI-SDR improvement and SNR improvement; STOI/PESQ only when the local dependency accepts the sample rate.
- Native-rate models are explicitly resampled with `scipy.resample_poly` (linear fallback only in minimal report-only installs); no hidden normalization.
- CPU p50/p95/p99, realtime factor, peak allocator memory, per-item timings, and latency hash are emitted for runnable adapters.
- Blocked baselines need a pinned executable, environment capture, model weight SHA-256, and license review before they can move to `reproduced_local`.
