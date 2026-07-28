# ClearHop model card

## Model details

ClearHop is a causal 16 kHz speech denoiser using a MobileOne-style encoder and stateful GRU mask/deep-filter path. The selected production model uses beta `0.03`, seed `71`, checkpoint SHA-256 `1305037b59438b1679f6202762001f52f5fb5bd80d6a7ee0e184bc7af46e4789`.

## Intended use

Local enhancement of mono or automatically converted WAV speech on CPU. V1 does not support live microphone input, speaker separation, dereverberation guarantees, forensic restoration, or safety-critical communication.

## Inputs and outputs

Input is WAV. The desktop pipeline explicitly converts to mono/16 kHz. Output is a WAV plus JSON receipt containing input/output hashes, conversion metadata, checkpoint hash, elapsed time, and status.

## Evaluation

Reproduced results and hardware-bound latency appear only in `reports/public/`. Literature claims remain separate in `docs/research-comparison.md`. External baselines without an executable, license-reviewed setup remain `blocked`.

The machine-readable comparison receipt is schema v2 and binds each baseline to
`configs/research_baselines.json` (source URL, pinned commit/version, native
sample rate, license, weight hash status, command/recipe, and environment). It
also binds the registry file SHA-256, tracked manifest path/SHA-256, ordered
item-ID fingerprint, and its own canonical hash. The research-readiness receipt
has a canonical self-hash plus hashes for every embedded evidence object.
DeepFilterNet3 is `reproduced_local` on all 500 frozen comparison IDs. Its
locally measured SI-SDRi/SNRi/STOI/PESQ are `6.390235 dB`, `4.362339 dB`,
`0.856658`, and `2.018783`; exact provenance, 16→48→16 kHz conversion,
environment, latency, memory, and canonical hash are bound in
`reports/public/deepfilternet3_reproduction.json`. Third-party weights are not
redistributed.

RNNoise remains numerically `blocked`; its base-image digest, source archive
SHA-256, exact direct package versions, Dockerfile/toolchain/setup hashes are
verified, not its enhancement metrics. DTLN and WebRTC NS remain
`blocked`. Audited external coverage is tier `one_plus_recipe`, which caps
research publish readiness at `9.0/10`. A blocked baseline may be promoted only
after rerunning the same IDs with pinned dependencies and model hashes.

## Limitations

Training/evaluation corpora may not represent every language, microphone, room, or noise type. SI-SDR improvement versus the frozen baseline was not statistically significant. Third-party weights are not redistributed.

## Privacy and license

Inference is local-only with no telemetry. Project code is Apache-2.0; datasets and third-party models retain their own licenses.
