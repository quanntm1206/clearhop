# Loss Refinement Design

## Evidence

The locked 4x3 screen completed without promotion. `complex_nmse_sisdr` was the
only near miss: paired SNRi improved by 6.3899 dB, SI-SDRi changed by -0.2862
dB, signed gain was positive for every seed, and STOI changed by -0.005286.
The sole failure was 0.000286 below the locked STOI floor.

## Alternatives

1. Increase the delayed SI-SDR weight slightly. Lowest risk; preserves the
   architecture, state, exporter, and all data controls.
2. Add a bounded causal gain head. More expressive, but changes the runtime
   contract and requires pumping, clipping, smoothing, and export decisions.
3. Apply a validation-derived scalar. Rejected: checkpoint-specific and unsafe
   under distribution shift or quantization.

## Decision

Run one predeclared refinement arm, `complex_nmse_sisdr_beta_0p015`. It keeps
the same normalized complex loss and steps 500-1000 warm-up, changing only the
terminal SI-SDR weight from 0.01 to 0.015. The observed difference between the
0.00 and 0.01 arms indicates this small increase should add shape margin while
retaining the gain/SNR correction. No test data may be read during refinement.

Screen seeds 17, 29, and 43 for 1,500 successful optimizer steps with the
unchanged 30,000-step scheduler. Compare against the frozen paired control rows
from `reports/generated/gain_calibration_screen.json`. Apply the exact original
promotion gates and ranking contract. Bind the refinement report to the
original screen report by SHA-256.

If refinement passes, it becomes the sole arm eligible for full 30,000-step
confirmation. If it fails, stop loss refinement. A causal gain head is then the
next mechanism, subject to the remaining 10-hour experiment budget.

### Evidence-driven amendment

Beta 0.015 improved the paired SI-SDRi delta from -0.286185 to -0.276493 dB and
the STOI delta from -0.005286 to -0.005065. It missed only STOI by 0.000065,
showing continued improvement rather than saturation. One final beta-0.02 arm
is therefore predeclared under the same three-seed protocol. No further weight
sweep is allowed: pass promotes beta 0.02; failure ends loss refinement.

## Safety And Verification

- New runs use `checkpoints/gain_calibration/refine/`; existing runs immutable.
- Dry-run starts no process and writes no artifact.
- Resume, hashes, manifest slice provenance, finite metrics, and exact seed
  coverage reuse the hardened ablation runner contracts.
- Tests cover stage validation, exact seed coverage, immutable screen binding,
  and independent gate recomputation.
