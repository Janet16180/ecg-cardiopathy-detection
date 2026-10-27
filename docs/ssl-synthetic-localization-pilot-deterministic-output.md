# Prospective deterministic output implementation for the original SSL pilot

18 October 2026, before full training or any localization score. The original batch-16 GPU profile
stopped in its initial replay: PyTorch 2.6.0+cu124 raised an error because linear interpolation backward
on CUDA has no deterministic implementation under the required deterministic-algorithm setting.
The attempt used 0.945 seconds wall/0.887 seconds CPU; no timed phase or full fit ran. Its receipt and
[failed report](ssl-synthetic-localization-profile-results.md) remain unchanged.

The original [pilot](ssl-synthetic-localization-pilot.md), bounded-generator amendment, patient identities,
initial encoder/head parameters, native/scoring grids, objective, schedules, seeds, batch size and
advancement rule remain fixed. A new `ssl_synthetic_model_v2.DenseWaveformModel` changes only how the
fixed factor-five linear output interpolation is evaluated. It must implement the same real-valued
linear map as `interpolate(mode='linear', align_corners=False)`; floating-point summation order need
not reproduce the failed operator bit-for-bit. No trainable parameter or extra input is introduced.

For latent length L, concatenate one copy of each endpoint onto its respective side. Apply a fixed
per-channel grouped transposed convolution with stride 5, padding 2 and the triangular nine-tap kernel
`[0.2,0.4,0.6,0.8,1,0.8,0.6,0.4,0.2]`; crop five samples from each end. Latent center i maps to output
center 5i+2. The endpoint copies implement align-corners-false boundary replication. The kernel is a
nonpersistent buffer; the encoder/head state dictionary and initialization RNG consumption are unchanged.
The new operation stays under deterministic-algorithm enforcement; warnings-only fallback is forbidden.

Before another actual profile, tests compare CPU float64 forward values and input gradients to the
original interpolation across lengths, ramps, constants, impulses and arbitrary gradients; tolerances
are 1e-12. Native 500-to-2,500 float32 values/gradients must agree within 1e-4 absolute/relative bounds.
The true CUDA two-update masked-real replay must then pass exactly using the successor, and all four
actual-path timings and original resource gates must pass. Original failed-profile costs are carried
forward. If the successor fails, retain it and stop before any full training.

This is a prospective numerical-implementation amendment, not a tuned architecture or model result.
The [PyTorch deterministic-algorithm documentation](https://docs.pytorch.org/docs/2.6/generated/torch.use_deterministic_algorithms.html)
describes the enforcement mechanism; actual compatibility is established only by the new local profile.
