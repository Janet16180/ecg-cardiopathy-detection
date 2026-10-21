# Prospective actual GPU profile v2

18 October 2026. This successor preserves the failed original profile and changes only the output
interpolation implementation as authorized by the separate deterministic-output pilot amendment.
The original linear CUDA backward was unavailable under mandatory deterministic algorithms. Never
edit or overwrite that source-bound attempt, weaken determinism, change batch size or use its weights.

Use the new inherited `ssl_synthetic_model_v2.DenseWaveformModel`, with the same seeded encoder/head
parameter bytes, scale, native grid and optimizer/update functions. Its fixed factor-five triangular
grouped transposed convolution with duplicated endpoints implements align-corners-false linear
interpolation. Source-bind the committed model-v2 file, its parity tests and prospective amendment
before execution. Root owns that model/amendment. All other pilot, cache, bank, label, schedule, loss,
resource and no-performance-scoring rules remain those in the original profile supplement.

New factory-taking measurement/replay helpers reuse original `profile_iteration`, CUDA batching,
`frozen_optimizer`, `load_profile_inputs`, `configure_cuda`, constants and `resource_forecast`.
No monkeypatch of pinned constructors and no import of a script is allowed. Each phase starts with
seed 20261018 and identical original encoder/head bytes. Reuse profile batch/mask seeds 20262018 and
20262019, and the full 600-step matched localization schedule seed 20261020/positive class weight.
Actual batch size remains 16; every phase has 10 warmup and 40 timed iterations with actual native
NumPy-to-CUDA transfers, followed by discarded weights. F remains head-only with its encoder digest
unchanged. Inference uses only prepared A-validation signals and never saves a score map or reads B.

Before the CUDA replay, compare original and v2 full CPU models on the first two positions of the
prospective actual-cache profile batch. Require exact encoder/head initialization hashes and float32
forward, input-gradient and parameter-gradient parity (`atol=rtol=1e-4`); record every maximum absolute
error. This is numerical implementation parity, not localization performance. The committed model
unit tests independently test the isolated operator and gradient in float64 at 1e-12, including tiny
lengths, edges, constants and impulses. Require a fresh exact two-update actual-cache CUDA replay:
identical model/optimizer/RNG state, exact losses and exact final parameter digest. Any parity, replay,
unsupported deterministic kernel, nonfinite value, OOM or other known failure stops admission and
writes a new failed receipt. Never silently switch to warn-only determinism or retry this version.

Bind the original failed receipt
`outputs/ssl_synthetic_localization/pilot_v1/profile/summary.json`, SHA-256
`688786d37f434fde28eb820dad8f53cc8d8485a7130dab496e2ab24ec0838b02`, and verify all its declared
sources, prerequisite sources and inputs. Add its 0.9454413469993597 measured wall seconds to the v2
profile wall charge in the unchanged GPU forecast. Add its 0.88682919 measured CPU seconds to the
previous 228.07547849299888 seconds of preparation charges in the unchanged CPU forecast. Preserve
all explicit charge entries and measured v2 times, so this successor does not erase failure cost.
The same 2x measured-update factor, 600-second reserve, 7,200 GPU-second, 14,400 CPU-second and 4 GiB
entire-stream ceilings apply. Forecasts are not measured full-training costs.

The exclusively created destination is `outputs/ssl_synthetic_localization/pilot_v1/profile_v2`.
Refuse any existing directory or symlink; acquire the same nonblocking shared GPU lock, reject busy
compute processes, use one CPU numerical thread, deterministic CUDA float32, TF32 off, cuDNN benchmark
false and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Write the source-bound `summary.json` last, retaining
failure details and completed phase information. No raw waveform is opened and no model score,
checkpoint or record identity leaves the machine. Commit this policy and implementation/tests before
execution, then write the executed report from live receipts. Passing profile-v2 establishes resource
and numerical readiness only; complete training admission still belongs to the combined pilot runner.
