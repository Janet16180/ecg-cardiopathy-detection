# SSL and synthetic localization: actual GPU profile failure

18 October 2026. The actual GPU profile failed its deterministic replay gate and stopped. PyTorch
rejected `upsample_linear1d_backward_out_cuda` because that backward operation has no deterministic
implementation under the required `torch.use_deterministic_algorithms(True)` setting. No timed phase
completed, no full pilot training started, and no localization metric or model prediction was scored.
The batch, architecture, determinism settings and protocol were not changed, and no rerun was made.
Full training remains unadmitted.

The [profile supplement](ssl-synthetic-localization-profile.md) was committed as `b8888c4`; the final
supplement, implementation and focused tests were committed as `5ea0c8a` before the actual CUDA call.
The coordinator's preparation-ready message preceded execution. The command used
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=/tmp/ecg-wt-ssl-inputaudit
.venv/bin/python -m scripts.validation.profile_ssl_synthetic_pilot`.

The run acquired the nonblocking shared GPU lock and found no active compute process. It verified
actual input/cache receipts, all cache outputs, every prepared A-bank file hash, all prerequisite
source bindings, the complete 3,060-case bounded-generator admission, the retained 28-case failed v1
admission, and the independent 1,012-case preparation replay receipt/script. The actual cache contains
2,048 permitted TRAIN patients, and the prepared banks contain 500 training and 512 A-validation
cases. The positive class weight from the shared 600-step matched training schedule was 20.0. The
ordered schedule SHA-256 is `8a19b5de52a531cbcf30e693bffc2f1aa95cd81e3d83271cc1650d4db6c82bb9`.
No raw waveform or B bank was opened by this profile.

| Observed resource | Executed value |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1060 3GB; compute capability 6.1 |
| Device memory | 3,129,409,536 bytes |
| PyTorch / CUDA runtime | 2.6.0+cu124 / 12.4 |
| cuDNN / driver | 90100 / 580.178.04 |
| Precision | float32; TF32 disabled |
| Determinism | Deterministic algorithms and cuDNN enabled; benchmark false |
| CPU numerical threads | 1 |
| Whole attempt wall / process CPU | 0.945441 / 0.886829 seconds |
| Peak PyTorch allocated / reserved CUDA memory | 69,285,888 / 79,691,776 bytes |
| Completed timed phases | 0 |

Execution ran from 21:22:43.092 to 21:22:44.037 UTC on 18 October. The failure occurred in the
mandatory actual-cache two-update SSL replay before the 10-warmup/40-timed update measurements for
SSL, trainable localization, frozen-head localization or inference. Exact replay losses/final state
hashes and the post-update F guard therefore remain unverified on CUDA. The observed memory peak is
only this failed attempt; it does not establish that the complete update workload fits the device.

The local immutable receipt is `outputs/ssl_synthetic_localization/pilot_v1/profile/summary.json`,
SHA-256 `688786d37f434fde28eb820dad8f53cc8d8485a7130dab496e2ab24ec0838b02`.
It retains the exact runtime error, detected device/backend settings, source/input bindings and prior
resource charges. Post-run metadata-only verification confirmed every recorded source, prerequisite
source and input hash against current bytes. No profile checkpoint or predicted map was saved, and
no profile weight may initialize a full run. The output directory refuses overwriting.

The receipt charges 228.075479 seconds for prior input audit, real cache, all three retained generator
qualifications, actual synthetic preparation and its independent replay audit. Generator receipts
lack process CPU measurements, so their measured wall times are explicitly charged as conservative
CPU fallbacks. No per-step times, GPU/CPU full-pilot forecast or complete profile cache-budget gate
could be produced after the replay failure. The latest preparation receipt's 4 GiB gate remains
separate evidence; it does not substitute for a completed real profile.

Thirty-two focused CPU-only profile/model/data/preparation tests passed before execution, including
16 profile tests covering projection arithmetic, independent resource limits, cache and bank hash
corruption rejection, exact default model initialization, an actual head-only optimizer update that
preserves every encoder byte, overwrite refusal, retained failure receipts and replay-failure stopping.
Focused Ruff passed. These CPU checks did not establish CUDA kernel support; this actual profile
exposed that missing compatibility gate. This task did not run the repository-wide suite.

A prospective successor must resolve deterministic output interpolation, preserve the intended
native-grid semantics, pass focused replay/geometry checks and obtain a new actual GPU profile before
training. That change cannot be silently folded into this failed source-bound attempt. No model
advancement or clinical conclusion follows from this profile failure.
