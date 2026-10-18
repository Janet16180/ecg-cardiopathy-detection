# SSL and synthetic localization: deterministic successor GPU profile

18 October 2026. Profile v2 passed the actual CUDA numerical and resource gates at the unchanged
batch size of 16. The original failed profile remains retained. Full-model CPU output/gradient
parity passed, two independent two-update actual-cache CUDA replays matched exactly, all phase
initial encoder/head hashes matched, and the F encoder remained byte-identical after its head-only
updates. The resource forecast is below two GPU hours, four CPU hours and 4 GiB of entire-stream
outputs. No full pilot fit, localization performance score or B-profile inference ran in this task;
complete training admission remains the combined pilot runner's responsibility.

The [prospective profile-v2 policy](ssl-synthetic-localization-profile-v2.md) was committed as `9e517c4`.
The root-owned deterministic-output amendment and model were committed as `831f7bf` and `7e5ba62`.
Profile implementation/tests were committed as `ab87418` before the new GPU attempt. It reused the
original actual update functions, NumPy batching, mask/sampling streams, matched localization schedule,
input verifier, frozen-head optimizer, CUDA configuration and resource arithmetic through explicit
factory-taking measurement helpers. No receipt-pinned constructor or original source was changed.

The executed command was `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONPATH=/tmp/ecg-wt-ssl-inputaudit .venv/bin/python -m scripts.validation.profile_ssl_synthetic_pilot_v2`.
It acquired the nonblocking shared GPU lock, found no competing compute process, and retained one
numerical CPU thread, deterministic algorithms/cuDNN, benchmark false, float32 and TF32 disabled.
The device was NVIDIA GeForce GTX 1060 3GB, compute capability 6.1, 3,129,409,536 bytes total memory;
PyTorch 2.6.0+cu124, CUDA runtime 12.4, cuDNN 90100 and driver 580.178.04.

On two actual cached TRAIN inputs, successor versus original full CPU model maximum absolute
errors were 6.016344e-7 for outputs, 4.973799e-13 for input gradients, and 1.862645e-9 for parameter
gradients, within the frozen 1e-4 absolute/relative bounds. Full initial state bytes were identical.
The two CUDA replay runs each completed two masked-real SSL optimizer updates with identical loss
lists and final parameter digest `8739b24a1f2a3afc7c43d0f8cff3c55131653baa4eb893925846d0a11af5ad58`.
These are implementation/replay checks, not model performance results.

Every phase used 10 warmup plus 40 timed iterations and actual native NumPy-to-CUDA transfers on
every iteration. Inference also materialized probabilities on CPU and checked finite values, then
discarded them without scoring. All model initializations used seed 20261018; profile sampling/masks
used 20262018/20262019, and the 600-step localization schedule used 20261020. The positive class
weight from that complete training schedule was 20.0. Training inputs were only signals; masks
entered the supervised loss, and paired baselines/parameters/prior maps never entered the model.

| Phase | Mean timed seconds / batch | Full phase wall seconds | Full phase CPU seconds | Peak allocated / reserved bytes |
| --- | ---: | ---: | ---: | ---: |
| Actual-cache masked SSL update | 0.02671819 | 1.342070 | 1.337648 | 103,609,856 / 115,343,360 |
| Trainable localization update | 0.02756722 | 1.372790 | 1.374105 | 100,419,584 / 111,149,056 |
| Frozen-encoder head update | 0.01472969 | 0.740265 | 0.741546 | 80,720,896 / 94,371,840 |
| A-validation inference | 0.01495175 | 0.749765 | 0.750285 | 77,379,584 / 94,371,840 |

The model contains 45,868 parameters. All are trainable for SSL/R/S; F trains only the 396 head
parameters. Its encoder digest before/after remains
`b65ea644feca87d9e02a749b56f4996845af4416b40c29e08b0dab69595a7cf7`.
All phases share this initial encoder digest and head digest
`e4498c214a55ec4875d023813a6d168dabd055cb94123982a28092e0c29bcbb7`.
The trainable and frozen localization phases used identical ordered batch digests. Their complete
600-step schedule hash remains `8a19b5de52a531cbcf30e693bffc2f1aa95cd81e3d83271cc1650d4db6c82bb9`.

The entire new attempt ran from 21:29:49.096 to 21:29:54.627 UTC on 18 October: 5.531138 seconds wall
and 5.495604 seconds process CPU. Whole-attempt peaks were 103,609,856 allocated and 115,343,360
reserved PyTorch CUDA bytes. Profile weights and optimizer states were discarded; no checkpoint was
saved or used to initialize a full fit.

The forecast's one-pass remaining work is 75.813505 seconds, from 1,000 SSL, 1,200 trainable R/S,
600 F and 480 inference batches at the measured means. Applying the fixed 2x safety factor and
600-second reserve, while charging the original failed attempt's 0.945441 wall/0.886829 CPU seconds,
the receipt forecasts 758.098355 GPU seconds and 986.079686 CPU seconds. Prior CPU charges total
228.962308 seconds, including every earlier input/cache/generator/preparation/replay charge and the
failed profile; generator wall-time fallbacks remain explicit. The receipt's forecast snapshot
precedes its final metadata by 0.005235 seconds. Replacing the snapshot times with the entire recorded
attempt times conservatively gives 758.103590 GPU seconds and 986.084922 CPU seconds; both gates pass.
These are full-pilot forecasts, not measured full-training costs.

The stream contained 1,217,298,033 bytes before this 21,515-byte summary; the profile's conservative
100,000-byte summary reserve passes the 4,294,967,296-byte ceiling. Future training/evaluation outputs
must still satisfy the complete live budget. All actual cache and prepared-bank file hashes, original
and successor sources, independent preparation replay, successful v2 admission and failed v1 admission
were verified. The retained failed-profile receipt and every declared source/prerequisite/input hash
also matched. No raw waveform or engineering-test/B bank was opened by the profile.

The local source-bound receipt is `outputs/ssl_synthetic_localization/pilot_v1/profile_v2/summary.json`,
SHA-256 `d585e5faed5145e7e4159ab8b9132025b300b819ae6b4deef3f26330ceae81e7`.
Post-run verification confirmed every recorded new source, input and prerequisite-source hash against
current bytes. The original failed receipt remains
`688786d37f434fde28eb820dad8f53cc8d8485a7130dab496e2ab24ec0838b02`.
All artifacts remain local; no push or PR update occurred in this task.

Thirty-seven focused CPU-only tests passed before this attempt: nine new profile-v2 tests, sixteen
original profile tests and twelve model-v2 tests. They cover factory identity, CPU outputs/input and
parameter gradients, a changed-model rejection, F encoder/kernel preservation, retained failure
charges, immutable failure/overwrite checks, stop-on-parity/replay-failure behavior, original resource
and payload guards, and independent float64/endpoint interpolation checks. Focused Ruff passed.
Actual CUDA replay and timings establish this local operation's compatibility in addition to those
CPU checks. The task did not run the repository-wide suite.

This admits the profiled deterministic interpolation recipe's numerical and resource requirements.
It does not establish SSL benefit, generator clinical realism, disease localization, unused-domain
confirmation or full training success. A subsequent matched fit must reconstruct original initial
weights and optimizer state, bind these receipts, acquire the shared GPU lock, and enforce its live
resource and integrity limits.
