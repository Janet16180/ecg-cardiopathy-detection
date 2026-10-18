# Actual GPU resource profile supplement

18 October 2026. Freeze this supplement and profile implementation before any GPU operation.
This is resource admission for the original pilot and its bounded-generator amendment. Profile
weights and optimizer states are disposable; no profile checkpoint may initialize the full pilot.
There is no AP, accuracy, threshold selection or evaluation-profile inference in this task.

Require successful source-bound real cache, complete bounded-generator admission of 3,060 cases,
retained failed-v1 receipt, and actual A-only training/validation bank preparation. Verify every
receipt's declared input/source/output hashes and explicitly verify every prepared bank-file hash.
Bind current pilot/configuration/amendment, model/data/preparation/profile code, shared GPU helper,
all prerequisite receipts and dependency files. Load cached real TRAIN signals and prepared A banks;
only `signals` is a network input, with training `support` consumed solely by the loss. Do not open
raw waveforms or any B bank. The real cache stays in mV; the model applies its frozen 2 mV entry scale.

Use the nonblocking shared `gpu_lock('cuda', blocking=False)` and reject an existing compute process
before allocating profile models. Never stop or preempt another process. Use one CPU numerical thread,
PyTorch deterministic algorithms, cuDNN deterministic mode with benchmark false, CUDA float32, TF32 off,
and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Record device/driver/backend versions and all resource measurements.
Refuse an existing directory or symlink at `outputs/ssl_synthetic_localization/pilot_v1/profile`.
A busy lock, unavailable/unsupported GPU, OOM, kernel error, nonfinite update or replay failure produces
a failure receipt and stop. Do not silently change batch size, configuration or rerun this version.

Every phase constructs a new model with seed 20261018 and identical initial encoder/head bytes.
Profile actual batch size 16 with 10 warmup and 40 timed iterations in each of four phases:
masked-real-SSL optimizer update, trainable-encoder localization optimizer update, frozen-encoder F
head-only localization optimizer update, and A-validation inference. Transfer actual NumPy batches
to CUDA on every iteration and synchronize at measurement boundaries. Use the shared actual update
functions, AdamW lr 0.001/weight decay 0.0001, no mixed precision, and no replacement workloads.
The 600-step matched localization schedule seed is 20261020; compute its fixed positive weight from
training masks and actual scheduled presentations. Profile SSL batch sampling uses seed 20262018;
masking uses 20262019. Inference cycles sequential batch-16 A-validation positions. Preserve phase
wall/process CPU time, timed mean step time and peak allocated/reserved CUDA memory.

Before phase timing, run two independent exact two-update CUDA SSL replays on the same actual cached
real batches: fresh identical model seed, optimizer, sampling seed 20262018 and mask seed 20262019.
Require exact loss-list equality and exact initial/final state hashes. F must set all encoder
parameters `requires_grad=False`, keep the encoder in eval mode, optimize only head parameters, and
require an unchanged encoder-state digest afterward. Initial component hashes must match across phases.
No profile weight, loss trajectory or predicted map is saved for model selection.

Let s, t, f and i be measured mean seconds per timed SSL, trainable-localization, frozen-localization
and inference batch. Define remaining one-pass work W = 1000*s + 1200*t + 600*f + 480*i. The inference
count conservatively covers all three arms on 128*4 validation and 256*4*2 A/B evaluation cases at
batch 16. GPU forecast = actual full profile wall seconds + 2*W + 600 seconds reserve, at most 7,200.
CPU forecast = prior preparation CPU charges + actual profile CPU seconds + 2*W + 600 seconds reserve,
at most 14,400. Prior charges include permitted-input audit, real cache, every retained generator
qualification (including failed v1), and actual synthetic bank preparation. Use `cpu_seconds` when
recorded, otherwise conservatively charge its `wall_seconds`; label that fallback explicitly. These
are forecasts, not measured full-training costs. Require all entire-stream files, including earlier
qualification outputs and a summary reserve, to stay within 4 GiB. Record formulas and each component.

Write a source-bound `summary.json` last, with per-phase measurements, replay/F integrity, counts,
resource gates and forecast, overall wall/CPU/peak memory, and any known failure. No arrays, weights
or record identifiers leave the machine. `profile_gate_passed` admits only the profiled recipe's
resource/replay requirements; complete training admission still belongs to the combined pilot runner.
CPU-only tests cover projection arithmetic and gate boundaries, refusal to overwrite, source binding,
and the actual F optimizer/encoder guard before code is committed. The executing agent then writes
a results report from the live profile receipt, including failures and forecast limitations.
