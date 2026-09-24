# Experiment 015: cached JEPA to CPC distillation

**Protocol frozen 24 September 2026, before the GPU profile or downstream results.** The initial pilot is development only. Test patients remain untouched until a later decision after repeatability checks.

## Question and population

Can fixed global JEPA representations improve a compact raw ECG CPC student under the same labeled training budget? The teacher cache covers 15,360 eligible PTB-XL training ECGs, including records whose labels are hidden in the 1,518-label setting. This is a documented subset of the 56,875-record SSL pool; no MIMIC teacher embeddings are claimed. Frozen teacher representations come from an external released checkpoint with uncertain pretraining overlap. They use no diagnostic labels from development, calibration or test here.

The ordinary 20-epoch Experiment 004 CPC SSL encoder initializes every arm. Its train-only, per-lead normalization, native 250 Hz twelve-lead input, independent five-second halves, causal CNN/GRU and 512-wide mean/max readout are retained. A binary linear head and a 512-to-768 projection are initialized with seed 42 identically in both arms. The projection is discarded at inference. The encoder and head are trained in both arms.

## Fixed matched pilot

Two arms run separately for each label budget: **BCE control** and **BCE plus 0.1 cosine loss** against the cached, frozen JEPA vector. BCE uses only exposed labels. Both arms process every one of the same 15,360 training ECGs once per epoch; the 1,518 labels are spread over the 120 fixed batches so each batch has a supervised term. All arms use batch 128, five epochs, AdamW weight decay 0.01, encoder learning rate 3e-4, head and projection learning rate 1e-3, no waveform augmentation, and the same epoch-wise batch schedule from seed 42. The projection forward graph is evaluated in the control with zero coefficient. Each arm receives 600 optimizer updates; full-label arms receive 76,800 labeled exposures and limited-label arms receive 7,590. Selection is best development AUROC across the five fixed epochs; all epochs and training times are reported.

The 1,306 development ECGs from patient-disjoint partitions support early screening. A five-fold patient-grouped development screen chooses a threshold on the other four folds for 95% sensitivity and reports held-out sensitivity/specificity. This screen is descriptive; best epoch still uses development AUROC. Development representation variance is recorded to detect collapse. Calibration (564 ECGs) and test (1,896 ECGs) are not used for the initial pilot. A positive seed-42 signal requires distillation to improve development AUROC by at least 0.002 and held-out fold specificity by at least 0.02 in a label budget, with no more than 0.002 AUROC loss or 0.005 held-out sensitivity loss in either budget and development representation variance at least 10% of its matched control. It then requires a second matched seed before calibration/test. A failure on this simple global target does not rule out relational or token targets.

## Execution gate and recovery

The CPU `check` stage verifies exact ECG and patient alignment, labels, source/cache hashes, train-only normalization, teacher cache metadata and the final CPC bootstrap state. A bounded RAM subset loader copies training and development waveforms in source order, then serves deterministic batches. It caps requested waveform RAM at 2.4 GB and leaves at least 1 GB available. This addresses the sustained random mmap I/O that invalidated Experiment 010's short profile.

The `profile` stage, under the shared V100 lock, runs a real complete training epoch and development pass for both full-label arms. It includes cache preload and checkpoint writes, then reloads and checks the saved state. It discards profile-trained states. Its conservative extrapolation covers five epochs across four arms and must fit the working two-hour gate before `train` starts. The `train` stage begins all four arms fresh, saves optimizer/model/RNG/batch position every 20 updates and on SIGTERM, and stops with a recoverable state at its wall-time deadline. It never changes the fixed batch size or subset based on development scores.

Commands:

```bash
.venv-pretrained/bin/python -m scripts.run_jepa_cpc_distillation --stage check --device cpu --threads 1
.venv-pretrained/bin/python -u -m scripts.run_jepa_cpc_distillation --stage profile --device cuda --threads 1
.venv-pretrained/bin/python -u -m scripts.run_jepa_cpc_distillation --stage train --device cuda --threads 1
```

Outputs are under `outputs/experiment015_jepa_cpc_distillation/`; `profile.json` is a runtime and correctness receipt, per-arm `resume.pt` supports recovery, and `report.md` is a development-only pilot result. No test performance claim follows from a check or GPU profile. The endpoint remains an ECG diagnostic annotation proxy; it does not establish clinical health or validate referral decisions.
