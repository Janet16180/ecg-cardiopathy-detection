# MLC-02 candidate S-R: xresnet1d101 trained from scratch (validation only)

10 November 2026. Draft for the MLC-02 results report. **No test-set quantity was computed.** Only train and
validation waveforms were loaded; no waveform of PTB-XL fold 10, the SPH evaluation half or the SPH calibration
half was read, and no prediction or metric was computed on them. Every number below is a validation or
training-data number, read from the outputs named in brackets.

Candidate S-R is row S-R of the frozen [MLC-02 protocol](mlc02-condition-models-protocol-2026-11-10.md): an
xresnet1d101 trained from scratch at 100 Hz, the PTB-XL benchmark recipe of Strodthoff et al. (2021). It is the
supervised baseline for the pretrained encoders.

## Code and runs

- Code and tests, committed before any run: `3b62a6e` (`ecg_experiment/mlc02_scratch_network.py`,
  `ecg_experiment/mlc02_scratch_runs.py`, `scripts/experiments/run_mlc02_scratch_network.py`,
  `tests/test_mlc02_scratch_network.py`). Later commits: `d437d5f` switched TF32 off after the first profile;
  `89c488c` stored the held-out record keys as text (departure 4); `60b0598` and `c948c59` added the
  published-file check and the memory error counters.
- Commits of the runs, each with no modified tracked file: input preparation and first profile `3b62a6e`;
  second profile, overfit check, all seven trainings and the first main publication `d437d5f`; held-out
  rescoring of the four leave-one-family-out runs, their first publication and the first `predict` check
  `89c488c`; republication of both directories (same files, receipts with the memory error counters) and the
  second `predict` check `60b0598`; the published-file check `c948c59`.
- Commands, from the worktree, with `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=<worktree>`:
  `.venv/bin/python -m scripts.experiments.run_mlc02_scratch_network <step>` with steps `prepare`, `profile`,
  `overfit`, `plan` (three main seeds, four leave-one-family-out runs, publication) and `check-predict`.
- Every GPU step held `gpu_lock(device="cuda", path=/tmp/ecg_project_gpu.lock)` and first checked that GPU 0
  had the profiled peak plus 1 GiB free.

## Recipe as run, with sources

The benchmark sources were read on 10 November 2026: `helme/ecg_ptbxl_benchmarking` (branch `master`:
`code/models/fastai_model.py`, `xresnet1d.py`, `basic_conv1d.py`, `timeseries_utils.py`,
`code/utils/utils.py`, `code/configs/fastai_configs.py`, `ecg_env.yml`, which pins fastai 1.0.61 and PyTorch
1.4.0) and `fastai/fastai1` at tag `1.0.61` (`fastai/train.py`, `callbacks/one_cycle.py`,
`callbacks/lr_finder.py`, `basic_train.py`, `basic_data.py`, `callback.py`, `torch_core.py`, `layers.py`).
fastai is not installed here, so the schedule is `torch.optim.lr_scheduler.OneCycleLR`; the mapping is below.

| Item | As run | Source |
|---|---|---|
| Records | v4 included records (`included_v4` of `features/rows.csv`): 56,250 train, 15,138 validation | Protocol; MLC-01 roles |
| Alignment | Manifest, `rows.csv` and the cache's `index_all.csv` agree on key order, source, role, block, row and `sha256_10s`; every loaded window reproduces its manifest `sha256_10s` (71,388 of 71,388) [`prepare.json`] | Brief |
| Missing samples | The 3,511 samples WFDB reads as missing (87 records, all train or validation) set to the stored rail value, -32.768 mV, before resampling | Stage 1 rule (`mlc02_features.model_input`) |
| 100 Hz | `scipy.signal.resample_poly(x, 1, 5)` over the whole 10 s in float64, stored as float32 (`resample.resample_full`) | See below |
| Normalization | One mean and one standard deviation over every sample and lead of the run's training records: mean 0.000260 mV, SD 0.25547 mV for the main runs (675,000,000 values) | Benchmark `utils.preprocess_signals`: `StandardScaler().fit(np.vstack(X_train).flatten()[:, np.newaxis])` |
| Architecture | `xresnet1d101(input_channels=12, num_classes=14, kernel_size=5, ps_head=0.5, lin_ftrs_head=[128])` from `third_party/ecg-fm-benchmarking/code/clinical_ts/models/xresnet1d.py`, imported through `sys.path` | Benchmark `fastai_model._get_learner`: `xresnet1d101(num_classes=..., input_channels=..., kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)` with defaults `kernel_size=5`, `ps_head=0.5`, `lin_ftrs_head=[128]`, `input_channels=12`; the config `conf_fastai_xresnet1d101` sets no parameter |
| Body | Stem of three convolutions (32, 32, 64 channels, kernel 5, first stride 2) and a max pool; bottleneck blocks [3, 4, 23, 3], expansion 4, width 64 in every stage (256 output channels), stride 2 from stage 2, average-pool shortcut | Both files: `block_szs = [int(o*widen) for o in [64,64,64,64] + ...]` (the `[64,128,256,512]` line is commented out in both) |
| Head | Concat pooling (max, mean) of 256 channels, BatchNorm(512), Dropout 0.25, Linear 512 to 128, ReLU, BatchNorm(128), Dropout 0.5, Linear 128 to 14 | `create_head1d` with `ps=0.5` expanded to `[0.25, 0.5]`, `bn=True`, `bn_final=False`; the benchmark takes `bn_drop_lin` from fastai 1.0.61 `layers.py` (batch norm, dropout, linear, activation), the order of the vendored copy |
| Parameters | 1,873,422 (unit test) | Counted on the built model |
| Initialization | `model.apply(weight_init)` after construction: Kaiming-normal convolution and linear weights, zero biases, every batch-norm weight 1 and bias 0 (this undoes the zero init of each block's last batch norm) | Benchmark `fastai_model.fit`, from scratch: `learn.model.apply(weight_init)` |
| Training crops | One random 2.5 s crop (250 samples) per record and epoch, start uniform in [0, 749] | `chunkify_train=False` (one entry per record), `TimeseriesDatasetCrops`: `random.randint(0, timesteps - self.output_size - 1)` |
| Batches | 128, shuffled per epoch, last partial batch dropped: 439 updates per epoch | `bs=128`; fastai `DataBunch.create` builds the training loader with `drop_last=True` |
| Loss | Binary cross-entropy with logits, averaged over the defined label entries of the batch; undefined entries contribute neither value nor gradient | Benchmark `loss="binary_cross_entropy"` (`F.binary_cross_entropy_with_logits`); masking from MLC-01 decision 4 |
| Optimizer | `torch.optim.AdamW(lr=1e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2)` on every parameter, biases and batch norms included | fastai 1.0.61 `Learner` defaults: `opt_func=AdamW = partial(optim.Adam, betas=(0.9,0.99))` (eps is torch's default, 1e-8), `true_wd=True` (decay `p *= 1 - lr*wd` on every parameter outside the Adam step, as in torch AdamW), `bn_wd=True`; `wd=1e-2` |
| Schedule | `OneCycleLR(max_lr=1e-2, total_steps=50*439, pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=1e4, cycle_momentum=True, max_momentum=0.95, base_momentum=0.85, three_phase=False)`: rate 4e-4 to 1e-2 over 30% of updates, then to 4e-8; Adam beta1 0.95 to 0.85 and back, by cosine | `learn.fit_one_cycle(self.epochs, self.lr)` with fastai v1 defaults `moms=(0.95,0.85)`, `div_factor=25`, `pct_start=0.3`, `final_div=div_factor*1e4`, `annealing_cos` in both phases. A unit test reproduces the fastai rule step by step within 1e-3 of the peak rate and 1e-3 in momentum; torch reaches the peak one update earlier |
| Epochs | 50 | Benchmark `epochs=50` |
| Learning-rate finder | Not run | The benchmark calls `lr_find` before `fit_one_cycle` but only plots it. In fastai 1.0.61 `LRFinder` saves the model and optimizer state when it starts and reloads them when it ends, so its only lasting effect is on the random number streams |
| Epoch choice | Highest validation selection metric (`mlc02_evaluation.selection_metric`), the earliest on equality | Protocol (documented difference, below) |
| Evaluation | Mean of float64 sigmoid probabilities over the windows starting at 0, 2.5, 5 and 7.5 s; batch norm in evaluation mode | Protocol (documented difference, below) |
| Seeds | 20261110, 20261111, 20261112: initialization, shuffling and crops (`np.random.default_rng([seed, epoch])`), dropout; the candidate is the mean probability of the three | Protocol |
| Precision | float32 throughout, TF32 off for convolutions and matrix products, no mixed precision | Benchmark trained in float32; see the profile |
| Augmentation, weights | Random crop only; no class weights | Protocol |

On the 100 Hz input. The benchmark reads PTB-XL's distributed 100 Hz files (`load_raw_data_ptbxl`,
`filename_lr`) and contains no resampling code, so its method cannot be taken from the code. The project's
PTB-XL notebook (`notebooks/01-jr-ptbxl.ipynb`, section 14) found those files smoother than the 500 Hz ones:
the largest difference per record, usually at a QRS peak, has a median of 0.4 mV. The Challenge sources have
no such files, so one method is needed for all sources from 500 Hz. Polyphase decimation by the integer
factor 5 applies an anti-aliasing low-pass FIR filter and does not treat the 10 s as periodic, as FFT
resampling does; it is the project's helper for new manifests (`ecg_experiment/resample.py`). The S-R input
therefore differs from the benchmark's PTB-XL input by this filter.

Documented differences from the benchmark, set by the protocol and followed as written:

- Epoch: the epoch with the highest validation selection metric. The benchmark keeps the final epoch
  (`early_stopping=None`, `learn.save` after `fit_one_cycle`).
- Evaluation: the mean probability over the four consecutive, non-overlapping 2.5 s windows. The benchmark
  default is the maximum (`aggregate_fn="max"`) over 2.5 s windows with a 1.25 s stride, seven windows on 10 s.

Architecture check. The vendored file is the later `clinical_ts` copy of the benchmark's `xresnet1d.py`. For
these arguments the two build the same network: same stem, block sizes, `ResBlock` (1x1, kxk, 1x1 convolutions
with batch norm before ReLU, the last batch norm zero-initialized before `weight_init`, shortcut average pool
with `ceil_mode=True` then 1x1 convolution), `init_cnn` and head. The vendored file adds options this run does
not use (self-attention blocks, `model_dim`); `input_size` only matters for those. The benchmark code was read,
not executed, so the parameter count is that of the vendored build.

## Profile and timings

A profile of 300 timed updates (after 20 warm-up updates) of the full recipe on the main training records,
plus one validation pass, gated the full run [`profile.json`, `profile_tf32_first.json`]. GPU 0 had
10,108 of 24,121 MiB free before each.

| profile | commit | TF32 for convolutions | seconds per update | validation pass, 15,138 records | peak allocated / reserved | forecast for the plan |
|---|---|---|---:|---:|---:|---:|
| first | `3b62a6e` | on (torch default) | 0.0175 | 1.53 s | 350 / 642 MiB | 0.77 h |
| as run | `d437d5f` | off | 0.0347 | 1.88 s | 436 / 692 MiB | 1.43 h |

Full float32 costs a factor of two and the whole plan still fits in under 1.5 hours, so TF32 was switched
off and mixed precision was not used. The plan (three main seeds, then the four leave-one-family-out runs, in
one process holding the GPU lock) ran from 17:23 to 18:42, when its last step failed (departure 4)
[`epochs.csv`, `run.json` of each run]:

| run | training records | updates per epoch | training s per epoch | validation s per epoch | epoch time, 50 epochs | peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| main, seed 20261110 | 56,250 | 439 | 14.1 | 1.77 | 792 s | 436 MiB |
| main, seed 20261111 | 56,250 | 439 | 13.6 | 1.70 | 763 s | 458 MiB |
| main, seed 20261112 | 56,250 | 439 | 12.8 | 1.65 | 723 s | 458 MiB |
| without PTB-XL | 38,832 | 303 | 8.9 | 1.43 | 515 s | 1,560 MiB |
| without Chapman and Ningbo | 29,234 | 228 | 6.8 | 0.69 | 376 s | 458 MiB |
| without Georgia | 50,133 | 391 | 12.0 | 1.51 | 677 s | 436 MiB |
| without CPSC and CPSC-Extra | 50,551 | 394 | 12.0 | 1.53 | 677 s | 1,762 MiB |

Epochs took 4,524 s in all, against a forecast of 5,136 s. The larger peaks of two later runs probably count
memory the process still held from earlier runs (my reading; reserved memory reached 3,728 MiB, far below the
free memory). The process's peak resident memory was 11,386 MiB. Preparing the inputs took 98 s (5,129 MiB).

## Validation results

Within-source macro AUROC on the validation records and the selection metric (their mean over the six
sources), for each seed at its chosen epoch and for the candidate, the mean probability of the three seeds
[`results.json` `selection`]:

| model | chosen epoch | PTB-XL fold 9 | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | selection metric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 20261110 | 27 | 0.9650 | 0.9813 | 0.9836 | 0.9472 | 0.9583 | 0.9404 | 0.9626 |
| seed 20261111 | 38 | 0.9668 | 0.9842 | 0.9854 | 0.9491 | 0.9614 | 0.9270 | 0.9623 |
| seed 20261112 | 41 | 0.9673 | 0.9837 | 0.9847 | 0.9496 | 0.9609 | 0.9268 | 0.9622 |
| **S-R (three-seed mean)** | | 0.9695 | 0.9851 | 0.9862 | 0.9531 | 0.9642 | 0.9405 | **0.9664** |

Entering conditions per source are those of the scoring library: 14, 14, 9, 12, 5 and 7. The three-seed mean
is above every seed by about 0.004.

The chosen epoch is the best of 50 noisy values. At the last epoch the seeds scored 0.9593, 0.9607 and
0.9608; the three best epochs of each seed lie within 0.0016 of its maximum [`epochs.csv`]. The chosen-epoch
numbers are therefore slightly optimistic for validation, as for every candidate whose epoch or setting is
chosen there.

Per-condition AUROC of the candidate by validation source [`results.json` `per_source.mean`]. Values in
parentheses do not enter the source's macro (fewer than 20 positives or negatives); a dash marks a condition
undefined or single-class there. The last column is the per-condition selection value (mean over the sources
where the condition enters).

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | mean over entering |
|---|---:|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.992 | 0.999 | 0.999 | 0.995 | - | 0.982 | 0.993 |
| sinus bradycardia | 0.973 | 0.998 | 1.000 | 0.997 | - | (0.983) | 0.992 |
| sinus arrhythmia | 0.890 | 0.951 | - | 0.884 | - | (0.614) | 0.908 |
| AF or flutter | 0.995 | 0.999 | 0.999 | 0.919 | 0.992 | 0.981 | 0.981 |
| PVC | 0.994 | 0.989 | 0.994 | 0.890 | 0.926 | 0.836 | 0.938 |
| first-degree AV block | 0.977 | 0.989 | 0.994 | 0.983 | 0.981 | 0.937 | 0.977 |
| complete RBBB | 0.997 | 0.999 | 0.990 | 0.992 | 0.988 | 0.991 | 0.993 |
| incomplete RBBB | 0.972 | 0.984 | - | 0.973 | - | (0.951) | 0.976 |
| LAFB | 0.987 | 0.985 | - | 0.973 | - | - | 0.982 |
| left axis deviation | 0.999 | 0.990 | 0.981 | 0.978 | - | - | 0.987 |
| LVH | 0.959 | 0.996 | (0.987) | 0.952 | - | 0.946 | 0.963 |
| ST depression | 0.957 | 0.969 | 0.958 | - | 0.934 | (0.907) | 0.955 |
| T-wave abnormality | 0.934 | 0.974 | 0.962 | 0.901 | - | (0.797) | 0.943 |
| any MI | 0.946 | 0.970 | (0.993) | - | - | 0.910 | 0.942 |

Positives and negatives behind each cell are in `results.json` (`per_source.<model>.<source>.conditions`).
Entering cells below 0.92 are sinus arrhythmia at PTB-XL and Georgia (0.890, 0.884), PVC at Georgia and
CPSC-Extra (0.890, 0.836), the T wave at Georgia (0.901), any MI at CPSC-Extra (0.910) and AF at Georgia
(0.919). The Chapman, Ningbo and Georgia validation records share patients with training records (6-13%,
MLC-01 Amendment 3), which may raise their numbers; PTB-XL fold 9 has no such overlap.

For orientation only: the published xresnet1d101 macro AUROC on PTB-XL fold 10 ranges from 0.896 (form
statements) to 0.957 (rhythm statements) over the six PTB-XL label sets, with 0.925 for all 71 statements
(benchmark README leaderboard). S-R scores 0.9650-0.9673 per seed and 0.9695 for the mean on fold 9. The label
sets (14 conditions here, with undefined entries removed), the training data (PTB-XL folds 1-8 plus the
Challenge sources here), the input filter, the window rule and the fold all differ, so the numbers are not
comparable; they only show that the network trains to the range expected of this architecture.

## Leave-one-family-out (seed 20261110)

Each run trains on the train records of three families, chooses its epoch on the validation records of the
same three families, and scores every train and validation record of the held-out family with its chosen
epoch [`lofo/S-R/lofo.json`]. The last column is the main run's validation macro of the same seed, for
reference: it uses only the validation records of that source and can have fewer entering conditions.

| held-out family | training records | selection records | chosen epoch | selection metric inside | held-out source | held-out records | entering | held-out macro AUROC | main run, validation |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| PTB-XL | 38,832 | 12,955 | 49 | 0.9574 | PTB-XL | 19,601 | 14 | 0.9314 | 0.9650 |
| Chapman and Ningbo | 29,234 | 6,130 | 35 | 0.9495 | Chapman | 8,133 | 10 | 0.9585 | 0.9836 |
| | | | | | Ningbo | 27,891 | 14 | 0.9522 | 0.9813 |
| Georgia | 50,133 | 13,095 | 30 | 0.9631 | Georgia | 8,160 | 12 | 0.9323 | 0.9472 |
| CPSC and CPSC-Extra | 50,551 | 13,234 | 37 | 0.9724 | CPSC 2018 | 5,014 | 5 | 0.9014 | 0.9583 |
| | | | | | CPSC-Extra | 2,589 | 10 | 0.8996 | 0.9404 |

Every held-out source loses 0.015 to 0.057 of within-source macro AUROC against the in-distribution
validation number. The losses concentrate in the ST and T-wave conditions, infarction and sinus arrhythmia:
held out, ST depression scores 0.69 in both CPSC sources and 0.85-0.87 elsewhere, the T wave 0.85-0.90,
any MI 0.81-0.95 and sinus arrhythmia 0.80-0.83, while rate, AF, complete RBBB and the axis stay at 0.93-0.998
(per condition in `lofo.json`). For xECG and ECG-JEPA the Chapman-Ningbo arm is not a clean held-out family
(their pretraining used those waveforms); for S-R, trained from scratch, it is.

## Sanity checks

- Loss decreases. Training loss fell from 0.585, 0.585 and 0.582 (epoch 1) to 0.070 at epoch 50 for all
  three seeds. Validation loss (binary cross-entropy of the four-window mean on defined entries) fell from
  0.257-0.266 to a minimum of 0.0767, 0.0752 and 0.0762 (epochs 40, 45, 46) and moved by less than 0.001 over
  the last 10 epochs.
- Validation selection metric per epoch [`epochs.csv`]:

  | epoch | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | seed 20261110 | 0.760 | 0.840 | 0.896 | 0.920 | 0.947 | 0.951 | 0.954 | 0.959 | 0.960 | 0.961 | 0.960 | 0.959 | 0.959 |
  | seed 20261111 | 0.756 | 0.843 | 0.876 | 0.924 | 0.946 | 0.954 | 0.956 | 0.960 | 0.962 | 0.962 | 0.961 | 0.962 | 0.961 |
  | seed 20261112 | 0.746 | 0.837 | 0.892 | 0.916 | 0.944 | 0.950 | 0.954 | 0.956 | 0.958 | 0.959 | 0.961 | 0.960 | 0.961 |

  The learning rate peaks at epoch 15 (1.0e-2) and ends at 4.0e-8, as scheduled.
- Overfit check [`overfit_check.json`]: 256 random training records, their own standardizer, the recipe's
  optimizer and schedule over 150 epochs (300 updates). Training loss went from 0.968 to 0.033 (training mode,
  dropout on, random crops); the evaluation-mode loss of the four-window mean on the same records went from
  7.11 at initialization to 0.017, with macro AUROC 1.000 on the 5 conditions with at least 20 positives and
  negatives there. The pipeline can fit its inputs.
- Label masking: `test_masked_bce_gives_zero_gradient_on_undefined_entries` checks that undefined entries
  have exactly zero gradient, defined ones a nonzero gradient, and that the loss equals the binary
  cross-entropy over the defined entries.
- Role guard: `check_roles` raises `PermissionError` on each of `test_same_source`, `test_external` and
  `sph_calibration`; `read_windows` raises before opening any file when a record's role or its block's role is a
  test role (tests `test_check_roles_refuses_every_test_role` and
  `test_read_windows_refuses_test_records_before_opening_any_file`). The input preparation opened only the 12
  train and validation block files [`prepare.json` `blocks_opened`].
- `predict` reproduces the saved three-seed validation predictions from the 500 Hz cache: on the GPU for all
  15,138 records the largest difference is 0, at `89c488c` and again at `60b0598` after the republication
  [`predict_check.json`]; on the CPU for 48 records it is within 1e-5
  (`test_predict_reproduces_the_saved_validation_predictions_on_cpu`).
- Published files [`files_check.json`, in both directories, `c948c59`, on the CPU because another job held
  the GPU lock]: every checkpoint matches its recorded SHA-256 and loads; recomputed from the prepared inputs,
  each seed reproduces its validation prediction file within 1.3e-6 and each leave-one-family-out model its
  held-out file within 2.0e-6 (CPU against the GPU that wrote them); the mean file equals the mean of the seed
  files exactly. The kernel's memory error counters at the check: `mc0/ue_count` 1, `mc0/ce_count` 0.
- Schedule: `test_one_cycle_follows_the_fastai_v1_schedule` (above); architecture:
  `test_model_matches_the_benchmark_configuration` (parameter count, head layers, dropout, batch-norm init).
- Tests: `tests/test_mlc02_scratch_network.py` 19 passed, with `tests/test_mlc02_evaluation.py` 55 passed;
  `ruff check ecg_experiment scripts tests conftest.py` passed. The full suite in the worktree: 1,712 passed, 2
  skipped, with `tests/test_xecg_encoder_motion_v11.py::test_graph_rejects_missing_and_conflicting_leaves`
  deselected as in the Stage 1 report (it rejects the worktree's symlinked `outputs`).

## How the test step calls `predict`

After the selection record is committed, a separate step loads the test windows itself (the S-R loader refuses
test roles by design), for example with `mlc02_waveforms.load_cache(index_name="index_all.csv")` and
`windows(...)` at the `included_v4` positions in manifest order, and calls

```python
from ecg_experiment.mlc02_scratch_network import CANDIDATE_DIR, CANDIDATES, MODEL_ID, SEEDS, predict
from ecg_experiment.mlc02_data import prediction_path, save_predictions

probability = predict(waveforms, CANDIDATE_DIR, device="cuda")  # [n, 12, 5000] float32, 500 Hz, mV
save_predictions(prediction_path(CANDIDATES, MODEL_ID, role, SEEDS), record_keys, probability, MODEL_ID, SEEDS)
```

`waveforms` are the cached windows exactly as stored (NaN allowed for missing samples; `predict` applies the
rail value, the 100 Hz filter, the saved standardizer, the four windows and the seed mean, and switches TF32
off for the process). The output is float64 `[n, 14]` in `mlc02_data.CONDITIONS` order. `settings.json` names
the three checkpoints and their SHA-256.

## Departures and doubts

All dated 10 November 2026. No rule of the protocol was changed. The two protocol differences from the
benchmark are listed under the recipe; the items below are departures from the benchmark that the data or the
software force, and points to keep in mind.

1. 100 Hz input by polyphase decimation instead of PTB-XL's distributed 100 Hz files (see the recipe).
2. One-cycle schedule from `torch.optim.lr_scheduler.OneCycleLR` (fastai is not installed). Its peak comes one
   update earlier than fastai's; otherwise it follows fastai 1.0.61 within 1e-3 of the peak rate.
3. The masked loss averages over the defined entries of each batch. With every entry defined it is the
   benchmark's loss (unit test). The learning-rate finder is not run (see the recipe).
4. The leave-one-family-out models were trained at `d437d5f` in the plan process. Its publication step failed
   because the held-out archives stored record keys as an object array, which loads only with pickle. The keys
   are now stored as text (`89c488c`), and the held-out scoring was repeated at that commit from the same chosen
   checkpoints (no epoch retrained). This overwrote the first `run.json` of these four runs, so their receipts
   show `89c488c` and the first records are not kept. For the two runs whose first records I had printed
   (PTB-XL 0.9314; Chapman 0.9585 and Ningbo 0.9522) the repeated held-out macros are equal at four decimals;
   Georgia and CPSC were not printed before the repeat. Training histories, chosen epochs and checkpoints are
   those of the plan process.
5. TF32 was off for all training runs; the first profile ran with the torch default (TF32 on for
   convolutions). Runs are not bit-reproducible: cuDNN may choose nondeterministic kernels. A resumed run sees
   the same batches and crops (seeded per epoch) but not bit-identical numerics. No run was interrupted.
6. Missing samples (3,511 at 500 Hz in 87 train and validation records) enter as the rail value -32.768 mV,
   the Stage 1 rule. They are a tiny share of the standardizer's 675 million values; their effect on it was
   not measured separately.
7. Metadata read: the record table reads the identity columns of every manifest row and the cache index, and
   the scoring library's `load_manifest` reads the label states of every role; test-role rows were dropped
   before any use. No test-role waveform file was opened.
8. The overfit check reached an evaluation loss of 0.017 rather than exactly zero, with 300 updates under the
   one-cycle schedule. Longer training was not tried.
9. Memory error. The kernel logged one uncorrectable ECC memory error at 17:36:20 (`EDAC MC0: 1 UE`), the
   second at which epoch 43 of seed 20261110 was logged; it killed another agent's process. None of the S-R
   processes failed, and seed 20261110's chosen epoch (27) was written before it. ECC
   detected the error, and the file check and both `predict` checks above ran after it. The counters
   (`/sys/devices/system/edac/mc/mc0/ue_count` 1, `ce_count` 0) are in both receipts and in
   `files_check.json`, so a later error will be visible.

## Files

- `outputs/mlc02_condition_models_20261110/candidates/S-R/`: `settings.json` (recipe, standardizer, the three
  chosen checkpoints with SHA-256 and commit), `normalization.json`, `seed-<seed>/model_epoch-<NNN>.pt`
  (`state_dict`), `validation__seeds-<seed>.npz` and `validation__seeds-20261110-20261111-20261112.npz`
  (library convention), `results.json` (selection tables and per-source, per-condition tables for each seed and
  the mean), `epochs.csv` (every epoch of the three seeds), `profile.json`, `profile_tf32_first.json`,
  `overfit_check.json`, `predict_check.json`, `files_check.json`, `receipt.json` (input hashes: manifest
  `25d75f8f...`, `rows.csv`, `index_all.csv`, cache receipts, prepared signals; each run's commit, code
  SHA-256, environment, timings, memory; the publication's commit and memory error counters).
- `outputs/mlc02_condition_models_20261110/lofo/S-R/`: `<family>/model_epoch-<NNN>.pt`,
  `<family>/held_out__seeds-20261110.npz`, `lofo.json`, `epochs.csv`, `files_check.json`, `receipt.json`.
- Local only (`/var/tmp/janetrivera/mlc02/scratch_network/`): prepared 100 Hz inputs (`inputs/`), every run's
  per-epoch weights, validation probabilities, `state.pt` and logs (`runs/`), and step logs (`logs/`).
