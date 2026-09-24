# Experiment 003: bounded MIMIC waveform adaptation

**Protocol date:** 23 September 2026  
**Status:** Data acquisition and environment preparation; this document specifies planned experiments, not completed model results.

## Question and scope

Does continued ECG-FM self-supervision on PTB-XL plus a substantially larger MIMIC waveform subset improve the existing PTB-XL abnormality-proxy task when all eligible public training labels are used?

The user requested approximately 200,000 additional recordings, not a 200 GB download. Download individually selected MIMIC waveform files under a **200,000-record selection cap**. Do not download the complete MIMIC archive. Estimated raw subset storage is **24–30 GB**, subject to actual file sizes and accepted counts. Stream waveform loading; do not create a second full float32 waveform cache. Download progress, files present, and validated records are operational counts and must not be presented as training or evaluation results.

This experiment jointly changes the number of available waveforms and the source mixture. Uniform record sampling means the larger source contributes most optimizer examples. It does **not** isolate a pure corpus-size effect at a constant source ratio. A subsequent nested-subset comparison with matched source proportions would be needed for that claim.

## Data selection and integrity

- PTB-XL SSL data: exactly the 17,418 official fold 1–8 recordings, including unresolved or hidden-label cases. Existing patient isolation checks remain required.
- MIMIC: deterministic patient-based selection with a recorded selection seed/rule and at most 200,000 selected recordings. Record source patient identifiers with a source namespace. Stop before the next whole patient would exceed the recording cap; do not truncate a selected patient's record list.
- Freeze the selected and accepted manifests before training. Report actual accepted ECG and unique-patient counts, rejection reasons, and manifest hashes. The selection cap is not a guarantee of 200,000 accepted records.
- MIMIC diagnosis labels and linked clinical tables are not required for this waveform-only experiment.
- Validate 500 Hz sampling, 5,000 samples, 12 unique standard lead names, physical signal units, and finite decoded values. Reorder leads to `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6` before preprocessing or identity hashing.
- Audit decoded waveform duplicates within the candidate pool and against all PTB-XL recordings, including held-out folds. Identity checks may use held-out waveforms solely to exclude duplicates, without using their labels or optimizing on those waveforms. Record audit results; duplicate removal does not prove cross-source patient independence.

Released ECG-FM and HuBERT have historical MIMIC pretraining exposure. This study is **continued adaptation of released weights**, not exposure to a previously unseen corpus. A MIMIC subset reserved now would not automatically constitute unseen-waveform evaluation for those encoders. New university data, when available, still require prospective patient separation before any optimization.

## Prespecified training comparison

All three arms start from the same released ECG-FM checkpoint. The two adaptation arms start independently from that checkpoint, not sequentially from one another.

| Arm | SSL pool | SSL optimizer updates | Downstream labels |
| --- | --- | ---: | ---: |
| Direct baseline | No continued SSL | 0 | 15,360 |
| PTB adaptation control | 17,418 PTB-XL training ECGs | 7,000 | 15,360 |
| MIMIC pooled adaptation | Same PTB data plus accepted MIMIC subset | 7,000 | 15,360 |

### Continued SSL

Use the existing CMSC-style symmetric temporal contrastive objective, with known same-patient cross-view pairs treated as positives. This is not the complete original ECG-FM WCR objective. Use mean final-layer token representations, L2 normalization for the loss, and no projection head. Keep per-lead z-score normalization over each ten-second recording followed by two nonoverlapping five-second views.

Fixed settings: seed 42, batch size 32, learning rate `1e-6`, temperature `0.1`, AdamW weight decay `0.01`, gradient clipping at `1.0`, and the final state after exactly **7,000 optimizer updates**. No development or test selection of SSL checkpoints or adaptation budget is allowed within this comparison.

Shuffle records uniformly across each entire pool. Do not impose source quotas or patient-balanced sampling. Use full batches (`drop_last=True`) in this update-budget mode, reshuffle at epoch boundaries, and stop exactly at update 7,000. Each adaptation arm therefore receives **224,000 recording exposures**. For the nominal 217,418-record pool, the first epoch contains 6,794 updates, so the pooled run requires a small part of a second pass. The PTB-only control needs part of its thirteenth pass.

Set epoch ceilings to 14 for the PTB-only control and 2 for the pooled run. Before launch, verify that `epochs * floor(accepted_pool_size / 32) >= 7000`; if filtering or an incomplete download violates this condition, resolve the data preparation or revise the documented epoch ceiling before training. Finishing fewer than 7,000 updates is an incomplete experiment, not a valid matched control. Log actual total recording exposures and pool composition. Source-specific exposure totals and per-record repetition counts are not instrumented in this implementation; any proportional estimates must be labeled as expectations, not observed counts.

The matching fixes update count, batch size, objective, optimizer settings, and initialization. PTB-only training repeats fewer distinct recordings more often, while pooled training changes both diversity and source composition. Same-patient positives depend on batch membership; different patient frequencies are another property of the source mixture.

The two-view agreement objective may suppress transient information visible in only one recording half. Larger sample size does not eliminate that limitation. Any changed objective would require a separately documented comparison.

### Supervised transfer

Fine-tune each of the three backbones using all **15,360 eligible public PTB-XL training labels**, seed 42, batch size 16, backbone learning rate `1e-5`, classifier learning rate `1e-3`, maximum 20 epochs, and early-stopping patience five. Select checkpoints using the existing development AUROC rule. Preserve the same preprocessing, downstream optimizer, and label definition across arms.

Keep the existing full-label frozen ECG-JEPA result (AUROC 0.9545) and supervised CNN result (0.9435) as practical references. They are not matched architectural controls for the adaptation contrast. The prior 10%-label ECG-FM fine-tune is also not the correct direct control for this full-label experiment.

## Evaluation and interpretation

Keep the existing development/calibration/test partition and patient assignments: 1,306 development, 564 calibration, and 1,896 retained test ECGs. Continue using the established binary ECG-annotation proxy and eligibility rules; do not describe it as confirmed cardiopathy or validated referral need.

For each final model, fit Platt calibration on the calibration partition only. Apply the prespecified rule selecting the highest calibrated-probability threshold achieving at least **95% sensitivity on the calibration ECGs**. Freeze that model's numerical threshold before test evaluation. The rule is common across arms; numerical thresholds need not be identical. Never adjust a threshold to achieve a desired test sensitivity.

Report AUROC, AP, sensitivity, specificity, Brier score, ECE, and confusion matrices. The primary contrast is pooled adaptation minus matched-update PTB adaptation; report both adapted arms against direct full-label fine-tuning as well. Align ECG IDs, patient IDs, and targets before paired patient-cluster bootstrap comparisons, using the existing 500-resample convention. Report differences and intervals, including negative or inconclusive findings. These intervals are conditional on fitted models and thresholds and do not include retraining uncertainty.

The PTB test cohort has already informed interpretation of 37 earlier evaluated runs. Although optimization continues to exclude its labels and waveforms, this next experiment is **exploratory on an already examined test cohort**, not a newly untouched confirmatory test. Completion of data acquisition alone does not add another evaluated run. Register any additional seeds, label budgets, or scale points separately before inspecting their results.

## Runtime, environment, and completion records

Prior completed V100 16 GB adaptation runs measured approximately 0.53–0.89 seconds per update and 8.53 GB peak allocated GPU memory. These measurements project to roughly **62–104 minutes per 7,000-update adaptation arm**, before any slower MIMIC file I/O. Downstream training, data transfer, and validation are additional. Profile an initial bounded segment and record observed throughput; do not change the comparative update budget in response to evaluation outcomes.

The pretrained environment's former Python 3.11.14 interpreter under `/tmp` disappeared after the storage change. Its interpreter symlink has been restored using `/usr/bin/python3.11` (Python 3.11.2, same Python minor-version ABI). Verification passed: Python 3.11.2, torch 2.6.0+cu124, successful torch/wfdb imports, CUDA available, and Tesla V100-SXM2-16GB detected. The direct baseline loaded the encoder and entered waveform preprocessing. Historical experiments retain their original recorded environments.

Preserve original-checkpoint and data-manifest hashes, source snapshots, complete optimizer settings, actual update and total recording-exposure counts, source pool counts, elapsed time, peak memory, final checkpoint hashes, downstream selection histories, calibration artifacts, and test predictions. Resume state must preserve the update position and random states so an interruption cannot silently reset the matched budget. Mark an arm complete only after its required update count and downstream evaluation artifacts are verified.


## Launch record and monitoring

On 23 September 2026, the following jobs were launched:

- Direct full-label baseline: PID 4607; log `outputs/experiment003_mimic/ecg-fm_direct_full_seed42.log`.
- Capped MIMIC preparation: PID 8560; log `outputs/experiment003_mimic/prepare_mimic_200k.log`. Metadata transfer was progressing at launch verification; selected-waveform acquisition and audit were not complete.
- Sequential GPU orchestration: PID 9583; log `outputs/experiment003_mimic/runner.log`, with stage state in `outputs/experiment003_mimic/status.json`. It first waits for the baseline, runs the PTB control, then waits for the verified MIMIC subset before pooled training.

These are historical launch PIDs, not permanent process identities. The runner verifies active process command lines and completed artifacts. Launch commands were:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.prepare_mimic_ssl \
  --max-records 200000 --workers 8 --seed 42 --retries 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-pretrained/bin/python -m scripts.run_mimic_scale \
  --wait-baseline-pid 4607 --wait-download-pid 8560
```

The runner serializes GPU stages and rejects concurrent runner instances. Its final comparisons are `paired_adaptation_comparisons.json` (both adapted arms versus direct fine-tuning) and `paired_pooling_comparison.json` (pooled versus PTB adaptation), under `outputs/experiment003_mimic/`. Exact final counts and results remain pending. Nine focused CPU tests passed for update budgeting and resume, dataset selection/audit, and orchestration validation.


### Restart after interruption, 23 September 2026 at 17:17 UTC

The earlier download, baseline, and runner processes had stopped. The incomplete baseline contained only two epochs of history and no trained-weight checkpoint; it was preserved under `outputs/experiment003_mimic/ecg-fm_direct_full_interrupted_20260923T171419Z/`. Its training restarted from the same released weights, seed, and cached inputs. It is not an evaluated run.

Download PID 3176 resumed the same frozen 200,000-record selection and reused verified files. Baseline PID 3768 and runner PID 3769 were launched in separate background sessions; current process records are stored in `download_process.json`, `baseline_process.json`, and `runner_process.json` under the experiment output directory.

Fine-tuning now atomically saves `resume.pt` after every completed epoch. It includes current and best model weights, optimizer state, all relevant random states, training history, elapsed time, and a strict configuration fingerprint. The same fine-tuning command with `--resume` resumes from the last completed epoch; an interrupted partial epoch is repeated. Completed results reject resume. The sequential runner detects incomplete downstream fine-tunes with checkpoints and adds `--resume` automatically. A deterministic CPU test verified identical uninterrupted and resumed weights and random draws; five runner tests also passed. The earlier two baseline epochs predated this checkpoint implementation and could not be recovered.
