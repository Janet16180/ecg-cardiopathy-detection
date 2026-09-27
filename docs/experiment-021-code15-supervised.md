# Experiment 021: supervised CODE-15 continuation of the starting CPC encoder

**Frozen 28 September 2026, before any CODE-15 training update or new PTB score.** The user asked to test
supervised pretraining on CODE-15 after [Experiment 020](experiment-020-full-development-readout.md). The
hypothesis: labeled examples help this project more than unlabeled ones (full versus limited PTB labels is
worth about 0.01-0.015 AUROC, while more or cleaner SSL data changed AUROC by about 0.002). CODE-15 has
345,779 exams with cardiologist-reviewed rhythm and conduction labels, far more than PTB-XL's 15,359. The
outcome remains an ECG diagnostic annotation proxy. Calibration and test patients stay closed.

## Data

- **Source:** all 18 parts of CODE-15% from the verified export archive
  (`outputs/data_export/code_15pct_waveforms_2026-09-25.tar.xz`, SHA-256 checked before use) and
  `exams.csv` (MD5 checked against Zenodo).
- **Records:** tracings with at least 4,000 non-padding samples (10 s at 400 Hz); 132,161 of 345,779 exams.
  The 7.3-second recordings are left out rather than padded. The central 4,000 active samples are kept.
- **Transform:** each five-second half is resampled separately from 400 to 250 Hz with
  `scipy.signal.resample_poly(up=5, down=8)`, giving 12 x 2,500 float32 samples in the canonical lead
  order, like the historical CPC input.
- **Units:** the EDA found CODE-15 amplitudes about twice those of every mV source. A single factor is set
  from training data only: the median over the 12 leads of (median lead standard deviation of the PTB-XL
  training rows of the historical 250 Hz cache) / (median lead standard deviation of the CODE-15 training
  rows). CODE-15 signals are multiplied by it before the historical CPC normalization.
- **Quality:** exclude a tracing with any nonfinite sample, a constant lead, a lead constant for at least
  1 s (400 native samples), or a scaled peak above 20 mV, as in `ecg_experiment/ecg_quality.py`.
- **Patients:** CODE-15 patients are split 95/5 into training and monitoring with seed 21001. Monitoring
  patients are used only to report CODE-15 loss and label AUROC, never to select anything.
- **Labels:** seven binary outputs: `1dAVb`, `RBBB`, `LBBB`, `SB`, `ST`, `AF` and `normal_ecg`.

## Training

- **Model:** the unchanged starting CPC encoder (`outputs/experiment004_cpc_40k/cpc_ssl/encoder.pt`) with
  a new linear head from the 512 pooled features to 7 logits (seed 21042). It is trained with the mean
  binary cross-entropy over the 7 labels.
- **Budget:** the Experiment 018/019 continuation budget: 115,359 exposures, batch 128, 902 updates. It
  uses AdamW with learning rate 1e-4 and weight decay 0.01, one pass in a seeded order (seed 21047) and no
  repeated record.
- **Checkpoints:** saved every 100 updates for exact resumption. Only the final encoder is read out; there
  is no checkpoint selection.
- **Cost gate:** 24 real V100 updates under the shared GPU lock. The projected full training (1.25 times
  measured, plus 300 s) must fit 7,200 s.

## Readout on PTB-XL

The final encoder is read out exactly like Experiment 020, with the same training rows, full development
rows, labels, fixed logistic head, historical PTB inputs and 250 Hz conversion of added records. The
comparator is the unchanged starting encoder's saved Experiment 020 predictions, reused after checking their
SHA-256 and row order.

- **Primary:** `cpc_standard` AUROC, CODE-15 continuation minus starting encoder, on full development with
  the standard superclass label. 2,000 paired whole-patient bootstrap draws, seed 21045.
- **Secondary:** the same contrast for `cpc_project_full` and `cpc_project_limited` on the original 1,306
  development ECGs with the project label (comparable with earlier experiments), and for `cpc_standard` on
  the added ECGs.
- **Descriptive:** CODE-15 monitoring AUROC for each of the 7 labels.

## Interpretation

A primary interval above zero supports labeled CODE-15 pretraining over this SSL encoder at a matched
update budget. An interval that includes zero does not establish a difference. This is one seed and one
encoder, and the development patients were inspected earlier. It does not compare against CPC continued on
CODE-15 without labels, so a gain would not separate the effect of labels from the effect of CODE-15's
population.

## Execution

1. `scripts.data.build_code15_cpc_cache` streams the archive once and writes
   `data/processed/code15_cpc_250hz_v1/` (signals, rows with exam, patient, labels, split and quality
   flags, and a metadata receipt). It needs about 16 GB and no GPU.
2. `scripts.experiments.run_code15_supervised021 --stage profile`, then `--stage train`, then
   `--stage readout`. Outputs go to `outputs/experiment021_code15_supervised_v1/`, and the results to
   `docs/experiment-021-code15-supervised-results.md`.
