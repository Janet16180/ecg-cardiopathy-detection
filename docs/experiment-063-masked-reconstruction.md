# Experiment 063: masked cross-lead and cross-beat reconstruction trained on clean normals

Frozen 12 October 2026, before any LocBench score. Part of the [localization programme](localization-program-2026-10-12.md);
evaluated on [LocBench v1](experiment-061-locbench.md).

## Question

Does a model trained only on clean normal ECGs to fill hidden lead-time positions give a calibrated local
error map that localizes abnormalities, including persistent ones that a within-ECG reference subtracts, and
is not drawn to noisy leads?

## Why this design

The 041-060 review found two blind spots: within-ECG references (052-056) subtract any change present in every
beat, and population references (U_B) see persistent change but cannot place it by lead. A reconstruction
model can combine both references if what it is allowed to see is controlled:

- a hidden block can be filled from the neighbouring beats (within-ECG reference: catches the odd beat);
- a phase mask hides the same R-relative window in every beat of the selected leads, so a persistent change
  cannot be copied from another beat and must be predicted from the other leads and the rest of the cycle
  (population reference learned from normals: catches persistent change);
- a whole-lead mask predicts a lead from the others for the whole record (catches lead-specific change).

Only hidden samples are scored, so the model cannot copy its input. The six limb leads are exact linear
combinations of I and II (checked on the cache: |III - (II - I)| at most 1 uV), so a single hidden limb lead
could be rebuilt exactly from two visible ones and would never show an error. The limb leads are therefore
always hidden together, as one mask unit; each precordial lead is its own unit (7 units).

## Data

- Training: the 2,666 `fit`-partition `clean_normal` records of `docs/locbench-partition-v1.json`, raw from the
  LocBench cache. 10% of their patients (seed 63063: 257 patients, 266 records) are held out for early stopping
  and calibration; 2,400 records train. No ECGAD record, `evaluation` record, fold 9/10 record outside ECGAD,
  SPH, Challenge or EchoNext record is used for fitting or selection.
- Challenge-train normals are not used. Their devices, filters and populations differ from PTB-XL (Chapman and
  Ningbo are heavily pre-filtered, Georgia and CPSC record differently), and the error map is meant to measure
  departure from PTB-XL normal morphology; a domain mix would inflate sigma everywhere. This is a declared
  choice, listed as a follow-up.
- Preprocessing: 0.5-40 Hz second-order Butterworth band-pass, zero phase, then decimation to 250 Hz (sample i
  is raw sample 2i). Amplitudes stay in mV (no per-record scaling), so amplitude abnormalities stay visible.
- R peaks: the frozen detector `fragment_localization.r_peaks` on the raw 500 Hz record, halved to 250 Hz.

## Model

`ecg_experiment/inpaint063.py`, `InpaintNet`: a 1D U-Net over time with leads and mask indicators as channels
(input 24 channels: the signal with hidden samples set to zero, and the 0/1 mask). Encoder widths 64, 96, 128,
192, 256 with four stride-2 stages (bottleneck at 15.6 Hz, 64 ms per token), each stage a residual pair of
kernel-7 convolutions with group norm and GELU; a depthwise positional convolution and a 4-layer pre-norm
transformer (d = 256, 4 heads, feed-forward 512, dropout 0.1) at the bottleneck, which sees the whole record
and can attend to the other beats; a mirrored decoder with skip connections; a 1x1 head giving a mean and a
log-variance for every lead-sample. The log-variance is bounded to (-9, 3) by a scaled tanh (sigma 11 uV to
4.5 mV). About 6.4 M parameters.

Loss: beta-NLL (Seitzer et al., 2022, beta = 0.5) of the Gaussian over hidden samples only. The beta weight
stops the variance from absorbing a poor mean fit.

## Training masks (one family per record)

- Block, probability 0.4: 2-7 blocks of 100-300 ms, each on 1-3 random units.
- Phase, probability 0.4: 1-2 windows centred at -300 to +500 ms from R, 80-250 ms wide, hidden in every beat
  of 1-3 random units.
- Whole-unit, probability 0.2: 1-2 units hidden for the whole crop.

## Augmentation and noise

Random 2,048-sample crops (8.2 s; time shift), a global amplitude scale log-uniform in 0.8-1.25 (the same for
every lead, which keeps the limb relations), and with probability 0.6 electrode noise: each of the nine
electrodes (RA, LA, LL, V1-V6) is noisy with probability 0.3 and gets white noise (3-50 uV), a 0.5-1.5 Hz
sinusoid (up to 0.15 mV) or differenced white noise (up to 50 uV); electrode noise is projected to the twelve
leads through the Einthoven, Goldberger and Wilson definitions. The noise is part of the target as well as the
input: the model must predict the observed signal, so it learns to raise sigma where the visible part of a lead
is noisy. This is the declared mechanism against attraction to noisy leads (LocBench C1).

## Optimisation and the GPU gate

AdamW (lr 1e-3, weight decay 0.01), 500-step linear warm-up then cosine decay, batch 64, bf16 autocast, gradient
clipping at 1. Validation every 500 updates on the 266 held-out records (full 10 s, two fixed seeded
training-style masks each, no noise): Gaussian NLL over hidden samples; the best checkpoint is kept and training
stops after 8 validations without improvement. A real GPU profile (seconds per update, peak memory, inference
seconds per record) is run first; the maximum number of updates is set from it so that training takes at most
about 75 minutes, and recorded in the results. Seed 63063. All CUDA phases hold `gpu_lock("cuda")` and release
it between phases.

## Inference and maps

For each record, a fixed set of 140 masks on the full 10 s (padded to 2,512 samples by reflection):

- block family: 200 ms blocks, every fourth block, at offsets 0 and 100 ms, for each unit (56 masks; every
  lead-sample hidden exactly twice);
- phase family: 160 ms R-relative windows centred from -280 to +520 ms in 80 ms steps, in every beat, for each
  unit (77 masks; samples from -360 to +600 ms around each R hidden about twice);
- whole family: each unit for the whole record (7 masks).

For each family, z = |x - mu| / sigma is averaged over the masks of that family that hide the sample (the same
for |x - mu| in mV). Maps are moving-averaged over 11 samples (44 ms) per lead, uncovered samples get the map's
minimum, and the 250 Hz map is linearly interpolated to the 500 Hz record grid. Maps are float16 (n, 12, 5000).

Declared variants (all scored and reported; only the primary decides):

| Variant | Map |
| --- | --- |
| `primary` | mean of the block, phase and whole family z maps (NaN-aware where the phase family does not cover) |
| `phase` | phase family z only |
| `block` | block family z only |
| `whole` | whole family z only |
| `raw` | mean of the three family absolute-error maps in mV (no sigma) |

Map sets: the 198 ECGAD development records and all 5,801 `evaluation`-partition records (region, noise and
detection sets), under `outputs/experiment063_inpaint_v1/maps_v1/`.

## Pre-score diagnostics (no LocBench labels)

On the 266 held-out fit normals, under the inference masks: mean and median z, share of z above 2 and 3 per
family (a calibrated Gaussian gives 4.6% and 0.27% for |z|), and the training and validation curves.

## Primary comparison and decision rule

Primary: paired A2 (within-record point AUROC, mean over the 94 marked ECGAD development records), `primary`
minus the bar of Experiment 061 (the baseline with the highest A2 frozen in `experiment-061-locbench-results.md`),
2,000 whole-patient bootstrap draws, seed 61061, computed by the LocBench scoring module. If no 061 results report
exists at scoring time, the comparison is against U_B and residual056, and the one with the higher A2 point
estimate plays the bar.

- Positive: the lower 95% bound of the paired A2 difference is above 0.
- Negative: the upper bound is below 0.
- Otherwise inconclusive.

A positive result is carried to Experiments 066 and 067 only if C1 (noisy-lead AUROC) is not above the bar's C1 by
more than 0.05; otherwise it is reported as positive but drawn to noise. A1, A3-A6, B1-B3 (macro and per family),
C1, C2 and D1 are reported for every variant beside the bar, with intervals. The variants are descriptive.

## Iteration

Iteration on the ECGAD development half is allowed. Any change after a score (model, masks, aggregation,
smoothing, variants) is written as a dated amendment below and committed before it is scored, and every scored
version is reported. The ECGAD holdout half is not read here.

## Integrity checks

1. The LocBench scorer, run on the bar baseline's maps (or recomputed baseline maps), reproduces the bar's A2 in
   the 061 result exactly.
2. Determinism: maps for 5 ECGAD development records regenerated in a separate process match the saved float16
   maps (maximum absolute difference at most 1e-2 z, allowing for GPU reduction order).
3. Data: no training or validation record is an ECGAD record or in the `evaluation` partition (asserted in code).
4. No-leak unit test: changing hidden input samples does not change the model output
   (`tests/test_inpaint063.py`).
5. Every map is finite with shape (n, 12, 5000).

## Outputs

`outputs/experiment063_inpaint_v1/`: `profile.json`, `train_v1.json`, `model_v1.pt`, `calibration_v1.json`,
`training_split.csv`, `maps_v1/` (maps and manifests), `result.json`, `run.log`. Code:
`ecg_experiment/inpaint063.py`, `scripts/experiments/run_inpaint063.py`.

## Amendments

None yet.
