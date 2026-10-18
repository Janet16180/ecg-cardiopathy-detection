# Original real-SSL and synthetic localization pilot v1

18 October 2026. User-authorized unattended work in PR #67, alongside a separate No Time to Train
session. This original engineering experiment implements the [research plan](ssl-synthetic-localization-plan.md).
The policy and configuration are committed before real input audit results, model scores or training.
No clinical disease truth, historical-method superiority or independent clinical confirmation is claimed.

## Question and data

Primary: does masked-waveform real-data pretraining improve localization of a programmed ST-region
voltage alteration on an edit profile absent from training and selection? Single-signal inference sees
only the edited ECG. A paired counterpart is retained solely to construct the intervention label;
neither it nor the target, phase landmarks, renderer identity or edit parameters enters the network.

Real data are PTB-XL TRAIN patients from the immutable v1 patient partition. Before the waveform audit,
sort canonical patient IDs by SHA-256 of `ssl_pilot_v1:real:<patient ID>` (lexicographic digest order;
patient ID breaks a tie), retain the first 2,048, then choose the lowest numeric ECG ID per patient.
A selected record failing the prospectively frozen input integrity/quality policy is skipped without
replacement; every failure remains in the receipts. Fewer than 1,900 usable patients rejects this pilot
before training and requires a prospective successor. Thus this is subset pretraining, not an all-cohort
SSL claim. Patient selection does not use diagnosis, demographics, audit values or any model score.

Freshly verified TRAIN waveforms are resampled whole-record from 500 to 250 Hz with `resample_full`,
kept in physical mV and divided by the fixed 2 mV scale at model entry. Cache only the selected usable
records. No per-record z-score, peak normalization, fitted transform or test-driven clipping is used.
Validation real waveforms may be audited for technical integrity but never enter this pretraining.
Real engineering-test waveforms, source folds 9–10 and reserved external tests remain closed.

Synthetic inference inputs are 12-lead, 10-second, 250 Hz traces from the separately qualified new
compact generator. They have known latent subjects and smooth controlled ST edits; these are engineering
interventions, not validated diseases. The support is the actual postprocessing paired difference
strictly above 0.005 mV, including coupled limb-lead changes. Unknown nonzero subthreshold tails are
reported by generator qualification and are not called complete pathology support.

Disjoint seed namespaces `ssl_pilot_v1:synthetic:<role>:<index>` map their first eight SHA-256 bytes
(big-endian integer) to the generator seed. Train, validation and engineering-evaluation subject seeds
must have no collision. Every variant of a latent subject stays in its role.

Training uses 500 independent latent subjects/examples: 375 positive paired edits, alternating focal
and persistent; 125 zero-intervention examples, alternating ordinary shams and stress-nuisance controls.
The ordinary nuisance stream is present through the same acquisition pipeline in every kind; declared
stress nuisances are balanced independently of target presence. Profile A supplies training/validation;
profile B is absent from optimization and threshold calibration. Design-only qualification of B is
allowed before this frozen model experiment. No generator parameter is adjusted after model scores.

Validation has 128 A subjects, each with focal, persistent, sham and nuisance versions. Engineering
evaluation has 256 B subjects, each with those four versions; secondary A versions on the same subjects
measure same-renderer behavior. Use identical subjects/parameters/draws for all arms. Count correlated
variants as one subject, never independent evidence. All model inferences and per-subject results stay
local; only aggregate reports enter Git.

## Matched model recipe

A 12-input temporal convolution (32 channels, kernel 11, stride 5, symmetric padding) feeds four
residual temporal blocks (two kernel-5 convolutions per block, dilations 1, 2, 4, 8, GELU) and a 1x1
12-output head. There is no dropout, normalization, explicit time coordinate or raw-input skip.
Output logits are linearly interpolated with `align_corners=False` to the native 2,500-point grid.
The latent sampling interval is 20 ms and the scoring grid 4 ms; interpolation does not establish 4 ms
independent model resolution. The named output channels retain lead coordinates.

| Arm | Encoder initialization | Localization updates |
| --- | --- | --- |
| R | Exact fresh encoder state from which SSL starts | Encoder and fresh common head |
| S | Last masked-real-SSL encoder state | Encoder and same fresh common head |
| F | Same final SSL state, frozen and in eval mode | Same fresh common head only |

SSL reconstructs the input waveform from joint all-lead masks: split 2,500 samples into fifty
nonoverlapping 50-sample blocks; mask thirteen randomly chosen blocks per record (26%, 200 ms blocks).
Replace masked samples with zero; minimize MSE on masked samples only. A separate reconstruction head
is discarded. Pretraining is 1,000 AdamW updates, batch 16 (16,000 record presentations), fixed final
checkpoint. No localization label selects an SSL checkpoint.

Supervision is 600 AdamW updates per arm with the same precomputed ordered batch identities and no
augmentation differences: 12 positive, two sham and two nuisance inputs per batch. Cycle shuffled
class-specific pools using independent seeded RNGs. Positive stress assignment is independent of edit
mode and location. This gives 9,600 presentations per arm, not 9,600 independent labels. AdamW uses
lr 0.001 and weight decay 0.0001 for both phases. Localization uses full-map binary cross entropy with
one fixed positive weight `min(20, training-negative/training-positive)` computed at schedule class
proportions, including all-zero controls. R/S have identical trainable architecture and optimizer budget;
F is a lower-capacity diagnostic, not a matched-capacity competitor. Head bytes match at initialization.
No tuning grid, early stopping or checkpoint competition: evaluate the fixed final 600-update state.
A shared initial seed is 20261018; each data/masking/sampling stream has an explicit separate seed.
Record parameter counts, losses, exposures, initialization hashes and optimizer settings.

## Frozen metrics and decisions

Compute scikit-learn average precision on the entire 12-by-2,500 support/score grid, with its standard
tied-score grouping. Do not restrict to candidate edit windows. Report mask prevalence next to AP.
For each positive subject, average focal and persistent AP equally. Primary is the mean of the 256
paired S-minus-R subject differences on B. Bootstrap whole latent subjects, shared paired draws, 2,000
draws, seed 20261019 using `ecg_experiment.intervals.patient_groups/patient_resample`; percentile bounds
are 2.5 and 97.5 with NumPy linear interpolation. All cases enter; any unscorable/nonfinite case rejects
evaluation instead of silently reducing the denominator. Report modes separately, F and A secondarily.

For each arm, compute the maximum full-map probability on each A validation control. For each subject
take the larger sham/nuisance maximum, then use the empirical 95th percentile (`method='higher'`) as
that arm's threshold. A recording flags only if its maximum is strictly greater; ties are negative.
Freeze thresholds before B inference. Report B sham and nuisance subject flag rates separately, plus
positive edited-record flag rates to expose vacuous thresholds. Report paired bootstrap differences.

Advancement requires B subject-macro AP S−R >=0.05, paired 95% lower bound >0, and neither sham nor
nuisance flag-rate increase S−R >0.05. This one-seed engineering rule does not establish clinical utility.
A passing result motivates a separately frozen replication; failure is retained and reported.

Contextual diagnostics are uniform scores (AP equals prevalence), raw absolute voltage, and an oracle
phase prior that highlights every broad candidate ST window for every beat and all 12 leads, without
using which beat/lead/width was actually edited. Use the generator's prospectively fixed broad candidate
domain, not realized support or edit parameters. The phase prior has privileged synthetic landmarks and
is a shortcut diagnostic, not a deployable competitor. A model gain over R alone does not prove precise
patient-specific evidence; compare these diagnostics and report limitations regardless of advancement.

## Integrity, profile and stop rules

This is an original architecture/training/generator combination with no frozen numerical predecessor.
Experiments 058 and 064 are motivation only; their absent artifacts have not been reproduced and their
scores are not compared. Before new scores, exactly reproduce the existing partition/declared-duplicate
audit and deterministic synthetic qualification invariants from pinned receipts. Verify source/input
hashes and exact few-update replay on artificial inputs. A separate unchanged historical method would
need its own exact predecessor reproduction before a comparison claim.

Before full training, commit implementation and obtain a real GPU profile of the actual cached-data,
masking, loss, optimizer and synthetic path, under the shared nonblocking GPU lock. Profile updates
must not enter final initial weights or scores. Project all phases, data preparation, evaluation and
reporting with a 2x measured-update safety factor plus setup reserve, at most two GPU hours and four
CPU hours and 4 GiB new cache. Log elapsed/CPU/peak CUDA memory. No busy GPU is preempted. If batch16
cannot fit, prospectively freeze a batch8 successor (including exposures and batch composition) before
training; do not silently change this protocol. No pinned dependency/source changes to force admission.

Hard input integrity failure, insufficient selected usable patients, generator gate rejection, missing
receipts, failed deterministic replay, unexpected nonfinite training or cost-gate failure stops full
training. Retain failure receipts and write the results report from executed outputs. Fixes after a
receipted scientific run use new files/versioned successors. No negative result is discarded or replaced.
