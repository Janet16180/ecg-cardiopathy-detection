# Experiment 001: 10% labeled training patients

## Question

How well can pretrained ECG representations distinguish a documented diagnostic abnormality
from a conservatively defined normal ECG when labels are exposed for only 10% of eligible
training patients? Does further self-supervised training improve the result?

The pending local dataset has approximately 20,000 raw 12-lead recordings and expert text reports
for approximately 1%. Begin with a 10% public-data simulation, then lower the fraction to 5% and
1%. This is simulated label scarcity using real signals, not synthetic ECG generation. Label
availability and disease prevalence are independent experimental settings.

## Data and provisional target

Use [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), including its official SCP
annotation table. Preserve its patient-exclusive folds: 1–8 training, 9 validation, 10 test.
The source includes both report text and structured annotations. SCP likelihood zero means
unknown, so code presence is not discarded merely because its likelihood is zero.

The implemented **diagnostic_abnormality** proxy uses:

- Target 1: at least one diagnostic code assigned to a non-NORM diagnostic superclass, with no
  NORM code. This denotes an annotated ECG finding, not confirmed disease or required referral.
- Target 0: explicit NORM and no other codes except SR (sinus rhythm).
- Unresolved: all other combinations, including NORM plus another finding, rhythm-only records,
  or absent diagnostic statements. Retain unresolved training waveforms for self-supervision;
  exclude unresolved records from this provisional supervised evaluation and report coverage.

The strict negative rule prevents NORM co-occurring with an arrhythmia from being silently
called normal. It also excludes some benign variants. This makes the initial task narrower and
potentially easier than real screening: results apply only to the retained cases. Expanding the
mapping to all rhythm/form codes needs clinical review. Neither absence of a diagnosis nor
sinus rhythm alone establishes health.

When local reports arrive, establish a clinician-reviewed normal/abnormal/referral mapping,
accounting for negation and uncertain statements. Text is annotation evidence; the deployed
classifier's input is the waveform. Evaluate any automatic report-label extraction against a
reviewed subset before using it as ground truth.

## Label masking and isolation

Select approximately 10% of eligible training patients uniformly at random with a recorded seed;
all eligible ECGs from selected patients receive labels. Patient grouping means the fraction of
recordings is approximate. Do not rebalance the population to 50/50. Report both classes and the
actual patient and recording fractions.

The unlabeled manifest and complete training waveform manifest contain only record identifiers,
patient identifiers, and waveform paths. They omit report text, diagnoses, and target labels.
Hidden annotations live in a separate audit directory and must never be loaded by a learner.
The full training waveform manifest includes labeled and unlabeled training ECGs, allowing both
to contribute to self-supervision.

Validation and test labels remain available to the evaluator. Their annotation cost is additional
to the 10% training budget; do not describe this as 10% of all annotations. Fit parameters on
training data, choose model settings and thresholds on validation data, then evaluate once on the
locked test set. Any calibration procedure must also be fitted without test labels. Repeat label
selection with seeds 42, 43, and 44 while keeping the same validation/test folds.

## Models and comparisons

Use ECG-FM, HuBERT-ECG, and ECG-JEPA SSL checkpoints with the same exposed labels. Their objectives
all learn from unlabeled signals; wav2vec is one appropriate option, not the only one.

The initial scope compares frozen features for all three released encoders, supervised
fine-tuning of HuBERT-small and ECG-FM, and continued self-supervision of ECG-FM followed by
the same downstream procedure. Retain a small CNN trained from scratch as a reference and
matched scratch/SSL compact transformers. Keep the labeled subsets and evaluation cases fixed.
Published encoders have different historical data, compute, and preprocessing, so this compares
practical systems rather than isolating architecture. Further SSL may help or hurt.

Following the expanded project brief, public training labels may all be used in a separate
full-label experiment. Local annotation scarcity does not require discarding public labels.
The 10% runs remain controlled comparisons. A custom architecture arm is described in
[its design note](custom-architecture.md). These additions make the overall study exploratory;
the whole model comparison was not prospectively locked before any test results were seen.

Follow each checkpoint's documented lead order, units, sample rate, normalization, and duration;
do not feed the 100 Hz convenience signals into an encoder requiring a different input without
the appropriate preprocessing. Bad signals need a separately assessed quality workflow.

Sources:

- [ECG-FM code and SSL weights](https://github.com/bowang-lab/ECG-FM)
- [HuBERT-ECG code](https://github.com/Edoar-do/HuBERT-ECG)
- [HuBERT-ECG SSL base checkpoint](https://huggingface.co/Edoardo-Coppola/hubert-ecg-base)
- [ECG-JEPA code and checkpoints](https://github.com/sehunfromdaegu/ECG_JEPA)

## The HuBERT data question and pretraining exposure

HuBERT-ECG's 9.1 million ECG corpus combines public and access-on-demand datasets; its GitHub
repository does not provide the whole waveform collection. Start from public subsets and
released SSL weights instead of waiting to assemble that corpus.

The [HuBERT methods](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v3.full) include
PTB-XL in SSL pretraining. ECG-FM also includes PTB-XL through the PhysioNet challenge data.
Consequently this experiment measures downstream label scarcity with prior waveform exposure
for those checkpoints, not completely unseen external generalization. Audit each exact
checkpoint and record its revision before comparing results. Avoid HuBERT's supervised
Cardio-Learning checkpoints here because their training also includes PTB-XL diagnostic labels.
The eventual local cohort can provide external evaluation if withheld from model development.

## Evaluation and current status

Report AUROC, average precision, sensitivity, specificity, precision, and confusion matrices.
Select a sensitivity target with the clinical collaborator; then report the resulting specificity
and referral volume. Give ECG-level metrics with patient-cluster confidence intervals and variation across label seeds.
Keep observed test prevalence visible. Any reweighting to an assumed student prevalence is a
scenario calculation, not demonstrated performance in students. Evaluate calibration separately
before presenting classifier scores as probabilities.

The preparation scripts write reproducible manifests and check patient isolation. Waveform
download, GPU training, calibration, and evaluation are now implemented. See the
[results report](experiment001-results.md) for completed runs and their limitations.

Verified preparation run (seed 42):

| Partition | Recordings | Annotated abnormal / normal |
| --- | ---: | ---: |
| Labeled training | 1,518 | 913 / 605 |
| Unlabeled training | 15,900 | Hidden from learner |
| Validation | 1,870 | 1,191 / 679 |
| Test | 1,896 | 1,195 / 701 |

The selected 1,335 patients are 9.9985% of the 13,352 eligible training patients. This gives
9.88% of eligible training recordings and 8.72% of all training recordings, because unresolved
records remain unlabeled. The proxy excludes 313 validation and 302 test recordings. The
retained evaluation population has many abnormal ECGs and does not simulate the low disease
prevalence expected among students. All 17,418 training waveforms can be used for self-supervision.
The generated `summary.json` also records metadata hashes and all exclusion counts.
