# Public-data training and local evaluation

The project can use all public training labels and additional unlabeled signals. The scarce
university annotations should support local adaptation and evaluation; their scarcity does not
require hiding most public labels in the final training strategy. The 10% PTB-XL experiment
remains a controlled representation-learning benchmark.

## Implemented expansion

- **Full-label PTB-XL reference:** all 15,360 eligible labeled training ECGs, with the same
  development/calibration/test partitions as the initial experiment. Frozen ECG-FM/ECG-JEPA
  probes and a supervised CNN provide practical reference systems.
- **Georgia SSL pilot:** download the 999 records in the official `g1` folder, validate official
  SHA256 checksums, require the existing 12-lead/500 Hz/10-second finite-signal contract, and
  remove exact decoded-waveform duplicates within the pool and against all PTB-XL folds.
  Accepted counts and exclusions are recorded in `data/processed/georgia_ssl_g1/metadata.json`.
  The completed audit retained **948** records: 10 failed the duration contract and 41 matched
  an exact waveform already represented in the audit/pool. The combined SSL pool therefore
  contains **18,366** ECGs. All 21,799 PTB-XL waveforms were included in the duplicate audit.
- **Controlled adaptation:** compare the released ECG-FM encoder against continued temporal
  contrastive adaptation on PTB-XL training signals alone and on PTB-XL plus the accepted
  Georgia pilot. Use the same exposed 10% downstream labels so the pooling comparison does
  not also change supervised data. This is a one-epoch feasibility experiment, with its budget
  chosen from timing measurements before downstream evaluation.

The Georgia pilot is a convenience subset, not a random or complete sample. It is small relative
to PTB-XL and cannot establish the value of full multicenter pretraining. No Georgia diagnoses
or reports enter the SSL manifest. Record identity is used for grouping when patient identity
is unavailable; same-person recordings and nonexact duplicate waveforms may remain. Exact
duplicate detection does not establish patient independence across sources.

## Sources and label compatibility

| Dataset | Available signals | Role and limitation |
| --- | --- | --- |
| [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | 21,799 twelve-lead ECGs | Existing patient-separated folds and the documented diagnostic proxy. |
| [Georgia, Challenge 2020](https://physionet.org/files/challenge-2020/1.0.2/training/georgia/) | 10,344 records; about 1.238 GB of waveform payloads | Additional SSL signals; the pilot uses only `g1`. |
| [Original CPSC 2018 cohort](https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/) | 6,877 records; about 1.316 GB of waveform payloads | Candidate separate evaluation cohort after defining a compatible endpoint and handling variable duration. Not evaluated here. |

The [challenge description](https://physionet.org/content/challenge-2020/1.0.2/) provides open
CC BY 4.0 downloads and an [official integrity manifest](https://physionet.org/files/challenge-2020/1.0.2/SHA256SUMS.txt).
The [organizers' analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC9469795/) documents heterogeneous
annotation procedures and approximate terminology harmonization. Sinus rhythm is not evidence
that an ECG has no other abnormality. Therefore these labels are not blindly pooled into a
single healthy/abnormal endpoint.

These public cohorts also differ from a young university population. The organizers report
mean ages around 60 years for Georgia and CPSC. The released
[ECG-FM checkpoint](https://github.com/bowang-lab/ECG-FM) and
[HuBERT-ECG pretraining corpus](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v3.full)
already include these public sources. Testing on a new downstream dataset would therefore be
external to task fine-tuning, but would not establish absence of historical waveform exposure.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.prepare_georgia_ssl
```

The pooling manifest contains exactly `ecg_id,patient_id,raw_dir,filename_hr,source`. The
`patient_id` column for Georgia explicitly encodes record grouping, not a known patient ID.
`scripts/adapt_ecgfm.py --help` describes how to add it with `--extra-ssl-manifest` and record
that limitation using `--extra-grouping-description`.

Example pooled adaptation followed by matched 10%-label fine-tuning (requires the official
ECG-FM checkpoint and extraction metadata described in the pretrained notes):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-pretrained/bin/python -m scripts.adapt_ecgfm \
  --manifest-dir data/processed/ptbxl/seed42_fraction0.1 \
  --raw-dir data/raw/ptb-xl/1.0.3 \
  --checkpoint third_party/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt \
  --checkpoint-metadata data/processed/pretrained/ecg-fm/metadata.json \
  --output-dir outputs/my_pooled_adaptation --epochs 1 --batch-size 32 \
  --extra-ssl-manifest data/processed/georgia_ssl_g1/ssl_manifest.csv \
  --extra-grouping-description "Georgia record grouping only; patient identities unavailable" \
  --device cuda
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-pretrained/bin/python -m scripts.finetune_pretrained \
  --model ecg-fm --manifest-dir data/processed/ptbxl/seed42_fraction0.1 \
  --raw-dir data/raw/ptb-xl/1.0.3 --output-dir outputs/my_pooled_experiment \
  --adapted-backbone outputs/my_pooled_adaptation/adapted_backbone.pt \
  --seed 42 --batch-size 16 --epochs 20 --patience 5 --device cuda
```

Omit the extra manifest and grouping arguments for PTB-XL-only adaptation; use a separate fresh
output directory. All adaptation starts from the same original released checkpoint. The pooled
run contains more examples and optimizer updates in its one pass, so a performance difference
would also include that additional compute.

## Pilot result

With the same 1,518 downstream training labels, original ECG-FM fine-tuning reached AUROC
0.9289, PTB-XL-only adaptation followed by fine-tuning reached 0.9315, and pooled adaptation
reached 0.9305. The direct pooled-minus-PTB-only change was −0.00103, with a paired
patient-bootstrap 95% interval of [−0.00239, +0.00028]. This pilot did **not** demonstrate a
benefit from adding the 948 Georgia signals. It does not establish that a larger or differently
balanced multicenter pool, another objective, or local adaptation would also fail.

Pooled adaptation used 574 optimizer updates versus 545 for PTB-XL-only adaptation. The
confidence interval is conditional on these fitted models and does not include retraining or
cohort-shift uncertainty. Full metrics and the fully labeled public-data comparisons are in
[the results report](experiment001-results.md).

For the real project, define the referral endpoint with the clinical collaborator, reserve local
patients for evaluation, and assess sensitivity, false referrals, and calibration in that cohort.
If unlabeled local evaluation waveforms participate in adaptation, label the evaluation as
transductive; maintain a separate untouched set to measure performance on future students.
