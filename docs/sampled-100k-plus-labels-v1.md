# Seeded 100k-plus-labels training cohort, version 1

The completed `data/processed/sampled_100k_plus_labels_v1/` cohort contains
**100,000 sampled unlabeled ECGs plus all 15,359 resolved PTB-XL training
labels**, for **115,359 unique training waveforms**. Its selection seed is
`20260926`. The fixed 1,518-label subset remains available for controlled
limited-label experiments. No PTB development, calibration, or test record is
in the training manifest; unknown cross-source patient overlap remains a
limitation of the public data.

All signals use the existing canonical 500 Hz, ten-second, twelve-lead,
float32 physical-mV contract. The manifest references verified shards from the
previous training union and Chapman view, plus original MIMIC WFDB files. It
does **not copy or modify** any waveform. The directory occupies about 22 MB;
the underlying source datasets must remain locally available. The loader checks
source shard hashes on first access in each worker and waveform hashes on every
read, including raw MIMIC records.

| Source | Seeded unlabeled sample | Resolved labels | Total |
| --- | ---: | ---: | ---: |
| MIMIC-IV-ECG | 70,000 | 0 | 70,000 |
| Chapman/Shaoxing | 9,561 | 0 | 9,561 |
| Georgia | 9,531 | 0 | 9,531 |
| CPSC 2018 | 6,157 | 0 | 6,157 |
| CPSC-Extra | 2,826 | 0 | 2,826 |
| PTB-XL | 1,925 | 15,359 | 17,284 |
| **Total** | **100,000** | **15,359** | **115,359** |

The program caps the planned MIMIC share at 70% of the unlabeled sample and
apportions the other 30% by available deduplicated source counts, using largest
remainders for an exact 100,000. A source-specific random stream samples its
records uniformly without replacement; a final seeded shuffle determines
manifest order. Sorting candidate IDs before sampling makes selection stable
if input table row order changes. The builder recorded **zero further exact
waveform duplicates** among the eligible candidates after their existing
source quality gates. Selected record IDs and waveform hashes are all unique.

The 15,359 resolved labels comprise **9,487 positive** and **5,872 negative**
annotations from **13,351 PTB patients**. The 70,000 selected MIMIC records
come from **26,010 identified patients**. In the selected canonical shard
records, **94** have retained review flags: 90 for amplitude above 10 mV and
four for a near-flat lead. These are review flags, not automatic rejection
reasons. The corresponding source manifests retain the record-level details;
the accepted MIMIC audit and this selection did not create new clinical labels.
The [aggregate EDA receipt](../outputs/data_quality/sampled_100k_plus_labels_v1/eda.json)
records these counts and the storage-backend mix without publishing patient rows.

A later complete waveform pass for the local CPC cache found **114 MIMIC ECGs
with a fully constant lead** among the selected records. They are finite and
hash-valid, and the original MIMIC acceptance audit did not exclude constant
leads. They remain in this immutable sampled cohort and are flagged as an SSL
quality limitation; they carry no mapped binary target. The sampled loader
now follows the MIMIC source policy while retaining its stronger constant-lead
check for the other sources. No raw waveform or selection row was changed.

The PTB labels are the existing resolved diagnostic-annotation proxy. The
other 2,058 original PTB training records lacked a resolved target under that
definition; 1,925 were sampled for SSL only. The unlabeled loader masks targets
even for labeled PTB rows. MIMIC diagnoses, Challenge/Chapman codes and CODE15
flags are **not** mapped to this binary target. CODE15 remains outside this
cohort because its native signal units/padding and label meaning need their own
validated policy. Source patient IDs are available for PTB and MIMIC; the
Challenge/Chapman prepared views lack verified patient IDs. No new patient-level
evaluation split is claimed for those sources.

The selection is a **new data-scaling cohort**, not a change to the frozen
56,875-record experiments or the existing 76,598-record training union. It
does not schedule or imply model training or a performance improvement. Future
comparisons must use a matched old-pool control and keep the existing PTB
development, calibration and test partitions separate.

## Rebuild and use

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -m scripts.data.build_sampled_training_dataset \
  --output-dir data/processed/sampled_100k_plus_labels_v1 \
  --unlabeled-records 100000 --seed 20260926 --mimic-fraction 0.7
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -m scripts.validation.verify_sampled_training_dataset \
  --dataset-dir data/processed/sampled_100k_plus_labels_v1 \
  --receipt outputs/data_quality/sampled_100k_plus_labels_v1/verification.json
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -m scripts.reports.summarize_sampled_cohort \
  --dataset-dir data/processed/sampled_100k_plus_labels_v1 \
  --output outputs/data_quality/sampled_100k_plus_labels_v1/eda.json
```

The builder refuses an existing output and publishes a complete manifest
atomically. Reproduction requires a new directory or deliberate removal of a
failed candidate, never editing this released manifest in place. The dataset
metadata SHA-256 is
`2e6ad95a7381e227efe3bbbbb3f722b1c7f698a7dbfac8fa4dabe6aa3b08f68c`;
the selection manifest SHA-256 is
`82a316a360a95dac3f24545c964afc1ad6b9654fa830541f6871b49371353288`.
The metadata binds input table hashes, source code hashes, source shard hashes,
seed, quotas and exact counts. The [verification receipt](../outputs/data_quality/sampled_100k_plus_labels_v1/verification.json)
checks all 115,359 selected references against the source manifest or MIMIC
audit, all referenced shard paths and input-table hashes, PTB patient
separation, label masks, and 16 real waveform reads. It has SHA-256
`d94f3436b6fc4c8b5da7dce02ae672ad5d330cb3f2ebc4981cbdbf40533b1f39`.
A two-worker CPU loader smoke also read all seven source/backend combinations.
Replaying the full candidate assembly and seed reproduced the published
115,359 record IDs in exactly the same order. Focused dataset tests and Ruff
passed.
The receipt is not a full independent reread of all 115,359 signals; the
upstream MIMIC, Chapman and union receipts supply their full source audits.

```python
from ecg_experiment.sampled_training_dataset import SampledTrainingECGDataset

directory = "data/processed/sampled_100k_plus_labels_v1"
ssl = SampledTrainingECGDataset(directory, purpose="ssl")
labeled = SampledTrainingECGDataset(directory, purpose="supervised", label_budget="1")
limited = SampledTrainingECGDataset(directory, purpose="supervised", label_budget="0.1")
```
