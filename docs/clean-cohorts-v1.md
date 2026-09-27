# Quality-filtered 25k, 50k and 100k training cohorts, version 1

27 September 2026. Three nested, manifest-only training cohorts built from the same candidates as
[`sampled_100k_plus_labels_v1`](sampled-100k-plus-labels-v1.md), after applying one waveform quality policy
derived from the [dataset EDA](../notebooks/README.md). No waveform is copied or modified, and no model was
trained. The analysis is in `notebooks/06-jr-clean-cohorts.ipynb`.

| Cohort | Unlabeled ECGs | Labeled PTB-XL | Total | MIMIC | Chapman | Georgia | CPSC 2018 | CPSC-Extra | PTB-XL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `data/processed/clean_25k_plus_labels_v1` | 25,000 | 15,349 | 40,349 | 17,500 | 2,395 | 2,386 | 1,533 | 704 | 15,831 |
| `data/processed/clean_50k_plus_labels_v1` | 50,000 | 15,349 | 65,349 | 35,000 | 4,791 | 4,772 | 3,066 | 1,407 | 16,313 |
| `data/processed/clean_100k_plus_labels_v1` | 100,000 | 15,349 | 115,349 | 70,000 | 9,582 | 9,544 | 6,131 | 2,815 | 17,277 |

- **Nested.** Each source's clean candidates are sorted by record ID and permuted once with seed
  `20260927`. A cohort takes the first records of every source's permutation, so 25k is inside 50k, which is
  inside 100k.
- **Same mix.** The unlabeled part uses the rule of the earlier 100k cohort at every size: MIMIC capped at
  70%, and the rest shared in proportion to each source's clean candidates. Only the amount of data changes
  between cohorts, which is what a data-scaling comparison needs.
- **Labels.** All clean resolved PTB-XL training proxy labels are included in every cohort: 15,349 for the
  full budget and 1,517 for the limited one. Ten labeled records failed the policy, one of them in the fixed
  1,518-label subset. PTB-XL development, calibration and test records are never candidates and are not
  filtered.
- **Unchanged format.** The directories use the schema of `sampled_100k_plus_labels_v1`, so
  `SampledTrainingECGDataset` reads them unchanged, hash-checking every waveform it serves.

```python
from ecg_experiment.sampled_training_dataset import SampledTrainingECGDataset

ssl = SampledTrainingECGDataset("data/processed/clean_50k_plus_labels_v1", purpose="ssl")
labeled = SampledTrainingECGDataset("data/processed/clean_50k_plus_labels_v1", purpose="supervised")
```

## Quality policy

`ecg_experiment/ecg_quality.py` scores one canonical ten-second, twelve-lead, 500 Hz mV waveform. It uses the
waveform only, never a diagnosis.

| Rule | Type | Definition |
| --- | --- | --- |
| `nonfinite` | exclude | any NaN or infinite sample |
| `constant_lead` | exclude | a lead with a single value for the whole recording |
| `flat_segment` | exclude | a lead constant for at least 1 s |
| `adc_rail` | exclude | any absolute value of at least 32.6 mV (16-bit saturation, including just below the exact limit) |
| `extreme_amplitude` | exclude | peak above 20 mV and below the rail |
| `near_flat_record` | exclude | median lead standard deviation below 0.02 mV |
| `noise_dominated` | exclude | more than half of the median lead power in 40-150 Hz |
| `amplitude_over_10mv` | review | peak above 10 mV |
| `limb_identity_violated` | review | largest II - I - III residual above 0.05 mV |
| `strong_baseline_wander` | review | more than 80% of the median lead power below 0.7 Hz |

Review flags are recorded but never exclude. The limb-identity flag marks 11.7% of MIMIC records: one cart
stores limb leads independently, and excluding it would remove a device rather than bad recordings.

## What the policy removed

All 244,632 records (15,359 labeled PTB-XL rows and 229,273 deduplicated candidates) were read, hash-checked
and scored. 806 (0.33%) failed at least one exclusion rule:

| Source | Records | Excluded | % | Main reasons |
| --- | ---: | ---: | ---: | --- |
| PTB-XL, labeled | 15,359 | 10 | 0.07 | extreme amplitude (8) |
| PTB-XL, unlabeled | 2,058 | 2 | 0.10 | extreme amplitude |
| MIMIC | 197,207 | 720 | 0.37 | flat segment (550, of which 307 constant leads), extreme amplitude (145), noise (67) |
| Georgia | 10,187 | 10 | 0.10 | extreme amplitude, flat segment, noise |
| CPSC 2018 | 6,581 | 43 | 0.65 | flat segment (23), saturation (23) |
| CPSC-Extra | 3,021 | 19 | 0.63 | flat segment (13), saturation (6) |
| Chapman/Shaoxing | 10,219 | 2 | 0.02 | flat segment |

Visual checks of the first failing public record for each rule, and of the excluded PTB-XL records, showed
artifacts in every case: lead dropouts, all-lead saturation, electrode pops and high-frequency noise. The
earlier `sampled_100k_plus_labels_v1` cohort contains 339 records that fail the policy (0.29%), including its
114 known constant-lead MIMIC records.

## Rebuild

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.data.build_clean_cohorts --workers 12
```

The build took about 45 minutes, limited by disk reads. It refuses existing output directories. It writes
the per-record quality table (`record_quality.csv.gz`, which is reused when it covers the same records) and
an aggregate `receipt.json` to `outputs/data_quality/clean_cohorts_v1/`. Each cohort's `metadata.json`
binds the input table hashes, source code hashes, quality policy, seed, quotas, table hashes and source
shards. A new policy or seed needs new output directories, never an in-place edit.

| File | SHA-256 |
| --- | --- |
| `clean_25k_plus_labels_v1/metadata.json` | `7ba6b1b09a3a3609f55f13490e0a920090682edc880c469218fbe0b21e7141df` |
| `clean_50k_plus_labels_v1/metadata.json` | `682242f868ca08398300b8f7e4effac6687e539d1bc8af13ef594ccb8792c70a` |
| `clean_100k_plus_labels_v1/metadata.json` | `067082eab9e2789ca7c12fd6774c5113c47c54a409efcca28f14b2a7dcf5e318` |
| `outputs/data_quality/clean_cohorts_v1/record_quality.csv.gz` | `f1ea4073808076c59165f34be7dd058c5157a8a0b95a001721c8213773bc50e4` |

## Limitations

- The policy targets obvious recording failures. It does not detect lead reversals, subtle noise or
  mislabeled recordings, and passing it does not mean an ECG is clinically clean.
- Exact-hash deduplication does not find shifted or resampled copies, and Challenge and Chapman records have
  no verified patient IDs.
- CODE-15 is not included: its amplitude unit and padding need their own validated preparation (see the
  [pipeline review](eda-pipeline-review.md)).
- These are new cohorts. Experiments frozen on earlier cohorts stay as they are; comparing against them
  needs a protocol that names both cohorts.
