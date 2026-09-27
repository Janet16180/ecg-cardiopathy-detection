# Review of the project's data pipeline

This file collects the findings that concern this project's own processing, as opposed to the datasets
themselves (those are in the notebooks under `notebooks/`). Every number was reproduced from the raw files
with the functions named below, which live in `ecg_experiment/eda/`; `notebooks/README.md` explains how to
run them. Function names are given relative to that package.

## Verdict

The data engineering is careful and largely correct. Files are read correctly, leads are ordered by name,
patients never leak across splits, the processed arrays match the raw files, and every exclusion can be
reproduced. The problems are in scientific choices that make the reported results look better, or more
general, than they are: an evaluation set without its hardest cases, no demographic baseline, an unexamined
device shortcut, and a label rule that narrows what "normal" means.

## What was checked and is correct

| Check | Result | Reproduce with |
| --- | --- | --- |
| PTB-XL binary label (`scripts/data/prepare_ptbxl.py`) | Matches its documented rule in all 8 manifests, 0 mismatches | `ptbxl_labels.reproduction_mismatches` |
| PTB-XL splits | Official folds; folds 9 and 10 (100% human-validated) for evaluation; no patient shared between train, validation and test; the 1,518-label subset is nested and patient-complete | `ptbxl_labels.split_integrity` |
| 100 Hz PTB-XL cache | Matches the raw 100 Hz files to float32 rounding (about 1e-7 mV) | `processed.compare_cache_100` |
| 500 Hz union dataset | Every sampled PTB-XL, MIMIC, Georgia and CPSC window matches its raw file to float32 rounding | `processed.compare_union` |
| MIMIC lead order | The swapped aVF/aVL header order is handled by name everywhere | `mimic.stored_lead_orders` |
| MIMIC selection | Random whole patients; the 40k subset is an exact prefix of the 200k selection | `mimic.load_records` and both `selected_records.csv` files |
| MIMIC exclusions | All 2,610 records with NaN samples and 183 exact duplicates excluded; counts match exactly | `mimic.mimic_summary` with `mimic.project_accepted` |
| Challenge union curation | All 885 absent records explained: 744 exact duplicates of a kept record, 74 shorter than 10 s, 35 constant leads, 26 exact rail hits, 6 NaN | `processed.challenge_union_reasons` |
| CODE-15 | Not used while units and duration were unresolved; the all-parts audit reproduces (670 all-zero tracings plus the 18 filler rows) | `code15.padding_table` against `data/processed/code15_quality/all_parts_manifest_v1/metadata.json` |

## Issues, most important first

### 1. The PTB-XL evaluation set excludes the hard cases

The label rule leaves 615 of the 4,381 records in folds 9 and 10 (14%) without a label, so they are dropped
from validation and test:

- 446 are NORM under PTB-XL's diagnostic grouping, with an extra rhythm or form statement (sinus arrhythmia,
  sinus bradycardia or tachycardia, PVCs, prolonged PR);
- 92 have an abnormal diagnosis alongside NORM (for example incomplete right bundle branch block);
- 77 have no diagnostic statement at all (paced rhythms, atrial fibrillation and flutter coded only as rhythm).

The reported AUROC of about 0.95 is therefore measured on an easier subset than PTB-XL contains, and it is not
comparable with published PTB-XL results.

Recommendation: also report results on all of folds 9 and 10 with the standard superclass label
(`ptbxl.superclass_label`), and state which labeling the headline numbers use.

Reproduce with `ptbxl_labels.label_rule_comparison` and `unlabeled_composition`.

### 2. No metadata baselines

For the project's label, age alone reaches a test AUROC of 0.79 (validation 0.79). Device and site each reach
about 0.65. No report compares a model against these.

Recommendation: report an age-only and an age-plus-sex baseline next to every model, and stratify results by
age band.

Reproduce with `ptbxl_labels.metadata_baselines`.

### 3. Device is a plausible shortcut, and it shifts between training and test

Among labeled records, the positive rate ranges from 24% (`CS-12 E`) to 76% (`AT-6 C`) by device. `CS100 3`
has a positive rate of 74% and a distinct filtering fingerprint (about 10x less low-frequency power). It is 35%
of the training folds but only about 3% of validation and test. A model can partly learn the device instead of
the disease, and the evaluation folds cannot detect it.

Recommendation: report performance per device, and check whether model scores predict the device.

Reproduce with `ptbxl_labels.positive_rates` and notebook 01, section 8.

### 4. The label rule restricts the negatives' heart rate

Because NORM records with sinus bradycardia or tachycardia become "unlabeled", every negative has an estimated
heart rate between 43 and 117 bpm, while positives range from 27 to 199 bpm. Among labeled records, an extreme
heart rate alone means "positive". This is created by the label definition and will not hold in a cohort where
normal-but-tachycardic ECGs count as normal. (The heart-rate estimator agrees with MIMIC machine measurements to
a median of 0.5 bpm.)

Recommendation: state this in the endpoint definition, and consider labeling normal ECGs with sinus brady- or
tachycardia as negative.

### 5. Waveform artifacts remain in the training data

The earlier cleaning rules were exact and source-specific, so several kinds of artifact passed:

- 488 local MIMIC records have a lead that is constant for all 10 seconds. The MIMIC audit only drops records
  with NaNs, so 307 remain in `mimic_ssl_200k` and 65 in `mimic_ssl_40k_cpc`. The union build removes those
  65, but `cpc_pool_40k` (built from the 40k manifest) and `sampled_100k_plus_labels_v1` (114 records) keep
  them.
- The union's rail rule only matches the exact values +32.767 and -32.768 mV. 29 CPSC windows contain
  clipped plateaus just below the rail (32.69-32.76 mV, a median of 84 samples each) and are still in the
  union.
- No rule catches leads that drop out for part of the recording, electrode pops above 20 mV, or recordings
  that are mostly noise.

Status: `ecg_experiment/ecg_quality.py` now applies one waveform-only policy to every source, and
`scripts.data.build_clean_cohorts` builds the nested `clean_25k/50k/100k_plus_labels_v1` cohorts with it
(see [clean cohorts v1](clean-cohorts-v1.md)). The policy removes 0.33% of the candidates; the earlier 100k
cohort contains 339 records that fail it. Frozen historical datasets were not changed.

Reproduce with `mimic.mimic_summary`, `mimic.project_accepted`, `processed.near_rail_windows` and
`notebooks/06-jr-clean-cohorts.ipynb`.

### 6. Chapman/Shaoxing is clean

Chapman was left out of the 76,598-record union as "incomplete processing". On every check it is the
cleanest Challenge source: every record is 10 s long, with no rail artifacts, no NaN and no limb-lead
violations, and only 15 records have a constant lead. It was later added to `sampled_100k_plus_labels_v1`
and is part of the clean cohorts, where the policy removes 2 of its 10,219 records.

### 7. CODE-15 documentation and preparation

- The waveforms were moved into `outputs/data_export/code_15pct_waveforms_2026-09-25.tar.xz` (verified; the
  SHA-256 matches `CODE_SHA256SUMS`), but `data/acquisition/code_15pct.json` and `docs/data-sources.md` still
  point to ZIPs in `data/raw/`. Update both.
- Before any use: rescale the amplitudes (about 2x mV; confirm the factor with the dataset authors), crop the
  zero padding, drop the 18 filler rows and 688 all-zero tracings, and keep the recording length (7.3 s vs
  10.24 s) as a covariate.

### 8. Smaller notes

- The earlier waveform EDA (`outputs/data_quality/local_audit.json`) sampled only 64 records per source. The
  notebooks here cover every record.
- `cpc_pool_40k` resamples each 5-second half separately, which leaves filter edge effects of up to about
  0.3 mV at the 5-second mark. This is consistent with the CPC model, which encodes each half on its own
  (`ecg_experiment/cpc.py`), so it is not a problem for CPC. A model that reads the full 10 s from this pool
  would see the join.
- One MIMIC cart (13% of local records) stores inexact limb leads, applies no baseline filter and produces no
  machine interpretation. Keep `cart_id` as a covariate for models that rely on lead relationships.
- If MIMIC machine measurements are ever used as labels: QTc is truncated at about 500 ms on the main cart
  system, and 29999 means "not measured".
- Every source is identifiable from noise statistics alone (65% balanced accuracy over seven sources). This is
  harmless for pretraining, but sources should not be mixed for supervised evaluation without per-source
  reporting.

## Suggested order of work

1. Re-evaluate on the full folds 9 and 10, and add age and age-plus-sex baselines (issues 1-2). It is cheap,
   and it changes how every result is read.
2. Check the device shortcut with per-device results (issue 3).
3. Use the clean cohorts for new pretraining and data-scaling runs (issue 5). The 25k, 50k and 100k cohorts
   differ only in size.
4. Update the CODE-15 receipt and docs, and treat a rescaled CODE-15 as a separate, recorded data-scaling
   experiment (issue 7).
