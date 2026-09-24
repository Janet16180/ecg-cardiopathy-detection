# Public ECG data quality and preparation

This is a reproducible **dataset suitability assessment**, not a clinical grading of individual ECGs. The current supervised endpoint is a PTB-XL diagnostic annotation proxy; it does not establish whether a person is healthy or needs a university referral. The intended future cohort is young Italian university patients, whereas these public data mostly come from older clinical populations. Counts below describe local files and fixed audits, not performance results.

## Source, local evidence and provisional score

Two separate 0–10 scores avoid treating clean waveforms as trustworthy labels. **Signal** measures verified provenance, ability to decode twelve leads, duration/rate compatibility, and observed signal defects. **Labels for our endpoint** measures annotation provenance, patient grouping, whether our current local selection actually carries usable labels, and transfer to the intended population. Scores are informed judgments, not estimated error rates; a 64-record random sample cannot certify the whole cohort. `outputs/data_quality/local_audit.json` and `outputs/data_quality/code15_part0.json` hold the reproducible counts and sampling seed.

| Dataset | Origin and local state | Signal /10 | Labels for our endpoint /10 | Key interpretation |
| --- | --- | ---: | ---: | --- |
| [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | German clinical ECGs; all 21,799 local 500 Hz pairs | **9** | **8** | 18,869 patient IDs and published patient folds; 16,056 records marked human validated. It defines our proxy, but the age mix is older than the intended cohort. |
| [MIMIC-IV-ECG 1.0](https://physionet.org/content/mimic-iv-ecg/1.0/) | BIDMC hospital ECGs; 39,996-record, 7,920-patient local selection audited; larger selection still downloading | **8** | **2 currently** | Strong patient identity and scale for self-supervision. Our selected waveform pool deliberately uses **no diagnosis labels or reports**, so it cannot currently support supervised endpoint labels. |
| [Georgia / Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/) | Emory/US Challenge release; 10,344 complete local waveform pairs | **8** | **5** | 10,292 exactly 10-second records; Challenge diagnosis codes are available, but cross-dataset label definitions and unavailable local patient IDs limit direct supervised pooling. |
| [CPSC 2018 / Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/) | Chinese clinical Challenge release; 6,877 complete local pairs | **7** | **5** | 4,451 recordings exceed 10 seconds and 10 are shorter; one sampled record has a fully constant V5 lead. Diagnoses need explicit harmonization. |
| [CPSC-Extra / Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/) | Additional Chinese Challenge training records; 3,453 complete local pairs | **7** | **4 provisional** | 2,038 exceed 10 seconds and 12 are shorter. Some exact waveform copies overlap CPSC with different raw code sets; this does not establish which annotation is correct. “Extra” is not a held-out CPSC test set. |
| [Chapman/Shaoxing / Challenge 2021](https://physionet.org/content/challenge-2021/1.0.3/) | Shaoxing People's Hospital, China; download in progress | **8 provisional** | **5 provisional** | Downloaded pairs checked so far are 10 seconds at 500 Hz with Challenge diagnosis codes. Full-cohort audit must follow completed acquisition. |
| [CODE-15%](https://zenodo.org/records/4916206) | Brazilian Telehealth Network of Minas Gerais; metadata for 345,779 exams / 233,770 patients; first waveform archive verified | **7 provisional** | **5 provisional** | Patient IDs and broad age coverage help. Six released diagnosis flags have unresolved row-level provenance here; `normal_ecg` is explicitly automatic. Its 400 Hz, padded 7/10-second representation needs its own validated windowing policy. |

The CODE scores remain provisional after a full first-archive native audit: 19,878 accepted exams passed semantic verification, but 23 excluded exact duplicate signals include eight pairs with different released patient IDs and four pairs with different automatic `normal_ecg` flags. These pairs warrant split and annotation review; they do not by themselves establish incorrect clinical diagnoses. The [processing review](data-processing-astra-review.md) records the checks and artifacts.

The [Challenge organizers' cross-database analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC9469795/) documents heterogeneous labeling and substantial database shifts. No Challenge SNOMED code or CODE automatic `normal_ecg` flag is silently mapped to the PTB-XL proxy. These scores would change for a different task: MIMIC is particularly useful for waveform self-supervision despite its low *current supervised-label* score. The [xECG checkpoint](https://huggingface.co/riccardolunelli/xECG_base_model_v1) is a model, not another source of ECG waveforms.

## Exploratory observations

The read-only inventory is generated with `python -m scripts.audit_dataset_quality`. For each available waveform source, a deterministic random sample of 64 complete records (seed 42) was decoded. All sampled records in PTB-XL, MIMIC, Georgia, CPSC, CPSC-Extra, and the downloaded Chapman subset decoded as finite twelve-lead waveforms. One CPSC sample had a near-flat lead: `training/cpsc_2018/g3/A2758` has **V5 exactly zero across the entire record**, so the preparation rule excludes it rather than inventing that lead. This is an observed case, not an estimate of the cohort-wide defect rate.

| Dataset | Median age; ages 18–30 among known ages | Exactly 10 s | Longer than 10 s | Shorter than 10 s |
| --- | ---: | ---: | ---: | ---: |
| PTB-XL | 61; 6.34% | 21,799 500 Hz waveform pairs | — | — |
| Georgia | 62; 4.46% | 10,292 | 0 | 52 |
| CPSC 2018 | 64; 8.46% | 2,416 | 4,451 | 10 |
| CPSC-Extra | 65; 3.36% | 1,403 | 2,038 | 12 |
| Chapman/Shaoxing | partial download; refresh audit after completion | partial | partial | partial |
| CODE-15% | 54; 15.87% | Requires 400 Hz padding inspection | — | — |

These percentages use only valid reported ages; they are not the fraction of the entire population. In CODE metadata, 134,657 of 345,779 exams have an automatic `normal_ecg=True` flag. That flag is not a cardiologist adjudication or our target label. MIMIC age is not reported here because the current waveform-only selection does not join patient demographics. The 64 decoded waveforms from CODE part 0 were finite and twelve-lead. Exact-zero edge counts were `(0,0)` for 29, `(581,581)` for 33, and two other patterns for one each. These differ from the [release description's](https://zenodo.org/records/4916206) illustrative 48/648-sample symmetric padding. Exact-zero runs alone may include real flat signal, so they cannot establish the true recording duration; no CODE view is materialized until its padding and units are validated.

The prior MIMIC 40k audit selected 39,996 records from 7,920 whole patients. It accepted 39,457 records from 7,892 patients, excluding 512 for the strict input contract and 27 exact internal duplicates. These counts are from `data/processed/mimic_ssl_40k_cpc/metadata.json`; the continuing 200k download has **not** received the same completed audit. The earlier Georgia g1 pilot accepted 948/999, with 10 duration failures and 41 exact waveform duplicates; this is a subset, not an independent estimate of the full Georgia cohort.

The completed **strict ten-second preparation** audited all 20,674 available Georgia, CPSC, and CPSC-Extra pairs after verifying both files of each pair against the official SHA256 list. It accepted **13,686** and excluded **6,988**. Exclusion reasons are mutually exclusive in this ordered pipeline: 6,563 failed the duration rule (mostly longer CPSC recordings, which may still be useful for SSL), 389 were exact duplicates of an earlier accepted waveform in this pool, 30 had a constant lead, and 6 contained nonfinite decoded values. No exact PTB-XL waveform copy was found. Another 24 accepted records carry low-amplitude or high-amplitude **review flags**, not an automatic rejection. The observed broken-lead example `cpsc_2018:A2758` is among the 30 constant-lead exclusions. The exact record IDs and reasons are in `data/processed/public_ecg_quality/strict_10s/exclusions.csv`; the accepted views are indexed in its `manifest.csv`. The source and output hashes and per-source accepted counts are in the adjacent `metadata.json`.

| Source | Strict ten-second accepted / audited | Constant-lead exclusions | Exact pool duplicates | Nonfinite exclusions |
| --- | ---: | ---: | ---: | ---: |
| Georgia | 10,187 / 10,344 | 9 | 90 | 6 |
| CPSC 2018 | 2,295 / 6,877 | 14 | 107 | 0 |
| CPSC-Extra | 1,204 / 3,453 | 7 | 192 | 0 |

An additional read-only check, `python -m scripts.audit_duplicate_labels`, compared original diagnosis-code sets on the 389 exact duplicates. **165 have different raw code sets** from the retained identical waveform. Of these, 155 are CPSC-Extra recordings that exactly duplicate a CPSC 2018 waveform but carry a different code set. A subsequent relationship audit found 82 subset/superset pairs and 83 non-nested pairs; an added code is not proof that either annotation is clinically wrong. The audit writes every comparison to `outputs/data_quality/duplicate_label_comparisons.csv` and a summary to `outputs/data_quality/duplicate_label_summary.json`. Such pairs are unsuitable as independent examples or for different train/test splits, and their labels remain quarantined pending harmonization. The 4/10 CPSC-Extra label score is a provisional project-utility judgment, not measured label accuracy.

The separate centered-window SSL pass over all 10,330 CPSC and CPSC-Extra recordings accepted **9,626** ten-second views and excluded 704: 22 shorter than ten seconds, 656 duplicate canonical views, and 26 constant-lead windows. This view has 92 accepted records flagged for amplitude review. Its manifest, exclusions, and metadata are in `data/processed/public_ecg_quality/cpsc_ssl_center_crop/`. It is a **separate pool** from the strict pass; the counts must not be added together. Labels on longer recordings remain original record annotations and are not valid as automatically assigned window labels.

## Read-only preparation policy

`scripts/prepare_public_ecg.py` writes manifests, exclusions, and a metadata receipt **outside `data/raw/`**. `load_view` is the transformation downstream code can use to obtain canonical `[12, 5000]` physical-mV `float32` at 500 Hz; it reorders leads and never edits the source. The script checks each source file against the official SHA256 manifest, rejects nonfinite or malformed records and any constant lead, screens near-flat leads (`std < 0.01 mV`) and amplitudes above 10 mV for manual review, and rejects exact waveform copies of any PTB-XL record or a previously accepted source record. Flags are review prompts, not clinical quality judgments.

`strict_10s` accepts only exactly ten seconds; it is the conservative starting point for label-sensitive analysis. `ssl_center_crop` makes a centered ten-second view of longer records for **self-supervised** use; a diagnosis attached to an entire longer recording must not automatically be assumed true for the cropped window. Short records are excluded in both modes. Challenge releases here lack reliable local patient identifiers, so the manifest marks the record-based surrogate explicitly and does not claim leakage-safe patient splits. Do not merge these records into the frozen experiment cohort or report supervised test results until identity and label mapping are resolved.

The current evidence supports lead ordering, duration handling, integrity checks, and rejecting broken leads. It does not establish a safe universal denoising filter or amplitude normalization. If filtering/scaling is evaluated later, compare it against the unfiltered view using fixed patient-separated splits and fit any learned parameters on training data only.

Run, after a source acquisition receipt says `complete`:

```bash
.venv/bin/python -m scripts.audit_dataset_quality
.venv-pretrained/bin/python -m scripts.audit_code15_pilot
.venv/bin/python -m scripts.prepare_public_ecg --source georgia --source cpsc_2018 --source cpsc_2018_extra --policy strict_10s --output-dir data/processed/public_ecg_quality/strict_10s
.venv/bin/python -m scripts.prepare_public_ecg --source cpsc_2018 --source cpsc_2018_extra --policy ssl_center_crop --output-dir data/processed/public_ecg_quality/cpsc_ssl_center_crop
```

The output `exclusions.csv` gives every excluded ECG ID and reason. The companion `metadata.json` records counts, source checksum-manifest hashes, preparation-source hash, and output hashes. `--limit N --allow-incomplete` is only for a clearly marked acquisition pilot; it must not be mistaken for a complete dataset.

CODE-15% now has the separate native preparation path below; conversion to the project's canonical 500 Hz/10 s physical-unit view remains unresolved. For MIMIC and PTB-XL, retain their existing audited pipelines and fixed patient/fold rules. No new public source enters the active experiment queue as a result of this analysis.

The subsequent derived-waveform workflow and its EDA are documented in [Challenge post-processing](challenge-postprocessing.md) and [processed ECG EDA](processed-ecg-eda.md). The EDA supplies a curated SSL-only manifest with exact cross-view deduplication and a separate ADC-rail exclusion overlay; it does not change the raw files or published shards.

## CODE-15% native preparation

`scripts/prepare_code15.py` is a separate, read-only CODE preparation path. It verifies `exams.csv` and each selected ZIP against the official MD5 recorded in `data/acquisition/code_15pct.json`, checks the HDF5 exam IDs against `trace_file` and `patient_id`, audits every selected waveform, and writes a manifest with the true patient ID. Its optional `--materialize-native` output copies accepted `float32 [4096,12]` traces exactly to a new HDF5 outside `data/raw/`; it does not filter, resample, scale, remove zeros, or infer original duration. The release states 400 Hz and lead order `{DI,DII,DIII,AVR,AVL,AVF,V1,V2,V3,V4,V5,V6}`. The author's [model README](https://github.com/antonior92/automatic-ecg-diagnosis) discusses input scale, but that wording and the local edge-zero patterns do not independently establish the amplitude unit or the true 7/10-second boundary in this archive. The prepared metadata therefore explicitly marks canonical 500 Hz/10 s eligibility **false**.

The output keeps `manifest.csv` (exam/patient identity, HDF5 locations, waveform hashes and QC), `demographics.csv` (age/sex and network-predicted age), `source_labels.csv` (six released diagnosis flags and the explicitly automatic `normal_ecg` flag), and `exclusions.csv` separate. The [Zenodo record](https://zenodo.org/records/4916206) calls `normal_ecg` an automatic annotation; it does not specify a per-row derivation for the six diagnosis fields. The associated [CODE study](https://www.nature.com/articles/s41467-020-15432-4) describes combining cardiologist text reports and automatic analysis for its training labels. We therefore preserve the released flags without assuming they are independently adjudicated or mapping them to the PTB-XL endpoint. No patient split is assigned here; any later supervised or self-supervised split must group by `patient_id` and check cross-archive duplicates first.

The completed first-archive pass used this command (choose a new output directory for any rerun, because the writer refuses overwrite):

```bash
.venv-pretrained/bin/python -m scripts.prepare_code15 --parts 0 --materialize-native --output-dir data/processed/code15_quality/part0_native
```

The completed first-archive pass (`data/processed/code15_quality/part0_native/metadata.json`) audited **20,001** HDF5 rows and retained **19,878** from **19,307** patients. It excluded **61 constant-lead**, **38 all-zero**, **23 exact duplicate native-signal**, and **1 missing-CSV-metadata** row; that last HDF5 row has `exam_id=0`. Another **843** accepted rows have review flags, not automatic rejections. The dominant exact-zero edge patterns are `(581,581)` for 11,651 accepted traces and `(0,0)` for 7,481. The former implies an observed nonzero span of 2,934 stored samples, but it is not proof of a 7-second acquisition; the latter similarly does not prove a 10-second acquisition. These differ from the Zenodo examples of `(648,648)` and `(48,48)`. The completed output receipt records preparation source SHA-256 `196ba481010f418b6c02db9c77b20c0f2a36b61c895cd6eae00b63561f5be502`. After that pass, the writer was hardened to stage future outputs and publish a directory only after all checks finish; the existing completed pass retains its own accurate source revision in metadata.

When more ZIP parts have official verified receipts, pass their numbers to `--parts` and use a new output directory. `--limit N` is only a pilot; its metadata records that it is incomplete. The quality audit excludes nonfinite or all-zero signals, a fully constant lead, missing or inconsistent metadata, duplicate exam IDs, and exact duplicate native traces within the selected parts. Exact-zero edge lengths are stored for investigation, **not** interpreted as recording duration. A near-flat lead or asymmetric edge run gets a review flag rather than an automatic exclusion.

The writer refuses an existing output path or `.inprogress` directory and publishes the final directory only when the manifest, exclusions, materialized HDF5, file hashes and metadata have all been written. Verify every materialized row and file hash with:

```bash
.venv-pretrained/bin/python -m scripts.verify_code15_prepared data/processed/code15_quality/part0_native --receipt outputs/data_quality/code15_part0_materialization_verification.json
```

That verification completed: all **19,878** accepted trace hashes and exam IDs, **123** exclusion rows, and all **5** output-file checksums matched the published metadata. Its receipt is `outputs/data_quality/code15_part0_materialization_verification.json`.

The staged writer revision was also exercised on the first three real rows with `--limit 3 --materialize-native --output-dir data/processed/code15_quality/pilot_atomic_3`. All three rows passed, the `.inprogress` directory disappeared on successful publication, and the row-hash verifier passed. This three-row check tests the publication behavior; it is not a second dataset-quality estimate.
