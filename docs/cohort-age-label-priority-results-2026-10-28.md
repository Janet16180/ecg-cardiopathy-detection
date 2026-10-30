# Recovered cleaning and quality-first age cohorts

The completed admitted pool contains 28,568 ECGs. Exactly 770 have an exact reported age from 18 through 25.
The strict primary construction therefore contains 770 records, and its general-age control contains the same
number with exactly matched source, label-evidence rank and binary target. Both have 506 negative and 264
positive ECG-annotation proxy labels. No eligibility rule was weakened to increase the sample size.

The previously constructed 25,000-record cohorts are preserved as secondary datasets. The age-preferred adult
25k cohort contains only 770 records aged 18–25 (3.08%); its mean exact age is 59.95 years. It should be described
as an age-preferred adult cohort, rather than a university-age cohort.

This report uses the completed local receipts and summary files listed below. No model was trained or scored
by these data stages. Protected Challenge access was restricted to the user-authorized mechanical reconstruction,
duplicate checks and fixed-window hash references. No EchoNext test or credentialed MIMIC/EchoNext data was used.

## Cleaning and original-split integrity

The recovered seed-42 PTB training manifest matches its original DVC MD5
`82d391124fdf28d5ee2a55d29e5dcf2c`. All 17,418 training waveforms were checked against official raw-file checksums
before and after decoding; 17,405 passed the frozen quality policy, exactly matching the documented historical
PTB count. The completed
[PTB quality receipt](../outputs/cohort25_recovery_20261028/ptb/quality/receipt.json) binds the inputs and unchanged
producer code. Its per-record CSV SHA-256 is
`8485ec71be1a2dc06a6cc68cc9134f4a07292aab3e9cad7f610d37118e7e645e`.

The original 65,824-row Challenge split was reconstructed with exactly its original rows SHA-256:
`5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8`.
The [reconstruction report](challenge-recovery-2026-10-28-results.md) and
[completed metadata](../outputs/cohort25_recovery_20261028/challenge/full/metadata.json) describe the original
producer snapshot and the resumed checkpoint provenance.

Only original Challenge training records entered the subsequent cleaning stage. Official checksums were checked
before and after each decoded view. The existing strict/centered view transforms and frozen signal-quality
policy were reused. Original duplicate decisions were preserved; exact strict-window copies were also checked
for mapped-label and known-sex conflicts. Longer CPSC crops retain their record-level annotation evidence for
SSL use; they do not acquire supervised crop labels. This is a conservative eligibility policy, not a finding
that all crop labels are wrong.

| Source | Original training records | Passed view, duplicate and QC gates | Strict 10 s supervised eligible | Negative | Positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chapman/Shaoxing | 6,147 | 6,127 | 3,050 | 819 | 2,231 |
| Ningbo | 20,941 | 19,803 | 9,952 | 2,581 | 7,371 |
| Georgia | 6,207 | 6,102 | 5,136 | 1,023 | 4,113 |
| CPSC 2018 | 4,121 | 3,765 | 832 | 197 | 635 |
| CPSC-Extra | 2,074 | 1,797 | 696 | 0 | 696 |
| Total | 39,490 | 37,594 | 19,666 | 4,620 | 15,046 |

The [full training-quality receipt](../outputs/cohort25_recovery_20261028/challenge_training_quality/receipt.json)
is complete. Its table SHA-256 is
`72e175e3d499f725a2d3a5a00cf3fe1263939ddd445dbd2a81e372fe0b32d92d`.
These are new supervised-eligibility artifacts, rather than a byte-identical historical 25k manifest with its
old storage pointers.

## Held-out reference coverage

Full-record hashes alone cannot establish whether two longer recordings share the same centered window.
The separate mechanical reference stage therefore recovered fixed canonical window hashes, without computing
model predictions or applying the quality-assessment function to held-out records. It reused the already
verified Ningbo whole-window hashes and read the other required public records with official checksum checks.

The [completed reference receipt](../outputs/cohort25_recovery_20261028/heldout_references/receipt.json) records
all 50,290 intended reference records: 26,334 Challenge calibration/test records irrespective of labels or
duplicate status; 6,554 PTB records from folds 8–10; and 17,402 existing SCREEN-01 SPH calibration/evaluation
records. Known patient IDs remain namespaced by dataset. No test diagnosis column is consumed by the reference
producer or the training-candidate loader.

There are 50,239 canonical window hashes. The remaining 51 records are retained with explicit view-contract
skip reasons: 25 constant-lead views, 23 short records and three nonfinite/shape failures. The reference-table
SHA-256 is `22f678f7b1329cca9799c634511c030fc69313abf585f5c0eb3b685c38371e5e`.
Every final cohort has zero observed overlap with the references by record ID, known patient ID, canonical
window hash or full-signal hash. The checks cover fixed canonical views; they are not an exhaustive search for
shifted, resampled or transformed copies.

## Label evidence and PTB target agreement

The [committed label-evidence policy](cohort-label-evidence-policy-2026-10-28.json) ranks documented annotation
process evidence before age. The admitted pool contains 13,002 rank-0 Chapman/Ningbo records and 15,566 rank-1
records; no rank-2 record was admitted. Source-process documentation is kept separate from PTB's per-record
human-validation flag. These ranks are not measured annotation error rates.

The current SCREEN-01 fitting roles restrict PTB to folds 1–7. The original stricter PTB supervised labels were
compared with current SCREEN-01 targets for all 13,432 common fitting records: there were zero disagreements.
The join normalizes integral patient identifiers exactly, without binary floating-point conversion, aligning
legacy `123.0` representations with current `123` representations while retaining dataset namespaces.

Among these 13,432 records, 4,526 lacked admitted label evidence and eight failed signal QC; four records meet
both exclusions. The resulting PTB contribution is 8,902 records. A false human-validation flag does not itself
establish that a label is wrong or has never been reviewed; exclusion follows the explicit evidence policy.
No proxy-label definition or high-voltage-only rule was changed.

## Strict 18–25 and matched general control

The [prospective derived-cohort protocol](cohort-strict-young-matched-protocol-2026-10-28.md) was committed before
selection. The strict set contains every admitted exact-age 18–25 record. The general control matches its
source, evidence rank and target counts exactly. Selection within each stratum uses seed 42. Shared training
records between arms are permitted and reported.

| Property | Strict 18–25 | Matched general age |
| --- | ---: | ---: |
| ECGs | 770 | 770 |
| Negative / positive | 506 / 264 | 506 / 264 |
| Exact age 18–25 | 770 | 31 |
| Exact-age mean | 21.51 | 57.46 |
| Exact-age median | 22 | 59 |
| Exact-age range | 18–25 | 5–92 |
| Under 18 | 0 | 7 |
| Missing age | 0 | 2 |
| Right-censored over 89 | 0 | 4 |
| Known patients | 370 | 378 |
| Records without patient identity | 392 | 392 |

Each cohort contains PTB 378, Ningbo 196, Georgia 112, Chapman 62, CPSC 2018 21 and CPSC-Extra one. Their
rank-0/rank-1 counts are 258/512. The cohorts share 31 records and canonical windows, and 25 known patients.
General-control age summaries exclude the two missing and four censored ages.

Local output: `outputs/cohort_strict_young_matched_20261028/`, with `strict_18_25.csv`,
`general_matched_size.csv`, `summary.json` and `receipt.json`.
The [receipt](../outputs/cohort_strict_young_matched_20261028/receipt.json) SHA-256 is
`e5f9c154f1f64f4c53ec00f09c94111bf10e025a86602ed3018f023345016ab4`.
The control preserves source/evidence/class composition; patient mix, disease severity and acquisition
conditions can still differ. A later difference does not establish a causal effect of age.

## Preserved secondary 25k cohorts

| Property | Age-preferred adults | General all ages |
| --- | ---: | ---: |
| ECGs | 25,000 | 25,000 |
| Negative / positive | 7,989 / 17,011 | 7,393 / 17,607 |
| Exact age 18–25 | 770 | 668 |
| Exact-age mean / median | 59.95 / 62 | 62.17 / 65 |
| Under 18 | 0 | 152 |
| Missing / censored over 89 | 0 / 0 | 135 / 136 |
| Known patients | 6,246 | 6,359 |

The age preference operates within evidence rank: ages 18–25 first, then older adults in ascending age.
The adult cohort excludes minors and unknown ages; the all-age cohort admits all eligible ages and unknowns.
These secondary sets share 22,216 records/windows and 4,892 known patients.

Local output: `outputs/cohort_age_label_priority_20261028/`, including `younger.csv`, `all_age.csv`,
`candidates.csv`, `exclusions.csv`, `summary.json` and `receipt.json`.
The [assembly receipt](../outputs/cohort_age_label_priority_20261028/receipt.json) SHA-256 is
`6538f3b8700e4d4b1a39cffd8f7489e42501653428b718b4b8bd69df099d5782`.

## Reproduction and validation

The new orchestration module includes every requested local prerequisite: original PTB full/limited label
preparation; checksum-verified public mapping download from pinned commit
`e2a75fc01f729cb74cc4e853e054ce81e28381fc`; real resource profiles; exact Challenge split recovery; full
fixed-window references; training quality; the 25k assembly; and strict matched-cohort derivation. It invokes
existing CLIs through subprocess arguments and imports no scripts. Missing public mappings are published only
after their pinned SHA-256 checks pass. Existing mismatched files and completed outputs are never overwritten.

Prerequisites are the local raw public releases, the project's pinned Python environment and the completed
current SCREEN-01 qualification metadata used for administrative fitting roles and existing SPH references.
The command performs data reconstruction and cohort selection only:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH="$PWD" uv run --locked python \
  -m scripts.data.reproduce_age_label_cohorts \
  --input-root "$PWD" --seed 42 --workers 4
```

Use `--plan` to inspect the stages without downloading or processing data. Original PTB sampling retains seed
42 and the original Challenge split retains seed 20260929; cohort selections use seed 42. Child processes set
`PYTHONHASHSEED=0` and numerical-library thread limits to one. Hash ordering, transforms, profiles and quality
assessment otherwise introduce no random sampling.

The executed [pipeline receipt](../outputs/cohort_age_label_priority_20261028/pipeline_receipt.json) binds all
12 stage receipts. Eleven completed stages were verified and reused; the derived stage was executed by the
orchestrator. A second complete invocation verified resumption successfully and left the pipeline receipt
unchanged, SHA-256 `f5673a5fb5628ab1707bd455b3ba47d80d3c6c1479dd6f68b233dff654b1d74c`.
This verification is distinct from rerunning every raw waveform stage a second time.

The 25k assembler captures policy, protocol and source hashes before interpretation and rechecks them before
completion. Tests reject concurrent mutations, target flips, ambiguous patient-ID conversions, protected-row
label ingestion, held-out overlaps and changed resumption artifacts. Strict-age tests verify exact membership,
matched source/evidence/target strata and seeded order invariance. The targeted recovery, assembly, selector,
derivation and reproduction suite passed 47 tests; Ruff passed for the new derivation and orchestration files. Raw data, row manifests and model artifacts
remain local; this report contains aggregate results only.

The full reference run recorded 306.48 seconds elapsed. The full Challenge cleaning receipt recorded 1,018.25
seconds elapsed, including the controlled pause; it is not a model-training runtime. Real profiles completed
before both stages. No clinical validity or referral performance follows from these data-construction checks.
