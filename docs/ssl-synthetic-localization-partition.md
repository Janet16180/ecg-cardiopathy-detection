# Patient partition for SSL and synthetic localization

18 October 2026. Prospective data-preparation policy for the independent SSL/synthetic stream.
This policy is committed before building the manifest, choosing a model or scoring localization.
It implements Stage 0 of [the research plan](ssl-synthetic-localization-plan.md).

## Scope and allocation

Use PTB-XL 1.0.3 metadata only. Interpret `ecg_id`, `patient_id`, `strat_fold` and `filename_hr`;
do not interpret diagnoses, reports, age or sex. Hash the original metadata bytes for provenance.
No waveform, diagnostic statement table or external dataset is read by this builder. Only check that
eligible `.hea` and `.dat` paths exist; existence is not waveform validation.

Eligible source folds are 1–8. Quarantine every record of a patient with any record in folds 9–10,
including that patient's otherwise eligible records. Fail on missing or malformed identifiers/folds,
duplicate ECG identifiers, unexpected filenames or missing eligible waveform/header paths. Patient
IDs written as integral decimals (for example `123.0`) normalize to the same integer as `123`.

Assign eligible patients deterministically, independently of row order, diagnoses and recording count:

1. UTF-8 encode `ssl_synthetic_partition_v1:20261018:<canonical patient ID>`.
2. Take SHA-256, interpret the full hexadecimal digest as an unsigned integer, and reduce modulo 10,000.
3. Buckets 0–7,999 are `train`; 8,000–8,999 are `validation`; 9,000–9,999 are `engineering_test`.

These are expected 80/10/10 patient proportions, not exact quotas. Never retry the seed to improve
balance or scores. The configuration fixes the seed, hash namespace and cut points. All repeat ECGs,
windows, original/edited pairs, synthetic variants and cached features inherit their background
patient's role. Synthetic latent subjects will need equivalent disjoint role-specific namespaces in
the generator protocol; this manifest does not assign identities to future simulated subjects.

Only `train` patients may enter SSL, generator fitting or supervised optimization. Validation may
select settings and thresholds. `engineering_test` cannot enter SSL, generator fitting, normalization
fitting or selection. Freeze its scoring protocol before reading its waveforms. It is a prospective
engineering evaluation within previously used PTB-XL, not an untouched clinical confirmation cohort.
Keep source folds 9–10 excluded regardless of future use of engineering-test results.

## Evidence and gates

Write the record-level manifest locally to
`data/processed/ssl_synthetic_localization_v1/partition.json` and a separate aggregate receipt to
`outputs/ssl_synthetic_localization/partition_summary.json`. Never commit the manifest or patient IDs.
Include exact metadata, policy and implementation hashes, row counts, role-level patient counts and
zero-overlap assertions. Reject an existing output instead of overwriting it. A repeated invocation
at new output paths must produce the identical manifest for identical inputs.

The manifest establishes metadata patient separation only. It does not establish cross-patient exact
waveform uniqueness, biological identity accuracy, waveform quality, clinical labels or absence from
an externally pretrained checkpoint. Consequently it records `training_admitted: false`. A forward
waveform/duplicate audit must qualify the permitted data path before any training. Do not read closed
Challenge or EchoNext payloads to fill this gap. No score or final-test permission follows from this
partition, and no new disease definition is introduced.

## Verification and decisions

Unit tests must demonstrate stable allocation under reordered rows, normalization of patient IDs,
repeat grouping, exclusion of a mixed-fold patient, and fail-closed behavior on bad metadata. They must
also prove that only the four permitted fields are interpreted and waveform files are never opened.
Test local output reproducibility and refusal to overwrite through the CLI on artificial data.

After execution, report all counts and limitations from the aggregate receipt. Empty roles, missing
eligible files or any overlap fail preparation; never silently discard affected eligible records.
This data-preparation result has no phenotype endpoint, predecessor accuracy comparison or GPU run.
Generator qualification and a prospective waveform/duplicate-audit policy are the next steps.
