# Astra review of ECG preparation and verification

Review date: 24 September 2026. Scope: the new public-data preparation and materialization code and its relationship to the existing frozen experiments. This review does not replace the frozen cohort, train models, modify raw recordings, or turn the diagnostic annotation proxy into a clinical endpoint.

## Findings and applied changes

| Finding | Consequence | Applied correction |
| --- | --- | --- |
| `prepare_public_ecg` wrote directly into its destination and permitted an existing directory | A rerun could silently replace evidence referenced by a materialized view; interrupted output could resemble a completed audit | Refuse existing destinations, build in a unique staging directory, and atomically publish only a complete result; remove staging on failure |
| PTB-XL exact-identity cache was keyed only by `ptbxl_database.csv` | Replacing a waveform while leaving the CSV unchanged could reuse a stale leakage reference | Version-2 reference verifies official checksums of the metadata and every referenced `.hea`/`.dat` before reuse; cache identity includes official checksum manifest, decoder source, NumPy/WFDB versions, record count and hash-list digest |
| Unknown direct-call crop policy silently fell through to the first ten seconds | A spelling error could produce a different waveform view with apparently valid dimensions | Reject unknown policy before reading; check decoded rank and channel dimensions; validate distinct source selection and positive limits |
| Materialized verification checked bytes and shape but not publication or label eligibility semantics | A self-consistent rewritten manifest could claim wrong units, wrong crop bounds, patient identity, or supervised eligibility | Verify complete/schema, canonical metadata and per-row units/leads/rate/duration, distinct identity/shards, safe shard paths, conservative annotation/patient eligibility, constant leads and independently recomputed amplitude/std statistics |
| Materialization published before an independent reread | Writer and storage problems could be discovered only after publication | Run the complete verifier on staging before atomic publication; record the canonicalization module hash alongside the writer hash |

The new PTB cache is `outputs/data_quality/ptbxl_reference_hashes_v2.json`. The earlier cache and published datasets are retained as historical evidence. Checksum verification is intentionally repeated: caching the result of an integrity check based only on file metadata would recreate the original problem. The cache avoids expensive repeated WFDB decoding, rather than avoiding the integrity check itself.

## Scientific interpretation and remaining gates

The existing strict representation preserves physical mV values and canonical lead order without filtering or scaling. Long records are center-cropped only into the SSL view; source diagnostic codes do not automatically label that window. These are appropriate conservative choices while label harmonization and diagnostic fidelity remain unvalidated. A bandpass, notch filter, baseline removal, clipping, or per-record scaling can alter clinically relevant morphology and amplitude; adding one should require an explicit downstream objective and development-only comparison, with preprocessing parameters and original views retained.

The current QC catches missing/nonfinite data, full-lead constants, extreme amplitudes, reviewed signed-16 recording rails and exact duplicates. A passing hash verifies identity, not clinical waveform quality. Intermittent dropout, electrode-motion artifact, lead reversal, baseline drift, mains interference, timing abnormalities and near-duplicate recordings are not fully resolved by those checks. New signal-derived measures should be descriptive review flags until their thresholds have source-stratified validation. A fixed standard-deviation threshold alone must not label an ECG pathological or healthy.

The published Challenge IDs are record identifiers. Setting them equal to `patient_id` does not make patient-disjoint splitting possible. The verifier enforces the current `patient_identity_known=false` and `patient_independent_eval_eligible=false` contract. Neither same-record deduplication nor different-dataset names establish patient independence.

The Challenge candidate pool also overlaps the Georgia pilot already used in the frozen training pool. Before any scaling experiment, compare record IDs and canonical waveform identities against **every** existing pool component, not just PTB-XL. Remove/group overlapping records during construction of a versioned union, retain lineage and source composition, and keep the existing comparison cohort unchanged. Exact identity does not rule out resampled, shifted, scaled or partial near-duplicates.

PTB-XL preparation checks official patient folds and its label subset is selected by patient. The older baseline `ecg_experiment/data.py` demeans each record and derives channel scale only from training rows; that train-only fitting is the right separation principle. This review leaves its frozen implementation unchanged. MIMIC's accepted IDs come from the official patient mapping, which is stronger than Challenge surrogate IDs, but the current waveform-only pool is not a supervised diagnostic dataset.

CODE-15% needs separate native-unit/duration and split gates. Its exact-zero edges are observations, not proof of the original signal boundaries. Retain native values until supported conversion is established, keep automatic `normal_ecg` distinct from adjudicated labels, and group records by the retained patient ID before assigning any downstream split.

The earlier 1–10 scores are qualitative project-suitability judgments. They are not calibrated clinical signal-quality ratings, measures of cardiologist agreement, or evidence of suitability for a young Italian university cohort. Source-stratified retention, missingness, demographics, duplicate/label conflict counts and representative waveforms provide more defensible evidence than a single score.

## Verification

Commands and results are recorded below after execution. All output artifacts are separate from the historical materializations and frozen experiment inputs.

```bash
.venv-pretrained/bin/python -m pytest -q tests/test_prepare_public_ecg.py tests/test_materialize_challenge_ecg.py
.venv-pretrained/bin/python -u -m scripts.prepare_public_ecg --source georgia --policy strict_10s --limit 32 --output-dir data/processed/public_ecg_quality/astra_v2_pilot32
```

The targeted suite passed 21 tests. It includes adversarial checks for raw-byte corruption with unchanged metadata, refusal to overwrite a published audit, cleanup after failed publication, and semantically invalid manifests whose file hashes have been recomputed. Full historical-waveform verification and the new end-to-end pilot are recorded in `outputs/data_quality/astra_review/`.

Full verification passed for every existing Challenge view: **13,686 strict records across 107 shards** (83.11 seconds) and **9,626 centered records across 76 shards** (63.99 seconds). The receipt is [`challenge_full_verification.json`](../outputs/data_quality/astra_review/challenge_full_verification.json); it binds the source, metadata and manifest hashes. These are integrity/QC-contract results, not clinical adjudication.

## Correction to the earlier duplicate-label interpretation

The 165 unequal code-set pairs are **annotation differences, not 165 established clinical contradictions**. Of all 389 duplicate pairs, 224 have identical raw code sets, 76 have an excluded-copy strict superset, 6 have a retained-copy strict superset, and 83 have non-nested differences. Added annotations can reflect richer labeling, and distinct SNOMED codes can sometimes require equivalence mapping. None of those pairs has been adjudicated here. Preserve the conservative unmapped-label gate, but do not conclude that CPSC-Extra labels are inherently less accurate solely from these counts. The earlier 4/10 label-utility score is therefore provisional rather than an evidence-based accuracy ranking. The reproducible counts and input checksum are in [`duplicate_annotation_relationships.json`](../outputs/data_quality/astra_review/duplicate_annotation_relationships.json).

The new 32-record Georgia audit and materialization both passed end-to-end. All 32 manifest rows are identical to their historical strict counterparts; rebuilding the PTB identity reference produced exactly the same 21,799 unique hashes. Warm cache reuse with current-byte official checksum verification took **14.90 seconds**; the cold full decode is more expensive and should be amortized across release preparation. See [`preparation_v2_verification.json`](../outputs/data_quality/astra_review/preparation_v2_verification.json). The immutable pilot outputs are `data/processed/public_ecg_quality/astra_v2_pilot32/` and `data/processed/challenge_ecg_views/astra_v2_pilot32/`.

## Explicit append-overlap gate

The new [`audit_public_pool_overlap.py`](../scripts/audit_public_pool_overlap.py) reconciles each candidate row with its published materialized view, verifies the curated-manifest receipt, compares against all PTB-XL reference identities and the frozen accepted MIMIC audit, and officially checksum-verifies/redecodes the 948-record Georgia pilot. It opens MIMIC SQLite read-only and checks its selection identity and exact agreement with the accepted manifest. Every metadata/manifest/audit input is hashed before and after the operation to detect concurrent changes. It does not rewrite a frozen manifest or existing arrays.

```bash
.venv-pretrained/bin/python -m pytest -q tests/test_public_pool_overlap.py
.venv-pretrained/bin/python -u -m scripts.audit_public_pool_overlap --output-dir outputs/data_quality/astra_review/pool_overlap
```

The append gate found **945 existing Georgia identities** among 19,789 candidates and **no exact PTB-XL or frozen-MIMIC waveform overlap**. Appending the unfiltered candidate would therefore duplicate training examples. The separate [`novel_challenge_ssl_manifest.csv`](../outputs/data_quality/astra_review/pool_overlap/novel_challenge_ssl_manifest.csv) contains **18,844 novel exact waveform identities**. This is an additional-data candidate only, not an updated or scheduled training pool. Its [receipt](../outputs/data_quality/astra_review/pool_overlap/receipt.json) records all input/output hashes, reference sizes and source counts. Both focused overlap tests passed, including duplicate signals under different record IDs and reused IDs whose signal bytes changed.

MIMIC comparisons use frozen audit hashes rather than rereading 39,457 raw recordings. The PTB comparison binds the completed v2 checksum verification snapshot. The append gate verifies manifest lineage rather than independently rereading all candidate shard bytes; use the full materialized verifier as the array-integrity gate. None of these identity checks establishes that different records belong to different people or detects all transformed near-duplicates.

The difference between 948 frozen Georgia records and 945 overlapping candidates is explained by **three full-lead-constant recordings already excluded by the newer QC**: `georgia:E00145`, `georgia:E00281`, and `georgia:E00282`. Their raw files were rechecked against official checksums and decoded again; per-lead evidence is in [`frozen_georgia_constant_leads.json`](../outputs/data_quality/astra_review/frozen_georgia_constant_leads.json). A future versioned clean-cohort construction should quarantine them, alongside its duplicate handling, and report the resulting training-count change. They remain in the historical frozen pool to preserve reproducibility of existing experiments; this review does not silently change past comparisons.

The combined preparation/materialization/overlap suite passed **23/23 tests** after the append gate was added.

## Additional CODE and selection review

The independent CODE verifier now recomputes waveform QC for every accepted row, checks identity consistency across the prepared tables, and checks the native HDF5 rate/type/shape and unresolved unit/duration attributes. The complete **19,878 accepted / 123 excluded** part-0 output passed. A semantic-tamper test changes the HDF5 rate while updating its file hash; the verifier correctly rejects it. Future CODE preparations validate source HDF5 type and integer exam IDs, sort requested parts, and refuse destinations anywhere under raw data. The published CODE output remains bound to its original source revision; no waveform was rewritten. See [`code15_semantic_verification.json`](../outputs/data_quality/code15_semantic_verification.json).

The checksum-verified [CODE duplicate audit](../outputs/data_quality/code15_duplicates/code15_duplicate_annotations.json) examined all 23 excluded exact native-waveform copies against their retained rows. **Eight have different released patient IDs**, four disagree on the automatic `normal_ecg` flag, and one pair has a different six-code diagnosis set. This is an identity and annotation ambiguity, not proof that a patient ID or label is clinically false. The prepared part-0 output keeps one copy per exact signal; a future raw-source or multi-part split must quarantine such pairs or group all linked IDs before train/development/test assignment.

The [age/sex retention audit](../outputs/data_quality/selection_bias.json) shows that the strict ten-second policy changes CPSC composition substantially: CPSC 2018 retains 19.1% of reported ages 18–30 versus 42.4% of ages over 70, and CPSC-Extra retains 15.5% versus 37.0%. Centered SSL retention for those CPSC age groups is much higher, but its labels remain quarantined. This is a record-level selection check, not a causal or patient-level fairness conclusion. The standalone analysis is `python -m scripts.audit_ecg_selection_bias`.

Final focused verification: **25/25 tests passed** across Challenge preparation/materialization, public-pool overlap, and CODE; source compilation and `git diff --check` passed.

## Readability follow-up

The subsequent code review separated the overlap gate into named candidate, published-view, PTB, MIMIC, Georgia, and publication steps; extracted materialized-row contract verification; and simplified preparation cache names. CODE preparation now removes only its newly created staging directory after failure, closes HDF5 handles reliably, and shares one native waveform-hash helper across preparation, verification and duplicate auditing. The selection-bias EDA now uses pandas for grouped age/sex outcome counts while retaining the small per-header age parser. The streaming per-waveform NumPy/HDF5 checks remain in that form because a pandas table would add memory use without clarifying signal validation.

The [refactor receipt](../outputs/data_quality/astra_review/refactor_verification.json) shows the two overlap CSVs are byte-identical to the prior audited outputs; the new novel manifest still has SHA-256 `7c22596fff865b3f5e30d3c2981bef88d3670448075d869124f4d830ffa67adf`. The full source-level selection-bias recomputation was also byte-identical to the prior JSON (SHA-256 `b98e9dfc9399d1b97f6b31d081fd4cad49d069ed01c32d583c0e9f6a96af48c5`). Two focused age/category tests passed. Published artifacts and frozen inputs were not rewritten; future regenerations record the newer source-code hashes.

The combined post-refactor data-processing suite passed **27/27 tests**, and source compilation and `git diff --check` passed.
