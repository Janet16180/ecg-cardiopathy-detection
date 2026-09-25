# Challenge ECG post-processing

The read-only audit in [data-quality-assessment.md](data-quality-assessment.md) selects valid ten-second views but does not store their arrays. `scripts/materialize_challenge_ecg.py` turns an audited manifest into immutable NumPy shards. It reads the original WFDB pairs, rechecks each pair against the official PhysioNet SHA256 list, applies the audit's canonical transformation, and checks every resulting signal hash against the audited manifest. It writes only under the selected output directory, never under `data/raw/` or the frozen experiment pool.

Each shard is a `.npy` array of `[records, 12, 5000]` little-endian float32 samples, at 500 Hz in physical mV. Lead order is `I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6`. The arrays retain the source waveform's unfiltered amplitude; no learned normalization, denoising, or clinical label remapping is applied. The output `manifest.csv` locates each recording by `shard` and `shard_index`, stores its canonical SHA256, original record-level SNOMED codes, window position, QC statistics, and label restrictions. `metadata.json` links the audited inputs, official release checksum manifest, source code hash, materialized manifest, and every shard hash. A complete output is published by renaming a staging directory only after all records succeed. Existing outputs are never overwritten.

The strict ten-second view and the centered-window view are **separate views**, not independent cohorts. A preflight join of their audited manifests found **3,499 exact canonical waveform overlaps**, with **zero additional differences in original code sets** among those overlapping pairs. Do not add the two view counts to estimate independent recordings. `overlap_by_record` and `overlap_by_signal` show exact overlap with the supplied other-view manifest; near-duplicate and patient overlap remain unresolved. The centered-window view has `label_scope=ssl_only_crop` for longer originals and `ssl_only_view` for exact ten-second originals. Its `record_annotation_available` and `endpoint_supervised_eligible` fields are false throughout. In the strict view, exact duplicate groups with conflicting original diagnosis sets carry `label_scope=conflicting_duplicate_annotations` and `record_annotation_available=false` on the **retained** waveform as well as exclusion of the other copies. Other strict records retain original record annotations for possible future adjudication, but `endpoint_supervised_eligible=false` for every row because no Challenge code has been mapped to the project's PTB-XL proxy. `patient_independent_eval_eligible=false` for every row because these local Challenge releases do not supply verified patient IDs. Exact waveform deduplication does not establish patient-independent splits.

Completed Challenge 2020 inputs can be materialized with:

```bash
.venv/bin/python -m scripts.materialize_challenge_ecg \
  --prepared-dir data/processed/public_ecg_quality/strict_10s \
  --output-dir data/processed/challenge_ecg_views/strict_10s \
  --duplicate-comparisons outputs/data_quality/duplicate_label_comparisons.csv \
  --overlap-manifest data/processed/public_ecg_quality/cpsc_ssl_center_crop/manifest.csv

.venv/bin/python -m scripts.materialize_challenge_ecg \
  --prepared-dir data/processed/public_ecg_quality/cpsc_ssl_center_crop \
  --output-dir data/processed/challenge_ecg_views/cpsc_ssl_center_crop \
  --overlap-manifest data/processed/public_ecg_quality/strict_10s/manifest.csv
```

To verify a published view independently:

```bash
.venv/bin/python -c 'from pathlib import Path; from scripts.materialize_challenge_ecg import verify_materialized; print(verify_materialized(Path("data/processed/challenge_ecg_views/strict_10s")))'
```

The verification reads every shard, checks its file SHA256 and shape, and recomputes every recording's canonical signal hash. The materializer requires `accepted + excluded = candidate` records in the audited receipt, verifies the audited manifest and exclusion hashes, and fails if an official raw pair or transformed view differs. `--allow-incomplete` exists only for an explicitly named pilot of a source still downloading; it cannot make that source a complete cohort. CODE-15% has a different 400 Hz padded format and is not processed by this Challenge WFDB pipeline.

The completed Chapman/Shaoxing strict view is `data/processed/challenge_ecg_views/chapman_strict_10s_v1/`. It contains **10,219 arrays in 80 shards**; independent verification checked every shard and signal hash (manifest SHA-256 `30af07277ba5ab4704500ffa070d851f8dd7470c15f209eee382c4c7d61e8313`). Its audit excluded 15 constant-lead and 13 exact-duplicate records from 10,247 candidates. The 13 duplicate pairs had identical original diagnosis-code sets. All retained source annotations remain unmapped to the project's binary endpoint, and no Chapman recording is marked endpoint-supervised eligible.

The completed strict output contains **13,686 arrays in 107 shards**. Its duplicate-label check identified **162 unique retained waveforms** with conflicting annotations among the 165 differing duplicate comparisons; all 162 have `record_annotation_available=false`. The 13,686 arrays passed an independent full-shard verification, including each array's canonical SHA256, under manifest hash `cad1859e50b34ea3cfd717aceaec5be14e3d6d94cc0268dc2823f3987a1b176c`.

The completed centered CPSC SSL output contains **9,626 arrays in 76 shards**: 6,127 ten-second crops of longer originals and 3,499 exact ten-second originals. All carry an SSL-only `label_scope`; 92 have amplitude review flags. Its independent full-shard verification passed under manifest hash `bcc59b6ece5f099c2dec439cac3a6a99813b794435d6034c58438c96ad5378b0`. The 3,499 original ten-second views are exactly the overlaps described above.
