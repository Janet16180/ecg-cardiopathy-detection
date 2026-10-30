# Fresh administrative roles and cohort reconstruction

The fresh raw-role bootstrap reproduced both existing baseline role tables byte for byte. The successor
cohort pipeline then reproduced all five candidate/cohort manifests byte for byte under new receipts that
consume the fresh roles. These are data reconstruction results. No model was fitted or scored by these stages.

## Executed outputs

The new raw-role receipt is
`outputs/cohort25_recovery_20261028/raw_roles/receipt.json`, SHA-256
`bebdc12bbecb800b4f6db29cf8414b893a13a4c9cb6e4e3f8ceef8d7f663682f`.
Its scope is `administrative_roles_raw_quality`; it does not claim completion of a historical scientific stage.

| Raw-derived table | Records | Baseline roles | Exact historical CSV SHA-256 |
| --- | ---: | --- | --- |
| PTB baseline rows | 19,229 | 14,954 fitting; 2,129 calibration; 2,146 development | `fbf7d098c4e651b86a758dabc9a1956107bb63db7f9818387f3b564f2cadd928` |
| SPH baseline rows | 17,402 | 6,968 normal calibration; 10,434 evaluation | `810f4fdd4f24f6a86534b2eee36d507a93aaec66421175b8e77c49fe06f36cba` |

The full bootstrap decoded all 19,601 PTB development records in folds 1–9 and all 25,770 SPH records.
PTB official file checksums and SPH individual HDF5 fingerprints were checked before and after decoding.
Sixteen windows from each source also passed independent canonical-reader checks. SPH descriptive quality
failures remain in the baseline: 70 admitted records have a quality flag, out of 98 flagged SPH records overall.
The SPH baseline excludes calibration positives; it does not exclude difficult evaluation signals.

The 128-record raw-role profile took 7.67 seconds. The full run took 945.37 seconds with four workers and one
numerical-library thread per worker; peak parent memory was 430.4 MiB and maximum worker memory 285.7 MiB.
These timings come from the new raw-role receipts, not a historical completion timestamp.

The successor data graph completed all 14 stages under
`outputs/cohort25_recovery_20261028/cold_start/pipeline_receipt.json`, SHA-256
`d60fddd0a49c7067aefa4ba45f2fcb14d72bb9e5190c3d06d08e8ad50b338f8c`.
Eight stages verified existing mechanical outputs; six successor stages executed. This invocation took
103.68 seconds with maximum measured resident memory 466.3 MiB. A subsequent invocation verified the same
receipts and left their artifacts unchanged.

## Raw work and cache reuse

The executed successor invocation explicitly requested `--reuse-verified-caches`. Its reference profile
decoded 80 raw held-out windows and combined them with previously verified Ningbo, PTB fold-8 and fresh SPH
hashes. Its training profile decoded and assessed 80 raw training windows. Both profile receipts record
`cache_used: false`.

The full successor reference and Challenge training-quality stages record `cache_used: true`: they reused
the earlier fully executed raw reconstruction tables after checking their table, source and available input
hashes. The new reference stage replaced the historical SPH administrative dependency with the freshly
reconstructed SPH rows. This run did not decode all Challenge windows a second time. The default command
omits cache reuse and dispatches the complete raw readers; two separate integrity fixtures exercise those
full reference and quality branches while forbidding optional legacy-cache access. The raw profiles exercise
the real decoders. The complete default branch was not redundantly run across every waveform during this
successor invocation.

The completed fresh references contain 50,290 records: 50,239 canonical window hashes and 51 explicit unusable
views, comprising 25 constant-lead, 23 short-duration and three nonfinite/shape failures. The new reference
receipt SHA-256 is `8123991fa69a094be8659c10c81e9a74b17a11959a201bcf68ab48277690fa64`.
The new training-quality receipt covers 39,490 original Challenge training records and admits 19,666 strict
10-second supervised records. Its SHA-256 is
`4f83c4dc59abfe643c39e42c3e7eafebc05770531e151e0b014b78038344660d`.
The restored original split remains gated by SHA-256
`5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8`.

## Cohort integrity

The new PTB loader compared 13,432 common strict supervised targets with fresh raw-derived superclass targets:
zero disagreements. Every common canonical PTB window hash also matched the completed training-quality table.
The admitted pool contains 28,568 records: 8,902 PTB and 19,666 Challenge records. Known held-out record,
patient, canonical-window and full-signal overlaps are all zero after admission.

| Training cohort | Records | Negative / positive | Exact ages 18–25 | Exact-age mean |
| --- | ---: | --- | ---: | ---: |
| Primary strict 18–25 | 770 | 506 / 264 | 770 | 21.51 |
| Primary source/evidence/target matched general control | 770 | 506 / 264 | 31 | 57.46 |
| Secondary younger-prioritized adult expansion | 25,000 | 7,989 / 17,011 | 770 | 59.95 |
| Secondary all-age | 25,000 | 7,393 / 17,607 | 668 | 62.17 |

The primary arms match exactly by source, label-evidence rank and target. They share 31 training records and
25 known patients, as allowed by the protocol; both have zero known held-out overlaps. The general arm includes
seven minors, two records with missing ages and four with ages censored above 89. Challenge patient identity
remains unknown, so exact-copy exclusion does not establish patient independence across those sources.

All five new CSVs, including the entire admitted candidate table, are byte-identical to the completed original
selections. The local comparison artifact is `cold_start/membership_parity.json`.
New cohort receipt paths and SHA-256 values are:

- `cold_start/cohorts_25k/receipt.json`:
  `60841a01f477b09dbc77dd452f7cdb9d0577cf45a4c542f259280fec8f25d5bb`.
- `cold_start/strict_matched/receipt.json`:
  `d434442acefbd4d1a5d76e10fa41fcfd7ade80b1d523ee373d4e58efde3684c9`.

Both paths are relative to `outputs/cohort25_recovery_20261028/`. The manifests preserve source paths, backend,
window offsets, sample counts, canonical hashes, label evidence and age interpretation for later frozen feature
extraction. Source-process reader evidence is not a measured per-record label-accuracy estimate. The binary
endpoint remains the existing ECG-annotation proxy; cropped record diagnoses are not promoted to supervised
window labels. No age-specific performance analysis or final Challenge/EchoNext score was produced here.

## Reproduction and validation

After installing the pinned environment, supply the local raw releases and their checksum/download manifests.
Run from the checkout containing the new code:

```bash
PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 .venv/bin/python \
  -m scripts.data.reproduce_cold_start_cohorts --input-root "$PWD" --workers 4 --seed 42
```

`--plan` prints the dependency graph. `--reuse-verified-caches` optionally accelerates full raw-only physical
stages. Missing caches fall back to decoding; historical SCREEN-01/041/042 scientific receipts, predictions,
features and fitted models are not prerequisites. Mapping downloads use a pinned public commit and expected
digests. Existing outputs are verified rather than overwritten. PTB sampling uses seed 42, the original
Challenge split uses 20260929, SPH patient roles use 20261018, and cohort ordering uses 42.

The raw-role and initial cold-stage tests passed six checks. Two additional full-raw dispatch integrity checks
passed. Ruff passed the new source and test files. All row data and raw fingerprints remain local; this report
contains aggregate results and artifact digests only.
