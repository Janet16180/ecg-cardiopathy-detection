# Clean Ningbo, version 1

29 September 2026. Built from the findings of [`notebooks/09-jr-ningbo.ipynb`](../notebooks/09-jr-ningbo.ipynb).
Raw files are never modified and no waveform is copied. No model was trained.

## `data/processed/ningbo_clean_v1/`

```bash
OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.data.build_ningbo_clean --workers 4
```

The build takes about 10 minutes on four processes. It refuses an existing output directory and writes:

- `rows.csv`, one row per record with a signal file (34,903);
- `metadata.json`, with the counts and hashes below;
- the overlap receipt `outputs/data_quality/ningbo_v1/exact_overlap_receipt.json`.

Waveforms are read from the raw WFDB files (`path` is relative to `data/raw/challenge-2021/1.0.3`).

- **Files.** Every local `.hea` and `.mat` file of `training/ningbo` is checked against the release's
  `SHA256SUMS.txt`: 69,807 files verified. The build stops unless the only absent official files are the
  three known upstream problems:
  - `JS13118.mat`, served with a hash different from `SHA256SUMS.txt` on two downloads
    (`data/acquisition/ningbo_verification.json`);
  - the unpaired `S23074.hea` and `JS23074.mat`, which form no WFDB record.
  The downloader receipt `data/acquisition/ningbo.json` stops at the first of these and still says `running`;
  the verification receipt is the one to cite.
- **Window.** The whole record: every Ningbo record is exactly 10 s at 500 Hz, in mV, in canonical lead order.
- **Demographics.** `age` is NaN for the 55 missing ages and the 247 records with age 0. No record is aged 1-3,
  and most Ningbo copies of Chapman records have age 0 where Chapman gives an adult age, so 0 is a
  placeholder. Ages are capped at 89 (681 records). `male` is 1, 0 or NaN (22 unknown).
- **Labels.** `snomed_codes` as listed in the header, and `primary`, `secondary` and one flag per superclass
  from the [Challenge label mapping](challenge-label-mapping.md).
- **Quality.** The project policy (`ecg_experiment/ecg_quality.py`) on the record gives `exclusion_reasons`
  and `review_flags`. `zero_leads` lists every lead that is exactly zero for the whole record.
- **Hashes.**
  - `signal_sha256` hashes the int32 µV samples, like the other Challenge sources in
    `outputs/eda/features/challenge_hashes.parquet`.
  - `window_sha256` hashes the float32 `(12, 5000)` window, like the training union, the Chapman view, the
    MIMIC audit and the PTB-XL reference list.

## Duplicates: Chapman and Ningbo as one source family

Chapman (`JS00001`-`JS10646`) and Ningbo (`JS10647` onward) come from one combined release and share one
signal fingerprint (notebook 09, section 7). Exact duplicates are therefore resolved over both sources
together, with the SPH rule extended by sex:

- In each group of records with identical µV samples, the lowest record name is kept when all copies have the
  same mapped labels (`primary`, `secondary` and the four superclass flags) and the same known sex.
- Otherwise every copy is dropped.
- The lowest name is always the Chapman copy in a cross-source group, so the Chapman record already used in
  the clean cohorts v1 is the one kept.

| Result | Ningbo | Chapman |
| --- | ---: | ---: |
| unique | 34,826 | 10,150 |
| kept | 2 | 75 |
| dropped as an agreeing copy | 64 | 13 |
| dropped for a conflict | 11 | 9 |

- **Ningbo against Chapman.** 71 Ningbo records are bit-identical to 71 Chapman records. 62 agree and the
  Ningbo copy is dropped. 9 disagree on the mapped labels and both copies are dropped. Sex agrees in every
  pair, age in only 17, and 53 of the Ningbo copies have age 0.
- **Within Ningbo.** There are 3 pairs, each with adjacent record numbers. Two agree (one copy kept). In the
  third (`JS33139`, `JS33140`) the copies list different sex and age, so both are dropped.
- **Within Chapman.** The same pass also resolves 13 identical Chapman pairs. The 9 conflicting and 13 copied
  Chapman record names are listed in the receipt; cohort builders drop them from Chapman.
- **Other sources.** No Ningbo record is identical to any record of:
  - PTB-XL (all 21,799 records, including the held-out folds 9 and 10);
  - the training union (76,598);
  - the MIMIC 200k audit (197,207);
  - Georgia, CPSC 2018 or CPSC-Extra (19,921 distinct µV hashes).
  SPH stores float16 samples, so an exact match with a µV-resolution source is impossible. SPH stays out of
  training anyway.
- **Near-duplicates.** A scale- and offset-free fingerprint of lead II (50 Hz means, z-scored) was compared at
  zero lag over all 45,150 Chapman and Ningbo records. At a correlation of 0.99 or more it finds the exact pairs
  and one other pair, `JS14182` and `JS14185` (0.991): two paced rhythms with atrial flutter, from patients aged
  62 and 50. They are more likely two similar paced ECGs than one recording, so they are kept and listed in
  `near_duplicate_of` for review. The fingerprint does not detect shifted copies.

## Use

| Set | Records | Negative | Positive | Undefined |
| --- | ---: | ---: | ---: | ---: |
| `use_evaluation` | 34,828 | 4,528 | 12,577 | 17,723 |
| `use_training` | 33,045 | 4,325 | 12,288 | 16,432 |

(primary label; the secondary label adds the sinus-variant-only records to the negatives)

- `use_evaluation` keeps every unique or kept record, including policy exclusions, as the policy prescribes
  for held-out data.
- `use_training` also removes the 1,783 non-duplicate records with an exclusion reason:
  - 1,479 with a lead that is zero for the whole record;
  - 97 with missing samples, one of which also has a zero lead;
  - 207 for the remaining rules: extreme amplitude (93), noise (73), a partial flat segment (35) and the
    rail (13).
- `metadata.json` has the full counts: 1,784 records have a reason before duplicate resolution.

## Choices and their evidence

- **Zero precordial leads exclude from training only.** 1,480 records (4.2%) have leads stored as exactly zero
  for the whole record: 965 have all six precordial leads zero, and 879 of the 1,480 are children aged 4 to 17.
  - The existing `constant_lead` and `flat_segment` rules already exclude them, so the policy needs no new rule.
  - Kept in evaluation, they would let a model score children with missing leads. Any protocol that evaluates on
    Ningbo should report results with and without `zero_leads`.
- **Missing samples.** The 97 records with WFDB invalid samples (-32768, read as NaN) are excluded from
  training. They stay in `use_evaluation` like every other policy exclusion. A runner that cannot take NaN must
  filter on `exclusion_reasons`.
- **Children.** 1,707 records are aged 4-17. They are kept, as every other source keeps its young patients, but
  half of them fail the policy. The training set is therefore older than the raw release.
- **Age 0.** It is treated as missing, not as infants (see above).

## Limitations

- Ningbo headers carry no patient identifier, so repeat recordings of one patient cannot be found. Exact and
  near-duplicate matching find copies of a recording, not different recordings of one patient.
- The labels are physician SNOMED codes read with a mapping that leaves half of the records undefined under the
  primary label (see the [mapping](challenge-label-mapping.md)). Ningbo codes no atrial fibrillation, and what
  its flutter code covers is unknown.
- Reading the waveforms for training needs a WFDB loader (`ecg_experiment/eda/ningbo.read_signal` reads one
  record). The CPC shard loaders read only the prebuilt union and Chapman shards; see
  [clean cohorts v2](clean-cohorts-v2.md).

| File | SHA-256 |
| --- | --- |
| `data/processed/ningbo_clean_v1/rows.csv` | `3767b8dc252ae41ac86c25acc73f475e3204f6a5feba069919f862df6cab822d` |
| `outputs/data_quality/ningbo_v1/exact_overlap_receipt.json` | `13f1e0d8055ddf38da283e5b7ce7b697e24e87364a6d8e2dd1f0fc0189cd4d6a` |
