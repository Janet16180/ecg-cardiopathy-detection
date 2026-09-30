# Quality-first nested cohorts, version 2 (25k to 1M)

29 September 2026. Seven nested, manifest-only pretraining cohorts, following the user's decisions of 28-29
September 2026:

- Quality first: fill from the curated human-read 500 Hz sources, preferring records without review flags.
- CODE-15 enters at 150k and 200k and fills 500k and 1M.
- MIMIC is the last filler. CODE-15 and MIMIC are unlabeled SSL data only.
- SPH stays out, because it is the external test hospital.

The analysis is in [`notebooks/10-jr-cohorts-v2.ipynb`](../notebooks/10-jr-cohorts-v2.ipynb). No waveform is
copied and no model was trained. [Clean cohorts v1](clean-cohorts-v1.md) stay as they are.

| Tier | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | MIMIC | ... pending | CODE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean_25k_v2` | 17,405 | 3,986 | 1,231 | 1,227 | 789 | 362 | 0 | 0 | 0 |
| `clean_50k_v2` | 17,405 | 17,105 | 5,284 | 5,268 | 3,384 | 1,554 | 0 | 0 | 0 |
| `clean_100k_v2` | 17,405 | 33,045 | 10,208 | 10,177 | 6,538 | 3,002 | 19,625 | 0 | 0 |
| `clean_150k_v2` | 17,405 | 33,045 | 10,208 | 10,177 | 6,538 | 3,002 | 19,625 | 0 | 50,000 |
| `clean_200k_v2` | 17,405 | 33,045 | 10,208 | 10,177 | 6,538 | 3,002 | 19,625 | 0 | 100,000 |
| `clean_500k_v2` | 17,405 | 33,045 | 10,208 | 10,177 | 6,538 | 3,002 | 292,357 | 95,870 | 127,268 |
| `clean_1m_v2` | 17,405 | 33,045 | 10,208 | 10,177 | 6,538 | 3,002 | 792,357 | 595,870 | 127,268 |

| Tier | Known patients | Records without a patient ID | Age known | Median age | Male share | Label available |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25k | 15,015 | 7,595 | 99.0% | 61 | 53.0% | 79.4% |
| 50k | 15,015 | 32,595 | 99.3% | 62 | 54.4% | 69.7% |
| 100k | 26,915 | 62,970 | 79.8% | 62 | 54.8% | 53.0% |
| 150k | 69,648 | 62,970 | 86.6% | 60 | 49.5% | 35.3% |
| 200k | 102,406 | 62,970 | 89.9% | 58 | 47.1% | 26.5% |
| 500k | 165,395 | 62,970 | 41.4% | 58 | 46.5% | 10.6% |
| 1M | 265,848 | 62,970 | 20.7% | 58 | 46.5% | 5.3% |

- **Nested.** Each tier is the first N rows of one saved candidate order, so every tier contains the smaller
  ones.
- **Size.** N counts every record, including the labeled PTB-XL rows. In v1, by contrast, "25k" meant 25,000
  unlabeled records plus 15,349 labels.
- **Labels.** Every tier carries the same PTB-XL label files as v1: 15,349 full-budget and 1,517
  limited-budget clean training proxy labels. `label_available` also marks Challenge records whose
  [mapped label](challenge-label-mapping.md) is defined, for reference; those labels are not in any label file.
- **Ages and sexes.** They come from each source's metadata. MIMIC has none locally, which is why the share of
  known ages falls in the large tiers.

## Known issue: Challenge evaluation records inside the cohorts (30 September 2026)

The cohorts were built before the [Challenge record split](challenge-splits-v1.md) existed, so they include
Challenge records that the split later assigned to its test and calibration groups: 1,512 test-group records in
`clean_25k_v2` and 12,603 in `clean_100k_v2` and every larger tier. No model has been trained on these cohorts
yet, so no result is affected. Before any self-supervised pretraining, rebuild the cohorts (v3) without the
Challenge test and calibration groups; see the backlog item `cohorts_v3_exclude_challenge_eval` and the
[audit](audit-2026-09-30.md).

## Build

```bash
OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.data.build_clean_cohorts_v2 build
```

The build takes a few minutes. It writes:

- `data/processed/clean_{25k,50k,100k,150k,200k,500k,1m}_v2/`, each with `train_manifest.csv`,
  `labels_fraction1.csv`, `labels_fraction0.1.csv` and `metadata.json`;
- the full candidate order and an aggregate receipt in `outputs/data_quality/clean_cohorts_v2/`.

It refuses existing directories. Each tier's metadata binds:

- the seed and the ordering rule;
- the SHA-256 of every input table and source file, and of the candidate order;
- its composition;
- the hashes of its own tables.

Each build reads 20 random rows of every locally readable backend and checks their waveform hash. All checks
passed.

Inputs:

- the v1 quality table (the same policy, already scored on every union, Chapman and MIMIC 200k record);
- [clean Ningbo v1](clean-ningbo-v1.md) and its overlap receipt;
- [clean CODE-15 v1](clean-code15-v1.md);
- the MIMIC record list (`outputs/eda/features/mimic_records.parquet`);
- the PTB-XL reference hashes and held-out references.

## Order

Blocks follow the tier rules:

1. **Curated.** Every clean PTB-XL training row first (17,405, of which 15,349 labeled). Then Ningbo, Chapman,
   Georgia, CPSC 2018 and CPSC-Extra, interleaved in proportion to their clean records: row `i` of a source
   with `n` rows sits at position `(i + 0.5) / n`, so every prefix keeps the proportions. 80,375 records.
2. **MIMIC top-up.** The curated sources are 19,625 short of 100k, so the first 19,625 local MIMIC records
   complete the 100k tier.
3. **CODE-15.** All 127,268 usable exams. 150k takes 50,000 and 200k takes 100,000.
4. **Local MIMIC.** The other 176,862 downloaded records that pass the policy.
5. **Pending MIMIC.** The 600,035 records of the 120,656 patients with no downloaded record, whole patients at
   a time. They are referenced by their official path and marked `quality_status = pending`.

Within every source, records without a review flag come first, and each group is ordered by the SHA-256 of
`"20260929:<record_id>"`. Pending patients are ordered by the SHA-256 of `"20260929:<subject_id>"`, and their
records by study ID. The key depends only on the seed and the identifier, not on row order, so rebuilding from
the same inputs reproduces the order.

## Exclusions

- **SPH** is never a candidate.
- **PTB-XL held-out records** (folds 9 and 10, which hold the development, calibration and test partitions)
  and their patients are excluded, and so is any record of another source whose waveform hash equals any of
  the 21,799 PTB-XL records. None was found.
- **Chapman and Ningbo** are resolved as one source family ([clean Ningbo v1](clean-ningbo-v1.md)):
  - the 62 Ningbo copies of Chapman records are dropped;
  - every copy in the 9 label conflicts is dropped. 9 of the 22 Chapman records dropped by that resolution are
    in the Chapman view. The v1 cohorts contain 1, 4 and 8 of the conflicting ones; they stay as they are.
- **Repeated waveforms** are removed in order, keeping the first copy. This removed 2 CODE-15 exams whose first
  10 s repeat another exam of the same patient and differ only in the last 0.24 s.
- **Policy failures** are never candidates: the v1 policy for PTB-XL, the union's Challenge rows, Chapman and
  local MIMIC, `use_training` for Ningbo, and the both-scales rule for CODE-15.

## Pending MIMIC records and the 1M tier

Only 200,000 of MIMIC's 800,035 records are downloaded. The 500k and 1M tiers therefore list records that are
not on disk yet: 95,870 (19.2%) and 595,870 (59.6%) of their rows.

Of the downloaded selection, 3,513 of 200,000 records (1.76%) failed the input contract, the duplicate audit
or the policy. At that rate:

- the 1,004,165 candidates would shrink to about 993,625 once the pending records are scored;
- **the 1M tier is expected to fall about 6,400 records short** even with the full MIMIC list and all of
  CODE-15, and the largest reachable tier would be all remaining candidates;
- every tier up to 500k is expected to fill.

On the machine that downloads the pending records:

```bash
uv run --no-sync python -m scripts.data.build_clean_cohorts_v2 score-pending \
    --results outputs/data_quality/clean_cohorts_v2/pending_results.csv --workers 4
uv run --no-sync python -m scripts.data.build_clean_cohorts_v2 resolve \
    --results outputs/data_quality/clean_cohorts_v2/pending_results.csv --suffix v2_resolved
```

- `score-pending` reads every pending record whose files exist, with the same input contract as the
  training loader (500 Hz, 12 named leads, 5,000 finite samples). It applies the quality policy and writes the
  exclusion reasons, review flags and waveform hash per record.
- `resolve` removes the failures and any repeated waveform from the saved order and writes new tier
  directories, `clean_<tier>_v2_resolved`. Rows that failed simply leave the order, so the replacements are
  the next candidates of the same seeded order and the tiers stay nested.
- When fewer than 1M candidates remain, `resolve` writes `clean_max_v2_resolved` with all of them instead of
  1M.
- The original v2 directories are never edited.

## Decisions made here, and alternatives

These are the decisions the brief left to this build. The most conservative option was taken where one had
to be chosen.

1. **N counts every record.** Alternative: v1's reading, N unlabeled records plus the labeled PTB-XL rows. The
   user has not confirmed either reading, for v1 or v2.
2. **PTB-XL first in every tier**, so the label files are the same in every tier and the probe does not change
   with the tier. The cost is that 25k is 70% PTB-XL. Alternative: include PTB-XL in the proportional
   interleave, which gives 25k a curated mix closer to 50k, but smaller tiers would then hold only part of the
   labels.
3. **Proportional interleave of the other curated sources.** Alternative: equal shares per source. That would
   give small sources a larger weight, but CPSC-Extra and CPSC would run out before 50k.
4. **Local MIMIC ordered by record; pending MIMIC by whole patient.** Records spread the 100k top-up over
   11,900 patients. Downloads happen per patient, so pending rows are grouped the same way.
5. **Seed 20260929.**
6. **CODE-15 amplitude unit unresolved; exams must pass the policy at 1.0 and 0.5 times the stored values**
   (see [clean CODE-15 v1](clean-code15-v1.md)).
7. **CODE-15 10 s only.** Allowing the 7.3 s exams would add about 200,000 SSL records and make 1M reachable,
   but it needs a second window length. Not done; this is for the user to decide.

## What loaders need

- **Shard and local MIMIC rows (`shard`, `mimic_wfdb`)** read unchanged with `SampledTrainingECGDataset`, as
  in v1.
- **Ningbo rows (`challenge_wfdb`)** point to the raw WFDB records under `data/raw/challenge-2021/1.0.3`.
  `ecg_experiment.waveforms.read_record(root, path)` reads them and reproduces their `signal_sha256`. The
  dataset class needs this backend added before it can serve them.
- **CODE-15 rows (`code15_archive`)** point to an HDF5 member of the compressed export archive, which has no
  random access. They need the materialization step described in [clean CODE-15 v1](clean-code15-v1.md) and
  an explicit amplitude scale.
- **Pending MIMIC rows** need downloading, `score-pending` and `resolve` first.

The tier directories themselves therefore do not pass `SampledTrainingECGDataset`'s backend check yet. The
build does not run that loader.

| File | SHA-256 |
| --- | --- |
| `outputs/data_quality/clean_cohorts_v2/candidate_order.csv.gz` | `65ab78da632b147a0537c81ea6e83928e033d67ad5e8018786404539b2467612` |
| `data/processed/clean_25k_v2/metadata.json` | `9d9d9eef914e9f8b5b7c4736756e2c863756c320337a4b45e897f74486f0a962` |
| `data/processed/clean_50k_v2/metadata.json` | `1d9491ae030615a9575902fabe648c3f917a82d0de75d0da99ac26d9b8de181f` |
| `data/processed/clean_100k_v2/metadata.json` | `7d9fdd0241bb6715c11fe40b1d73e300fe4c76954401c6ea7519511831a57b7e` |
| `data/processed/clean_150k_v2/metadata.json` | `b7eb7673761548d5a0449d5249dcd0bdb503b715eb935e3127fa45b73909d1d7` |
| `data/processed/clean_200k_v2/metadata.json` | `5051a70dc9bb6d31d106387e5bc57ac8644f79ae30a09116be3123eb8ac3ef7f` |
| `data/processed/clean_500k_v2/metadata.json` | `7480700577c300fbed3608191586d4c1f92f1db8592ce74a5ded2e9ebe8a9082` |
| `data/processed/clean_1m_v2/metadata.json` | `b365b62bf41709dd99c97933ed6cd10cc1e7198f9c98cffa37a0a3e538e6d5fd` |
