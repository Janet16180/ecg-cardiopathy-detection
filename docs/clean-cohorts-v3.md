# Quality-first nested cohorts, version 3 (25k to max)

30 September 2026. [Cohorts v2](clean-cohorts-v2.md) without any record of the Challenge test or calibration
groups ([Challenge splits v1](challenge-splits-v1.md)). Use v3, not v2, for any self-supervised pretraining.
V2 stays unchanged for provenance. No waveform is copied and no model was trained.

| Tier | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | MIMIC | ... pending | CODE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean_25k_v3` | 17,405 | 3,982 | 1,232 | 1,227 | 791 | 363 | 0 | 0 | 0 |
| `clean_50k_v3` | 17,405 | 17,090 | 5,287 | 5,266 | 3,394 | 1,558 | 0 | 0 | 0 |
| `clean_100k_v3` | 17,405 | 19,803 | 6,127 | 6,102 | 3,933 | 1,805 | 44,825 | 0 | 0 |
| `clean_150k_v3` | 17,405 | 19,803 | 6,127 | 6,102 | 3,933 | 1,805 | 44,825 | 0 | 50,000 |
| `clean_200k_v3` | 17,405 | 19,803 | 6,127 | 6,102 | 3,933 | 1,805 | 44,825 | 0 | 100,000 |
| `clean_500k_v3` | 17,405 | 19,803 | 6,127 | 6,102 | 3,933 | 1,805 | 317,557 | 121,070 | 127,268 |
| `clean_max_v3` | 17,405 | 19,803 | 6,127 | 6,102 | 3,933 | 1,805 | 796,522 | 600,035 | 127,268 |

| Tier | Known patients | Records without a patient ID | Label available |
| --- | ---: | ---: | ---: |
| 25k | 15,015 | 7,595 | 19,857 |
| 50k | 15,015 | 32,595 | 34,860 |
| 100k | 34,692 | 37,770 | 37,936 |
| 150k | 77,425 | 37,770 | 37,936 |
| 200k | 110,183 | 37,770 | 37,936 |
| 500k | 170,454 | 37,770 | 37,936 |
| max | 266,741 | 37,770 | 37,936 |

"MIMIC" includes the pending records; "... pending" is the part of it not downloaded yet.

## What changed from v2

- The v2 order held 12,603 test-group and 12,597 calibration-group Challenge records, matching the
  [audit](audit-2026-09-30.md). Per source (test / calibration): Ningbo 6,628 / 6,614, Chapman 2,038 / 2,043,
  Georgia 2,036 / 2,039, CPSC 2018 1,302 / 1,303, CPSC-Extra 599 / 598.
- The build drops them from the curated candidates before the v2 order is built. The curated pool falls from
  80,375 to 55,175 records, exactly 25,200 fewer. The waveform-hash guard removed nothing further.
- Every v2 tier rule still holds: the 25k and 50k tiers are curated only, the MIMIC top-up completes 100k, and
  CODE-15 enters at 150k.
- **The 100k tier is now 45% MIMIC** (44,825 records, against 19,625 in v2), because the curated sources lost
  a third of their records. A scaling study that wants curated data only must stop at 50k.
- The 500k tier lists 121,070 records that are not downloaded yet (v2: 95,870).
- The candidate count is 978,965, so 1M is out of reach. As in v2's `resolve`, the build writes
  `clean_max_v3` with every candidate instead.

## Checks

- On this machine, the v2 functions rebuild the saved v2 candidate order row for row (1,004,165 rows,
  identical), so v3 differs from v2 only by the exclusion.
- No v3 tier holds a Challenge test or calibration record, by record ID or by waveform hash, and every tier is
  the first rows of the next larger one. The build refuses to publish otherwise.
- Challenge duplicate groups never span split groups, so dropping the evaluation records by ID also drops every
  exact copy of one. The code refuses to run if that stops being true.
- Each tier's build re-read 20 random rows per local backend and matched their waveform hashes, as in v2.
- The build takes about 50 seconds with a peak of 1.2 GB of memory; writing the CSV manifests is about
  half of it.

## Build

```bash
OMP_NUM_THREADS=4 uv run --locked python -m scripts.data.build_clean_cohorts_v3 build
```

It writes the tier directories `data/processed/clean_<tier>_v3/` and the candidate order and receipt in
`outputs/data_quality/clean_cohorts_v3/`. The receipt binds the SHA-256 of every input, including
`data/processed/challenge_splits_v1/rows.csv`, and of every source file. It refuses existing directories.

The v2 build needs `dx_mapping_scored.csv` and `dx_mapping_unscored.csv` from the public Challenge evaluation
repository (see [the label mapping](challenge-label-mapping.md)). They were missing after the machine
migration and were fetched again; both match their pinned SHA-256.

Pending MIMIC records are resolved as in v2: run v2's `score-pending` with
`--work-dir outputs/data_quality/clean_cohorts_v3`, then

```bash
uv run --locked python -m scripts.data.build_clean_cohorts_v3 resolve --results <pending_results.csv>
```

The order is written with a fixed gzip timestamp, so a rebuild from the same inputs gives the same bytes;
two builds were compared file by file. The uncompressed order has SHA-256
`a5beca9a1b503bcc49e933c4c209787bf5d3f75bdf9e935f024d581fbf8870b4`.

| File | SHA-256 |
| --- | --- |
| `outputs/data_quality/clean_cohorts_v3/candidate_order.csv.gz` | `061e3c9562c9c8a11c063df54ee7e311d5e12f593a313b48705999bd0cb231ca` |
| `data/processed/clean_25k_v3/metadata.json` | `629c123f6b4bd2dfc62a3677b588c3a2fb1635268dab0b65c89f824a4ed583ea` |
| `data/processed/clean_50k_v3/metadata.json` | `2da3d7e226afb8800693138543bdfa659a99ba897f2dda9e579852fa5b4b3414` |
| `data/processed/clean_100k_v3/metadata.json` | `095ee9d6dd9e762998a3e388eb8c57f5c6ad61f369a1a1d45b3c85fcdeb6e4ac` |
| `data/processed/clean_150k_v3/metadata.json` | `099f25318e809de6cda5a91e681f24cb4a67239f16dab36896133dc05abad015` |
| `data/processed/clean_200k_v3/metadata.json` | `01011aa1acd1c077dfa08cd5f355412be80b18c0ef2f1574048bb67c91dde634` |
| `data/processed/clean_500k_v3/metadata.json` | `c153f07c1dd1bfb254c530d0a80eedcd96682c2be7b8e1fa5babc4e9c3318109` |
| `data/processed/clean_max_v3/metadata.json` | `b94b74125b9c337f56fe30b3e9b230be30145f89b3cfcd8fa0c91a1bfd0ffdb9` |

## Still open, carried over from v2

- Whether N counts every record (decision 1 of v2) is unconfirmed by the user.
- Ningbo and CODE-15 rows still need loader backends, and CODE-15 needs an explicit amplitude scale (see
  [clean CODE-15 v1](clean-code15-v1.md)).
