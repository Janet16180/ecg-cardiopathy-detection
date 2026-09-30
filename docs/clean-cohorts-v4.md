# Quality-first nested cohorts, version 4 (25k to 1M)

30 September 2026. [Cohorts v3](clean-cohorts-v3.md) with the usable EchoNext training ECGs added as the
second-best source, right after the curated human-read sources and before CODE-15 and MIMIC. V3 stays
unchanged: Experiment 038 pins it. No waveform is copied and no model was trained.

| Tier | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | EchoNext | MIMIC | ... pending | CODE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean_25k_v4` | 17,405 | 3,982 | 1,232 | 1,227 | 791 | 363 | 0 | 0 | 0 | 0 |
| `clean_50k_v4` | 17,405 | 17,090 | 5,287 | 5,266 | 3,394 | 1,558 | 0 | 0 | 0 | 0 |
| `clean_100k_v4` | 17,405 | 19,803 | 6,127 | 6,102 | 3,933 | 1,805 | 44,825 | 0 | 0 | 0 |
| `clean_150k_v4` | 17,405 | 19,803 | 6,127 | 6,102 | 3,933 | 1,805 | 71,823 | 0 | 0 | 23,002 |
| `clean_200k_v4` | 17,405 | 19,803 | 6,127 | 6,102 | 3,933 | 1,805 | 71,823 | 0 | 0 | 73,002 |
| `clean_500k_v4` | 17,405 | 19,803 | 6,127 | 6,102 | 3,933 | 1,805 | 71,823 | 245,734 | 49,247 | 127,268 |
| `clean_1m_v4` | 17,405 | 19,803 | 6,127 | 6,102 | 3,933 | 1,805 | 71,823 | 745,734 | 549,247 | 127,268 |

| Tier | Known patients | Records without a patient ID | ECG-annotation label available |
| --- | ---: | ---: | ---: |
| 25k | 15,015 | 7,595 | 19,857 |
| 50k | 15,015 | 32,595 | 34,860 |
| 100k | 34,977 | 37,770 | 37,936 |
| 150k | 62,241 | 37,770 | 37,936 |
| 200k | 99,824 | 37,770 | 37,936 |
| 500k | 181,799 | 37,770 | 37,936 |
| 1M | 282,236 | 37,770 | 37,936 |

"MIMIC" includes the pending records; "... pending" is the part of it not downloaded yet.

## What changed from v3

- **25k and 50k are byte-identical to v3** (same `train_manifest.csv` SHA-256), so they stay comparable with
  Experiment 038.
- **100k holds no MIMIC.** The 44,825 MIMIC top-up records of v3 are replaced by the first 44,825 EchoNext
  ECGs (19,962 patients; median 1 ECG per patient, at most 97).
- **200k holds no MIMIC either.** CODE-15 still enters at 150k, with 23,002 exams instead of 50,000 at 150k
  and 73,002 instead of 100,000 at 200k.
- **1M is reachable again.** There are 1,050,788 candidates (v3: 978,965). If the pending MIMIC records fail
  at the rate of the downloaded ones (1.76%, [v2](clean-cohorts-v2.md)), about 10,540 would drop out and
  about 1,040,200 would remain. That is a projection, not a count.
- Without the EchoNext rows, the v4 order holds exactly v3's 978,965 records in v3's order per source; only
  the v3 MIMIC top-up moves behind CODE-15, where the other local MIMIC records already were.

## Why EchoNext goes second

The user asked for the sources with more quality to come first. The order ranks each source by how its
signal and labels were obtained:

| Rank | Source | Signal | Labels | Patient IDs |
| ---: | --- | --- | --- | --- |
| 1 | PTB-XL, then Ningbo, Chapman, Georgia, CPSC, CPSC-Extra | 500 Hz, mV | Human ECG reading | PTB-XL only |
| 2 | EchoNext train | 250 Hz, filtered, clipped, standardized (no unit) | Echocardiogram-confirmed | Yes |
| 3 | CODE-15 | 400 Hz, amplitude unit unresolved | Not used (SSL only) | Yes |
| 4 | MIMIC-IV-ECG | 500 Hz, mV | Cart software, not trusted | Yes |

- EchoNext has the strongest labels in the project and known patients, and only 652 of its 72,475 training
  ECGs (0.9%) fail the unit-free quality rules ([clean EchoNext v1](clean-sph-echonext-v1.md)).
- It ranks below the curated sources because its waveforms are the least raw: the authors filtered,
  clipped and standardized them, and they have 20 times less power below 0.7 Hz than PTB-XL
  ([EDA review](sph-echonext-eda-review.md), finding 7). The amplitude rules of the quality policy cannot be
  applied to it for the same reason.
- It ranks above CODE-15 and MIMIC because they are already the fillers in the user's v2 decisions, and
  neither has trustworthy labels.

## Which EchoNext ECGs are candidates

- Only `train` rows with `use = True` in `data/processed/echonext_250hz_v1/rows.csv`: 71,823 ECGs of
  26,023 patients.
- `val` is the evaluation split of Experiments 023 and 028, and `test` is closed data. `no_split` shares its
  patients with both (EDA review, finding 6). None of them is a candidate.
- The build reads only the `ecg_key`, `patient_key` and `split` columns of the release metadata, never a
  label or a waveform outside `train`. It checks that none of the 26,218 training patients has an ECG in
  `val`, `test` or `no_split` (none has), and that every EchoNext row in the order is a release `train` ECG.
- EchoNext rows have `label_available = False`. That column marks the binary ECG-annotation label, which
  EchoNext does not have. Its echo labels stay in its own `rows.csv`, so 37,936 is the same count as in v3.
- EchoNext is credentialed. The tier manifests list its ECG and patient keys, so they stay under
  `data/processed/` and `outputs/`, outside Git, like the MIMIC rows.

## Checks

- The build refuses to publish unless the v4 curated block equals v3's row for row (55,175 rows), and v4
  without EchoNext holds v3's records in v3's order for every source.
- As in v3, no tier holds a Challenge test or calibration record, by ID or by waveform hash. Each tier is
  the first rows of the next larger one.
- `train.npy` matches the SHA-256 in its cache metadata, and the release metadata file matches the release's
  verified SHA-256.
- Each tier's build re-read 20 random rows per local backend, including `echonext_npy`, and matched their
  waveform hashes.
- The build took 2 min 13 s with a peak resident size of 8.8 GB, mostly pages of the memory-mapped
  `train.npy` (8.7 GB).
- A second build from the same inputs gave the same bytes for the order and for every tier's manifest and
  metadata.

## Build

```bash
OMP_NUM_THREADS=4 uv run --locked python -m scripts.data.build_clean_cohorts_v4 build
```

It writes the tier directories `data/processed/clean_<tier>_v4/`, and the candidate order and receipt in
`outputs/data_quality/clean_cohorts_v4/`. It refuses existing directories. The receipt binds the SHA-256 of
every v3 input, of the EchoNext `rows.csv`, `metadata.json` and `train.npy`, of the v3 candidate order and
of every source file.

Pending MIMIC records are resolved as in v3: run v2's `score-pending` with
`--work-dir outputs/data_quality/clean_cohorts_v4`, then

```bash
uv run --locked python -m scripts.data.build_clean_cohorts_v4 resolve --results <pending_results.csv>
```

The uncompressed order has SHA-256 `3914fc0443dfb4aede6ae968dc5e6094909e96d225ac2922055b02d9431bb115`.

| File | SHA-256 |
| --- | --- |
| `outputs/data_quality/clean_cohorts_v4/candidate_order.csv.gz` | `e8814d92a55ce5418001aa1f6521984ed888fb8d63b077fdcafeab3436be0f38` |
| `data/processed/clean_25k_v4/metadata.json` | `6fc39c42f66c951857c4d7469d352a6f3b485fff1e5be2c360824dc506e6ff9a` |
| `data/processed/clean_50k_v4/metadata.json` | `aed1bca62621ca2911f53b426febe669c4e08a92a299a1190926d4ce0d842eee` |
| `data/processed/clean_100k_v4/metadata.json` | `d332363d885e4a62cbae8483d91df91ab8bfd419c762a3f2f19b92e8c25bcc21` |
| `data/processed/clean_150k_v4/metadata.json` | `f27cf6b89353d64993a29e9f2dba805b157e152faad120a097a47e34069e6014` |
| `data/processed/clean_200k_v4/metadata.json` | `7b676ab7cd2f1cac8d8574a8859682bf5607e61560e3174ad794df4247a61f08` |
| `data/processed/clean_500k_v4/metadata.json` | `f5f91be6e9281125afbce14891d63e2baa63ba2d082d27a9eb6a12039a8e0420` |
| `data/processed/clean_1m_v4/metadata.json` | `fac9a8f232ca9da96729691d937ec9ad9e7f3d431bd75e14e9b622a7e3bb43fb` |

## Decisions made here, and alternatives

1. **EchoNext after the curated sources.** Alternative: interleave it with the curated sources from 25k.
   EchoNext would then take about two thirds of the rows after PTB-XL (71,823 of 109,593), and the 25k and
   50k tiers of Experiment 038 would change.
2. **EchoNext before CODE-15.** Alternative: after CODE-15, as another source without a physical unit. The
   100k tier would then hold MIMIC again.
3. **EchoNext ordered by ECG, not by patient**, like the local MIMIC records. Repeat ECGs stay in, as in the
   EchoNext training split itself; the 100k part averages 2.2 ECGs per patient.
4. The v3 rules still hold: N counts every record (still unconfirmed by the user), PTB-XL comes first, seed
   20260929.

## What loaders need

- **EchoNext rows (`echonext_npy`)** point to a row of `data/processed/echonext_250hz_v1/train.npy`: float32
  `(12, 2500)` at 250 Hz, in the release's standardized values. The CPC pipeline already runs at 250 Hz, so
  no resampling is needed there; `ecg_experiment.echonext.to_cpc_scale` maps the amplitudes as Experiment 023
  did. A 500 Hz encoder would need the rows upsampled, and nothing above 125 Hz can be recovered.
- The other backends need what the [v3](clean-cohorts-v3.md) and [v2](clean-cohorts-v2.md) notes list:
  Ningbo WFDB rows, CODE-15 materialization with an explicit amplitude scale, and the pending MIMIC
  downloads.
- The Experiment 038 cohort caches serve 25k and 50k; since those tiers are identical, the caches serve v4
  too. Larger tiers have no cache yet.
