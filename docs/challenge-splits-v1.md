# Challenge record split, version 1

29 September 2026. A frozen train / calibration / test assignment of every record of the five PhysioNet
Challenge sources: Ningbo, Chapman/Shaoxing, Georgia, CPSC 2018 and CPSC-Extra. Every later experiment that
fits, calibrates or tests on these sources uses this split, so their groups stay comparable. No waveform is
read and no model score is used.

```bash
OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.data.build_challenge_splits
```

Code: `ecg_experiment/challenge_splits.py` (tested in `tests/test_challenge_splits.py`) and
`scripts/data/build_challenge_splits.py`. The builder refuses an existing output directory.

## Rule

- **Unit: the record.** The Challenge headers carry no patient identifier, so a patient with two recordings
  can land in two splits. This is the default the user was told about. Evaluation intervals on these
  sources are therefore record-level and somewhat too narrow.
- **Duplicate groups.** Records that share a signal hash (the int32 µV samples) or a window hash (the
  float32 `(12, 5000)` window, Ningbo only) are joined transitively into one group, and a group is assigned
  as a whole. No signal appears in two splits. 816 groups hold more than one record (1,656 records); 406 of
  them span two sources: 71 Ningbo–Chapman pairs and 335 CPSC 2018–CPSC-Extra groups.
- **Source families.** Chapman and Ningbo are one family (`chapman_ningbo`): one combined release with
  shared copies ([clean Ningbo](clean-ningbo-v1.md)). CPSC 2018 and CPSC-Extra are also one family (`cpsc`):
  both come from the 2018 China Physiological Signal Challenge collection, and 335 of their signals are
  identical. Georgia is its own family. A leave-one-source-out analysis should hold out a family, not a
  source.
- **Strata.** Source by primary label (positive, negative, undefined), taken from the lowest record of each
  group. Within a stratum the groups are sorted, shuffled by one `numpy` generator with seed `20260929` (the
  strata are visited in sorted order), and cut into 60% / 20% / 20%, the calibration and test shares
  rounded half up. The assignment depends only on the records, their strata and the seed.
- **Every record keeps its assignment**, including undefined labels and dropped duplicates, so a later
  label version or quality policy never moves a record between splits.
- **`evaluable`** marks the records an evaluation may use: a defined primary label and a duplicate status of
  `unique` or `kept`. Duplicates are resolved over all five sources with the Ningbo family rule
  (`ningbo.duplicate_status`: in each identical group the lowest name is kept when the mapped labels and
  known sex agree, otherwise every copy is dropped). The build stops unless this reproduces the clean Ningbo
  manifest's statuses and the Chapman lists of its overlap receipt, which it does.

## Why 60 / 20 / 20

- **Calibration and test need negatives.** The primary negative (sinus rhythm as the only code) is rare:
  Chapman has 1,366, Georgia 1,752, CPSC 2018 918 and CPSC-Extra none. A 20% share gives each source with
  negatives 183 to 906 evaluable negatives per group, enough for a specificity to within about ±0.03 to
  ±0.07 at the observed rates. A 10% share would halve that.
- **The 95%-sensitivity threshold rests on the missed positives.** With 593 to 2,515 evaluable positives per
  source in each 20% group, the 5% tail holds 30 to 126 positives per source, against 17 in the whole
  PTB-XL calibration set of Experiment 027.
- **Training keeps the majority.** Readouts fitted on the Challenge sources later (`multisource_lso`) get 60%,
  about 22,900 evaluable records.
- Equal calibration and test shares let a later experiment move between "fit on calibration" and "confirm
  on test" without changing precision.

## Counts

All records, by split:

| Source | Train | Calibration | Test | All |
| --- | ---: | ---: | ---: | ---: |
| Ningbo | 20,941 | 6,979 | 6,983 | 34,903 |
| Chapman/Shaoxing | 6,147 | 2,050 | 2,050 | 10,247 |
| Georgia | 6,207 | 2,066 | 2,071 | 10,344 |
| CPSC 2018 | 4,121 | 1,375 | 1,381 | 6,877 |
| CPSC-Extra | 2,074 | 692 | 687 | 3,453 |

Evaluable records, positive / negative on the primary label:

| Source | Train | Calibration | Test |
| --- | --- | --- | --- |
| Ningbo | 7,547 / 2,716 | 2,515 / 906 | 2,515 / 906 |
| Chapman/Shaoxing | 2,234 / 820 | 745 / 273 | 743 / 273 |
| Georgia | 4,138 / 1,039 | 1,379 / 347 | 1,379 / 347 |
| CPSC 2018 | 2,067 / 551 | 690 / 183 | 692 / 183 |
| CPSC-Extra | 1,781 / 0 | 593 / 0 | 595 / 0 |

`metadata.json` also counts undefined labels, all records per label, and duplicate statuses per source.
CPSC records range from 6 to 144 s; a runner that needs exactly 10 s (such as the frozen-encoder features in
`outputs/features_challenge_v1/`) uses fewer CPSC records than this table, and the 97 Ningbo records with
missing samples have no features.

## Access rule

- The **test** groups stay closed until an experiment's final frozen stage: no score of a test record is
  computed or read before that stage, and the protocol must say it opens them.
- **Calibration** groups may be used to fit calibration and thresholds, and to evaluate a method that was not
  fitted on them (for example a leave-one-family-out arm).
- **Train** groups are for fitting readouts.

## Files

| File | SHA-256 |
| --- | --- |
| `data/processed/challenge_splits_v1/rows.csv` | `5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8` |
| `data/processed/challenge_splits_v1/metadata.json` | `e29807c909bae00f74001659a95ff6432a69c37ec87196a2c53ce66654b5d2a1` |

`rows.csv` columns: `source`, `family`, `record`, `path`, `duration_s`, `primary`, `secondary`, `MI`, `STTC`,
`CD`, `HYP`, `signal_sha256`, `window_sha256` (Ningbo only), `duplicate_group` (the group's lowest record),
`duplicate_status`, `evaluable` and `split`. `metadata.json` holds the seed, proportions, families, counts,
and the hashes of every input (the clean Ningbo manifest and receipt, the Challenge EDA header and hash
caches, the official SNOMED tables) and of the code.

The manifest is local and not yet tracked by DVC, like the clean Ningbo manifest; adding both is the user's
choice.
