# Clean EchoNext, version 2

30 September 2026. A review of [clean EchoNext v1](clean-sph-echonext-v1.md) against its EDA
(`notebooks/08-jr-echonext.ipynb`, [review](sph-echonext-eda-review.md)) found one EDA finding that v1 could
not support and one exclusion rule the EDA never examined. Version 2 is a new rows table over the unchanged v1
arrays; v1 stays as it is, because Experiments 023 and 028 and cohorts v4 pin it.

RESULTS_PLACEHOLDER

## Decisions

The user left these decisions to the agent on 30 September 2026. They were written and committed before the
v2 build ran. An exploratory scratch check earlier the same day had already shown the numbers that prompted
them: the noise-dominated ECGs are 95% positive, and the rows line up with the metadata.

1. **Add `most_recent_ecg` and `ecgs_in_split`.** EDA finding 4 says calibration and threshold work must use one
   ECG per patient, the most recent one, because training keeps every repeat ECG. V1 dropped the column, and
   the acquisition year cannot break ties, so the finding could not be applied from the clean rows.
2. **Split `use` into `use_training` and `use_evaluation`.**
   - `use_training` excludes every v1 reason: nonfinite samples, a constant lead, a flat segment of 1 s or
     more, and noise-dominated recordings. It equals v1 `use`.
   - `use_evaluation` excludes only recordings with part of the signal missing (nonfinite, constant lead,
     flat segment). EDA finding 3 says the flat-segment records must not be in primary results, and a
     missing lead is a broken recording.
   - Noise-dominated recordings stay in evaluation. The quality policy (`ecg_experiment/ecg_quality.py`) says
     held-out partitions should keep their difficult cases, and SPH's `use_evaluation` keeps its policy
     exclusions the same way.
3. **Noise-dominated ECGs stay out of training.** More than half of their power lies above 40 Hz, so the
   ECG is mostly masked. They are also almost all positive, so a model trained on them could learn "noise
   means disease". The rule was built for 500 Hz data; at 250 Hz its 40-150 Hz band is in effect 40-125 Hz.
   It keeps its v1 threshold because it removes under 0.1% of ECGs, and changing it would change the v1
   training set that Experiments 023 and 028 used.
4. **Clipping gets no rule.** The release clips every lead at its 0.1 and 99.9 percentiles. A rule would
   need a threshold nobody has examined, and the checks below measure how many ECGs sit at a bound.
5. **Experiments 023 and 028 are not rerun.** They used v1 `use` for training (the same as `use_training`)
   and for primary evaluation. 023 also reports all validation rows as a secondary result: AUROC 0.807
   against 0.812 on usable rows. Future protocols should name `use_evaluation` for the primary analysis.

## Cohort entry gate

EchoNext enters the pretraining cohorts ([cohorts v4](clean-cohorts-v4.md)) only if every criterion holds.
Otherwise no v4 cohort is published.

| Criterion | Threshold |
| --- | --- |
| Share of training ECGs with `use_training` | at least 95% |
| Waveform heart rate within 10 bpm of the cart rate, 3,000 usable rows per split | at least 95%, and at most 50% with the rows shuffled |
| Standard lead order: the four limb relations fitted on 2,000 usable training ECGs | expected signs and R² of at least 0.9 for each |
| Identical waveforms among train and val ECGs | none |
| Training patients with an ECG in val, test or `no_split` | none |

## Build

```bash
OMP_NUM_THREADS=4 uv run --locked python -m scripts.data.build_echonext_rows_v2
```

It writes `data/processed/echonext_250hz_v2/rows.csv` and `metadata.json`, and refuses an existing directory.
It checks the SHA-256 of the v1 arrays, the v1 rows and the release metadata. It reads only the `ecg_key`,
`split`, `patient_key` and `most_recent_ecg` columns of the release metadata, and never a test waveform or
label. EchoNext is credentialed: the directory stays local and outside Git, and only aggregates are reported.
