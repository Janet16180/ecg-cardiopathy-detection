# Exploration notebooks

Name notebooks with an order, author initials, and purpose, for example
`01-jr-signal-quality.ipynb`. Move reusable code into `ecg_experiment/` and
record reproducible commands in a protocol. Clear sensitive or large outputs
before committing. Notebooks are optional; the experiment pipeline runs in Python.

## Dataset EDA

An exploratory analysis of every ECG source, built from the raw files. The notebooks describe the
datasets themselves; read them in order. Findings about this project's own pipeline, with
recommendations, are in [the pipeline review](../docs/eda-pipeline-review.md), and the quality policy
derived from these notebooks is described in [clean cohorts v1](../docs/clean-cohorts-v1.md).

| Notebook | Source | Main questions |
| --- | --- | --- |
| `01-jr-ptbxl.ipynb` | PTB-XL | Missing and implausible values, diagnoses by age and sex, signal integrity, devices |
| `02-jr-mimic.ipynb` | MIMIC-IV-ECG (800k metadata, 200k signals) | Repeat recordings, machine measurements and their consistency, lead order, devices |
| `03-jr-challenge.ipynb` | Georgia, CPSC 2018, CPSC-Extra, Chapman | Format, diagnoses by age and sex, duplicates, rail artifacts, filtering |
| `04-jr-code15.ipynb` | CODE-15% (all 18 parts) | Labels and mortality by age and sex, padding and duration, amplitude unit, sampling rate |
| `05-jr-cross-dataset.ipynb` | All | Populations, the same findings across sources, source fingerprints |
| `06-jr-clean-cohorts.ipynb` | Training candidates | What the quality policy removes per source, and the 25k, 50k and 100k cohorts |
| `07-jr-sph.ipynb` | SPH (Shandong Provincial Hospital) | AHA codes and the standard label, duplicates with conflicting codes, artifacts, filtering |
| `08-jr-echonext.ipynb` | EchoNext (credentialed) | Splits, echo-based labels and missing values, partially filled leads, filtering |
| `09-jr-ningbo.ipynb` | Ningbo (Challenge 2021) | SNOMED codes and the binary label, age placeholders, zero precordial leads in children, filtering, copies of Chapman records |
| `10-jr-cohorts-v2.ipynb` | Training candidates | CODE-15 cleaning and amplitude, and the source, age, sex, label and pending composition of the nested 25k-1M cohorts |

## Experiment demonstrations

Notebooks that show an experiment's method on single ECGs. The numbers that decide whether the method works are in the experiment's results report.

| Notebook | Experiment | What it shows |
| --- | --- | --- |
| `11-jr-section-maps.ipynb` | [041](../docs/experiment-041-fragment-localization-results.md) | Abnormal 0.25 s sections painted red on healthy ECGs, benign variants and twelve kinds of cardiopathy; needs `outputs/experiment041_fragment_localization_v1/` and runs on the CPU |
| `12-jr-beat-wave-maps.ipynb` | [042](../docs/experiment-042-lead-wave-maps-results.md) | The beat-aligned map: each beat cut into P, QRS, ST and T per lead, compared with healthy beats, with red waves on the same examples as notebook 11; needs `outputs/experiment042_lead_wave_maps_v1/` and refits its references on the CPU in about a minute |
| `13-jr-pipeline-v4-explained.ipynb` | [046](../docs/experiment-046-pipeline-v4-results.md), [048](../docs/experiment-048-pvc-switch-results.md), [049](../docs/experiment-049-focal-switch-results.md) | Pipeline v4 run live on the CPU: referral by E and F with `combined_50` at 5%, and the PVC-switch explanation (U_B in red, attention in blue) on the same groups and examples as notebooks 11 and 12, with the per-group referral table, the experiments' numbers and the costs; needs the 042-049 outputs and runs in one to two minutes |

The MIMIC and EchoNext notebooks show aggregate statistics only (credentialed data), so their outputs can
be kept in Git. Findings for the project from notebooks 07 and 08 are in
[the SPH and EchoNext review](../docs/sph-echonext-eda-review.md). The clean manifests built from notebooks 09 and 10 are described in
[clean Ningbo v1](../docs/clean-ningbo-v1.md), [the Challenge label mapping](../docs/challenge-label-mapping.md),
[clean CODE-15 v1](../docs/clean-code15-v1.md) and [clean cohorts v2](../docs/clean-cohorts-v2.md). Plots use seaborn. The analysis code is in
`ecg_experiment/eda/`.

Build the feature caches once (about 80 minutes on CPU: MIMIC, then CODE-15 streamed from its export
archive), then execute a notebook:

```bash
uv run --no-sync python -m scripts.reports.build_eda_caches
uv run --no-sync jupyter nbconvert --to notebook --execute --inplace notebooks/01-jr-ptbxl.ipynb
```

Caches go to `outputs/eda/`, which is ignored by Git. Delete `outputs/eda/features/` to recompute
from the raw files.

| Module in `ecg_experiment/eda/` | Purpose |
| --- | --- |
| `signals.py` | Dataset-independent features: flat leads, amplitude, limb-lead identities, noise bands, heart rate; parallel caching; ECG plots |
| `ptbxl.py`, `ptbxl_labels.py`, `ptbxl_signals.py` | PTB-XL loading, an independent re-implementation of the project label, label and signal audits |
| `mimic.py` | MIMIC record list, machine measurements, name-based lead ordering |
| `challenge.py` | Challenge headers, SNOMED names, signal hashes |
| `code15.py` | CODE-15 metadata; streams all 18 parts from the export archive one at a time |
| `sph.py` | SPH metadata, code dictionary and label mapping, full-record features, quality policy on the first 10 s |
| `echonext.py` | EchoNext metadata and splits, label consistency, waveform statistics streamed from the ZIP (train and val only) |
| `processed.py` | Comparisons of the project's processed arrays with the raw files |
| `cross.py` | All sources in one table; six common findings on one vocabulary; source fingerprinting |
| `stats.py` | Prevalence with Wilson intervals, chi-square tests, logistic-regression odds ratios, prevalence plots |
