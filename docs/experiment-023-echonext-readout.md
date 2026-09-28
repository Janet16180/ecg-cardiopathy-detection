# Experiment 023: echo-confirmed structural heart disease readout on EchoNext

**Frozen 28 September 2026, before any EchoNext waveform is featurized or scored.** Every earlier project
label comes from reading the ECG itself. EchoNext labels each ECG with structural heart disease measured on an
echocardiogram, which is an independent test of whether the heart is diseased and is closer to the project's
cardiopathy aim. This readout asks how much of that the frozen CPC features already carry. No encoder is
trained. The user holds the PhysioNet credential and data use agreement; like MIMIC, only aggregates leave the
machine.

## Question

Does a fixed logistic head on the unchanged starting CPC encoder detect echo-confirmed structural heart disease
better than the ECG cart's own measurements and demographics?

## Data

- Source: `data/raw/echonext/echonext-...-1.1.0.zip`. Every member is checked against the shipped
  `SHA256SUMS.txt`. Waveforms are 10 s, 12 leads, 250 Hz, in standard lead order (checked on validation data
  with the Einthoven identities), and already preprocessed by the dataset authors: median filtered, clipped at
  the 0.1 and 99.9 percentiles and standardized. Physical units are not recoverable.
- Partitions: the official patient-disjoint splits. `train` (72,475 ECGs, 26,218 patients) fits every head,
  `val` (4,626 ECGs, one per patient) is the development set, `test` stays closed and is only hash checked,
  and `no_split` is not used (it shares patients with `val` and `test`).
- CPC input: each lead is standardized with the mean and standard deviation of that lead over all training
  samples, then mapped onto the historical PTB-XL per-lead scale, so that the historical normalization gives
  the same per-lead z-scores. No other transform. Records with a nonfinite sample would stop the build.

`scripts/data/build_echonext_cache.py` writes the train and val waveforms as float32 `(N, 12, 2500)` arrays,
the rows, the per-lead training statistics and a receipt to `data/processed/echonext_250hz_v1/`.

## Labels

- **Primary:** `shd_moderate_or_greater_flag`, the dataset's composite, on all rows.
- **Secondary:** each component flag with at least 50 validation positives, restricted in both splits to rows
  where the underlying echo value is present (the dataset only fills components for ECGs within a year of an
  echo, and sets missing ones to 0).

## Heads

All use the Experiment 020 fixed readout (`full_development.fit_logistic`): train-only scaler and L2 logistic
regression with `C=0.01`, float64, one BLAS thread. Nothing is tuned on validation patients.

| Head | Inputs |
| --- | --- |
| `age_sex` | age / 10, male |
| `tabular` | age / 10, male, ventricular rate, atrial rate, PR, QRS, QTc (train-median imputed, with a missing indicator for atrial rate and PR) |
| `cpc` | 512 pooled CPC features |
| `cpc_tabular` | CPC features and the tabular inputs |
| `ptb_cpc_standard` | Experiment 020 `cpc_standard` head, fitted on PTB-XL, applied unchanged (no EchoNext fitting) |

## Analyses

1. **Primary:** AUROC and average precision of `cpc` on validation.
2. **Primary contrasts:** `cpc` minus `tabular`, and `cpc_tabular` minus `tabular`.
3. Secondary: `cpc` minus `age_sex`; `ptb_cpc_standard` AUROC, which asks whether the PTB-XL
   ECG-abnormality head already flags echo-confirmed disease.
4. Secondary: `cpc` and `tabular` AUROC for each eligible component label.

Differences use 2,000 paired bootstrap draws with seed 23023, by patient (one ECG each in validation).
Results are exploratory: one encoder and one seed, and the input preprocessing differs from the encoder's
pretraining data.

## Execution

`scripts/experiments/run_echonext_readout023.py` has two stages. `profile` checks the cache receipt, hashes the
inputs, reproduces the Experiment 020 `cpc_standard` head, and times extraction for 512 training ECGs on the
V100 under the shared GPU lock, projecting with a 1.5 margin plus 900 seconds against 3,600 seconds. `run`
requires a matching passed profile, extracts features, fits the heads on CPU, and writes local features,
predictions and an aggregate `result.json` to `outputs/experiment023_echonext_v1/`. Results go to
`docs/experiment-023-echonext-readout-results.md`.

## Addendum v2, 28 September 2026, before any EchoNext waveform is featurized or scored

The user asked for an EDA of EchoNext before running (`notebooks/08-jr-echonext.ipynb`,
[review](sph-echonext-eda-review.md)). Changes:

- Heads are fitted and scored only on rows with `use` in `data/processed/echonext_250hz_v1/rows.csv`. That
  removes the records failing the unit-free quality rules, including the 629 whose leads hold only 2.5-5 s of
  signal. Those records are half as often positive and would reward a shortcut.
- The CPC input statistics are the per-lead mean and standard deviation of the usable training rows, from the
  cache receipt.
- A secondary result reports the primary AUROC of `cpc` on all validation rows, including excluded ones.
- Component labels are restricted to rows with a measured echo value, as already specified; the EDA confirmed
  that missing values are coded 0.

Nothing else changes.

## Addendum v3, 28 September 2026, before any EchoNext waveform is featurized or scored

Experiments 022 and 025 found the frozen released ECG-JEPA and xECG clearly stronger than CPC:
- on PTB-XL at every label budget, for example +0.035 AUROC at 1,000 labels;
- on the independent SPH hospital (+0.036 and +0.039).

The useful question is now which representation carries echo-confirmed structural heart disease. v1 and v2
never ran. Changes:

- **Added heads**, all using the same fixed readout, the usable training rows and nothing tuned on validation:

  | Head | Inputs |
  | --- | --- |
  | `jepa` | 768 pooled ECG-JEPA features |
  | `xecg` | 1,024 pooled xECG features |
  | `jepa_tabular`, `xecg_tabular` | encoder features plus the tabular inputs |
  | `ptb_jepa_standard`, `ptb_xecg_standard` | the Experiment 022 v3 PTB-XL heads, applied unchanged |

- **Encoder input.** EchoNext has no physical units, so each lead is standardized with the usable-train
  statistics, as for CPC. It is then mapped onto the per-lead mean and standard deviation, in mV, of that
  encoder's PTB-XL input:
  - The statistics come from 1,000 evenly spaced Experiment 020 training ECGs: first 10 s, converted from
    500 Hz to 250 Hz with `resample_poly`, all samples pooled per lead.
  - Each encoder then applies its own preprocessing from 250 Hz: ECG-JEPA takes the 2,500 samples and its
    8 leads directly; xECG resamples to its 100 Hz with its own resampler.
  - A reported diagnostic, not a gate: for 32 evenly spaced PTB-XL training ECGs, compare features computed
    through this 250 Hz path with the cached 500 Hz-path features (mean and minimum cosine similarity).
- **Encoder contrasts (prespecified, secondary):**
  - `jepa` minus `cpc` and `xecg` minus `cpc`;
  - `jepa_tabular` minus `tabular` and `xecg_tabular` minus `tabular`.
  Same bootstrap as analysis 2. Analysis 4 is also reported for `jepa` and `xecg`.
- **Runtime ceiling:** 7,200 s, the project's adopted two-hour planning gate, instead of 3,600 s. Three
  encoders over about 76,000 ECGs do not fit the one-hour ceiling with the 1.5 margin. The profile times all
  three encoders on 512 training ECGs.
- **Outputs** go to `outputs/experiment023_echonext_v3/`. As EchoNext is credentialed, features and
  predictions stay local; only aggregates appear in documents.

The primary analyses (1 and 2) are unchanged. The test split stays closed.
