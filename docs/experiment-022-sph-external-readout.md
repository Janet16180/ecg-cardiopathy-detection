# Experiment 022: external readout of the Experiment 020 heads on SPH

**Frozen 28 September 2026, before any SPH waveform is featurized or scored.** The user added the Shandong
Provincial Hospital (SPH) ECG database and asked for the recommended experiments. Every earlier readout used
PTB-XL development patients, which the project has inspected many times. SPH is a different country, hospital,
device and labeling team, and no project model has seen it. It is used only as an external test here: nothing
is fitted, tuned or selected on it, and the run is done once.

## Question

How well do the Experiment 020 heads on the unchanged starting CPC encoder rank abnormal versus normal ECGs
in an independent hospital, and does the encoder still add over age and sex there?

## Data

- Source: `data/raw/sph/sph.zip`, downloaded with the ECG-FM benchmark manifest. Every file is checked
  against the manifest MD5 before use. 25,770 ECGs of 24,666 patients, 500 Hz, 10 to 60 s, HDF5.
- Signal: the first 10 s (5,000 samples) of each record, in the canonical lead order, converted to 250 Hz
  with the historical CPC transform (`ecg_experiment.cpc_input_audit.historical_resample`). The lead order and
  mV units are checked on the data before the run (Einthoven identities and amplitude range).
- Quality: as the quality policy says for held-out data, no record is excluded for quality. Records with a
  nonfinite sample would stop the run. `ecg_quality.assess` is run on every 10 s window and its exclusion
  reasons and review flags are reported, with AUROC also reported without policy-excluded records.

## Label

SPH lists AHA statement codes, with `+` modifiers. The label follows the standard PTB-XL superclass label of
Experiment 020 as closely as the code sets allow: rhythm and form statements are ignored.

| Group | AHA codes |
| --- | --- |
| Negative | `1` (normal ECG) and nothing else |
| MI | 160, 161, 165, 166 |
| STTC | 145, 146, 147, 148, 153 |
| CD | 82, 83, 84, 87, 88, 101, 102, 104, 105, 106, 108 |
| HYP | 140, 142, 143 |
| Ignored | 20-23, 30-37, 50-54, 60, 80, 81, 85, 86, 120, 121, 125, 152, 155, and all modifiers |

A record is positive when it has any MI, STTC, CD or HYP code, negative when its only code is `1`, and
undefined otherwise; undefined records are counted and dropped. Codes 85 and 86 are ignored because every SPH
use of them describes atrial flutter conduction (always with code 51), not AV block disease.

SPH never gives `1` together with a rhythm statement, so an otherwise normal ECG with sinus bradycardia is
coded `22` alone. PTB-XL would call such a record NORM. A prespecified secondary label therefore also counts
records whose codes are only 21, 22 or 23 as negative.

## Heads

The Experiment 020 heads are refitted exactly as there, from its saved training features
(`outputs/experiment020_full_development_v2/features.npz`, hash checked against its `result.json`) and the same
training rows and labels: `cpc_project_full`, `cpc_standard`, `age_sex_standard` and `cpc_age_sex_standard`.
Before SPH is scored, each refitted head must reproduce its saved Experiment 020 development probabilities to
1e-9. SPH ages are used as given (18 to 95); sex `M` is male. No age is missing.

## Analyses

1. **Primary:** AUROC and average precision of `cpc_standard` on SPH with the primary label.
2. **Primary contrast:** `cpc_standard` minus `age_sex_standard` AUROC.
3. Secondary: all four heads on the primary and the secondary label; `cpc_age_sex_standard` minus
   `cpc_standard`; `cpc_standard` minus `cpc_project_full`.
4. Secondary: `cpc_standard` AUROC for each superclass (records with that superclass versus negatives).
5. Secondary: primary AUROC without the records the quality policy would exclude.

Differences use 2,000 paired whole-patient bootstrap draws with seed 22022. The PTB-XL development AUROC is
quoted for context only; the two populations differ, so no test compares them.

## Execution

`scripts/experiments/run_sph_external022.py` has two stages. `profile` verifies the sources, checks the label
counts and the lead identities on 64 evenly spaced records, reproduces the Experiment 020 heads, and times CPC
extraction for 512 SPH records on the V100 under the shared GPU lock. It projects the run with a 1.5 margin
plus 900 seconds against a 3,600-second ceiling. `run` requires a matching passed profile, extracts features,
scores the heads, and writes local predictions and an aggregate `result.json` to
`outputs/experiment022_sph_external_v1/`. Results go to `docs/experiment-022-sph-external-readout-results.md`.

## Addendum v2, 28 September 2026, before any SPH waveform is featurized or scored

The user asked for an EDA of SPH before running (`notebooks/07-jr-sph.ipynb`,
[review](sph-echonext-eda-review.md)). It found 168 duplicate uploads, 25 with conflicting codes. Changes:

- The scored set is `use_evaluation` of `data/processed/sph_clean_v1` (hash checked against its receipt):
  25,577 ECGs, 13,818 negative and 7,190 positive on the primary label. One copy of each agreeing duplicate is
  kept, and both copies of a conflicting pair are dropped.
- Analysis 5 uses the manifest's `use_training` rows (the policy exclusions removed) instead of recomputing
  the policy in the runner.
- The EDA also showed that SPH is far more band-pass filtered than any training source. This does not change
  the analyses; the results document must mention it when interpreting the external score.

Nothing else changes. The run writes to `outputs/experiment022_sph_external_v2/`.
