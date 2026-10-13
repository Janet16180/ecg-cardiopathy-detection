# Experiment 061 amendment 3: lead-standardized noise reading

12 October 2026, written after Experiment 061's scores and before any score of Experiments 062-066. It adds a
secondary reading and changes no 061 definition, set or result.

## Why

C1 is the within-record AUROC of raw lead scores against the leads a noise annotation names. In the noise set,
61% of records name only limb leads, and limb leads have small signals (median amplitude lead score of clean
normals 0.11-0.22 mV against 0.39-0.45 mV in V2-V4; `outputs/experiment061_locbench_v1/descriptives.json`). Any
map whose level differs by lead therefore moves C1 away from 0.5 without reacting to noise: `amplitude` scores
C1 0.333 only because noisy leads are small leads.

## Definition

`C1_std`: for each lead, take the median and interquartile range of the method's lead score (95th percentile of
the lead's map over samples 100-4899) across the 1,400 `evaluation` clean normals of the noise control. For each
noise-set record, standardize its 12 lead scores by those per-lead values and take the within-record AUROC
against the named noisy leads. Report the mean over the 299 records with 2,000 whole-patient bootstrap draws
(seed 61061). Near 0.5 means the map is not drawn to noise beyond its usual lead levels; above 0.5 means it is.
C1 stays reported unchanged.

Code: `ecg_experiment/locbench_noise.py`; CLI `scripts/experiments/run_locbench_noise.py <tables>...`, which
writes `noise_std.json` and `noise_std.csv` into each tables folder.
