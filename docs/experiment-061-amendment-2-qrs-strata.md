# Experiment 061 amendment 2: QRS-stratified ECGAD reading

12 October 2026, written after Experiment 061's scores and before any score of Experiments 062-066. It adds a
secondary reading; it changes no 061 definition, set or result. It is post hoc for 061 and prospective for
062-066.

## Why

Experiment 061 found that the 059 hybrid QRS covers 16.1% of scored ECGAD development samples but holds 35.9%
of the marked lead-samples, so the disease-blind `qrs_prior` map is the best baseline on A2 (0.742). A method
that only learns where QRS complexes are would match it without localizing anything. The primary comparison of
062-066 stays the paired A2 against `qrs_prior`; this reading shows where any gain comes from.

## Definition

For each marked ECGAD development record, the QRS mask is `qrs_prior >= 0.5` (the 059 hybrid QRS intervals),
from `outputs/experiment061_locbench_v1/tables/qrs_prior/ecgad_maps.npy`. On samples 100-4899:

- `A2_qrs`: within-record AUROC of the map, marked against unmarked lead-samples, inside the QRS mask only. It
  asks whether the map picks the marked complexes and leads among all QRS complexes.
- `A2_nonqrs`: the same outside the QRS mask. It asks whether the map finds marked P, PR, ST and T regions.

A record enters a stratum only if that stratum holds both marked and unmarked lead-samples. Each stratum is the
mean over its records, with 2,000 whole-patient bootstrap draws (seed 61061). Paired differences use the records
where both methods are defined. `qrs_prior` itself is constant inside and outside the mask apart from its 1e-3
amplitude tiebreak, so its strata measure that tiebreak only.

Code: `ecg_experiment/locbench_strata.py`; CLI `scripts/experiments/run_locbench_strata.py <reference tables>
<method tables>...`, which writes `strata.json` and `strata.csv` into each tables folder.
