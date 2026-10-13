# Experiment 067: one-time confirmation of the frozen localization maps on the ECGAD holdout half

Frozen 13 October 2026 (UTC), before any holdout map was built and before any holdout mark was read. Part of
the [localization programme](localization-program-2026-10-12.md). It applies the holdout rule of
[ecgad-benchmark-v1.md](ecgad-benchmark-v1.md): the holdout half is scored once, at the end of the programme,
for the methods frozen by then, and the scores are reported whatever they are.

Before this freeze, only the following were read: the protocols, results reports and code of 061, 062, 065, 066
and 068, the programme note and the benchmark note; the ECGAD split table (record IDs, patient IDs and
partitions); and the saved receipts and development outputs of 061-068. No holdout map, holdout mark or
holdout score exists. The holdout PTB-XL statements have not been read.

## Question

Does the development ranking of the frozen localization maps hold on the 201 ECGAD holdout records, which no
method has seen? In particular, does the best development map, 068 `units_v2`, beat the disease-blind bar
`qrs_prior` on within-record localization?

## Maps

Candidates (declared by 066 and 068):

| Map | Source | Development A2 |
| --- | --- | ---: |
| `qrs_prior` (bar) | 061 `locbench_baselines061.qrs_prior_map` | 0.7419 |
| `qrs_lexicographic` | 062 `normative_map062.record_maps` with `outputs/experiment062_normative_v1/normative.pkl` | 0.7530 |
| `factor` | 066 `combination066.factor_sum` and `finish`, 065 three-seed `mil_lse` weights | 0.7247 |
| `units_learned` | 066 unit combiner, weights `maps.combiner_weights` of `outputs/experiment066_combination_v1/result.json` | 0.7604 |
| `units_v2` (primary) | 068 `combiner068`, weights `fit.models.units_v2` of `outputs/experiment068_combiner_v2/result.json` | 0.7671 |

References, reported beside them and never candidates: `random`, `amplitude`, `U_B` and `residual056` (061
code; `U_B` and `residual056` with the saved `outputs/experiment061_locbench_v1/ub_references.pkl`), and 066
`textbook`. The 066 diagnostic `textbook_true` is also reported, as a diagnostic: it paints the textbook region
of each holdout record's true PTB-XL statements (065 label rule), so it reads the holdout statements (not the
marks) in the map stage. It is never a candidate.

Every map is produced by the frozen code with the frozen fitted quantities: the 065 network weights
(`outputs/experiment065_mil_v1/weights_lse_6506[5-7].pt`), the 066 lead-quality model
(`outputs/experiment066_combination_v1/quality/model.pkl`), the 062 normative model, the 061 U_B references
and the 066 and 068 combiner weights. Nothing is refitted and no parameter changes. The 061-068 runners
refuse holdout records and keep their per-record steps inside the scripts, so a thin wrapper
(`ecg_experiment/holdout067.py`) repeats those steps by calling the frozen module functions:

- signals from the LocBench cache (`locbench_cache.open_cache`); the exact float64 signal
  (`locbench_baselines061.exact_signal`) for the 061 maps, amplitude, residual and quality features, and the
  float32 cache values cast to float64 for the 062 normative maps and the 066 delineation, as in 062 and 066;
- age and sex from the cache index;
- MIL bag logits and evidence from the three 065 networks on GPU with bf16 autocast, deterministic cuDNN and
  the 066 batching (128-record chunks, `mil065.predict` batches of 64), summarized as in 066 (bag logits, lead
  means over the scored steps, rhythm-head evidence stored as float16, combined map);
- lead quality q from the saved 066 model on `combination066.quality_features`;
- the 062 maps stored as float16, as 062 stored and scored them (the `normative_v1` field of the unit features
  too, as in 066 and 068);
- the 066 and 068 maps exactly as `run_combination066._record` and `run_combiner068._features` and `_paint`
  built them.

Holdout records: the 201 `holdout` rows of `ecgad.load_split()`, in benchmark-index order. Benchmark index 162
(partition `duplicate`) is never mapped or scored. Maps are float32, shape (201, 12, 5000), canonical lead
order, written outside Git (`/tmp/ecg067-work/maps_<name>.npy`). A manifest with the SHA-256 of every map file
and the record order is written to `outputs/experiment067_holdout_v1/maps_manifest.json` and committed as
`docs/experiment-067-maps-manifest.json` before scoring.

## Metrics

LocBench v1 ECGAD metrics ([061 protocol](experiment-061-locbench.md) and its Amendment 1), computed with
`locbench.ecgad_record_metrics`, `a1_histograms`, `metric_interval` and `paired_difference` on the holdout
records, samples 100-4899. A record is marked if any scored lead-sample is marked.

- A1 pooled point AUROC over all 201 holdout records; A2 within-record point AUROC (mean over marked records,
  skipping records whose every scored lead-sample is marked); A3 lead AUROC (records with 1-11 marked leads);
  A4 time AUROC; A5 record AUROC (marked against unmarked records); A6 precision at the top 5%, beside the
  marked share.
- Intervals: 2,000 whole-patient bootstrap draws, seed 61061, 2.5 and 97.5 percentiles; paired differences use
  the same draws for both maps.

## Primary comparison

Paired holdout A2 of `units_v2` minus `qrs_prior` over the holdout marked records, 2,000 whole-patient
bootstrap draws, seed 61061.

- **confirmed** if the 95% interval lies above 0;
- **matches** if it contains 0;
- **below** if it lies below 0.

## Secondary readings

All descriptive. Secondary comparisons are not adjusted for multiplicity.

1. Paired A2 against `qrs_prior` of every other candidate (`qrs_lexicographic`, `factor`, `units_learned`) and
   of every reference; `units_v2` minus `textbook`; `units_v2` minus `units_learned`.
2. A1-A6 with intervals for every map, and paired differences against `qrs_prior` for A1-A6.
3. QRS strata (061 Amendment 2 definition): `A2_qrs` and `A2_nonqrs`, with the QRS mask taken from the holdout
   `qrs_prior` maps (value at least 0.5). A record enters a stratum only if the stratum holds marked and
   unmarked lead-samples. Means with intervals (`locbench_strata.strata_summary`) and paired differences
   against `qrs_prior` (`strata_paired`).
4. Statement-stratified A2. Each holdout record's PTB-XL statements at likelihood at least 50 are mapped to
   the families of `locbench_data.REGION_TABLE` (`locbench.FAMILIES`); a marked record with no such statement
   is in `other`. A record counts in every family it has. Mean A2 per map for each group with at least 5
   records that have an A2. Descriptive only, no intervals.
5. Development-to-holdout shift of A2 for every map: holdout A2 minus development A2 (the development value
   recomputed from the saved development tables). Its 95% interval comes from independent bootstraps of the two
   halves: holdout draws with seed 61061, development draws with seed 61062, 2,000 each, interval of the
   draw-wise differences.

## Integrity checks

Before any holdout map is built (stage 2):

- The wrapper maps the 198 development records, and its maps must equal the scored development maps saved by
  061 (`tables/<map>/ecgad_maps.npy`), 062 (`locbench/qrs_lexicographic/tables`), 066
  (`locbench/<map>/tables`) and 068 (`locbench/units_v2/tables`): exactly for the 061 and 062 maps, and to at
  most 1e-5 absolute for the 066 and 068 maps (they depend on GPU MIL evidence). If a check fails, no holdout
  record is mapped and the failure is reported.
- The recomputed 065 bag logits of the development records must equal 065's saved `bags_ecgad_dev.npz` to at
  most 1e-3 (066's tolerance), and q must equal the saved 066 `q.npz` values to at most 1e-12.
- The SHA-256 of every frozen fitted input is recorded and compared with the receipts that record it (065
  weights, 062 model and 061 references in 066's `result.json`).
- No ECGAD record enters any fitted quantity: nothing is fitted in this experiment.

At scoring (stage 3), before any holdout mark is read:

- The SHA-256 of every map file and the record order must match the committed manifest.
- The ECGAD label file must match the hash in `ecgad-benchmark-v1.md`.
- The development A2 of each map, recomputed with `ecgad_record_metrics` from the wrapper's development maps,
  must match the saved per-record development A2 to 1e-12 and the saved development means (`qrs_prior`
  0.7419383358406554, `qrs_lexicographic` 0.7530198339166797, `factor` 0.7246870157102797, `units_learned`
  0.7603613741425232, `units_v2` 0.7671251079351179, `textbook` 0.7493730086608462) to 1e-12.

After scoring: holdout `random` A2 must lie in [0.48, 0.52]; if not, the metric code is wrong, the fix is
treated as a crash fix (below) and recorded.

## Single scoring and reporting rule

One run of the scoring script loads the holdout marks (rows of the holdout records only), checks the hashes,
computes everything declared above and writes `outputs/experiment067_holdout_v1/result.json` and per-record
tables. If it crashes after reading the labels, only the crash is fixed (no parameter, map or definition
changes), it is rerun, and every attempt is recorded in the report. The results are reported whatever they
are. Nothing is retuned, reselected or rescored afterwards, and the holdout half is then spent: it cannot
select or tune any later method.

## Not read

The Challenge test and calibration groups, the EchoNext test set, and PTB-XL fold 9 and fold 10 records
outside ECGAD. No fitting on any ECGAD record.

## Code and outputs

- `ecg_experiment/holdout067.py`: the wrapper and the holdout metric helpers.
- `scripts/experiments/run_holdout067.py`, phases `maps` (stage 2, GPU0 with the shared lock for the MIL
  evidence, CPU 8 processes) and `score` (stage 3, CPU).
- `tests/test_holdout067.py`: unit tests that never read holdout labels.
- Outputs: `outputs/experiment067_holdout_v1/` (`maps_manifest.json`, `maps_check.json`, `result.json`,
  per-record tables); maps in `/tmp/ecg067-work/`.
- Report: `docs/experiment-067-holdout-results.md`, written from the outputs.
