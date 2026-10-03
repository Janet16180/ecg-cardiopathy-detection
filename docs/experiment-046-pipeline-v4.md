# Experiment 046: pipeline v4, pipeline v3 with the ensemble readout

**Frozen 1 October 2026, before any score of this experiment is computed.** Experiment 045 found that the
unfitted ensemble E, the mean of pipeline v3's readout logit (`logistic_concat`) and the `attention_jepa`
3-seed mean logit of Experiment 043 Stage 2, beats v3's readout by 043's rule: SPH AUROC 0.9442 against
0.9404 (+0.0038 [+0.0028, +0.0049]), full PTB-XL development 0.9405 against 0.9347 (+0.0058). An AUROC gain
does not have to show at the top 5% of normals, where the referral rule works. This experiment asks whether
pipeline v3 with only its binary readout replaced by E, **pipeline v4**, catches more athlete-criteria
abnormal ECGs at the same referral budget. The user authorized follow-up experiments when results show
improvements. Before this freeze only the protocols and results reports of 037, 043, 044 and 045, the
aggregate `result.json` fields of 043 Stage 2 (profile, cache receipt, validation counts, best epochs and
AUROCs), 044 (output hashes and primary contrast), 044's rate-matched check and 045 (ensemble AUROCs), and the
key names and shapes of their saved prediction files, were read. No `attention_jepa` score of the 4,569 SPH
ECGs without a binary label exists, and no score, threshold or metric of v4 on any draw was computed.

## The three pipelines

| | v3 (current candidate) | v4 (candidate) | v2 (reference) |
| --- | --- | --- | --- |
| Binary readout | R3: 044's readout on xECG + JEPA features (1,792) | E = (R3 logit + A logit) / 2 | 037's R: xECG features (1,024) |
| Score used to rank | R3 probability | expit(E) | R probability |
| Finding heads, standardization | 032's xECG PVC and WPW heads, 033's z-scores | the same | the same |
| Rule | 033 `combined_50` and `binary` alone | the same | the same |

A is `attention_jepa`: the 3-seed mean logit of 043 Stage 2's attention head on frozen ECG-JEPA tokens.
E has no fitted weight, scale or threshold. The thresholds are order statistics of the local normals' scores,
so ranking by expit(E) is ranking by E; expit is used only so that v4's binary column is a probability like
v3's. Everything else (rows, labels, weights, solver settings, the threshold rule, the draws, the budgets, the
seeds and the resamples) is 044's, unchanged.

### R3 and the frozen v2

- v3 in the draws is the frozen v3: its binary score is 044's saved `sph_v3_binary` and E uses 044's saved
  R3 logits `sph_v3_logit` and `development_v3_logit` (hash checked). R3 is refitted exactly as 044
  (`multisource_readout.fit_readout` on 043's `training_data` rows, labels and 035 `dropped_upweighted`
  weights, one BLAS thread, OpenBLAS Haswell kernels) only to check them.
- v2's binary score is 037's saved `sph_v2_binary` (hash checked). The finding scores of all three pipelines
  are 033's `load_scores`, as 037 and 044 used them.

### A, retrained (043's weights were not saved)

- The 044 pipeline scores every SPH evaluation ECG (25,577, including the 4,569 without a binary label), and
  its local normals are SPH ECGs. 043 scored only the 21,008 labeled SPH ECGs and saved no weights, so A is
  trained again with 043 Stage 2's recipe, unchanged: `ann_heads.AttentionHead(768)` (LayerNorm, linear to
  128, GELU, dropout 0.1, a 128-64-1 tanh scorer, per-token logits), AdamW, learning rate 3e-4, weight decay
  1e-2, batch 64, at most 30 epochs, patience 4, weighted binary cross-entropy with R's sample weights,
  `ann_heads.train` (best validation-AUROC epoch kept), seeds 43043, 43044 and 43045 with
  `torch.manual_seed(seed)` before each network is built, as 043's `train_seeds`.
- Rows: 043's 35,638 fit rows and 3,939 validation rows (043's `shared_validation`, seed 43043; the mask must
  equal 043 Stage 1's saved `validation_mask`).
- Tokens of the training, development and labeled SPH rows: 043's float16 token cache
  (`<scratchpad>/exp043_cache/`), reused only if its row keys equal 043's `extraction_table` and both data
  hashes equal its receipt (rehashed).
- Tokens of the 4,569 SPH evaluation ECGs without a binary label: extracted with 043's `extract_cache` into a
  new scratch cache (`<scratchpad>/exp046_cache/`), with 043's SPH reader (`sph.read_window`, the window hash
  checked against the manifest's `signal_sha256`, `external_encoders.jepa_input`,
  `lead_wave_maps.jepa_tokens`). Every extracted token mean must equal 022's saved SPH JEPA feature to 1e-4.
- A scores the 1,604 development ECGs and all 25,577 SPH ECGs (21,008 from 043's cache and 4,569 from the
  new one). Each seed's weights are saved to `attention_jepa_weights.npz`; a fresh network loaded from the
  file must reproduce that seed's development logits to 1e-6.

### Reproduction checks (the run stops if any fails)

- Every input file is checked against its receipt: 037's `predictions.npz`, `draws.csv` and
  `pipeline_v2_heads.npz`; 043 Stage 1's and Stage 2's `predictions.npz`; 044's `predictions.npz`,
  `draws.csv` and `pipeline_v3_heads.npz`; 035's `predictions.npz`; and those that 033's `load_rows` and
  `load_scores` check themselves (030, 032). The hashes of 044's rate-matched `result.json` and 045's
  `result.json` are recorded.
- The xECG rows, labels, weights and evaluation rows equal 037's `training_data` (043's `check_against_037`).
- The R3 refit reproduces 044's saved `sph_v3_logit` (25,577) and `development_v3_logit` (1,604) and the
  probabilities `sph_v3_binary` and `development_v3_binary` to 1e-10, and 043 Stage 1's `logistic_concat`
  logits to 1e-10 (044's `check_v3`). Scoring the `v3_*` parameters of 044's `pipeline_v3_heads.npz`
  reproduces the refit's probabilities to 1e-12.
- 037's saved `sph_v2_binary` equals 044's saved `sph_v2_binary` exactly, and 033's z-scores equal 044's
  saved `sph_v2_z_pvc`, `sph_v2_z_wpw` and `sph_v2_z_combined` exactly.
- **Retrained A against 043 (GPU training need not be bit-identical, so this is a tolerance, not an
  equality).** On the 21,008 labeled SPH ECGs and on the 1,572 full-development ECGs, the retrained 3-seed
  mean logits must correlate with 043's saved `sph_attention_jepa` and `development_attention_jepa` at
  Pearson r ≥ 0.999, and their AUROCs must be within 0.001 of 043's (0.94039 SPH, 0.93683 full development).
  Reported beside it: whether the logits are bit-identical, their largest absolute difference, each seed's r
  and AUROC against 043's seed, and each seed's best epoch against 043's.
- The v2 and v3 draws reproduce every v2 and v3 row of 044's `draws.csv` exactly (every threshold, outcome
  and local-pool column, difference 0), and the v2 rows reproduce 037's (044's `check_v2_draws`).
- v2's AUROCs on 037's four sets equal 037's (043's `check_comparator_aurocs`); v3's equal 043 Stage 1's
  `logistic_concat` to 1e-12.
- 033's local-pool selection with v2 and with v3 selects `combined_50`, as in 044.
- The matched-rate check below reproduces every v2 and v3 number of 044's rate-matched `result.json`
  (observed rates and interval bounds) to 1e-12.

## The simulated site, draws and rules: 044's, exactly

- **SPH is development data** (read by 022-045). This simulates a new site; it is not a final test.
- Rows, split and outcomes: 033's `load_rows`; evaluation half 12,759 ECGs of 12,320 patients (6,895 normal,
  3,584 binary positive, 4,052 athlete-criteria composite, 1,812 "other"); local pool of 6,923 normals.
- Threshold rule (030), `combined_50` (033, `finding_screen.split_thresholds`), 200 draws per m from
  `numpy.random.default_rng([30030, m, draw])`, m in {50, 100, 200, 500, 1,000, 2,000}, budgets
  {1%, 2%, 5%, 10%}; `combined_50` only where k = floor(b m) ≥ 1. The other 033 rules run at 5% with 200
  normals only, for the local-pool selection step. The same draws serve every pipeline (paired).
- Intervals: 037's, unchanged. Whole-patient bootstrap over the 12,320 evaluation patients, 2,000 resamples
  from `numpy.random.default_rng(41041)` (`pipeline_v2.resample_counts`), each resample's statistic the mean
  over the 200 draws with the draw thresholds fixed, 2.5th and 97.5th percentiles. AUROC contrasts use
  `intervals.paired_auroc_difference`, 2,000 draws, seed 41041.

## Primary comparison and decision

- **Primary:** the share of athlete-criteria composite ECGs (4,052) caught by rule `combined_50` at a 5%
  budget with 200 local normals, **v4 minus v3**, paired over 044's identical draws and resamples, with its
  95% interval.
- **Decision.**
  - `adopt_v4` if the lower bound of that interval is above 0;
  - otherwise `no_worse_keep_v3` if the lower bound is above −0.005 (037's margin): v4 is reported as no
    worse, but not adopted;
  - otherwise `keep_v3`.

## Secondary outcomes (no decision)

- At 5% with 200 normals, for v2, v3 and v4: binary-label sensitivity (3,584); the achieved false-referral
  rate (share of the 6,895 evaluation normals referred); **the share of the 1,812 "other" ECGs referred**
  (sinus bradycardia or tachycardia, atrial premature beats and similar; neither normal nor abnormal by the
  label). At a student site many of these would be normal variants, so this is the student-relevant cost of
  a pipeline; it is reported in the first results table beside the primary, whatever the decision. Also PVC,
  WPW, frequent PVC, AF/flutter, high-grade AV block, long QT, composite without a binary label and the four
  superclasses; referrals and cases caught per 1,000 at an assumed 5% prevalence (037's definition).
- The same outcomes at every other budget and m, for `combined_50` and `binary` alone.
- Contrasts, each paired: v4 − v3, v4 − v2, v3 − v2 (which reproduces 044), and each pipeline's
  `combined_50` − `binary`.
- 033's local-pool selection step repeated with v4.
- Matched false-referral rate (044's post hoc check, prespecified here, descriptive): at each budget, for
  `combined_50` and `binary`, thresholds set on the evaluation normals themselves (an oracle no site has),
  re-estimated in each whole-patient resample; normal, composite, binary-positive, PVC and "other" referral
  rates, v4 − v3 and v4 − v2 with intervals. Also the evaluation rates with the thresholds set on the whole
  local pool (6,923 normals).
- AUROC of the binary readouts on 037's sets (PTB-XL full development 1,572 ECGs, ordinary 1,306, hard 266;
  SPH 21,008): v2, v3, v4 and A alone, with the paired v4 − v3 and v4 − v2 intervals. Beside them, 045's E and
  043's A, so the effect of retraining A is visible.
- Cost: the GPU profile and run times, and the inference cost per ECG of each part of v4 (xECG features,
  ECG-JEPA tokens, the three attention heads), timed on 128 of the newly read SPH ECGs. The xECG features
  computed there are compared with 022's saved ones (descriptive).

**What to expect (reasoning before any score).** E's SPH AUROC gain over R3 (+0.0038) is about twice R3's
over R (+0.0017), which gave +0.008 composite at 5% in 044, so a gain of about +0.005 to +0.01 is plausible.
037 showed that AUROC and budget sensitivity can disagree, so the sign is not certain. A is a token-level
morphology head, and in 045 its map marked benign variants more often than `U_B`; v4 may therefore refer
more "other" ECGs than v3, adding to 044's +0.026. The retrained A may move E's AUROC by about 0.001 from
045's.

## Closed data and exclusions

- The PTB-XL calibration and test ECGs, the Challenge test groups and the EchoNext test set are not read. The
  Challenge calibration groups are not read: the finding scores are 033's saved z-scores.
- The SPH ECGs are 044's 25,577 evaluation ECGs; only the waveforms of the 4,569 without a binary label are
  read again (their features were extracted by 022).
- No age or other demographic subgroup analysis. No label definition changes.
- New files only: `ecg_experiment/pipeline_v4.py`, `tests/test_pipeline_v4.py`,
  `scripts/experiments/run_pipeline_v4_046.py`, this protocol and the results report. 033's, 030's and 032's
  functions are reached through the 043 and 044 runners, which import them. If v4 is adopted, the
  specification goes to a new `docs/pipeline-v4.md`; `docs/pipeline-v2.md` and `docs/pipeline-v3.md` are not
  edited.

## Caveats written into the results

- SPH is development data, an older Chinese hospital cohort at 34% prevalence; the local pool and evaluation
  half come from the same hospital, the most favourable case.
- 045 chose E on SPH and full development AUROC, both read here again; this experiment is a third look at the
  same data, not an independent confirmation.
- A is a retrained network; its scores can differ slightly from 043's, and three seeds of one recipe are one
  realization of it. The network saw 90% of the training rows.
- WPW (15), high-grade AV block (12) and long QT (10) have few evaluation ECGs.
- The labels are ECG annotation proxies, not confirmed disease or referral decisions.
- v4 needs both frozen encoders, ECG-JEPA's token output (not only its pooled feature) and three small
  attention networks at the student site.

## Execution

```bash
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    CUDA_HOME=.venv/lib/python3.11/site-packages/nvidia/cuda_runtime \
    .venv/bin/python -u -m scripts.experiments.run_pipeline_v4_046 --smoke \
    --output outputs/experiment046_pipeline_v4_smoke
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    CUDA_HOME=.venv/lib/python3.11/site-packages/nvidia/cuda_runtime \
    .venv/bin/python -u -m scripts.experiments.run_pipeline_v4_046
```

One GPU (about 11 GB free beside the vLLM server), held under `ecg_experiment.gpu.gpu_lock("cuda")` for all
CUDA work; at most four CPU threads (logistic fits with one). Before any extraction or training the runner
profiles the extraction (128 records) and 30 optimizer steps, and stops if the projected time of extraction,
training (every epoch run) and scoring exceeds 2 hours. The smoke mode uses training rows only: 043's
`smoke_split` (4,000 binary training rows, a held-out quarter split by group into a local pool and an
evaluation half), the readouts and finding heads fitted as in 044's smoke, A trained for 2 epochs on the
smoke rows' cached tokens, 64 training rows extracted into a smoke cache with the token-mean check, sizes 50
and 100 and 200 resamples; no development, calibration or SPH ECG is read or scored. The runner hashes every
input, source and this protocol into the result, performs the checks above, writes to `<output>.partial` and
renames it when complete, and refuses to overwrite an existing output. The full run writes
`outputs/experiment046_pipeline_v4_v1/` (`result.json`, `draws.csv`, `predictions.npz`,
`attention_jepa_weights.npz`, `run.log`). Results go to `docs/experiment-046-pipeline-v4-results.md`.
