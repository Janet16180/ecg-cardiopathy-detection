# Pipeline v2: the candidate student screening pipeline

Adopted by [Experiment 037](experiment-037-pipeline-v2-results.md) on 30 September 2026 (decision
`adopt_v2`). This file lists everything a later one-time final test (`final_frozen_test`) would freeze. It
describes a candidate, not a validated screen: every number behind it comes from development data (PTB-XL
development, the Challenge calibration groups and SPH, all read by earlier experiments). No final test has
been run. The Challenge test groups have not been read by any readout. The PTB-XL test set is not untouched:
Experiments 001-008 evaluated on it, and the encoders were chosen after those readings
([audit](audit-2026-09-30.md)).

On SPH, v2 caught 0.005 fewer athlete-criteria abnormal ECGs than v1 (the 022b readout) at a 5% budget with
200 local normals (−0.0048 [−0.0072, −0.0025]). That is within the pre-registered tolerance for adoption but
fails a stricter non-inferiority reading. v1 is described at the end so that the final test can report both
if the user wants it to.

## 1. Features

- One frozen encoder: the released xECG backbone (`third_party/checkpoints/xecg/`: `model.safetensors`
  SHA-256 `812dec69…722c`, `config.json` `23ba68fc…9a4a`, `xECG.py` `373fed5a…e266f`), loaded with
  `ecg_experiment.external_encoders.load_xecg_backbone`.
- Input: a 10 s, 12-lead, 500 Hz window in mV, canonical lead order, converted to the official 100 Hz
  time-major input by `external_encoders.xecg_input` (`ptb_xecg_input` for raw PTB-XL records); for longer
  Challenge records, the centred 10 s window (start = (samples − 5000) // 2). Features: the pooled
  1,024-dimensional representation from `external_encoders.xecg_features`, float64 at readout time.
- The feature files used to fit the heads: the PTB-XL xECG cache (`features.npy` `f33d89d0…a7f4`,
  receipt `de79664f…7760`), `outputs/experiment022_sph_external_v3/features.npz` (`f6f82fe6…1857`, SPH) and
  `outputs/features_challenge_v1/` (`metadata.json` `9bcd2f7c…21ee`).

## 2. Binary readout R (035 `dropped_upweighted`)

- Model: a `StandardScaler` and an L2 logistic regression, `C = 0.01`, `lbfgs`, `tol = 1e-8`,
  `max_iter = 5000`, float64, with sample weights in both the scaler and the fit
  (`multisource_readout.fit_readout(x, y, weights)`). Converged in 316 iterations.
- Training rows (39,577, 27,360 positive), in this order:
  - the 17,083 PTB-XL training ECGs of `full_development.cohorts` with a standard label (9,840 positive),
    including the 1,724 without a project label (353 positive);
  - the 22,494 Challenge training-group rows (frozen split `data/processed/challenge_splits_v1`, duplicate
    status `unique` or `kept`) with a primary label that pass the training quality policy: 022b's
    `training_rows.csv` rows with an empty `reasons` (`d216d93e…f407`); Ningbo `use_training`, the other
    sources `ecg_quality` on the saved feature window (026b's `quality_reasons`).
- Labels: the PTB-XL standard label and the Challenge primary label (`challenge_labels`).
- Weights (average 1): each of the 1,724 PTB-XL rows without a project label 2.3167476, every other row
  0.9400292, so that those rows carry 1,724 / 17,083 = 10.09% of the weight, their share in a PTB-XL-only
  fit (`hard_subset.share_weights`, arm spec `ARM_SPECS["dropped_upweighted"]`).
- Output: the positive-class probability. It is used only to rank; no calibration step is applied.

## 3. Finding heads (032) and the combined finding score F

- Two heads on the same xECG features: `ventricular_ectopy` (PVC) and `preexcitation` (WPW), each a
  `StandardScaler` and L2 logistic regression with the same settings, unweighted
  (`full_development.fit_logistic`), trained on PTB-XL training rows plus the Challenge training rows that
  pass the quality policy, with 032's group labels (`rhythm_findings.GROUPS`; rows undefined for a group are
  left out). The rows and reasons are 032's `training_rows.csv` (`66a55a47…e43a`). PVC converged in 233
  iterations, WPW in 81.
- Standardization: each head's logit (probabilities clipped to [1e-15, 1 − 1e-15]) minus its mean, divided by
  its SD (ddof 0), both on the 1,707 Challenge calibration clear normals (standard-label negative, positive in
  no 032 group): PVC mean −5.539442, SD 0.825770; WPW mean −9.857676, SD 1.768115
  (`finding_screen.finding_z`).
- F = max(z_PVC, z_WPW).

## 4. The referral rule and threshold (030 and 033 `combined_50`)

- A site first records m local ECGs that the cardiologist reads as normal. From their scores, with a budget
  b and k = floor(b m) computed in integers:
  - F's threshold is the (r + 1)-th highest normal F, r = floor(50 k / 1000);
  - R's threshold is the highest-ranked normal R such that at most k − 1 of the m normals lie above either
    threshold (`finding_screen.split_thresholds(normals, budget_per_mille, (50,))`, columns R then F).
- An ECG is referred when R or F is **strictly above** its threshold. The rule needs k ≥ 1 (for example
  m ≥ 50 at 2%, m ≥ 20 at 5%).
- Budget choices: 1%, 2%, 5% or 10% of normals flagged; 5% is the reference. The budget is the
  cardiologist's choice of workload, not a tuned value.

## 5. The local-normal requirement

- The threshold must come from the site's own normal ECGs. A threshold taken from other hospitals' normals
  referred 1-20% of normals in 030; this pipeline has no source-quantile option.
- The normals must be read as normal by the cardiologist with the athlete criteria in mind: the F threshold
  rests on the most extreme normal pilot ECG, so an unrecognised PVC or WPW among them hides real ones.
- Size: 200 normals give a usable mean but a wide achieved rate (3.0-7.7% for a 5% target across draws);
  about 1,000 are needed to keep a 5% budget within ±1 point in about 9 of 10 pilots (88% of draws with v2;
  96% with 2,000). At 1-2%, 500 normals suffice (94-96%). At 10%, even 2,000 normals keep the rate within
  ±1 point in only 82% of draws.

## 6. Expected operating numbers (SPH, development data)

Evaluation half of the SPH simulated site, 200 local normals, per 1,000 ECGs at an assumed 5% prevalence of
athlete-criteria abnormal ECGs:

| Budget | Referrals per 1,000 | Abnormal (binary label) caught | Athlete-criteria findings caught | Normals referred |
| --- | ---: | ---: | ---: | ---: |
| 1% | 47 | 0.61 | 0.64 | 1.52% |
| 2% | 58 | 0.68 | 0.71 | 2.43% |
| 5% | 91 | 0.77 | 0.79 | 5.45% |
| 10% | 138 | 0.84 | 0.85 | 10.07% |

PVC ECGs are referred 0.96-0.99 of the time and WPW 0.76-0.88 (15 ECGs). These are the numbers a final test
would check, not guarantees.

## 7. Code paths and hashes to reproduce

Frozen parameters (load these to score new ECGs; do not refit):

- `outputs/experiment037_pipeline_v2_v1/pipeline_v2_heads.npz`, SHA-256
  `ca2deef4c0553cdeaa80b6ed75f9349688b9415ca7e4b75cbe12954569bcc5af`. Keys `v2_*` (R), `pvc_*`, `wpw_*`
  (`mean`, `scale`, `coef`, `intercept`), `pvc_logit_mean`, `pvc_logit_sd`, `wpw_logit_mean`,
  `wpw_logit_sd`; also `v1_*` for the comparison. Score with `ecg_experiment.pipeline_v2.score_parameters`
  (reproduces the fitted heads to 6e-16).
- Experiment 037 result: `outputs/experiment037_pipeline_v2_v1/result.json` `36ac7f6a…c857`;
  `predictions.npz` `3f8f35eb18523b003ea83c1c646cc5aaa81e6a7f8bdee2ab204787baad7686a2`; `draws.csv`
  `eb26c95b5d2313af51b3648a4c9a81ff4415047b25cbb73d71a8e58342d4d8fd`.

Code (hashes as recorded by the 037 run, commit `780d6f6`):

| Path | SHA-256 |
| --- | --- |
| `ecg_experiment/pipeline_v2.py` | `1748c062efa9cb71630a50122d268c74e1ddaa87ec5d6e3f6bf52439dc3c8b53` |
| `ecg_experiment/finding_screen.py` | `2cb793ac7094e5fba4d3a239f154ff8e8bdb8ee780f63e6fc9dace3658687648` |
| `ecg_experiment/referral_budget.py` | `c2af973802e269c29c143b30e54147c05bba5513548149b8174269aec13ff56c` |
| `ecg_experiment/hard_subset.py` | `3f8f04be40202902747636da6e909f3983ecd025de09285d381278f37cafacb2` |
| `ecg_experiment/multisource_readout.py` | `6cf20dff174ace45e870fb4f65776c654f954fb96a82a7c739c70946b7887bc5` |
| `ecg_experiment/full_development.py` | `ae7f37ae76063220785dc4031ee5ba206b0ca75134a9f37c858c66ef2dff83c4` |
| `ecg_experiment/hybrid_score.py` | `e0206b3f545afb83f378972780694993ebc1dbc61e8a41003f8b1de3336f1e7d` |
| `ecg_experiment/rhythm_findings.py` | `9fd88e4a25f0951253e16ed3a84d8976a96b999b361f0e69bacc91adeef48bde` |
| `ecg_experiment/intervals.py` | `b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af` |
| `scripts/experiments/run_pipeline_v2_037.py` | `d05868abbd982b283bb5e06329a8ed62b1a19d25db5a18b8ddf7681fbc3a3bde` |
| `scripts/experiments/run_rhythm_findings032.py` | `de5fa98aabb8787ba09057c8be31da6cb32c0d4c91d011b6870fa0202270fc5e` |
| `scripts/experiments/run_finding_screen033.py` | `0a15d541ab8dae3c554880f708bcc101f7d62bf457775941e3b3926d487ec879` |
| `scripts/experiments/run_referral_budget030.py` | `7f23495f2c0916f7c91d33949f49fdb36a19af17fda28808086f9235953a1f17` |
| `scripts/experiments/run_hard_subset035.py` | `85366059ab4f607e9c14552cff7da2441f11c95c5b8a9839f7d18e884f7f362c` |
| `scripts/experiments/run_multisource_manifold026b.py` | `5dc7b21202fc6fe940589bd98b412ff91741028169786f4738bfb77d23b31823` |
| `pyproject.toml` | `9df6d56e8f27348d067eb0cf3b6506f5efbbc03154b96b6b3987f4bb4854279e` |
| `uv.lock` | `20c6e373cba70e7fe859ba59697f6145c34b42418eddc955f82d29a165d5db7c` |

Inputs checked by the run against their receipts:

| Output | File | SHA-256 |
| --- | --- | --- |
| 022b | `predictions.npz` | `7c5a84820f6f74461be0f871fa74d0cebb4cd15b5803076ddb981089200aa4d1` |
| 022b | `training_rows.csv` | `d216d93e9f57319d8988488a8591421f62aa593486b9685d5a7811f30ab6f407` |
| 030 | `draws.csv` | `79edb856a7cf4401b1731f8d12de4d499ba3551741e996e3aa888be6a03ec380` |
| 032 | `predictions.npz` | `d150909f90b9828d82ddcb6cbdb513b94e606d2d05d9224363c55b9a179c4350` |
| 032 | `training_rows.csv` | `66a55a47b6e4b236ea96d26e62585c8841dd10aa52839d07e8ebcc526a39e43a` |
| 032 | `calibration_rows.csv` | `38b75990370bb0973425ef255d6610955d001515b1e76bbefe705d290eceee60` |
| 033 | `draws.csv` | `e928088d2f89552fa6b84ae0215b2cedd1f8b42794a495c79b436116069f24e2` |
| 035 | `predictions.npz` | `f51b440fd20360326f9f5952f919c4a27631bcdaa75e1aea31f6549e56323394` |
| SPH manifest | `data/processed/sph_clean_v1/rows.csv` | `2cb064ced7fec1c8981ddf4e886876a64306ab77f597bd1383874282279fc432` |

Refit from scratch (reproduces the saved heads exactly, CPU, about 4 minutes):

```bash
PYTHONPATH=. OMP_NUM_THREADS=3 uv run --no-sync python -u -m scripts.experiments.run_pipeline_v2_037
```

## 8. What a final test must still decide

- Which held-out set. The Challenge test groups are unread by readouts, but the split is per record and the
  released encoders saw Chapman/Ningbo waveforms in pretraining. The PTB-XL test set was already evaluated in
  001-008. The only fully untouched site would be new data, such as the student pilot. Also: which of its normals
  play the local normals, and which budget is primary. The local-normal draw must come from the test site's
  normals only, and nothing may be refitted on it.
- Whether to report v1 beside v2. v1 is the same pipeline with the unweighted 022b readout (`v1_*` in the
  heads file; 022b `pooled`, 326 iterations).
- The label question from 035 and the cardiologist meeting: which "normal" the student screen should use
  (sinus bradycardia, ectopic beats). Pipeline v2 keeps the Challenge primary label.
