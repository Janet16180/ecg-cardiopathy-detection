# Pipeline v3: the candidate student screening pipeline

Adopted by [Experiment 044](experiment-044-pipeline-v3-results.md) on 1 October 2026 (decision `adopt_v3`).
It is [pipeline v2](pipeline-v2.md) with one change: the binary readout R is refitted on the concatenated xECG
and ECG-JEPA features. The finding heads, their standardization, the threshold rule and the local-normal
requirement are v2's, unchanged. This file lists everything a later one-time final test (`final_frozen_test`)
would freeze. It describes a candidate, not a validated screen: every number behind it comes from development
data (PTB-XL development, the Challenge calibration groups and SPH, all read by earlier experiments), and 043
chose the concatenated readout partly on SPH. No final test has been run.

On SPH, v3 caught 0.008 more athlete-criteria abnormal ECGs than v2 at a 5% budget with 200 local normals
(+0.0080 [+0.0040, +0.0120]). It also referred slightly more normal ECGs (5.74% against 5.45%) and more
"other" ECGs such as sinus bradycardia (25% against 22%). v2 stays documented in `pipeline-v2.md` so that a
final test can report both.

## 1. Features

- Two frozen encoders, both applied to a 10 s, 12-lead, 500 Hz window in mV, canonical lead order (for longer
  Challenge records, the centred 10 s window, start = (samples − 5000) // 2):
  - xECG, exactly as v2 (`pipeline-v2.md` section 1): the released backbone in `third_party/checkpoints/xecg/`
    (`model.safetensors` `812dec69…722c`), `external_encoders.xecg_input` and `xecg_features`, 1,024 features.
  - ECG-JEPA: the released multiblock encoder `third_party/checkpoints/ecg-jepa/multiblock_epoch100.pth`
    (SHA-256 `61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e`), loaded with
    `external_encoders.load_jepa` from `third_party/ECG_JEPA`. Input: leads I, II, V1-V6, Fourier-resampled to
    2,500 samples (`external_encoders.jepa_input`; `ptb_jepa_input` for raw PTB-XL records). Features: the
    pooled 768-dimensional `encoder.representation` (`external_encoders.jepa_features`), float32 cached,
    float64 at readout time.
- The readout input is the concatenation `[xECG (1,024), JEPA (768)]`, 1,792 features, in that order.
- The feature files used to fit the heads: the PTB-XL caches (JEPA `features.npy` `214ca700…309a`,
  `metadata.json` `2e3f9a00…b7de`, data/processed/pretrained/ecg-jepa-full-public; xECG `features.npy`
  `f33d89d0…a7f4`), `outputs/experiment022_sph_external_v3/features.npz` (`f6f82fe6…1857`, SPH, both
  encoders) and `outputs/features_challenge_v1/` (`metadata.json` `9bcd2f7c…21ee`, both encoders).

## 2. Binary readout R3 (the 035 `dropped_upweighted` fit on xECG + JEPA)

- Model: a `StandardScaler` and an L2 logistic regression, `C = 0.01`, `lbfgs`, `tol = 1e-8`,
  `max_iter = 5000`, float64, with sample weights in both the scaler and the fit
  (`multisource_readout.fit_readout(x, y, weights)`), on the 1,792 concatenated features. Converged in 422
  iterations. It is 043 Stage 1's `logistic_concat`, reproduced to 1e-14 in the logits.
- Training rows (39,577, 27,360 positive), labels and weights: exactly v2's (`pipeline-v2.md` section 2): the
  17,083 PTB-XL training ECGs with a standard label and the 22,494 Challenge training-group rows that pass the
  training quality policy; the 1,724 PTB-XL rows without a project label weigh 2.3167476 each, every other row
  0.9400292.
- Output: the positive-class probability, used only to rank.

## 3. Finding heads and the combined finding score F (unchanged from v2)

- The xECG `ventricular_ectopy` (PVC) and `preexcitation` (WPW) heads of 032, refitted here and identical to
  v2's (`pipeline-v2.md` section 3): PVC 233 and WPW 81 iterations.
- Standardization on the 1,707 Challenge calibration clear normals: PVC mean −5.539442, SD 0.825770; WPW mean
  −9.857676, SD 1.768115.
- F = max(z_PVC, z_WPW).
- Not adopted: the v3b heads (the same fits on the 1,792 concatenated features). They are saved for reference
  (`v3b_pvc_*`, `v3b_wpw_*`), carry no decision and are not part of pipeline v3.

## 4. The referral rule and threshold (unchanged from v2)

`finding_screen.split_thresholds(normals, budget_per_mille, (50,))` with columns R3 then F, as in
`pipeline-v2.md` section 4: k = floor(b m); F's threshold is the (r + 1)-th highest normal F with
r = floor(50 k / 1000); R3's threshold is the highest-ranked normal R3 such that at most k − 1 of the m normals
lie above either threshold; an ECG is referred when R3 or F is strictly above its threshold; k ≥ 1 is required.
Budget choices 1%, 2%, 5% (reference) or 10%.

## 5. The local-normal requirement (unchanged from v2)

- The threshold must come from the site's own normal ECGs, read as normal by the cardiologist with the athlete
  criteria in mind.
- Size: with 200 normals the achieved rate at a 5% target ranges 3.5-8.5% across draws; with 1,000 normals
  4.4-6.6%, within ±1 point in 77% of draws (88% with v2). On SPH at 5%, v3 refers 0.13-0.28 points more of the
  evaluation normals than v2, at every size from 50 to 2,000.

## 6. Expected operating numbers (SPH, development data)

Evaluation half of the SPH simulated site, 200 local normals, per 1,000 ECGs at an assumed 5% prevalence of
athlete-criteria abnormal ECGs:

| Budget | Referrals per 1,000 | Abnormal (binary label) caught | Athlete-criteria findings caught | Normals referred |
| --- | ---: | ---: | ---: | ---: |
| 1% | 47 | 0.62 | 0.65 | 1.56% |
| 2% | 60 | 0.69 | 0.72 | 2.55% |
| 5% | 94 | 0.78 | 0.80 | 5.74% |
| 10% | 143 | 0.84 | 0.86 | 10.49% |

PVC ECGs are referred 0.96-0.99 of the time and WPW 0.77 at 5% (15 ECGs). "Other" ECGs (neither normal nor
abnormal by the label) are referred 0.25 of the time at 5%. These are the numbers a final test would check,
not guarantees.

## 7. Code paths and hashes to reproduce

Frozen parameters (load these to score new ECGs; do not refit):

- `outputs/experiment044_pipeline_v3_v1/pipeline_v3_heads.npz`, SHA-256
  `54c5049aa0830b05a6c971c2a5234c18bddd136ad9c7592da0b4549b0786f8ce`. Keys `v3_*` (R3, 1,792 features),
  `pvc_*`, `wpw_*` (`mean`, `scale`, `coef`, `intercept`), `pvc_logit_mean`, `pvc_logit_sd`,
  `wpw_logit_mean`, `wpw_logit_sd`; also `v3b_pvc_*` and `v3b_wpw_*` with their constants, not part of v3.
  Score with `ecg_experiment.pipeline_v2.score_parameters` (reproduces the fitted heads to 7e-16). The
  finding heads score the same as those in v2's `pipeline_v2_heads.npz` to within 6e-17.
- Experiment 044 result: `outputs/experiment044_pipeline_v3_v1/result.json` `f3ec9b58…c16d`;
  `predictions.npz` `93973a48eac00765abe97cd3885a14a42f7921f1a9d13781f7dd6b67d6d5ac1b`; `draws.csv`
  `32081958f476f0056c58bee5c3843d302244a0e49ba98bc1585e2e65040fc5f0`.

Code (hashes as recorded by the 044 run, commit `5f291cf`):

| Path | SHA-256 |
| --- | --- |
| `ecg_experiment/pipeline_v3.py` | `dce65303c6c2e8d131664782ce94f66e839b4d01a83f4aaf71425336a182dcd8` |
| `ecg_experiment/pipeline_v2.py` | `1748c062efa9cb71630a50122d268c74e1ddaa87ec5d6e3f6bf52439dc3c8b53` |
| `ecg_experiment/finding_screen.py` | `2cb793ac7094e5fba4d3a239f154ff8e8bdb8ee780f63e6fc9dace3658687648` |
| `ecg_experiment/intervals.py` | `b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af` |
| `ecg_experiment/referral_budget.py` | `c2af973802e269c29c143b30e54147c05bba5513548149b8174269aec13ff56c` |
| `ecg_experiment/hard_subset.py` | `3f8f04be40202902747636da6e909f3983ecd025de09285d381278f37cafacb2` |
| `ecg_experiment/multisource_readout.py` | `6cf20dff174ace45e870fb4f65776c654f954fb96a82a7c739c70946b7887bc5` |
| `ecg_experiment/full_development.py` | `ae7f37ae76063220785dc4031ee5ba206b0ca75134a9f37c858c66ef2dff83c4` |
| `ecg_experiment/hybrid_score.py` | `e0206b3f545afb83f378972780694993ebc1dbc61e8a41003f8b1de3336f1e7d` |
| `ecg_experiment/rhythm_findings.py` | `9fd88e4a25f0951253e16ed3a84d8976a96b999b361f0e69bacc91adeef48bde` |
| `ecg_experiment/ann_heads.py` | `760f4347c6de1e7bb8df747f8e8c5b9200b18ea5de127ff297a30cd522ffd179` |
| `scripts/experiments/run_pipeline_v3_044.py` | `78a903b51d563627eacd73d9cf9f407b870bc08daaec40dc86cafcb3e626939e` |
| `scripts/experiments/run_ann_heads043.py` | `1b14f4e1e52e638770d913ca7290dc3e7ae158087c92864762d56f214fac1921` |
| `scripts/experiments/run_pipeline_v2_037.py` | `d05868abbd982b283bb5e06329a8ed62b1a19d25db5a18b8ddf7681fbc3a3bde` |
| `scripts/experiments/run_finding_screen033.py` | `0a15d541ab8dae3c554880f708bcc101f7d62bf457775941e3b3926d487ec879` |
| `scripts/experiments/run_rhythm_findings032.py` | `de5fa98aabb8787ba09057c8be31da6cb32c0d4c91d011b6870fa0202270fc5e` |
| `scripts/experiments/run_referral_budget030.py` | `7f23495f2c0916f7c91d33949f49fdb36a19af17fda28808086f9235953a1f17` |
| `scripts/experiments/run_lead_wave_maps042.py` | `faf48641699e1cbb089be6c01bd72067550f86358e799501354bd674c0ad10d6` |
| `pyproject.toml` | `9df6d56e8f27348d067eb0cf3b6506f5efbbc03154b96b6b3987f4bb4854279e` |
| `uv.lock` | `20c6e373cba70e7fe859ba59697f6145c34b42418eddc955f82d29a165d5db7c` |
| `docs/experiment-044-pipeline-v3.md` | `2426971d3e3e4be54947914fb2ae377a23aa33d4666836a1104829064dd62026` |

Inputs checked by the run against their receipts:

| Output | File | SHA-256 |
| --- | --- | --- |
| 037 | `predictions.npz` | `3f8f35eb18523b003ea83c1c646cc5aaa81e6a7f8bdee2ab204787baad7686a2` |
| 037 | `draws.csv` | `eb26c95b5d2313af51b3648a4c9a81ff4415047b25cbb73d71a8e58342d4d8fd` |
| 037 | `pipeline_v2_heads.npz` | `ca2deef4c0553cdeaa80b6ed75f9349688b9415ca7e4b75cbe12954569bcc5af` |
| 043 Stage 1 | `predictions.npz` | `53c1fcd121646aa9db2bba5be1701d3715944c565cb609ece8cd05f088014463` |
| 035 | `predictions.npz` | `f51b440fd20360326f9f5952f919c4a27631bcdaa75e1aea31f6549e56323394` |
| SPH manifest | `data/processed/sph_clean_v1/rows.csv` | `2cb064ced7fec1c8981ddf4e886876a64306ab77f597bd1383874282279fc432` |

033's `load_rows` and `load_scores` also check the 030 and 032 outputs against their receipts, as in v2.

Refit from scratch (reproduces the saved heads exactly, CPU, about 3 minutes):

```bash
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
    .venv/bin/python -u -m scripts.experiments.run_pipeline_v3_044
```

## 8. What a final test must still decide

Everything in `pipeline-v2.md` section 8 (which held-out set, which of its normals play the local normals, which
budget is primary, the label question), and in addition:

- Whether to report v2 beside v3. v2 is the same pipeline with the xECG-only readout; on SPH v3 catches
  0.003-0.014 more composite and binary-label ECGs with `combined_50` at 1,000 or fewer normals.
- Both encoders must run at the test site; the JEPA input path (lead selection, Fourier resampling) must be the
  one above.
- The "other" ECGs: v3 refers more of them (sinus variants, atrial premature beats), which at a student site
  may mostly be normal variants. The cardiologist should see this before a final test fixes v3.
