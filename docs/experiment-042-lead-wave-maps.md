# Experiment 042: per-lead maps from ECG-JEPA patches and beat-aligned waves

**Frozen 30 September 2026, before any score of this experiment is computed.** Experiment 041 marked
abnormal 0.25 s xECG sections across all leads. Its unsupervised map found premature beats (+0.669 over
chance), but it could not say which lead was abnormal, and its worst section was a weak screen (AUROC 0.777).
That triggered the prespecified ECG-JEPA fallback (041b). After reading the 041 report, the user asked for
both per-lead approaches on 30 September 2026, with a new notebook only if the result improves. This
experiment is 041b plus a second arm.

- **Arm J (ECG-JEPA patches):** the released ECG-JEPA encoder cuts 8 leads (I, II, V1-V6) at 250 Hz into
  50 patches of 200 ms each, so one token is one lead at one moment. Its pooled feature is the mean of the
  400 output tokens.
- **Arm B (beat-aligned waves):** each detected beat is cut into P, QRS, ST and T windows in each of the 12
  leads, and each piece is compared with the same piece of healthy beats. There is no neural encoder: a unit
  is a raw-signal wave.

## Questions

1. Does a per-lead map point anterior infarcts to V1-V4 more often than inferior infarcts?
2. Does it keep 041's premature-beat localization, and does it detect better or mark benign variants less?

## Rows

The Experiment 041 rows, unchanged: fit set 5,872 NORM-only training ECGs (5,537 patients); readout pool
15,359; binary evaluation 1,306 (843 positive, 463 NORM-only); premature-beat set 84 (81 patients);
benign-variant set 52; full development 1,604. Added, counted on metadata before this freeze:

- **Anterior-only infarcts:** full-development ECGs with the AMI subclass and none of IMI, LMI or PMI: 146
  (133 patients).
- **Inferior-only infarcts:** IMI and none of AMI, LMI or PMI: 149 (139 patients). No patient is in both
  groups.

Calibration and test patients stay closed.

## Arm J maps

- Input: `external_encoders.ptb_jepa_input`. Tokens: the official `representation` without its final mean,
  `[400, 768]`, token `50 * lead + patch`. Patch p covers seconds [0.2 p, 0.2 p + 0.2).
- `U_J` (primary): per-position centring (the fit-set mean of each of the 400 positions), then a streaming
  version of Experiment 026's Mahalanobis on the 2,348,800 centred fit tokens: per-feature mean and standard
  deviation, the covariance of the standardized tokens accumulated in chunks, its top 64 eigenvectors as the
  PCA, and scikit-learn `LedoitWolf` on the projected tokens. This equals `normal_manifold.fit_mahalanobis`
  without holding 2.3 million float64 rows twice.
- `U_J_kmeans` (secondary): `KMeans(n_clusters=32, n_init=4, random_state=42042)` on the whitened 64
  coordinates, as in 041.
- `G_J` (secondary): the Experiment 026 `jepa` `probe_all` head applied to each token, c = w · (x - m) / s + b.
  The mean of the 400 contributions is the ECG's logit.

### Amendment 1, 30 September 2026, before any development score

The training-only smoke test showed that one covariance pooled over the 8 leads makes lead II dominate. In
a training check (1,000 fit-set ECGs as reference; 150 normal and 150 abnormal other training ECGs, no
development row), lead II held the top token for 79 of 150 normal ECGs, and the worst-token AUROC was 0.319.
Lead II tokens vary more among normal ECGs, so they look far from a covariance shared with the other leads.
Fitting one reference per lead gave 0.735, with the top lead spread across all eight.

`U_J` and `U_J_kmeans` therefore use **one reference per lead**. For each of the 8 leads,
position centring, the streaming Mahalanobis and the k-means (32 clusters, `random_state=42042`) are fitted
on that lead's 293,600 fit tokens (5,872 ECGs by 50 patches) exactly as described above, and the lead's tokens
are scored against its own reference. Everything else is unchanged.

## Arm B maps

- Signal: `read_ptb_float64` (500 Hz, mV). R peaks: `fragment_localization.r_peaks` on the raw signal.
  The signal is high-passed (second-order Butterworth, 0.5 Hz, `filtfilt`). A beat is kept if
  [R - 0.30 s, R + 0.45 s] lies inside the record; per beat and lead, the mean of [R - 0.10, R - 0.06) s is
  subtracted.
- Waves, fixed offsets from R: P [-0.25, -0.06) s, QRS [-0.06, +0.08) s, ST [+0.08, +0.20) s and T
  [+0.20, +0.45) s, sampled every second sample (250 Hz): 48, 35, 30 and 63 values.
- `U_B` (primary): one `LedoitWolf` reference per lead and wave (48 references), fitted on every kept beat of
  the fit set. A unit is one (beat, lead, wave), scored by squared Mahalanobis distance.
- `U_B_median` (secondary): the median over an ECG's kept beats of each lead and wave piece, scored against
  48 references fitted on the fit set's median pieces. One unit per lead and wave; no time.
- `G_B` (secondary): `full_development.fit_logistic` on the 2,112 concatenated median pieces of the readout
  pool. A unit is a lead and wave block, scored by its share of the logit, the sum over its coordinates of
  w · (x - m) / s, plus the intercept divided by 48. The 48 shares sum to the logit.
- ECGs with fewer than two kept beats are excluded from every arm-B metric and from the paired contrasts
  that involve arm B, and are counted. Pool ECGs without two kept beats are left out of the `G_B` fit.

## Common definitions

- ECG score: the maximum over the ECG's units. Red units: above the 95th percentile of the ECG scores of
  the 463 NORM-only binary-evaluation ECGs, as in 041.
- Premature beats: 041's rule and windows. `hit` is 1 if the top unit's time span overlaps a premature-beat
  window; `chance` is the share of the ECG's units whose spans overlap one. Units without a time span
  (`U_B_median`, `G_B`) are not scored here.
- Lead localization: the top lead is the lead of the highest unit. An anterior hit is a top lead in
  {V1, V2, V3, V4}; an inferior hit is {II, III, aVF} (arm J has only II).
- 95% intervals: 2,000 whole-patient bootstrap draws, seed 42042. Two-group contrasts resample patients within
  each group.

## Statistics

- **Primary, per arm:** lead contrast of the primary map, anterior-hit rate in anterior-only infarcts minus
  in inferior-only infarcts.
- Secondary, for every map: inferior-hit rate in inferior-only minus anterior-only infarcts; binary AUROC and
  AP; premature-beat hit, chance and hit - chance; share with any red and mean red units in the NORM-only,
  positive, premature-beat and benign sets; sensitivity at the 5% budget.
- Paired against 041's `U` (saved `section_scores.npz`, hash checked against the 041 receipt): AUROC
  difference (`intervals.paired_auroc_difference`); per-ECG difference of hit - chance on the premature-beat
  ECGs included by both; per-ECG difference of benign any-red.

## Prespecified reading: does an arm improve on Experiment 041?

An arm improves if its primary map:

1. keeps premature-beat localization: the lower bound of its hit - chance minus 041 `U`'s is above -0.10; and
2. gains at least one of: (a) the lead contrast's lower bound is above 0; (b) the AUROC minus 041 `U`'s lower
   bound is above 0; (c) the benign any-red difference's upper bound is below 0.

A new demonstration notebook is written only if at least one arm improves, and shows only the arms that do.
The secondary maps change no reading.

## Integrity

Before any score:

- The JEPA cache matches its receipt (`external_encoders.jepa_cache`). For every extracted ECG present in it,
  the token mean equals the cached feature to 1e-4.
- Token order: on four fit-set ECGs, adding 1 mV to lead k in patch p changes token 50 k + p the most, for
  (k, p) in {(0, 0), (1, 25), (7, 49)}.
- The whole-ECG `jepa` Mahalanobis and `probe_all` refitted from the cache reproduce Experiment 026's
  `development_scores.csv` to 1e-8 (OpenBLAS Haswell kernels, as in 041).
- The streaming Mahalanobis fitted on the 026 pooled JEPA fit features gives scores within a relative 1e-6 of
  `normal_manifold.fit_mahalanobis` on the evaluation rows.
- The mean of the `G_J` contributions equals the cached-feature logit to 1e-3; the 48 `G_B` shares sum to the
  `G_B` logit to 1e-8.
- The 041 section scores and result match the 041 receipt.

## Figures

The 041 examples (ECGs 47, 219, 8, 184, 287, 30, 69), with the 12 leads and red marks per lead for `U_J`,
`U_B` and `G_J`. Arm J marks only its 8 leads.

## Caveats

- Infarct location is a PTB-XL statement for the whole ECG, not a marked region, and old infarcts can be
  subtle. The lead test asks only whether anterior infarcts point to anterior leads more than inferior ones.
- Arm J cannot mark III, aVR, aVL or aVF, which are linear combinations of I and II.
- Arm B's wave windows are fixed offsets from R. They do not adapt to heart rate (the T wave moves with the
  QT interval), and a wide QRS spills into the ST window.
- The R-peak rule is unvalidated; the premature-beat check is 041's automatic rule.
- Development data were read by earlier experiments; the results are exploratory.

## Execution

```bash
OPENBLAS_CORETYPE=Haswell CUDA_HOME=$PWD/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime \
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.experiments.run_lead_wave_maps042
```

`scripts/experiments/run_lead_wave_maps042.py` uses `ecg_experiment/lead_wave_maps.py` and the 041 module.
It takes the GPU lock for arm J and profiles 128 records first; it stops if the projected total exceeds
3,600 seconds. It hashes every input, source and this protocol, refuses to overwrite a run, and writes
`outputs/experiment042_lead_wave_maps_v1/` (`result.json`, `unit_scores.npz`, `figures/`, `run.log`). The
results go to `docs/experiment-042-lead-wave-maps-results.md`.
