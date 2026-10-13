# Experiment 041: marking the abnormal sections of an ECG

**Frozen 30 September 2026, before any section score is computed.** The user's thesis tutor proposed that
the model should show where an ECG is abnormal, not only whether it is. The user agreed this scope on
30 September 2026: time sections across all leads, an unsupervised map and a label-guided map, PTB-XL
development data only, a check on premature ventricular beats, and ECG-JEPA only as a prespecified fallback.

The released xECG backbone (the pipeline v2 encoder) cuts a 10 s, 100 Hz, 12-lead input into 40 patches of
25 samples (250 ms, all leads), runs a bidirectional xLSTM over them and averages the 40 output tokens into
the 1,024-dimensional feature every earlier experiment used. This experiment scores each of the 40 tokens.
Section t covers seconds [0.25 t, 0.25 t + 0.25) of the 10 s window. A section is a time band across all
12 leads, not a lead.

## Questions

1. **Unsupervised map (Part 1).** Without any abnormal label, does the distance of each section from normal
   sections point at the abnormal part of the ECG? Tested on premature beats, the only finding whose time
   can be located without an expert.
2. **Label-guided map (Part 2).** Does splitting the supervised readout into per-section contributions
   flag abnormal ECGs better, and mark benign rhythm variants less, than the unsupervised map?

## Rows

Counted on metadata only, before this freeze. `full_development.cohorts` and the Experiment 025/026 selection
(standard label defined, in the xECG and ECG-JEPA caches).

- **Fit set:** the 5,872 NORM-only training ECGs of Experiment 026 (5,537 patients). Only these fit the
  unsupervised reference. None has a PVC, PAC, sinus bradycardia, tachycardia or arrhythmia statement.
- **Readout pool:** Experiment 026's 15,359 training ECGs (9,487 positive), only to refit its `xecg`
  `probe_all` head from the cached pooled features. No token of these ECGs is extracted except the fit set.
- **Binary evaluation:** Experiment 026's 1,306 original development ECGs (1,173 patients, 843 positive,
  463 NORM-only).
- **Premature-beat set:** the 84 full-development ECGs (81 patients) with the PTB-XL `PVC` statement, any
  likelihood. 59 are in the binary evaluation, all positive on another superclass.
- **Benign-variant set:** the 52 full-development ECGs whose only diagnostic statement is NORM, whose
  statements are all in {NORM, SR, SBRAD, SARRH} and which list SBRAD or SARRH. The 2017 international
  athlete criteria count sinus bradycardia and sinus arrhythmia as normal findings. None is in the binary
  evaluation (the project target excludes them), and none has a standard label of 1.
- Tokens are extracted for the fit set and the 1,604 full-development ECGs. Calibration and test patients
  stay closed.

## Section maps

All maps give 40 scores per ECG; a higher score means more abnormal. Fitted in float64 on the fit set only,
with no hyperparameter search.

- `U` (primary unsupervised map, per-section Mahalanobis): subtract the fit-set mean token of each of the 40
  positions, then `normal_manifold.fit_mahalanobis` (StandardScaler, PCA with 64 components, Ledoit-Wolf) on
  the 234,880 centred fit tokens; the score is `normal_manifold.mahalanobis_scores`. Position centring stops
  the first and last sections, which have one-sided context, from looking abnormal in every ECG.
- `U_kmeans` (secondary, the tutor's clustering): the same centring, scaler and PCA; coordinates divided by
  the square root of each component's explained variance; scikit-learn `KMeans(n_clusters=32, n_init=4,
  random_state=41041)` on the fit tokens. The score is the squared distance to the nearest centroid.
- `U_uncentred` (sensitivity): `U` without the position centring.
- `G` (primary label-guided map, readout contribution): the Experiment 026 `xecg` `probe_all` head
  (`full_development.fit_logistic` on the readout pool's cached features). With scaler mean m, scale s,
  weights w and intercept b, section t scores c_t = w · (x_t - m) / s + b. Because the head is linear and the
  feature is the token mean, the mean of the 40 c_t is the ECG's logit.
- `G_gated` (secondary): `U` where c_t > 0, and 0 elsewhere: unusual and pushing toward abnormal.

ECG score of a map: the maximum over its 40 sections.

**Red sections.** For each map, the threshold is the 95th percentile of the ECG scores of the 463 NORM-only
binary-evaluation ECGs, so 5% of them have at least one red section. This mirrors Experiment 030's referral
budget fitted on local normals. A section is red when its score exceeds the threshold.

## Premature beats (localization ground truth)

No dataset here marks where a finding is. For the premature-beat set, beats are found by a fixed rule on the
raw 500 Hz signal, written before any score:

- R peaks: band-pass 8-20 Hz (second-order Butterworth, `filtfilt`), absolute value summed over the 12 leads,
  `scipy.signal.find_peaks` with a 0.3 s minimum distance and a height of 0.35 times the 99th percentile,
  the rule of `eda.signals.heart_rate`.
- A beat is premature when the interval from the previous R peak is below 0.8 times the record's median
  RR interval. Its window is [R - 0.10 s, R + 0.40 s], covering the QRS and T wave.
- ECGs with fewer than four R peaks or no premature beat are excluded and counted.

For each included ECG: `hit` is 1 if the highest-scoring section overlaps a premature-beat window, and
`chance` is the fraction of the 40 sections that overlap one (the hit rate of a randomly chosen section).
The statistic is mean(hit - chance).

## Metrics and uncertainty

95% intervals from 2,000 whole-patient bootstrap draws, seed 41041 (`intervals.patient_groups` and
`patient_resample`; paired AUROC differences with `intervals.paired_auroc_difference`). Draws with one class
are skipped and counted.

- **Part 1, primary:** mean(hit - chance) of `U` on the premature-beat set.
- **Part 1, detection:** binary AUROC and AP of the `U` ECG score, next to Experiment 026's whole-ECG `xecg`
  Mahalanobis (reproduced here), and their paired AUROC difference.
- **Part 2, primary:** binary AUROC of `G` minus `U` (paired).
- **Part 2, benign marks:** share of the benign-variant set with any red section, `G` minus `U`, with a
  patient bootstrap interval.
- Secondary, for every map: binary AUROC and AP; mean(hit - chance) and hit rate; share with any red section
  and mean red sections per ECG in the NORM-only, positive, premature-beat and benign sets; share of
  positives with a red section (sensitivity at the 5% budget); mean section score per position in the
  NORM-only set.

## Prespecified reading

- **Part 1 localizes** if the lower bound of `U`'s mean(hit - chance) is above 0.
- **Part 1 keeps detection** if the `U` ECG-score AUROC is at most 0.05 below Experiment 026's whole-ECG
  `xecg` Mahalanobis AUROC (point estimate).
- **Part 2 improves detection** if the lower bound of `G` minus `U` AUROC is above 0.
- **Part 2 marks fewer benign variants** if the upper bound of `G` minus `U` benign any-red share is below 0.
- `U_kmeans`, `U_uncentred` and `G_gated` are secondary and change no reading.
- **JEPA fallback:** if Part 1 fails to localize or to keep detection, Experiment 041b repeats this protocol
  on ECG-JEPA's 400 patch tokens (8 leads by 50 patches of 200 ms), adding per-lead maps. It gets its own
  frozen protocol and runs only after this report. If both Part 1 readings hold, the fallback is not run
  and JEPA per-lead maps go to the backlog.

## Integrity

Before any section score:

- The xECG cache matches its receipt and checkpoint (`external_encoders.xecg_cache`).
- For every extracted ECG present in the cache, the mean of its 40 tokens equals the cached feature to 1e-4
  (largest absolute difference, the Experiment 022 tolerance).
- Section order: on eight fit-set ECGs, adding 1 mV to all leads in the first 25 input samples changes token
  0 more than any other token, and in the last 25 samples changes token 39 the most.
- The whole-ECG `xecg` Mahalanobis and `probe_all` refitted from the cache reproduce Experiment 026's
  `development_scores.csv` to 1e-8 (relative for the distance, absolute for the probability).
- For every binary-evaluation ECG, the mean of the `G` sections equals the cached-feature logit to 1e-3.

## Figures

Rule-based examples, not chosen by score: the lowest ECG ID in the full development set of each group:
NORM-only, PVC, IMI, AMI, CLBBB, LVH and the benign-variant set. Each figure shows the 12 leads with the red
sections of `U` and of `G`.

## Caveats

- The red band is a time window across all leads. It cannot say which lead is abnormal.
- The xLSTM is bidirectional, so every token also carries information from the rest of the ECG. A finding
  may spread to nearby sections or to the whole record.
- Many findings (bundle branch block, hypertrophy, old infarction) are in every beat. For them, the map can
  only say "every beat", and nothing checks it locally.
- The premature-beat rule is automatic and unvalidated; it also catches premature atrial beats and can miss
  PVCs during an irregular rhythm. The hit counts the top section only.
- PTB-XL NORM is an annotation of an older clinical population, not a student cohort. The development
  patients were read by earlier experiments, so the results are exploratory.
- The benign-variant set is small (52) and defined from PTB-XL statements, not by a cardiologist.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.experiments.run_fragment_localization041
```

`scripts/experiments/run_fragment_localization041.py` uses `ecg_experiment/fragment_localization.py`. It
takes the GPU lock and profiles 128 records before the full extraction; the run stops if the projected total
exceeds 1,800 seconds. It hashes every input, source and this protocol, refuses to overwrite an existing run,
and writes `outputs/experiment041_fragment_localization_v1/` (`result.json`, `section_scores.npz`, the
fitted `reference.npz`, `figures/` and `run.log`). The results go to
`docs/experiment-041-fragment-localization-results.md`.
