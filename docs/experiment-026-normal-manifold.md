# Experiment 026: one-class screening by distance from the normal-ECG manifold

**Frozen 28 September 2026, before any development or SPH score is computed.** A university screening
cohort is mostly healthy, and each rare cardiopathy has few or no labels. A supervised probe can only learn
the conditions it is shown. This experiment asks whether a score fitted only on normal ECGs, the distance
from the normal-ECG manifold in a frozen feature space, ranks abnormal ECGs well, including conditions it
was never labeled for. It is CPU only: no encoder is trained and no feature is extracted. Calibration and
test patients stay closed.

## Questions

1. **Primary.** How well does a Mahalanobis distance from the normal ECGs, in the ECG-JEPA feature space,
   rank abnormal versus normal development ECGs, and how does it compare with a supervised probe?
2. **Unseen conditions.** For a superclass that a supervised probe never saw in training, does the one-class
   score catch it as well as, or better than, that probe?
3. **External.** How does the same score, fitted on PTB-XL only, rank abnormal ECGs at SPH?

## Encoders

Cached frozen features, loaded exactly as in Experiment 025 (`scripts/experiments/run_label_efficiency025.py`:
`identity`, `select_rows`, `load_features`). Rows are selected by PTB-XL ECG ID, and every cache must match its
extraction receipt. The JEPA and xECG caches also hold calibration and test ECGs; they are memory-mapped and
only the selected training and development rows are read.

| Name | Cache | Width |
| --- | --- | ---: |
| `jepa` | `data/processed/pretrained/ecg-jepa-full-public` | 768 |
| `xecg` | `outputs/experiment016_xecg_probe_finetune/features` | 1024 |
| `cpc` | `outputs/experiment020_full_development_v2/features.npz` | 512 |

Released ECG-CPC and age/sex, which Experiment 025 also reported, are not used here.

## Rows

Counted on metadata only, before this freeze:

- **Training pool:** Experiment 025's pool, 15,359 training ECGs from 13,351 patients, 9,487 positive on the
  standard label.
- **Fit set:** the pool's NORM-only ECGs (standard label 0): 5,872 ECGs from 5,537 patients. No abnormal
  ECG is used to fit any one-class score.
- **Evaluation:** Experiment 025's 1,306 original development ECGs (1,173 patients, 843 positive, 463
  NORM-only).
- **Label:** the PTB-XL standard superclass label (`ptb_table()["standard"]`). A superclass or subclass is
  present when any listed diagnostic SCP code maps to it (`scp_statements.csv`, `diagnostic == 1`), with no
  likelihood threshold, as in Experiment 024.

## One-class scores

Each score is fitted per encoder on the fit set only, in float64, with no hyperparameter search. A higher
score means farther from normal.

- `mahalanobis`: a `StandardScaler` fitted on the fit set, then `PCA(n_components=64, svd_solver="full")`
  fitted on the standardized fit set, then scikit-learn `LedoitWolf` fitted on the 64 components. The score is
  the squared Mahalanobis distance under that covariance (`LedoitWolf.mahalanobis`), which ranks exactly as
  the distance does.
- `knn`: the same fit-set `StandardScaler`, then each row scaled to unit length. The score is the mean cosine
  distance (1 - cosine similarity) to the 25 nearest fit-set ECGs. No evaluation ECG belongs to the fit set,
  so no self-match is possible.

## Supervised comparators

All comparators are the fixed Experiment 020 readout (`fit_logistic`: train-only `StandardScaler`, L2
logistic regression, C = 0.01, L-BFGS, `max_iter=5000`, `tol=1e-8`, float64, seed 42), fitted on the binary
standard label (abnormal versus NORM-only).

- `probe_all`: the Experiment 025 N = all probe of the same encoder, refitted on the whole pool.
- `probe_100`: the Experiment 025 N = 100 probes, one per draw, on the same 20 draws (seeds 25025-25044,
  `draw_subset`). Its metric is the mean of the 20 per-draw metrics.
- `probe_without_S`, for each superclass S in MI, STTC, CD and HYP: the same head fitted on the pool after
  removing every ECG that has S, that is NORM-only ECGs versus ECGs with abnormal superclasses other than S.
  Training rows (metadata count): without MI 10,982 (5,110 positive), without STTC 11,198 (5,326), without CD
  11,784 (5,912), without HYP 13,242 (7,370). It never sees an ECG with S.

## Tasks

Each task scores development ECGs only.

- **Binary (primary task):** all 1,306 evaluation ECGs, standard label.
- **Superclass S versus NORM:** development ECGs with S (positive) and NORM-only ECGs (negative). Positives:
  MI 390, STTC 360, CD 348, HYP 180; 463 negatives each.
- **Held-out condition:** the same S-versus-NORM rows, scored also by `probe_without_S`. Secondary: an
  S-only version, keeping only positives with no other abnormal superclass, which removes the positives the
  held-out probe could catch through a co-occurring condition it did see. S-only positives: MI 164,
  STTC 179, CD 124, HYP 41.
- **Subclass versus NORM:** each diagnostic subclass other than NORM with at least 20 development positives:
  AMI 229, IMI 227, STTC 152, LAFB/LPFB 143, LVH 136, ISC_ 81, IRBBB 74, ISCA 66, _AVB 60, IVCD 47,
  NST_ 47, CLBBB 40, CRBBB 37, ISCI 32 and LAO/LAE 32 (15 subclasses). The comparators are the binary probes;
  no subclass-specific probe is fitted.

## Metrics and uncertainty

- AUROC and average precision per task, score and encoder.
- **Primary statistic:** development AUROC and AP of `mahalanobis` on `jepa`, binary task.
- **Differences:** one-class minus comparator, for AUROC and AP, from 2,000 paired whole-patient bootstrap
  draws with seed 26026. Within a task, every score and comparator is evaluated on the same resampled
  patients; draws with one class are counted and skipped. The interval is the 2.5 and 97.5 percentiles.
  In a draw, `probe_100` is the mean over the 20 draws of the resampled metric.
  - Binary task: each one-class score minus `probe_all` and minus `probe_100`.
  - Superclass tasks: each one-class score minus `probe_all`, `probe_100` and `probe_without_S`.
  - S-only tasks: each one-class score minus `probe_without_S`.
  - Subclass tasks: each one-class score minus `probe_all`. `probe_100` is reported as a point estimate only,
    to keep the run within about half an hour.

## External readout (secondary, scored once)

- Features: `outputs/experiment022_sph_external_v3/features.npz` (`ecg_ids`, `cpc`, `jepa`, `xecg`), which
  must match that run's `outputs_sha256`. They come from the same extraction code as the PTB-XL caches.
- Labels: `data/processed/sph_clean_v1/rows.csv` (hash equal to Experiment 022's), rows with `use_evaluation`
  and a defined `primary` label: 21,008 ECGs, 7,190 positive. Joined by `ecg_id`.
- The one-class scores fitted on the PTB-XL fit set score SPH unchanged. Nothing is fitted, tuned or
  selected on SPH.
- Reported: AUROC and AP per score and encoder on the primary label, next to the Experiment 022 probes from
  its `result.json` (JEPA 0.911, xECG 0.915, CPC 0.876). Also AUROC per superclass versus negatives, next to
  Experiment 022's. No interval is computed for SPH.

## Integrity

Before any one-class score is computed:

- The caches match their receipts (Experiment 025's `identity`).
- The refitted `probe_all` of each encoder reproduces Experiment 025's saved N = all development
  probabilities (`all_budget_predictions.npz`, `primary_<encoder>`) to 1e-8.
- Each refitted `probe_100` draw has the same subset hash as `draws.csv` and reproduces its development AUROC
  to 1e-10.
- The SPH features and rows match Experiment 022's recorded hashes, and the SPH counts match (21,008 ECGs,
  7,190 positive).

## Prespecified reading

Applied to the primary statistic (`mahalanobis`, `jepa`); the other scores and encoders are read the same
way as secondary results.

- **Competitive:** the one-class AUROC is at most 0.05 below `probe_all` (point estimate of one-class minus
  `probe_all` at least -0.05).
- **Useful for unseen conditions:** for a held-out superclass S, the bootstrap interval of one-class minus
  `probe_without_S` AUROC includes or exceeds 0 (its upper bound is at least 0). Reported per superclass; the
  S-only version is a sensitivity analysis and does not change the reading.
- Otherwise the approach is reported as not competitive.

The `probe_100` contrasts are descriptive.

## Caveats

- The development patients were inspected by earlier experiments; results are exploratory.
- NORM in PTB-XL is an ECG annotation, not proof of health. The fit set may contain people with heart
  disease that the ECG does not show, and PTB-XL is a clinical, older population, not a student cohort.
- A distance from normal flags anything unusual, including noise, device and demographic differences
  (Experiment 024 found device and heart rate strongly encoded). A high score is not a diagnosis.
- Each encoder is one feature set: one checkpoint, preprocessing and pooling.
- `probe_without_S` still sees conditions that co-occur with S, so the S-versus-NORM task can overstate how
  well it detects S itself; the S-only version addresses this.
- The SPH label follows the PTB-XL superclass rule only as closely as AHA codes allow, and SPH is heavily
  band-pass filtered compared with PTB-XL.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.experiments.run_normal_manifold026
```

`scripts/experiments/run_normal_manifold026.py` uses `ecg_experiment/normal_manifold.py`. It runs one process
per encoder with one BLAS thread each. It hashes every input, source file and this protocol into the result,
refuses to overwrite an existing run, and writes `outputs/experiment026_normal_manifold_v1/result.json`,
per-row development scores (`development_scores.csv`) and SPH scores (`sph_scores.csv`). The results go to
`docs/experiment-026-normal-manifold-results.md`.
