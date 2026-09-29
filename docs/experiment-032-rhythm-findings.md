# Experiment 032: detecting the rhythm findings the athlete criteria call abnormal

**Frozen 29 September 2026, before any score of this experiment is computed.** This runs the backlog item
`rhythm_findings_detector`. The user asked to keep experiments running while they prepare a machine
migration. Before this freeze only aggregate results of 022, 022b and 030 and metadata were read: code lists,
labels, record and patient counts, feature record lists, and the training quality-policy reasons of the
Challenge training rows. The counts below come from one pass of the runner with `--counts`, which stops
before any readout is fitted; it applied the quality check (a waveform check that uses no score and no model)
and confirmed that its reasons equal 022b's `training_rows.csv` on every row 022b used.

## Why

The target is screening university students. The 2017 international criteria for athletes' ECGs
([meeting guide, section 2.3](cardiologist-meeting-prep.md)) call these findings abnormal:

- 2 or more premature ventricular contractions (PVCs) per 10 s tracing; couplets, triplets, non-sustained VT;
- ventricular pre-excitation (WPW pattern);
- atrial tachyarrhythmias: supraventricular tachycardia (SVT), atrial fibrillation, atrial flutter;
- Mobitz type II or third-degree AV block;
- prolonged QTc (≥ 470 ms in men, ≥ 480 ms in women).

Our binary label ([challenge-label-mapping.md](challenge-label-mapping.md), section 2.4 of the meeting guide)
never makes an ECG positive for a PVC, atrial fibrillation or flutter, or SVT: they are rhythm statements,
ignored in PTB-XL and SPH and "other" in the Challenge mapping. Pre-excitation, AV block and long QT are
positive in the binary label (CD, CD and STTC). The questions:

1. Can the frozen encoders detect each finding with a simple linear readout?
2. Does that readout transfer to an unseen hospital (SPH)?
3. How many of these ECGs does the current binary screen already refer, by accident or because they carry
   another finding?

## Finding groups

One binary target per group. A record is **positive** if it lists any positive code of the group,
**undefined** for that group (left out of that group's training and evaluation) if it lists an ambiguous
code and no positive one, and **negative** otherwise. PTB-XL codes count at any likelihood, as in the binary
label; SPH codes are the base AHA codes without modifiers. The lists are in
`ecg_experiment/rhythm_findings.py` (`GROUPS`) and tested in `tests/test_rhythm_findings.py`; every SNOMED
code is in the official Challenge tables.

| Group | Criterion | PTB-XL positive | SNOMED positive | SPH positive | Ambiguous (undefined) |
| --- | --- | --- | --- | --- | --- |
| `ventricular_ectopy` | ≥ 2 PVCs, couplets, NSVT | `PVC` | 427172004 PVC, 17338001 ventricular premature beats, 164884008 ventricular ectopics, 11157007 ventricular bigeminy, 251180001 ventricular trigeminy, 425856008 paroxysmal VT, 164895002 VT | 60 PVC(s) | PTB-XL `BIGU`, `TRIGU`, `PRC(S)` (origin unknown); SNOMED 13640000 fusion beats |
| `preexcitation` | ventricular pre-excitation | `WPW` | 74390002 WPW pattern, 195060002 ventricular pre-excitation | 108 ventricular pre-excitation | SNOMED 49578007 shortened PR; SPH 80 short PR |
| `af_flutter` | atrial fibrillation, flutter | `AFIB`, `AFLT` | 164889003 AF, 164890007 atrial flutter, 195080001 AF and flutter, 426749004 chronic AF | 50 AF, 51 atrial flutter | none |
| `svt` | SVT | `SVTAC`, `PSVT` | 426761007 SVT, 713422000 atrial tachycardia, 233897008 AVRT, 251166008 AVNRT, 426648003 junctional tachycardia | 54 junctional tachycardia | PTB-XL `SVARR` (supraventricular arrhythmia) |
| `high_grade_av_block` | Mobitz II, third degree | `3AVB` | 426183003 Mobitz II, 27885002 complete heart block | 84 Mobitz II, 87 advanced (high-grade), 88 complete | PTB-XL `2AVB`; SNOMED 195042002 2nd degree, 233917008 AV block, 50799005 AV dissociation; SPH 85 2:1, 86 varying conduction |
| `long_qt` | prolonged QTc (optional) | `LNGQT` | 111975006 prolonged QT | 148 prolonged QT | none |

- **CPSC 2018 annotates only nine classes** (normal, AF, first-degree AV block, LBBB, RBBB, PAC, PVC, ST
  depression and elevation). Its records are defined only for `ventricular_ectopy` and `af_flutter`; for the
  other groups they are undefined, because an unlisted finding there is unannotated, not absent.
- **The labels do not match the criteria exactly.** A PVC code means at least one PVC, not two per 10 s. The
  long-QT statements use the reader's or cart's general-medicine limit, lower than the athlete limits. PTB-XL's
  `2AVB` and the Challenge "2nd degree AV block" do not say Mobitz I (normal in athletes) or Mobitz II, so they
  are ambiguous. SPH has no SVT code; its only SVT-group code is junctional tachycardia.
- **Ningbo codes no atrial fibrillation and codes atrial flutter on 21.8% of records**
  ([clean-ningbo-v1.md](clean-ningbo-v1.md)), which suggests fibrillation coded as flutter. Merging the two
  in `af_flutter` makes that confusion harmless for the label; the secondary readout below measures whether
  Ningbo's rows help or hurt.
- **SPH secondary subset, frequent PVCs.** SPH qualifies some PVC statements: 60+310 frequent, 60+340
  couplets, 60+341 bigeminal, 60+342 trigeminal. These 372 ECGs are the closest to the "≥ 2 PVCs" criterion.
  The `sph_frequent_pvc` set holds them as positives against the `ventricular_ectopy` negatives (other PVC
  ECGs, 60 alone or 60+308 occasional, are left out).

## Rows

- **PTB-XL training**: the 17,417 clean-union training rows of `full_development.cohorts` (Experiment 020's
  selection), every one, whatever its binary label. 022 used the 17,083 of them with a binary label; an ECG
  with atrial fibrillation and no diagnostic superclass has none, so the full set is needed here.
- **PTB-XL development**: the 1,604 full-development rows (022's 1,572 labeled ones plus 32 without a binary
  label). Patients are disjoint from training.
- **Challenge**: the frozen [Challenge split](challenge-splits-v1.md), train and calibration groups, every
  record whose `duplicate_status` is `unique` or `kept` (not only `evaluable` ones, for the same reason), with
  a feature row in `outputs/features_challenge_v1/`. Codes come from the clean Ningbo manifest and the Challenge
  EDA header cache (hash checked against the split's inputs).
- **Training quality policy, as 022b.** Ningbo training rows must be `use_training`; Chapman, Georgia, CPSC
  2018 and CPSC-Extra training rows must pass `ecg_quality` on the saved feature window (026b's
  `quality_reasons`). Exclusions: Ningbo 1,031, Chapman 10, Georgia 12, CPSC 2018 31, CPSC-Extra 18. The
  calibration groups are used at all quality, as in 022b.
- **SPH**: all 25,577 `use_evaluation` ECGs of `data/processed/sph_clean_v1` (24,642 patients), with 022's
  saved features. 022 scored 21,008 of them, those with a binary label.

## Positive counts

Positives / rows defined for the group, from the `--counts` pass. The runner stops if any count differs.

| Group | PTB-XL train | PTB-XL dev | SPH | Ningbo train | Chapman train | Georgia train | CPSC 2018 train | CPSC-Extra train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ventricular_ectopy` | 915 / 17,370 | 84 / 1,600 | 1,058 / 25,577 | 663 / 19,750 | 177 / 6,126 | 239 / 6,102 | 360 / 3,765 | 83 / 1,797 |
| `preexcitation` | 64 / 17,417 | 6 / 1,604 | 27 / 25,566 | 42 / 19,788 | 11 / 6,127 | 3 / 6,101 | - | 4 / 1,796 |
| `af_flutter` | 1,257 / 17,417 | 115 / 1,604 | 762 / 25,577 | 4,445 / 19,803 | 1,339 / 6,127 | 439 / 6,102 | 584 / 3,765 | 75 / 1,797 |
| `svt` | 32 / 17,291 | 4 / 1,592 | 13 / 25,577 | 167 / 19,803 | 435 / 6,127 | 33 / 6,102 | - | 8 / 1,797 |
| `high_grade_av_block` | 12 / 17,405 | 1 / 1,603 | 27 / 25,497 | 43 / 19,688 | 0 / 6,023 | 5 / 6,044 | - | 12 / 1,787 |
| `long_qt` | 94 / 17,417 | 5 / 1,604 | 24 / 25,577 | 191 / 19,803 | 34 / 6,127 | 814 / 6,102 | - | 2 / 1,797 |

Challenge calibration groups by family (positives / rows):

| Group | Chapman/Ningbo | Georgia | CPSC (2018 + Extra) |
| --- | ---: | ---: | ---: |
| `ventricular_ectopy` | 293 / 8,970 | 66 / 2,041 | 137 / 1,868 |
| `preexcitation` | 14 / 8,987 | 0 / 2,041 | 0 / 602 |
| `af_flutter` | 1,989 / 8,992 | 133 / 2,041 | 246 / 1,868 |
| `svt` | 216 / 8,992 | 15 / 2,041 | 4 / 603 |
| `high_grade_av_block` | 24 / 8,915 | 1 / 2,020 | 4 / 599 |
| `long_qt` | 75 / 8,992 | 304 / 2,041 | 1 / 603 |

Training rows per readout (PTB-XL plus Challenge training, positives): `ventricular_ectopy` 54,910 (2,437),
`preexcitation` 51,229 (124), `af_flutter` 55,011 (8,139), `svt` 51,120 (675), `high_grade_av_block` 50,947
(72), `long_qt` 51,246 (1,135), `af_flutter_without_ningbo` 35,208 (3,694), and the binary readout 39,577
(27,360).

### Pre-registered thresholds

- **A group's readout is fitted only with at least 50 training positives.** All six pass;
  `high_grade_av_block` (72) is closest.
- **A set is evaluated for a group only with at least 20 positives**; below that only the count is reported.
  No group is merged: SVT and AF/flutter stay apart because the readouts would learn different things, and
  the criteria list them as one line only for reporting.
- With these thresholds the evaluated sets are:
  - SPH: `ventricular_ectopy`, `preexcitation`, `af_flutter`, `high_grade_av_block`, `long_qt` (not `svt`,
    13), and `sph_frequent_pvc`.
  - PTB-XL development: `ventricular_ectopy`, `af_flutter`.
  - Chapman/Ningbo: every group except `preexcitation` (14).
  - Georgia: `ventricular_ectopy`, `af_flutter`, `long_qt`.
  - CPSC: `ventricular_ectopy`, `af_flutter`.

## Readouts

- **Head**: `full_development.fit_logistic`, the 022 and 022b recipe: a `StandardScaler` on the training rows,
  then L2 logistic regression, `C=0.01`, `lbfgs`, `tol=1e-8`, `max_iter=5000`, float64, unweighted. Nothing is
  tuned or searched.
- **Training**: PTB-XL training plus the Challenge training groups (the 022b `pooled` design), every row
  defined for the group.
- **Encoders**: the frozen `xecg`, `jepa` and `cpc` features of 022 and 022b. **xECG is primary**; JEPA and
  CPC are secondary.
- **The binary screen**: 022b's adopted `pooled` binary readout, refitted on exactly 022b's training rows. It
  must reproduce 022b's saved SPH and Challenge calibration probabilities to 1e-9, or the run stops. It is then
  scored on every row of every set, including the rows 022b did not score.
- **Secondary readout** `af_flutter_without_ningbo`: `af_flutter` fitted without any Ningbo training row.

## Evaluation

Nothing is fitted or selected on an evaluation set. Each set is scored once.

- **Metrics per group and set**: AUROC and average precision of each score, for the group's positives against
  every other defined row of the set (normal or abnormal).
- **Scores compared**: the group's readout and the binary readout, per encoder; for `af_flutter` also
  `af_flutter_without_ningbo`.
- **Budget sensitivity (030-style)**: the share of the group's positives referred at a referral budget. The
  threshold is the 030 rule (`referral_budget.budget_threshold`: the (k + 1)-th highest normal score,
  k = floor(b m), refer strictly above) on the normal ECGs of the same set. Normal means a binary-label
  negative that is positive in no group: SPH code 1 alone (13,818), PTB-XL NORM without another diagnostic
  superclass, and Challenge sinus rhythm alone. Budgets 2%, 5% and 10%; **5% is the one reported first**.
  With 13,818 SPH normals this is the large-pilot case of 030 (m ≥ 2,000), where the achieved rate stays
  within about ±1 point of the budget. Only normal scores set the threshold; no positive is used. CPSC-Extra
  has no normals, so CPSC's thresholds come from CPSC 2018's.
- **What the binary screen already catches**: (a) the share of each group's positives that the binary label
  already marks positive (they carry another finding), and (b) the binary readout's sensitivity at the same
  budget on the same positives.

### Bootstrap

- 2,000 paired draws, seed 36036 (`normal_manifold.patient_resamples`): whole patients for SPH and PTB-XL
  development, records for the Challenge families (no patient IDs). Every score of a (group, set) uses the
  same draws; draws with one class are skipped and counted. In each draw the budget threshold is recomputed
  from that draw's normals. Fits are held fixed. Intervals are the 2.5th and 97.5th percentiles.
- **Contrasts, per group and set**: group readout minus binary readout (each encoder), xECG minus JEPA and
  xECG minus CPC (group readout), and for `af_flutter` each encoder's readout minus its
  `af_flutter_without_ningbo` readout; for AUROC, AP and every budget sensitivity.

## Primary question and decision rule

- **Primary**: for each group with at least 20 SPH positives, the xECG readout's SPH AUROC with its
  patient-bootstrap interval.
- **Decision, per group:**
  - **usable** if AUROC ≥ 0.90 and the interval's lower limit ≥ 0.85;
  - **not usable** if the interval's upper limit is below 0.90;
  - **undetermined** otherwise.
  - `svt` has too few SPH positives and gets no SPH reading; its Chapman/Ningbo result is described
    without a decision.
- Everything else (JEPA, CPC, AP, budget sensitivity, PTB-XL development, the Challenge families,
  `sph_frequent_pvc`, the binary-screen contrasts, the Ningbo readout) is described by its value and where its
  interval lies, with no decision.
- "Usable" means the frozen features rank the finding well at an unseen hospital. It does not mean the
  finding is detected reliably in students, and it sets no operating point.

## Closed data and exclusions

- The Challenge test groups, the PTB-XL calibration ECGs and the PTB-XL test ECGs stay closed: no score of
  them is computed.
- No age or other subgroup analysis.
- New files only: `ecg_experiment/rhythm_findings.py`, `scripts/experiments/run_rhythm_findings032.py` and
  `tests/test_rhythm_findings.py`. No frozen, hashed module changes.

## Caveats written into the results

- **SPH is development data now** (022, 022b, 024b, 026, 026b, 027, 027b, 029 and 030 read it), though never
  for these targets. It is a Chinese hospital cohort, older and sicker than students.
- **The Challenge families are not unseen.** The readouts were trained on the same families' training groups,
  and the split is by record, so a patient may appear in both. ECG-JEPA and xECG also saw Chapman and Ningbo
  waveforms, without labels, in pretraining. SPH is the only external read.
- The positive counts at SPH are small for pre-excitation (27), high-grade AV block (27) and long QT (24),
  so their intervals are wide. The training positives are few for high-grade AV block (72) and
  pre-excitation (124).
- The labels are annotation statements, not rhythm-strip adjudication; a PVC code does not count PVCs, the
  long-QT codes use general-medicine limits, and the Ningbo flutter code is probably AF in part.
- A 10 s window may miss an intermittent finding that another part of the record shows; CPSC records longer
  than 10 s are read through their centred window.
- One fit per readout and encoder; the bootstrap holds the fits fixed.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_rhythm_findings032
```

CPU only, at most three processes with one BLAS thread each (another experiment runs in parallel). The runner
hashes every input, source and this protocol into the result, checks the counts above, the quality reasons
and the 022b reproduction, refuses to overwrite an existing run, and writes
`outputs/experiment032_rhythm_findings_v1/` (`result.json`, `predictions.npz`, `training_rows.csv`,
`calibration_rows.csv`, `run.log`). Results go to `docs/experiment-032-rhythm-findings-results.md`.
