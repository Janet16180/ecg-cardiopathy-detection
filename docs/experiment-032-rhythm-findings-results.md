# Experiment 032 results: detecting the rhythm findings the athlete criteria call abnormal

Completed 29 September 2026 under the [frozen protocol](experiment-032-rhythm-findings.md) (frozen at commit
`ac8bdf2`; the run recorded its file hash, `aeed4b51…dbf8`, which matches). Run once at commit `3264c7c` on
CPU in 1,478 s (260 s of it the training quality check), three processes with one thread each. Local outputs
are in `outputs/experiment032_rhythm_findings_v1/` (`result.json` SHA-256 `16fdda1e…20ab`, `predictions.npz`,
`training_rows.csv`, `calibration_rows.csv`, `run.log`). No Challenge test-group ECG, no PTB-XL calibration
ECG and no PTB-XL test ECG was scored. This covers the backlog item `rhythm_findings_detector`.

Before the freeze, one pass of the runner with `--counts` stopped before any readout was fitted; it produced
the counts in the protocol. No other pass was made.

## Integrity

- Every count matched the protocol, including the quality-policy exclusions, and the quality reasons equalled
  022b's on every row 022b used.
- The refitted binary readout reproduced 022b's `pooled` probabilities: the largest difference was 0.0 on SPH
  and 2.2e-16 on the Challenge calibration groups.
- All 24 fits (8 readouts × 3 encoders) converged (74-326 iterations). No bootstrap draw was skipped in any set.

All intervals are 95% paired bootstrap intervals (2,000 draws, seed 36036): by patient for SPH and PTB-XL
development, by record for the Challenge families. "Sensitivity at 5%" is the share of the finding's ECGs
above the threshold that refers 5% of the same set's normal ECGs (13,818 at SPH).

## 1. Primary: xECG at SPH, an unseen hospital

AUROC for the finding against every other SPH ECG (normal or abnormal):

| Group | SPH positives | xECG AUROC | AP | Sensitivity at 5% | ECG-JEPA AUROC | CPC AUROC | Reading |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `ventricular_ectopy` | 1,058 | 0.990 [0.986, 0.993] | 0.945 | 0.980 [0.971, 0.988] | 0.990 [0.986, 0.994] | 0.976 [0.970, 0.982] | usable |
| `preexcitation` | 27 | 0.992 [0.985, 0.997] | 0.565 | 1.000 [1.000, 1.000] | 0.984 [0.962, 0.996] | 0.927 [0.873, 0.972] | usable |
| `af_flutter` | 762 | 0.9999 [0.9999, 1.0000] | 0.997 | 1.000 [1.000, 1.000] | 0.9999 | 0.9998 | usable |
| `high_grade_av_block` | 27 | 0.9999 [0.9997, 1.0000] | 0.882 | 1.000 [1.000, 1.000] | 0.9998 | 0.9996 | usable |
| `long_qt` | 24 | 0.934 [0.879, 0.976] | 0.166 | 0.792 [0.607, 0.944] | 0.952 [0.922, 0.980] | 0.940 [0.907, 0.969] | usable |
| `svt` | 13 | - | - | - | - | - | too few positives |

**Every evaluable group is usable by the pre-registered rule** (AUROC ≥ 0.90, lower limit ≥ 0.85). Long QT
passes narrowly: its lower limit is 0.879, from 24 positives. SVT has 13 SPH positives (all junctional
tachycardia, the only SVT code SPH has) and gets no reading.

The SPH subset closest to the criterion, PVCs marked frequent, in couplets, or in a bigeminal or trigeminal
pattern (372 ECGs, secondary), is detected almost perfectly: AUROC 0.998 [0.995, 1.000], AP 0.982,
sensitivity at 5% 0.997.

Average precision is low for the rare groups (WPW 0.565, long QT 0.166) even with high AUROC. At SPH's
prevalence of 27 in 25,566, a readout that ranks well still puts many other ECGs among the top scores. This
matters for a direct alarm on one finding, less for a referral budget.

## 2. What the binary screen already catches

The same positives scored by the adopted binary readout (022b `pooled`, xECG). "Labeled abnormal" is the share
of the finding's ECGs that the binary label marks positive. For pre-excitation, AV block and long QT that is
100% by definition (they are CD or STTC codes); for PVCs and AF/flutter it is the share that carries another
finding.

| Group (SPH) | Labeled abnormal | Binary AUROC | Binary sensitivity at 5% | Finding readout at 5% | Difference at 5% |
| --- | ---: | --- | --- | --- | --- |
| `ventricular_ectopy` | 37.6% | 0.735 [0.721, 0.748] | 0.616 [0.583, 0.646] | 0.980 | +0.364 [+0.332, +0.398] |
| frequent PVCs (secondary) | 31.5% | 0.777 [0.760, 0.794] | 0.715 [0.668, 0.759] | 0.997 | +0.282 [+0.237, +0.329] |
| `preexcitation` | 100% | 0.772 [0.702, 0.835] | 0.704 [0.519, 0.882] | 1.000 | +0.296 [+0.118, +0.481] |
| `af_flutter` | 59.2% | 0.948 [0.943, 0.952] | 0.999 [0.996, 1.000] | 1.000 | +0.001 [+0.000, +0.004] |
| `high_grade_av_block` | 100% | 0.944 [0.922, 0.965] | 1.000 [1.000, 1.000] | 1.000 | +0.000 [+0.000, +0.000] |
| `long_qt` | 100% | 0.913 [0.882, 0.940] | 1.000 [1.000, 1.000] | 0.792 | −0.208 [−0.393, −0.056] |

- **The binary screen already refers almost every AF/flutter, high-grade AV block and long-QT ECG** at a 5%
  budget. These ECGs look far enough from a normal ECG that the screen flags them, even the 41% of AF ECGs
  that the label leaves undefined.
- **It misses about 4 in 10 PVC ECGs and 3 in 10 pre-excitation ECGs.** A PVC ECG whose other beats are
  normal looks mostly normal, and pre-excitation had only 124 training examples in a readout trained for
  everything abnormal. The finding readouts catch 98% and 100% of them at the same budget.
- **Long QT is the reverse.** The binary screen refers all 24 (prolonged QT is itself an STTC statement, so the
  screen was trained to flag it), while the long-QT readout, which learned to separate long QT from other abnormal ECGs, refers 79%.
  A dedicated QT readout adds nothing to the screen here.
- The finding readout ranks its finding above the other abnormal ECGs far better than the binary screen does
  (AUROC +0.05 to +0.26, every interval above 0), except for long QT (+0.021 [−0.036, +0.069]).

## 3. Challenge families and PTB-XL development (secondary)

These sets are **not unseen**: the readouts were trained on the same families' training groups (split by
record), and JEPA and xECG saw Chapman and Ningbo waveforms in pretraining. xECG AUROC:

| Group | Chapman/Ningbo | Georgia | CPSC | PTB-XL development |
| --- | --- | --- | --- | --- |
| `ventricular_ectopy` | 0.986 [0.980, 0.990] (293) | 0.875 [0.820, 0.924] (66) | 0.893 [0.857, 0.928] (137) | 0.994 [0.989, 0.998] (84) |
| `af_flutter` | 0.998 [0.998, 0.999] (1,989) | 0.927 [0.893, 0.958] (133) | 0.994 [0.991, 0.997] (246) | 0.993 [0.985, 0.998] (115) |
| `svt` | 0.993 [0.990, 0.995] (216) | (15) | (4) | (4) |
| `high_grade_av_block` | 0.994 [0.987, 0.999] (24) | (1) | (4) | (1) |
| `long_qt` | 0.959 [0.944, 0.972] (75) | 0.900 [0.879, 0.919] (304) | (1) | (5) |
| `preexcitation` | (14) | (0) | (0) | (6) |

Positives in parentheses; below 20, only the count is reported. SVT is detected at Chapman/Ningbo (AUROC 0.993,
sensitivity at 5% 1.000), but that is an in-distribution read.

**Georgia is the weak site** for PVCs (0.875) and AF/flutter (0.927), well below SPH, which the readouts never
saw. CPSC is also lower for PVCs (0.893). The pattern for the binary screen is the same as at SPH: at a 5%
budget it refers 0.64 of Georgia and CPSC PVC ECGs against 0.79 and 0.83 for the PVC readout, and it already
refers 0.87-1.00 of the AF/flutter ECGs.

## 4. Encoders

At SPH, xECG and ECG-JEPA are tied in practice for every group: the AUROC differences are at most 0.018 and
every interval includes 0, except AF/flutter, where xECG is ahead by 0.00004 (an interval above 0 at the
ceiling, of no practical meaning). JEPA has a slightly higher PVC AP (0.957 against 0.945). CPC trails for
PVCs (0.976, xECG minus CPC +0.013 [+0.009, +0.018]) and pre-excitation (0.927, +0.065 [+0.024, +0.114]); for
AF/flutter and AV block it trails by 0.0001-0.0002, and for long QT it ties. By the same rule, the CPC
readouts would also be usable for every group (pre-excitation lower limit 0.873).

## 5. The Ningbo flutter coding

The AF/flutter readout fitted without any Ningbo row gives the same SPH ranking (AUROC 0.99985 against
0.99992; AP 0.995 against 0.997). Ningbo's 4,445 training "flutter" ECGs, probably fibrillation in part, neither help nor
hurt at an unseen site. They help only at Chapman/Ningbo itself (+0.010 [+0.007, +0.014], xECG), an
in-distribution gain.

## Surprises

- **How easy AF/flutter and high-grade AV block are at SPH.** AUROC 0.9999 for all three encoders, with
  only a linear head. These are hospital cases, likely mostly persistent and often with fast or slow
  ventricular rates; a paroxysmal AF in a student would be harder, and one 10 s strip can miss it entirely.
- **Pre-excitation transfers from 124 training examples**, to AUROC 0.992 at SPH. The binary label already
  calls WPW abnormal, yet the binary screen misses 30% of these ECGs at a 5% budget.
- **Georgia, not SPH, is where transfer is weakest**, even though Georgia's training rows were in the fit.
  Georgia's labels or recordings may differ; this experiment cannot say which.
- Long QT is the one finding where a dedicated readout refers fewer ECGs than the general screen.

## What this means for student screening

- **The frozen encoders already contain the rhythm information.** A plain linear readout per finding, trained
  on public labels, separates PVCs, pre-excitation, AF/flutter and high-grade AV block from every other ECG at
  an unseen hospital with AUROC 0.99 or more. Long QT is weaker (0.93).
- **The current screen is blind mainly to PVCs and partly to pre-excitation.** At a 5% referral budget it
  already refers nearly every AF, flutter, complete heart block and long-QT ECG, because they look abnormal
  overall. It refers only about 60% of PVC ECGs and 70% of pre-excitation ECGs. Adding a PVC and a
  pre-excitation readout closes most of that gap; an AF or AV-block readout would mainly add a named finding
  for the reader, not extra referrals.
- **What is not measured here.** Combining the binary screen with finding readouts would raise the share of
  normals referred above 5%; that combined operating point is not computed here and needs its own
  pre-registered test, ideally on young normals. The labels mark "PVCs present", not "2 or more per 10 s";
  the frequent-PVC subset suggests the criterion's cases are the easier ones. SVT could not be tested at an
  unseen site. The positives come from hospital patients, so detection in young, fit people, with higher
  voltages and more sinus arrhythmia, is not established.
- **For the cardiologist meeting:** the model can name these findings, which is useful because the athlete
  criteria refer on them regardless of the rest of the ECG. Which of them should trigger a referral on its own
  (for example isolated PVCs versus two or more), and how a flagged PVC should be confirmed (a longer strip
  or Holter), are clinical choices for them.

## Caveats

- SPH is development data for other targets (022-030) and was used here for the first time for these; it is
  an older, sicker Chinese hospital cohort. The Challenge families are in-distribution reads.
- The rare groups have 24-27 SPH positives, so their intervals are wide and one mislabeled ECG moves the
  AP noticeably.
- The labels are annotation statements, not adjudicated rhythm strips; the long-QT codes use
  general-medicine limits, not the athlete limits.
- One fit per readout and encoder; the bootstrap holds the fits fixed. The budget threshold uses the set's own
  normals (the large-pilot case of 030); a small local pilot would add threshold noise.
