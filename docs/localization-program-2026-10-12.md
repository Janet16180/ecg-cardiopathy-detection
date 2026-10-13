# Localization programme, October 2026

12 October 2026. Goal set by the user: highlight where in a 12-lead ECG an abnormality is (which beats, which
leads, which wave), without localization labels for training, and show that the highlighted regions are
cardiac abnormalities rather than noise or benign variation. The user authorized new architectures,
preprocessing and training strategies, DINO-style and otherwise, compared on one benchmark.

## Why Experiments 041-060 did not get there

From a full review of the 041-060 protocols, reports and receipts:

1. Two kinds of reference, each blind where the other sees. A within-ECG reference (052, 054, 056) finds the
   odd beat (INCART top beat 89.6%) but subtracts any change present in every beat: persistent synthetic ST
   was found in 0-14% of cases, below chance (058). A population reference (U_B, 042) sees persistent change
   but could not place it by lead (AMI/IMI lead contrast not significant). No method combined them.
2. Fixed windows instead of physiology. U_B's P, QRS, ST and T windows are fixed offsets from R (190, 140,
   120 and 250 ms) against true durations of about 96, 94 and 180 ms; a wide QRS spills into ST. Its 93%
   premature-beat hit falls to 55% under a strict rule (051).
3. Coarse tokens. xECG tokens are 250 ms and mix all leads; ECG-JEPA patches are 200 ms on 8 leads.
4. Supervision decides where a head points: a binary-label attention head hit premature beats 2.7% of the
   time, a PVC-label head 84% (043, 050).
5. A contaminated normal reference: 22% of the 5,872 reference normals carry a noise annotation and 189 an
   extra-beat annotation. Added noise of 0.02 mV raised flags by 54-88 points (058).
6. Evaluation targets that do not measure the goal: an automatic premature-beat proxy (73 ECGs), a top-1-lead
   test for MI, Holter data. Nothing tested lead and wave truth for persistent findings on 10-second 12-lead
   ECGs, and most sets had 30-150 cases, so several near-misses (050, 055, 060) are unresolved.

## What changes

- One benchmark for every method, LocBench v1 (Experiment 061), with real localization truth:
  - the ECGAD cardiologist marks, lead by sample, on 399 PTB-XL records ([ecgad-benchmark-v1.md](ecgad-benchmark-v1.md));
    development half for iteration, holdout half scored once at the end;
  - a clinical region test on 1,000+ PTB-XL records from a held-out third of folds 1-8: each localized
    statement (inferior MI, anterior MI, bundle branch blocks, LAFB, AV block, long QT, ST injury, ischemia)
    has a fixed expected lead set and ECG component, with component boundaries from PTB-XL+ (ECGdeli);
  - a noise control: normal ECGs whose noise annotation names the affected leads. A cardiac map should not be
    drawn to those leads;
  - detection on the same held-out third, so that no method buys localization with detection.
- Clean references: normals with NORM at likelihood 100, no other statement, clean sinus rhythm, and no noise,
  extra-beat or pacemaker annotation, from the fit two thirds of folds 1-8 (2,666 records).
- Baselines on the same benchmark: random, signal amplitude, physiological wave priors, U_B and the 056
  residual. A method counts only if it beats the best of these.

## Planned experiments

| Exp | Method | Question |
| --- | --- | --- |
| 061 | LocBench v1 and baselines | How well do the existing maps and trivial priors localize on real truth? |
| 062 | Normative phase-aligned map | Does a population model of delineated, phase-warped beats per lead (conditioned on heart rate, age and sex), combined with the within-ECG residual, localize persistent and focal findings? |
| 063 | Masked cross-lead and cross-beat reconstruction | Does a model trained on clean normals to fill masked lead-time blocks give a calibrated local error map that localizes? |
| 064 | SimDINO patch tokens with phase-matched kNN | Do self-supervised beat-level lead x 40 ms tokens, scored AnomalyDINO-style against a normal token bank, localize? |
| 065 | Finding-specific multiple-instance heads | Do weakly supervised heads over beat x lead x component units, trained on PTB-XL statements, point to the clinically expected region? |
| 066 | Combination and artifact suppression | Does combining label-free and supervised maps, with a lead-quality map learned from the noise annotations, improve localization without being drawn to noise? |
| 067 | Holdout confirmation | One scoring of the frozen candidates on the ECGAD holdout half. |

Each experiment commits its protocol before any score and reports from its outputs. The primary comparison
of 062-066 is the paired within-record localization AUROC on the ECGAD development half against the best
061 baseline, with patient bootstrap intervals; the clinical region, noise and detection readings are
reported beside it. Self-supervised pretraining uses cohorts v4 minus the LocBench evaluation third of folds
1-8; ECGAD records are already outside v4.
