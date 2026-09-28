# SPH and EchoNext: EDA findings and what they mean for the project

28 September 2026. The dataset descriptions are in `notebooks/07-jr-sph.ipynb` and
`notebooks/08-jr-echonext.ipynb`; this file keeps the project-specific consequences apart from them. No model
was trained or scored for this review.

## What the two sources are

| | SPH | EchoNext |
| --- | --- | --- |
| ECGs, patients | 25,770, 24,666 | 100,000, 36,286 (train 72,475; val 4,626; test 5,442; no_split 17,457) |
| Signal | 500 Hz, 10-56 s, mV | 250 Hz, 10 s, standardized (no unit), median filtered and clipped |
| Labels | Cardiologist AHA codes (ECG reading) | Echocardiogram findings (structural heart disease) |
| Access | Open | PhysioNet credentialed; aggregates only |

## Findings that change what we do

1. **SPH has 168 duplicate uploads, 25 of them with conflicting codes.** Evaluating on raw SPH would count
   those ECGs twice and score 10 pairs against contradictory labels. The clean manifest keeps one copy when
   the copies agree and drops both when they conflict.
2. **SPH has about 120 ECGs with non-cardiac spikes of up to 1,590 mV.** 98 fall in the first ten seconds and
   are caught by the existing quality policy. Because they are less often abnormal (22% against 34%), keeping
   them in an external evaluation does not remove hard positives, but it does add nonsense inputs; report
   results with and without them.
3. **629 EchoNext records hold only 2.5-5 s per lead, with a constant filler elsewhere,** almost all from
   2008-2012, and they are half as often positive (26% against 52%). This is a shortcut: a model can learn
   "filler means healthy". The flat-segment rule of the quality policy removes them; do not train or report
   primary results with them.
4. **EchoNext training is more often positive than validation (52% against 43%) because of repeat ECGs.**
   Training keeps every ECG, and patients with five or more ECGs are 61% positive. Thresholds or calibration
   fitted on training ECGs will not transfer to one-ECG-per-patient populations. AUROC is unaffected, but
   calibration work must use one ECG per patient.
5. **EchoNext component labels are 0 when the echo value is missing** (for example 43% of PASP labels).
   Component analyses must be restricted to rows with a measured value; the composite is complete.
6. **`no_split` shares its patients with EchoNext val and test.** It must never be used for training.
7. **Both new sources are far more filtered than the training data.** Their share of power below 0.7 Hz is 20
   (EchoNext) to 35 (SPH) times smaller than PTB-XL's. The CPC encoder never saw such smooth signals, so
   external scores measure robustness to a different recording chain as well as disease detection.
8. **The SPH "normal ECG" code is exclusive.** An otherwise normal ECG with sinus bradycardia is coded with the
   rhythm alone, so the standard label treats 2,824 such ECGs as undefined unless the secondary label counts
   them as normal. Report both.
9. **EchoNext cart measurements already separate the classes partly** (QRS 94 against 86 ms, QTc 459 against
   440 ms), so a tabular baseline is required before claiming anything for a waveform model.

## Clean data built from these findings

See [clean SPH and EchoNext v1](clean-sph-echonext-v1.md).
