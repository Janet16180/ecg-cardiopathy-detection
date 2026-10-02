# Experiment 054: aligned residuals with normal record-max phase calibration

Frozen 2 October 2026 before any score. Experiments 052 and 053 were negative. This distinct prospective
successor asks whether ordinary QRS alignment variability and phase-dependent residual scale obscure
small local waveform changes. No model, referral rule, clinical label, download or closed test changes.

## One primary recipe

Only existing R peaks align beats; RR timing is never an anomaly feature. For each ECG with at least
three complete beats, extract raw beat offsets [-0.27, +0.47) seconds (370 samples at 500 Hz), with NaN
only outside record boundaries. Subtract each beat/lead's original pre-QRS median at R - 0.10 through
R - 0.06 seconds. Compute a recording-level lead scale: median QRS peak-to-peak across complete beats,
with a 0.1 mV floor. Never fit per-beat amplitude, since that could erase a real amplitude change.

Build an initial median template. Align each beat by one shared multilead shift in {-5,...,+5} samples
(at most 10 ms), minimizing mean squared error on QRS offsets [-0.06,+0.08), after division by the
recording-level lead scales. Ties prefer the smallest absolute shift, then the negative shift. Rebuild
the median once. The leave-one-out prediction is the median of other aligned beats on extended offsets
[-0.26,+0.46) (360 samples). For every original canonical sample t in [-0.25,+0.45), predict at t minus
its own chosen shift, and compare to the original observed sample. All residuals and targets stay on
the original waveform timeline. Extended templates avoid padding/scattering artifacts at displayed
edges; require every prediction finite and explicitly record failures.

Square residuals divided by recording-level lead scale and average in 140 ms windows at 10 ms stride.
Candidate eligibility is exactly Experiment 052's midpoint-owned complete-beat mask intersected with
actual saved U_B piece coverage; the comparator uses identical candidate windows. No wider supports.

For each of 1,000 calibration-normal patients, compute the maximum window energy for every lead and
phase channel. Phases are fixed strata of a window center relative to its nearest *original complete*
R anchor: P [-0.25,-0.06), QRS [-0.06,+0.08), ST [+0.08,+0.20), T [+0.20,+0.45). These fixed offsets are
alignment strata, not expert wave delineations. Assert every eligible center has one stratum.
Each of the 48 channels has the same N = 1,000 reference record maxima. Score an observed window with
`-log((count(reference >= energy) + 1)/(N + 1))`, using `searchsorted(side='left')`, which conservatively
includes equal reference values. This per-record-max reference accounts for within-phase window
multiplicity and avoids unequal pooled-window CDF caps. Report saturation at log(1001) and ties.

`aligned_phase` is the only primary method. The uncalibrated aligned residual is a descriptive ablation;
it cannot rescue a failed primary. U_B fixed support is the comparator. No sweep or outcome-driven
choice is permitted.

## Frozen patients

The committed [cohort manifest](experiment-054-cohorts.json) pins exact ECG IDs before scoring. All are
unique training-normal patients, disjoint from development patients and Experiment 052's 200 synthetic
patients. Take the next 200 eligible metadata rows in ascending ECG ID as new synthetic holdout. From
the remainder, a seed-54054 permutation assigns 1,000 CDF-calibration patients and 300 threshold-control
patients. The sets are mutually patient disjoint. Previous project normal-reference exposure is not
new 054 leakage; no 053 parameters are reused. Patient separation applies to every fit in 054.

Synthetic eligibility requires five complete beats, as in 052. If a frozen synthetic ID fails this
requirement, record the exclusion without replacement or changing the target. Calibration or control
failures are explicit; fewer than the promised 1,000 finite channel references prevents promotion.

## Integrity before new scores

Require 052's protocol and all recorded source/input/output hashes unchanged. Receipt-check and
reproduce the entire historical U_B result as in 052. Reconstruct 052's full primary metrics from its
saved per-record table with seed 52052 and 2,000 patient draws; reproduce its complete descriptive map
metric dictionary from saved maps. Reproduce its synthetic summary from saved tables. Any difference
stops before any new score. Existing 052 sources are frozen and never edited.

## Decision rule and controls

Reuse 052's 73 automatic early-beat QRS targets (R - 0.06 through R + 0.08), center-inside hit, uniform
eligible-window chance, tied-maximum expectation with `isclose(rtol=1e-10, atol=1e-10)`, actual-R distance
and support overlap. On development data, promote only if the primary map's paired hit-minus-chance
gain over U_B fixed is at least +0.10 and its patient-bootstrap lower bound exceeds zero.

On the 200 new synthetic normals, seed 54054 chooses an interior complete beat and leads. Keep all R
peaks fixed to the unchanged ECG. Use exactly 052's edit magnitudes and supports: one-lead ST bump
0.15 mV at R + 0.10..+0.20 seconds; three-lead 1.6 times stretched median QRS transplant supported at
R - 0.096..+0.128 seconds. The transplant reference is the unmodified ECG's other beats, not a learned
pathology. Joint success requires top-window center within the exact edited support and the selected
lead among edited leads, averaging tied maxima. Both ST and QRS must separately reach at least 80%
joint localization AND have a strictly positive lower bound on paired score change from the unchanged
copy. The real rule and both synthetic rules must all pass; thresholds cannot be loosened.

The 300 disjoint unchanged controls set each map's worst-score 95th percentile threshold. Report
unchanged and offset-plus-drift nuisance red shares on synthetic holdout (offset 0.15 mV, drift 0.05 mV
at 0.2 Hz), score changes, and ST time-shift/lead-permutation controls.
The lead-permutation control permutes calibration reference channels together with waveform leads;
it checks coordinate equivariance, not incorrect canonical lead assignment.
A nuisance increase exceeding 0.05 is an explicit review concern, descriptive rather than a replacement
gate. Patient intervals use
2,000 draws, seed 54054. Development targets remain automatic and exploratory, never clinical truth.

## Execution and review

New files only: `ecg_experiment/raw_residual054.py`, `scripts/experiments/run_raw_residual054.py`, and
meaningful tests including shared-shift mapping, lead/time invariance, constant offsets and preservation
of amplitude changes. Import shared library helpers, never scripts. Run CPU only with two threads,
after this protocol and the cohort manifest are committed and scientific tests/Ruff pass.

Write receipt-backed outputs to `outputs/experiment054_aligned_phase_residual_v1/` through a `.partial`
directory without overwriting older results. Persist calibration references, maps, per-record tables,
cohort identities, waveform/source/protocol hashes, shifts, exclusions and `run.log`. Save local review
figures showing original, aligned and calibrated location marks for all seven established examples
(47, 219, 8, 184, 287, 30, 69), deterministic lowest-ID successes/failures, and template overlays.
Waveforms stay local. The running agent writes `docs/experiment-054-aligned-phase-residual-results.md`
from executed outputs, including every negative result and limitation. Persistent ECG abnormalities
can still disappear in a within-record template: this map cannot replace referral detection or
identify an anatomical heart lesion.
