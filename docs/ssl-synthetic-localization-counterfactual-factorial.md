# Counterfactual localization loss factorial

18 October 2026. The user requested continued local work until an important finding. This protocol
freezes the next engineering comparison before new patient selection, qualification, GPU profiling,
training or performance scores. Nothing is pushed or uploaded. Earlier failed advancement decisions
remain failed. No clinical label, referral rule or closed-test access changes.

## Question and comparison

Does explicitly teaching the difference between an edited ECG and its matching original improve
localization of the introduced change? Four arms cross the original random (R) or existing real-ECG
self-supervised (S) encoder initialization with ordinary weighted BCE (B) or BCE plus a paired loss
(P): RB, SB, RP, SP. All use the same original localization head and the unchanged deterministic v2
single-input model. A model always receives one observed ECG; paired originals and support masks
enter the training objective only, never inference features.

The single primary endpoint is the initialization-averaged paired-loss effect on DIRECT edited-map
AP: one half of [(RP minus RB) plus (SP minus SB)]. Compute each difference per patient after equal
averaging of focal and repeated cases, then average patients. AP ranks exact introduced support;
it is not diagnostic accuracy. A response gain caused only by lower baseline-site AP cannot improve
this primary endpoint. Predeclare loss effects within R and S, the initialization-by-loss
interaction [(SP−SB)−(RP−RB)], and S−R under each loss as secondary comparisons. Report them even
when the primary fails; no selection of whichever contrast looks best.

## Fixed objective and fitting

Reuse the exact existing 500 TRAIN patients, 2,000 cases, native support/candidate masks, rendered
inputs, actual-schedule positive class weight and 600-by-16 ordered case schedule of the completed
paired-background study. Reuse initial.pt and ssl.pt from the original pilot. No new SSL, GAN,
augmentation, patient filtering, target generator, optimizer setting, training seed or checkpoint
search. Batch 16 contains four patients and all four adjacent focal/repeated/sham/nuisance variants;
each arm sees 9,600 presentations in exactly 600 updates. AdamW lr 0.001 and weight decay 0.0001,
all parameters trainable, same fixed 2 mV scale, float32 deterministic CUDA, one numerical thread.

Let p_e and p_b be sigmoid native-grid outputs for a positive case and its exactly matching
unchanged/nuisance control already present in the SAME batch. Let T be the nonempty exact introduced
support and notT its nonempty complement on all twelve leads and 2,500 samples. For each positive
case calculate mean_T relu(0.20−(p_e−p_b)) plus mean_notT abs(p_e−p_b). Average those sums equally
across the eight positive cases. P minimizes existing weighted BCE plus 1.0 times this paired term;
B minimizes exactly existing weighted BCE. Gradients flow through both outputs with no detach.
The per-cell probability hinge asks every introduced cell to respond; outside-support consistency
and BCE on controls oppose a global score shift. Margin 0.20 and weight 1.0 are fixed engineering
choices, not optimized values. Report BCE, hinge, consistency and total losses without choosing a
checkpoint from them. Record tensor and component hashes, exact pairing, exposures and all updates.

RB and SB are deliberate deterministic reproductions. Require their final model/component hashes
to match the predecessor R/S final states EXACTLY before any new calibration or evaluation. Any
mismatch stops this study; do not reinterpret a changed baseline as a successful reproduction.

## Prospective decision and uncertainty

A positive engineering finding eligible for a separately frozen replication requires ALL:

- Primary direct-AP mean gain at least 0.02 and patient-bootstrap 95% lower bound strictly above 0.
- Within EACH initialization the P−B direct edited-AP lower bound strictly above−0.01.
- EACH paired-loss arm has strictly positive lower bounds for within-arm AP response and for
  target-minus-unedited-candidate probability response. AP response subtracts matching baseline-map
  AP against the SAME support; selectivity is mean_T(p_e−p_b)−mean_U(p_e−p_b), U=candidate minus T.
- Within EACH initialization P−B point flag-rate increase at most 0.05 on EACH unchanged and nuisance
  control under arm-specific prospectively calibration-frozen thresholds.

Report every gate and their conjunction. These are engineering guards, not clinical thresholds.
Report focal/repeated edited AP, baseline-site AP and response separately. A pooled finding does not
establish both modes improved. Decompose AP response changes into edited-AP and baseline-AP changes.
Report target/U/outside candidate probability levels, selectivity, whole-map absolute response,
top 1% precision/recall, raw-voltage/uniform/phase baselines and positive/control record flags.
All probability-magnitude comparisons across arms remain uncalibrated descriptions.

Use 2,000 paired whole-patient bootstrap draws with shared intervals helpers, seed 20261027,
2.5/97.5 percentile bounds, no patient drops and explicit skipped-draw count. Apply the same patient
resamples to all four arms. Intervals are conditional on this one initialization/pretraining seed
and schedule; they do not estimate training variability. Secondary intervals are descriptive with
no multiplicity-adjusted claim. A failed conjunction stops replication admission without tuning,
extra seeds, altered thresholds or automatic successor fits. A failed primary does not demonstrate the prespecified benefit of this fixed recipe under this
setup; it does not establish zero effect or reject every paired method.

## New calibration and evaluation data

The original real-background transfer and paired-background follow-up used 384 distinct VALIDATION
patients. Exclude ALL 384. Among remaining original VALIDATION patients choose each patient's lowest
numeric ECG ID without quality filtering, sort SHA-256 of UTF-8
`ssl_counterfactual_factorial_v1:validation:<canonical patient ID>` (numeric patient ID tie break),
take first 64 calibration and next 128 evaluation patients. Freeze the 192-row selection before any
selected waveform read; no replacements. All selected patients must be disjoint from 500 TRAIN,
2,048 SSL and every protected group. Existing original engineering_test, PTB-XL folds 9–10,
Challenge closed groups and EchoNext test remain CLOSED. No clinical fields or ages are used.

Reuse the exact previous TRAIN cache/masks/cases by reference; do not duplicate them. Cache ONLY 192
new backgrounds with local indices 0–191, explicit virtual subject indices 500–691, global case indices
2000–2767 and map/mask rows 0–767. Calibration occupies rows 0–255 and evaluation 256–767. All index
adapters must be explicit and tested; no monkeypatching frozen globals. New case seeds are first 8
big-endian SHA-256 bytes of `ssl_counterfactual_factorial_v1:<role>:<patient ID>`, role calibration or
evaluation, existing edit/nuisance streams 101/303. Calibration uses profile A and evaluation B.
Reuse unchanged native 250Hz anchoring/edit/nuisance/limb-delta/once-cast math and qualification.

Bind official checksums and original audit for each of 192 headers/data, validate units/lead order,
decode/resample 500→250Hz with frozen helpers. Require all 192 backgrounds and 768 variants pass
finite geometry, at least 3 anchors, support<=0.18, peak added delta<=0.50 mV, algebra, control identity,
exact replays and 384 matching baselines. Strict final float32 observed-difference magnitude>0.005 mV
defines support. Native findings remain UNKNOWN. A zero intervention mask does not mean healthy;
these controlled pulses are not expert-verified disease annotations or waveform boundaries.
Any failure retains its receipt and stops the whole selection without replacement.

Freshly SHA-bind the passed parent training/metrics/diagnostic audits and their compared values;
reproduce the full predecessor summary, frozen thresholds and all nine gates at 1e−12 tolerance,
retaining its failed verdict. Existing independent numerical reproduction plus fresh binding is
sufficient; do not rescore old evaluation patients. Read only permitted artifacts needed for this
study. Unrelated old waveform/checkpoint declarations may be verified through explicitly anchored
passed-audit catalogs, with exceptions recorded; conflicting/missing authority stops admission.

## Resources and execution gates

Before fits profile the ACTUAL training path for all four arms under the shared nonblocking GPU
lock:10 warm-up+20 timed updates with real fixed TRAIN batches, regeneration/checks, H 2D, model,
objective, backward and AdamW. Disposable states are discarded. Require exact two-next-update replay
after a one-step model/AdamW/RNG save/restore for each arm. Profile 10 warm+20 timed calibration-A
inference batches through the actual path; discard predictions and compute NO target metrics.
No evaluation-B inference during profile. Freeze/review/root-integrate source before each stage.

Forecast spent preparation/profile plus twice measured 600-update cost for each arm, twice the
four-arm 192-batch inference cost and 120seconds evaluation/report reserve. Admit only within 900new
stageGPU-wall seconds,1800new stageCPU seconds and cumulative 7200GPU/14400CPU seconds. Include
retained failures and verification once; enforce live ceilings with reporting reserve. Costs enter
from exact SHA-bound latest diagnostic audit, not rounded literals. Latest prior cumulativeCPU is
569.9337830239988 and GPU-wall 226.41281207299835 seconds; confirm against live receipts.

New arrays are 192×12×2500float32 backgrounds(23,040,000bytes), two 768×12×2500bool masks(46,080,000)
and four 768×12×2500float32 prediction banks(368,640,000). Total 437,760,000 payload bytes plus 896 NPY
header bytes and 64 MiB metadata/checkpoint reserve =504,869,760 prospective bytes. Require live stream
bytes PLUS remaining forecast <=4 GiB before preparation/profile/fits/evaluation and guard actual
size at boundaries. No cleanup/deletion of predecessor artifacts to manufacture room. All new study
artifacts, including failed attempts and audit outputs, share the 64 MiB reserve. Refuse overwrite.

After exact 600-step fits and deterministic BCE closure, fill all four calibration slices only.
For each 64patient/arm take joint unchanged+nuisance maximum; empirical higher 95th percentile and
strict greater-than flags. Save/hash calibration bytes, all slices and thresholds BEFORE ANY B
inference. Then run all 128evaluation patients/all 512cases, retaining native maps and verifying
calibration unchanged. Do not select new thresholds or choose a winning checkpoint after scores.

End with the runner's executed qualification/profile/training/results reports, an independent audit
of saved metrics/decision and training identity, both queue updates and ranked follow-ups. No case
identifiers, waveforms, masks, maps or weights enter Git or external APIs. No clinical validity
claim follows, even if every engineering gate passes. No further fit starts automatically.
