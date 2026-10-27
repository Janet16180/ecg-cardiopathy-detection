# Real-background introduced-voltage qualification and frozen-model transfer

18 October 2026. A separate prospective engineering follow-up, after the original SSL pilot failed
its frozen advancement rule. This protocol is committed before selecting new patients, reading their
payloads for this study, deriving anchors or scoring new maps. Prior technical quality auditing did
read training/validation records; no model has evaluated these real-background patients in this stream.
The original pilot, thresholds, checkpoints and failed verdict remain unchanged. All work stays local.

## Question and scope

Do the existing frozen R/S toy-trained localizers recover a precisely added, beat-timed voltage
perturbation on real ECG backgrounds? This tests a new engineering endpoint, not native disease
localization or clinical ST boundaries. No fitting, additional SSL, checkpoint selection, generator
search or GAN training is authorized by this protocol. It cannot establish that background mismatch
caused the original null SSL gain. Existing profile B is spent development evidence; the new patient
endpoint is exploratory engineering development, not independent clinical confirmation.

Native abnormalities and normal variants remain unknown. A zero intervention mask means no introduced
perturbation; it does not mean healthy tissue or absence of clinically important waveform features.
Natural morphology resembling an edit can make intervention history ambiguous from a single signal.
A detected algorithmic anchor is not an expert-verified R, J or ST landmark.

## Freeze patients before payload access

Use the immutable original patient partition and full technical input audit. Reserve 500 quality-eligible
TRAIN patients disjoint from ALL 2,048 originally selected SSL-cache patients, including any not actually
sampled during SSL. For each remaining patient choose the lowest numeric quality-eligible ECG ID.
Order by SHA-256 of `ssl_real_background_v1:train:<canonical patient ID>`; retain the first 500.
These patients are reserved only; this stage does not read their waveforms or train on them.

For VALIDATION patients, choose the lowest numeric ECG ID without filtering quality-review/exclusion
flags, sort patients by SHA-256 of `ssl_real_background_v1:validation:<patient ID>`, and take the first
64 for calibration and the next 128 for engineering evaluation. Preserve every chosen recording and
flag. No replacement after a technical, anchoring or generation failure. Record the complete selection
manifest locally before reading any selected payload. Every repeat and variant inherits its patient's
role; verify no overlap with each other or the SSL cohort. Keep original engineering-test patients,
folds 9–10, Challenge and EchoNext closed sets and all real diagnostic fields untouched.

Reverify both selected .hea/.dat hashes and canonical units/geometry against the original audit and
official table. Decode and whole-record resample 500 to 250 Hz with the shared helper. Cache exactly
192 physical-mV, 12-by-2,500 backgrounds, without waveform clipping, fitted normalization or diagnosis
filtering. Model entry retains the original fixed 2 mV scale. A hard integrity/decode failure rejects
this study rather than silently reducing the patient denominator.

## Fixed algorithmic anchors

On an anchor-only copy of the processed unedited background:

1. Apply scipy third-order Butterworth 5–25 Hz bandpass, SOS `sosfiltfilt`, along time, using defaults.
2. First-difference each lead, prepending its first sample. Divide by that lead's RMS first difference,
   floored at 1e-8 mV, square, and take the median across the twelve leads.
3. Smooth by convolution with twenty equal weights, mode `same`. Set height to median smoothed energy
   plus three times its median absolute deviation. `find_peaks` uses minimum distance 75 samples.
4. Refine each candidate within plus/minus ten samples to the earliest maximum of summed absolute
   filtered voltage. Deduplicate and sort. Keep indices from 100 through 2,375 inclusive.
5. If more than seventeen remain, keep `np.rint(np.linspace(0,n-1,17)).astype(int)` positions. Require
   at least three. Retain all counts/failures and freeze the rule without per-patient adjustment.

Anchors are computed before augmentation or editing and never enter network features. Qualification
checks deterministic replay and finite geometry; it cannot adjudicate physiological beat validity.
Failure to qualify every selected patient stops transfer scoring and is reported as a task-domain limit.

## Paired voltage edits and exact truth

Operate at the cached native 250 Hz grid; the scoring preprocessing after addition is identity.
Reuse the original continuous A raised-cosine plateau and B beta pulse functions. Start at anchor
+100 ms; width is uniform 0.14–0.19 seconds, signed amplitude uniform 0.04–0.30 mV. Focal edits choose
one retained anchor; repeated edits use all retained anchors. These names describe program timing,
not expert cardiac-wave diagnoses.

Draw one of four independent perturbation patterns: chest-only with one to three selected chest
coordinates; I-only; II-only; or I=II, with the same coefficient. Each limb-active pattern may add one
chest coordinate. Derive III, aVR, aVL, aVF from the added I/II delta with the shared algebra helper.
This changes at most six observed leads. Preserve the actual measured background; never reconstruct
or correct its original limb leads. Check algebra on the ADDED delta only, not on native physiology.

Use seeds from the first eight big-endian SHA-256 bytes of
`ssl_real_background_v1:<role>:<patient ID>` and separate SeedSequence streams 101 (edit), 303 (nuisance).
Nuisance on/off is `bool(seed & 1)` for both positive modes; controls use false and true. The enabled
nuisance stream draws scalar gain U(0.8,1.2), noise SD U(0,0.025) mV, drift amplitude U(0,0.12) mV,
frequency U(0.05,0.40) Hz and phases U(0,2pi) on eight independent coordinates, then couples the six
limb leads. Both counterparts share all nuisance values. Added-nuisance controls do not claim to be
clinically abnormal; unchanged controls do not claim normality.

Calibration patients use A, evaluation patients use B. Each supplies focal, repeated, unchanged and
added-nuisance variants. The baseline of a positive must equal its matching same-nuisance control
exactly. Save immutable parameters and processed-input hashes; regenerate inputs exactly from cached
backgrounds rather than duplicating full waveform banks. No paired baseline or parameter enters inference.

Truth is `abs(edited_float32.astype(float64)-baseline_float32.astype(float64)) > 0.005 mV` on all
lead/time cells. Candidate windows use every anchor+[0.06,0.39] on every lead. Require exact support
reconstruction, nonempty positives, zero controls, support contained in candidate, support occupancy
<=0.18, added delta peak<=0.50 mV, identical paired controls, finite geometry, and exact deterministic
replay for ALL 768 cases. Check delta-algebra rounding error against the prospective bound
`64*eps_float32*(1+max_abs_baseline+max_abs_edited)` mV; do not enforce the toy generator's absolute
full-waveform voltage or limb-identity gates on real backgrounds. Report floor/quantization residuals.
No discarded/retried cases or thresholds relaxed after qualification. Any gate failure stops scoring.

## Conditional frozen transfer and reporting

Transfer requires a successful original numerical audit/binding closure, complete new qualification,
and the counterfactual diagnostic showing positive R/S mean AP response and target-versus-unedited
candidate response with positive paired 95% lower bounds. This is a prospective admission for this
follow-up, not a new advancement rule for the closed original pilot. No results from the real endpoint
select this condition. If admission fails, report qualification only.

Use original final R/S states and the deterministic v2 output implementation. Calibrate each arm's
strict maximum-map threshold on the 64 new-domain calibration patients: maximum of paired unchanged/
nuisance controls, empirical higher 95th percentile, then flag only scores strictly above threshold.
Freeze calibration bytes before any evaluation inference. Evaluate all 128 patients, both positives
and both controls, using the same original full-grid AP, prevalence and top-1%-area tie convention.
Report contextual uniform/raw-voltage/algorithmic-anchor phase priors; the latter has privileged
algorithmic timing. Macro equally over two positive modes within patient, then over patients.

Report S−R AP and paired 95% intervals using shared whole-patient helpers, 2,000 draws, seed 20261023;
report absolute AP/modes, unchanged/nuisance flag rates and paired differences. Use existing matched
control maps to report input-response diagnostics without extra inference. Invent no pass criterion
or clinical threshold; retain failure and low coverage. No direct difference of old-domain and new-domain
AP is interpreted as a causal training-domain effect. A later domain-alignment fitting experiment would
need a separately frozen same-endpoint difference-of-differences comparison and is not launched here.

## Storage, resource and integrity gates

Keep the 192 processed backgrounds, 768 support/candidate masks and two-model probability maps.
The conservative array-payload forecast is 253,440,000 bytes, plus exact NPY headers and 64 MiB metadata
reserve. Freeze/regenerate input hashes to permit independent inference replay; do not store duplicate
edited waveform banks. Require whole-stream storage including this forecast below the original 4 GiB.
Use only aggregate reports in Git; real backgrounds, masks, patient IDs and predictions stay local.

Before scoring, bind every source/dependency/input/selection hash and verify original model states.
Profile only A-calibration inference on the actual regenerated input path, discarding probe outputs,
under the shared nonblocking GPU lock. Admit the new stage only if a 2x measured forecast plus 120
seconds reserve fits 900 GPU seconds and 1,800 CPU seconds, and cumulative original ceilings of 7,200
GPU-wall/14,400 CPU seconds and 4 GiB remain satisfied. Enforce live deadlines and one numerical/interop
thread. No GPU preemption, new packages, data downloads or remote actions. Commit policy/code before
qualification/profile/scoring; verify hashes again afterward. Retain every failed stage in new outputs.
Write executed qualification and transfer reports from the local receipts, and rank measured follow-ups.
