# Pulse2Pulse background and intervention qualification

19 October 2026. Prospective protocol; no results are claimed. The companion
`configs/ssl_synthetic_generated_quality_v1.json` fixes all numerical choices. This study asks whether
one publicly released generator produces a reproducible bank with measurable temporal structure and
whether that bank can carry numerically verified, controlled lead-time interventions. It neither
establishes disease localization nor compares GANs with diffusion models. No training is authorized.

## Admission and provenance

Run only after the standalone Pulse2Pulse six-forward CPU probe passes and its source-bound report is
reviewed. Reuse its exact verified weights and reviewed architecture; do not download another model,
change packages, call an external inference service, or read any real ECG. Replay all four predecessor
CPU inputs through the same CPU float32 path, requiring exact raw, integer-export and canonical-output
equality before new generation. The old factorial evidence is historical: missing original outputs
prevent live reproduction on this computer. Do not invent a new audit or combine reported old costs
with measured current-machine costs. This standalone successor reproduces its available immediate
predecessor and records the older evidence limitation.

No patient split is necessary because no fitting, threshold selection or patient inference occurs.
The synthetic seeds are the resampling units; they are not patients. Public checkpoint provenance,
unknown training-record overlap and unknown physical calibration remain unresolved. No clinical
labels or clinician recruitment are needed for this engineering study. Weights and signal arrays
remain local. Generated traces are not proof of privacy or freedom from memorization.

## Fixed generation and actual GPU profile

Generate 128 native eight-lead, 5,000-sample outputs using seeds 2026103000 through 2026103127, one
sample at a time. Construct noise with the CPU torch.Generator and exact uniform[-1,1] convention
of the admitted probe, then transfer that noise to CUDA. The model is in evaluation/inference mode,
float32; deterministic operations are required. Do not replace seeds or exclude awkward outputs.
No transformations, examples or gates may be chosen from the resulting metrics.

After CPU predecessor closure, acquire `ecg_experiment.gpu`'s shared lock and profile one disposable
first-seed GPU forward followed by the complete CPU standardization, interventions, masks and metric
path. Synchronize CUDA around timing. Require at least 2 GiB free device memory at admission and
peak allocated memory strictly below 2 GiB; leave unrelated processes untouched. Record driver,
GPU, PyTorch, source hashes and actual timings. Profile feasibility requires spent wall plus twice
profile full-path wall multiplied by 130, plus 30 seconds reporting reserve, below 1,200 seconds;
apply the analogous full-path CPU forecast against 1,800 seconds. Also forecast saved bank/metadata
bytes against 512 MiB. Failure ends this version with its receipt; no device or batch-size search.

The main bank has 128 forwards and one exact first-seed replay on the identical CUDA path. Together
with one profile forward and four CPU predecessor replays, these are all authorized forwards. Save
all bank seeds and raw outputs. CPU-versus-GPU differences may be reported descriptively if already
available for an identical input, but equality across device types is not a pass condition. No new
forward may be added for that comparison. Exact CUDA replay is a gate. The discarded profile sample
is charged but not included in the bank or its denominator.

## Dimensionless representation and intervention truth

For each independent lead I, II, V1–V6, compute the float64 median and linear-quantile q95−q05 from
its 5,000 raw values. Require scale >1e-6, subtract the median, divide by that scale, and cast to
float32. Retain both raw data and all centers/scales. This deliberately changes between-lead amplitude
ratios; it is a dimensionless engineering representation, not a calibrated ECG. Derive III=II−I,
aVR=−(I+II)/2, aVL=I−II/2, aVF=II−I/2 from the transformed independent leads, and store canonical
I, II, III, aVR, aVL, aVF, V1–V6 order. Verify all variants satisfy the same float32 limb equations
with atol=2e-6, rtol=1e-6. Algebraic closure is construction integrity, not physiological validation.

Each bank member has four variants:

1. Sham: exact unedited canonical array, with zero intervention mask.
2. Nuisance: add the common signal 0.10*sin(2*pi*0.33*t+phase) to all eight independent leads,
   with t=arange(5000)/500 and phase from a fresh NumPy default_rng(2026103100+i).uniform(0,2*pi).
   Derive all twelve leads anew. Its intervention mask is zero because the target is the local offset.
3. Focal: start from the nuisance independent-lead array and add 0.5*np.hanning(81) in lead II,
   center sample 1250, endpoints center±40 inclusive; derive all twelve leads anew.
4. Repeated: start from the same nuisance array and apply the same bump at centers
   750,1250,1750,2250,2750,3250,3750,4250; derive all twelve leads anew.

Use float32 arrays for final observed and baseline signals, and evaluate their numerical difference
in float64. For each positive variant, save strict nonzero support (difference !=0) and primary
thresholded support (abs(difference)>0.01). These are masks of the introduced change, including
changes propagated to derived limb leads. They are not abnormality masks or waveform annotations.
Require both masks nonempty, thresholded support a subset of strict support, and no strict changes
outside the constructed temporal windows and affected lead hull. Require at least 99% of squared
intervention energy retained by thresholded support. Report below-threshold energy and maximum
residual rather than calling the thresholded mask complete numerical support.

Sham closure is exact. Nuisance has zero target mask despite a nonzero waveform change. Fixed timing
and lead are intentional qualification fixtures: a position-only prior could identify these masks.
Consequently these inputs must not be used as a localization learning benchmark, training set or
proof of usefulness for a learned localizer. A separately frozen successor would vary positions,
leads, signs, amplitudes and edit families before any model comparison. The smooth offsets here do
not assert ST elevation, ischemia or any other physiological mechanism.

## Primary comparison and descriptive quality

The primary engineering comparison is native minus circular-shift-control cross-lead activity
coincidence. Use only the eight independent standardized leads, not algebraically derived leads.
For each lead, take the float64 first difference with the initial sample prepended (first difference
zero), square it, and convolve with np.ones(20)/20, mode='same'. Define activity by strict greater
than its linear 90th percentile. For all 28 lead pairs, calculate intersection/union Jaccard of
these binary masks, defining zero-union Jaccard as zero; average the 28 values per bank member.
The control shifts the completed masks independently using
`default_rng(2026103200+i).integers(1,5000,size=8)` and np.roll, then repeats the same pair calculation.
This keeps each lead's activity pattern and rate while disrupting its alignment with other leads.

Report the 128-member paired mean difference and a 95% percentile interval using 2,000 whole-seed
resamples, seed 2026103301. Reuse `patient_groups` and `patient_resample` as generic grouped-bootstrap
helpers; label the result seed-bootstrap, never patient-bootstrap. No single-class filtering applies;
report zero skipped draws. A positive result indicates aligned derivative-energy activity relative
to this shift control, not clinical fidelity, true QRS detection or superiority to real recordings.
The contrast is descriptive, without a clinical quality or advancement threshold.

Also report nonfinite/flat/duplicate counts, leadwise scales, native saturation fraction
(abs(raw)>=0.99), RMS and standard-deviation summaries, activity occupancy and power fractions.
Use a float64 SciPy periodogram, fs=500, boxcar window, detrend='constant', scaling='density';
sum bins in [0.5,100) for the denominator and half-open [0.5,5),[5,25),[25,100) bands. Report a zero
power denominator explicitly. These scale-free summaries are not normality thresholds or waveform
boundary validation. Report diversity through exact raw-output hashes and descriptive pairwise
RMS distances after standardization, with no near-duplicate cutoff or memorization claim.

For each intervention mode separately, report support occupancy, strict-support occupancy, retained
energy, and full-grid AP using absolute observed amplitude versus the thresholded mask. Compare
with paired absolute-difference AP and mask prevalence. Paired-difference AP is a truth-construction
oracle, expected to be one, and has unavailable-at-inference information; it is not an eligible
localizer. Amplitude AP describes overlap/difficulty for this fixture, not screening performance.
Do not pool away focal/repeated differences or compare these numbers with historical physical-mV AP.

Save figures for the first eight fixed seeds, unselected: native/standardized traces with units
clearly identified; nuisance/focal/repeated overlay with support; a bank-wide coherence paired plot;
and distributions of power ratios and scales. Prefer PNG/SVG artifacts generated by ordinary plotting
code. No example selected after metric inspection may replace the first eight.

## Operational decision and accounting

Operational PASS requires predecessor closure, verified model/source identity, finite full geometry,
nonflat native and standardized leads, 128 distinct raw output hashes, exact same-device replay,
all limb-equation checks, sham closure, valid nonempty supports, no changes outside the constructed
hull, energy retention and resource compliance. Coherence, spectra, amplitude AP and visual appearance
do not participate in this conjunction. A PASS means the bank and intervention mechanism are usable
for further engineering design, not that the generator is clinically qualified.

New output path: `outputs/ssl_synthetic_localization/generated_quality_v1`. Maximum 512 MiB including
noise, raw/native/transformed arrays, four variants, masks, figures, metrics, partial artifacts and
receipts, with 1 MiB final-receipt reserve. Reused predecessor assets count in the separate predecessor
ledger and are not copied. Whole-stage monotonic wall limit is 1,200 seconds, CPU limit 1,800 seconds,
and charged GPU-lock wall limit 1,200 seconds. Use one numerical and interop thread. Start measurement
before admission/replays, charge profile and replay once, preserve every failure and forbid overwrite.
A child deadline or supervisor must preserve a failure receipt if execution cannot finish. No automatic
retry. Missing legacy artifacts/costs are unknown, not zero; report present-machine storage separately
from historical reported storage and do not assert compatibility with the old stream's 4 GiB ceiling.

The executing coordinator writes a results report from actual saved outputs, including source hashes,
primary comparison, every operational gate, fixed figures, resource use, failures and limitations.
An independent reviewer checks code before execution and saved arrays/receipts afterward. Update both
queue documents and the ranked backlog. Follow-up candidates are varied-support qualification,
public expert-wave-boundary datasets, and a separately profiled localization design; no follow-up fit
or clinical label change is automatically authorized by this protocol.
