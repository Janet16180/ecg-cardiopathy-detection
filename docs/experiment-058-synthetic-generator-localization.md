# Experiment 058: paired raw generator qualification and frozen localization stress tests

Prospective protocol, 2 October 2026. This study qualifies a reproducible raw ECG generator and tests
the frozen 056 aligned residual against frozen U_B. It does not train a new localizer or classifier.
No synthetic result establishes clinical accuracy or replaces independent expert validation.

## Established source and implementation

Use the published ECGSYN equations of McSharry, Clifford, Tarassenko and Smith (IEEE TBME 50(3),
289–294, 2003), with the [official ECGSYN 1.0.0 resource](https://physionet.org/content/ecgsyn/1.0.0/)
as algorithm documentation. The official archive is local at `third_party/ecgsyn058/ecgsyn.tar.gz`,
SHA-256 `8775dd65e42d46fb25bfc672873079771405287c1519631e0b02eea4d307f720`, with original licenses
preserved. The official C source lacks its nonredistributable Numerical Recipes dependencies; its
32-bit Linux executable lacks the local loader, and Java/Octave are not installed. No source is copied
into the repository. New code independently implements the published mathematical model with NumPy
and SciPy. ECGSIM's official downloads are desktop Windows/macOS builds and are not used.

The first generator family, `ecgsyn`, uses the ECGSYN limit-cycle phase and five Gaussian-derivative
P/Q/R/S/T forcing terms. Default angles are [-70,-15,0,15,100] degrees, amplitudes
[1.2,-5,30,-7.5,0.75], and widths [.25,.1,.1,.1,.4]. Apply the official heart-rate scaling: widths
multiply sqrt(HR/60), P/T angles multiply the square root of that factor, and Q/S angles multiply that
factor. The RR spectrum has Gaussian peaks at .1 and .25 Hz, width .01 Hz, LF/HF .5; generate 128
one-Hz values with random phases, normalize to mean 60/HR and standard deviation 60*HRstd/HR².
Polyphase-interpolate RR to 500 Hz and hold its value over each latent cycle.

Numerics differ explicitly from the official Matlab ODE solver: integrate the unit-cycle angle
directly, and the linear relaxation z' = forcing - z using an exponential recurrence with averaged
adjacent forcing samples. Wave components start at zero; a separate unchanged background carries
the official .04 initial state and respiratory baseline .005*sin(2*pi*.25*t), projected along the
mean component direction. Thus inverting the T source cannot invert respiratory or initial-state
background. Render sixteen seconds
at 500 Hz and discard four seconds of burn-in, yielding twelve-second traces. Save latent anchors for
generator audit and interventions only; localization receives no latent phase or anchors.

The second family, `compact`, is an explicitly separate engineering control. On the same phase and
RR construction, render finite-support Gaussian pulses with cosine taper at three widths, signed
amplitudes [.12,-.13,1,-.2,.24]. It is not ECGSYN or a validated physiological model. Never pool the
two generator families to hide a failure.

## Coupled lead projection and subjects

Both source families use the same fixed five component directions, virtual RA/LA/LL and six chest
electrode positions committed in `ecg_experiment/synthetic_generator058.py`. Each electrode potential
is the dot product with its inverse-cube distance vector in a homogeneous dipole construction.
Randomize heart orientation and electrode coordinates; derive I/II/III and augmented leads
algebraically and chest leads relative to the Wilson terminal. This is a simplified engineering
projection, not a validated heart/torso geometry or disease model. Normalize the clean Lead-II
peak-to-peak amplitude once to 1.2 mV and reuse exactly that scale on its interventions. Add paired
electrode noise with standard deviation .003 mV before lead algebra.

Use seed 58058 and independently spawned streams [seed,subject,1] for parameters, 2 for RR, 3 for
paired background noise, 4 for focal beat selection and 5 for extra-noise control. Subject IDs 0–199
are calibration; 200–399 are IID tests; 400–599 are shifted-family tests. Each subject is rendered
once per generator family and cached for all edits. No test subject is replaced or selected by score.

Calibration and IID heart rates are uniform 55–85 bpm, orientation Euler xyz uniform -20 to +20
degrees, width multiplier .9–1.1, per-component amplitude .85–1.15, electrode-coordinate jitter
-.025 to +.025, and HR standard deviation 1–3 bpm. Shifted subjects use equal random choices between
HR 45–55 or 85–105, width .8–.9 or 1.1–1.2, and independent signed orientation magnitudes 20–35
degrees on each axis. Other distributions are identical. Calibration is normal-only; test clean
counterparts and programmed changes never fit a reference or scoring transformation.

## Paired interventions and observable targets

Apply interventions to latent sources before lead projection. Focal edits use one seeded latent beat
between one and eleven seconds, selected before inference. Preserve RR, geometry, noise and scale.

- `qrs`: stretch the Q/QRS source components Q/R/S by 1.5 around that beat, crossfaded with a compact
  raised-cosine pulse from R-.12 to R+.12 seconds.
- `st`: add a source-direction [.3,1,.25] raised-cosine pulse from R+.10 to R+.24 seconds. Its source
  amplitude is chosen so the maximum observable lead effect is .15 mV at the pulse maximum, using
  only generator geometry and the fixed clean scale.
- `t`: subtract twice the T source, tapered from R+.08 to R+.48 seconds.
- `persistent_st`: the same ST change on every latent beat in the trace, including edge beats.
- `persistent_t`: invert the T component across the entire trace, leaving no normal reference beat.

Ground truth comes from the actual noise-free paired lead difference, not desired wave tags or
ECGSYN peak labels. Save its squared energy D and mask abs(difference)>=.01 mV. A lead is affected
only if at least twenty milliseconds exceed that floor; other leads' masks are cleared. If no
observable target remains, count the case as uninformative and give zero localization credit, without
replacement or exclusion. Noise/artifact differences never become target masks.

Negative controls are processed identically: sham has no source alteration; drift adds virtual
electrode .15+.05*sin(2*pi*.2*t), with chest multipliers [.8,.88,.96,1.04,1.12,1.2]; common gain
multiplies all observed leads by 1.1; extra electrode noise has standard deviation .02 mV. These are
specified engineering nuisances, not claims that every gain or noise change is clinically benign.
Shifted clean morphology includes broader normal QRS and serves as an additional morphology control.

## Frozen inference, endpoints and decision

Load the saved 056 U_B normal reference; do not refit it on simulations. Use exact frozen
`score_signal` from `incart_localization056`, which detects waveform-only R peaks independently on each
trace and constructs aligned residual and U_B on identical shared support. No clean counterpart,
latent phase, programmed target, intervention name or support mask enters inference. Classifier
weights and diagnosis labels remain frozen.

Use all shared 140 ms / 10 ms lead-time candidates. Top localization averages over all maximal ties
with isclose(rtol=1e-10,atol=1e-10); a hit requires the selected lead and window center to be inside the
observable mask. Report each generator × test family × intervention separately, with whole-subject
2,000-draw intervals, seed 58058. No usable map means zero hit and zero captured energy; retain that
case and report the scorer failure. Secondary endpoints are exact-mask IoU, changed energy captured
in the single fixed-area highlight, distance to the nearest observable support and tied-top counts.
Persistent families remain distinct; template-subtraction failures are findings, not grounds for
changing the algorithm.

For every case, report uniform eligible-candidate chance and full observable-mask occupancy. Wide
persistent support can make raw hit easy; neither support occupancy nor chance is hidden by the
80% illustrative threshold.

For each generator and method, fit a 95th-percentile record-max threshold on its 200 unchanged
calibration subjects, then freeze it before all test scores. An unscorable calibration record has
record maximum zero and is counted. Report test-sham false marks and paired nuisance-minus-sham
flag differences with whole-subject intervals.

Generator qualification requires bit-identical processed shams (maximum absolute difference zero),
limb algebra errors below 1e-10, and masks exactly reconstructed from saved clean differences. These
must pass before interpreting localization. No zero-observable case is removed. A useful provisional
synthetic robustness improvement requires, separately in BOTH generators and BOTH test distributions:
mean paired transient QRS/ST/T hit gain >=.10 with lower interval >0; each transient kind's hit >=.80;
and every nuisance flag increase upper interval <=.05. Report persistent ST/T against these same
thresholds separately without letting transient gains conceal them. For transient gain, average the
three kinds within each subject first, then bootstrap the 200 independent subjects in that generator
and test distribution. Repeated edits or generator renders never count as independent subjects.
No result promotes a clinical localizer. Any future simulator-developed localizer must itself beat
its frozen comparator on an independent real expert endpoint; separate 059/060 studies answer
different questions and cannot validate this simulator or imply its clinical transfer.

## Integrity and execution

Before new scores, hash-check and exactly reproduce the full 056 benchmark dictionary from saved
patient tables, nuisance fields from paired rows, thresholds from normal scores and coverage counts.
Verify its source/output hashes and U_B reconstruction receipt. Also reproduce 057's primary and
stress summaries from live tables using its shared library, and verify its source/input/output
receipt hashes. Do not import scripts, access closed data or alter pinned predecessor files.

New source modules, runner and meaningful tests are committed before execution. Test limb identities,
paired sham identity, exact target reconstruction, deterministic subjects, no counterfactual access,
and persistent intervention coverage. CPU only, two numerical threads, no GPU or dependency change.
Forecast pure generator runtime before launching scores; cache each clean subject/family only once.
Keep official sources, raw simulations and plots local. Output identity records relative paths via
`paths.py`, source/source-archive/input hashes, all seeds, equations, simulator parameters, paired
traces and masks, candidates/maps, thresholds, per-case tables and all negative outcomes. Write through
`outputs/experiment058_synthetic_generator_v1.partial/` and rename on success, refusing overwrite.
The executor writes `docs/experiment-058-synthetic-generator-localization-results.md` from final live
outputs. Parent owns queue/backlog and the existing PR; no new PR is created.
