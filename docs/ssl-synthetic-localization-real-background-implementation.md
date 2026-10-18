# Real-background qualification implementation choices

18 October 2026. Freeze these details before qualification and selected raw payload access. The
original study `7843be6` and its authoritative clarification `7378493` govern this implementation.
No model inference, fitting or reserved TRAIN waveform access occurs in qualification.

Selection stores the complete 692 original rows and audit flags before opening a selected header.
TRAIN reservation hashes use role `train`; case seeds use roles `calibration` and `evaluation`.
Hash strings use canonical patient IDs, UTF-8, SHA-256 hexadecimal ordering and numeric ID tie-breaks.
The 500 TRAIN reservations use each remaining patient's lowest numeric quality-eligible ECG ID and
exclude every patient in all 2,048 original SSL selections. Validation uses each patient's lowest
numeric ECG ID without quality filtering, first 64 for calibration and next 128 for evaluation.
Cache indices/subject indices are 0–63 for calibration and 64–191 for evaluation; role_subject_index
is separately zero-based within its role. Every patient's four cases remain adjacent in variant
order `focal`, `repeated`, `sham`, `nuisance`. Controls have mode `none`; their kinds are sham/nuisance.
All patient identities and parameters remain local.

Anchoring is exactly the study's float64 SOS-filter/difference/RMS/median/smoothing algorithm at 250 Hz.
RMS is sqrt(mean(diff**2)) per lead; the first difference is zero because the first sample is prepended.
Refinement windows include both +/-10 endpoints and use the earliest argmax. Retained indices are
100 through 2,375 inclusive. Candidate masks include integer samples anchor+15 through anchor+97
on every lead, equivalent to the prescribed [0.06,0.39]-second windows. No anchor, record or case is
replaced after a failure.

For a patient seed, edit RNG is default_rng(SeedSequence([seed,101])). Draw in this exact order:
pattern integer in [0,4); chest count integer in [1,4) only for chest-only (pattern zero), otherwise
one chest; uniformly select those chest coordinates without replacement from primitive indices 2–7;
magnitude U(0.04,0.30); sign uniformly from [-1,+1]; width U(0.14,0.19); and focal anchor position
integer in [0,number_of_anchors). Pattern 1 activates I, pattern 2 II, pattern 3 I=II. All active
primitive coordinates share the same signed amplitude. Repeated mode uses every retained anchor.
One common draw supplies both positive modes and all label-only control parameters. Pulse time is
`arange(2500)/250`, start `anchor/250+0.10`; no interpolation, filter or resampling follows addition.

Nuisance RNG is default_rng(SeedSequence([seed,303])). Always draw, in order, gain U(0.8,1.2), noise SD
U(0,0.025), drift amplitude U(0,0.12), frequency U(0.05,0.40), eight phases U(0,2pi), then a native
(8,2500) normal array at that SD. Coupled disturbance is normal noise plus per-coordinate sinusoidal
drift, expanded through the shared limb algebra. Disabled nuisance applies gain one and zero disturbance;
the drawn metadata is retained but not applied. Enabled positive nuisance is bool(seed&1); controls
are disabled/enabled respectively. Common parameters/noise match across every patient's variants.

With float64 cached-background copy x, coupled analytic edit delta and shared gain g/disturbance n,
compute baseline=float32(g*x+n), edited=float32(g*(x+delta)+n). Each observation is independently
rounded once. A rounded baseline NEVER feeds an edit. Preserve native measured lead relationships;
only the added analytic/observed deltas receive limb-algebra checks. Actual truth is the strict 0.005 mV
floor on float64 subtraction of the final float32 observations. Test observed algebra/quantization
against 64*eps32*(1+max_abs_baseline+max_abs_edited); test analytic g*delta against
64*eps64*(1+max_abs_analytic_delta). No full-waveform amplitude/algebra gate is imported.

Array hashes are SHA-256 of contiguous C-order bytes; dtype and exact shape are separately pinned.
Freeze every case parameter and background-input hash in cases.json, with baseline/edited/support/
candidate hashes from qualification. A second render must reproduce every array byte for all 768
cases. Positive baselines must equal their matching shared-nuisance control byte-for-byte. Save only
192 processed backgrounds plus support/candidate masks, never duplicated baseline/edited banks.
Failure placeholders remain locally with explicit reasons and block the entire qualification gate.

The exact seven eventual array headers (background/support/candidate plus four R/S role probability
files) are included with 253,440,000 payload bytes and 64 MiB metadata reserve in storage admission.
Use one numerical CPU thread, hidden CUDA, a new output directory and 1,800-second stage CPU ceiling;
retain input/source/output hashes, all failures, measured costs and cumulative original resource gates.
Conditional future transfer separately requires the four B/all R/S counterfactual lower-bound gates
in the clarification. Qualification never grants model inference or training admission by itself.
