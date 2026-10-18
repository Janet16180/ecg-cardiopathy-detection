# Controlled ST intervention generator v1: prospective qualification

18 October 2026. Commit this policy before the design qualification. This is a new, independently
implemented pure simulation and intervention-truth check, not an Experiment 058 reproduction,
model comparison, disease simulation or clinical validation. It reads no real ECG, header or label.
Experiment 058 at `8902c1306d090e6fd8d338a47bab5d45965ef52d` motivates controls; its missing local
artifacts prevent exact numerical reproduction. No predecessor sources are copied.

## Frozen generation and truth

Each latent background is a deterministic ten-second, 500 Hz simulated ECG with known beat centers.
Independent random substreams set background morphology/rate, intervention and nuisance. Beats have
analytic P/Q/R/S/T components with randomized amplitude and width. The independent coordinates are
I, II and V1–V6. All limb leads, including every intervention and nuisance, obey
III=II-I, aVR=-(I+II)/2, aVL=I-II/2 and aVF=II-I/2, in canonical I, II, III, aVR, aVL, aVF, V1–V6 order.
An edit changes one or both primitive limb coordinates and/or one to three chest coordinates; the
reported truth retains every induced limb change. Values are mV, without clipping, per-record
normalization or a target-dependent preprocessing branch.

Profile A uses independent cosine entrance/exit ramps with a plateau. Profile B uses a compact
quartic beta pulse, 16u²(1-u)². B is qualified here as generator design, then excluded from all
training, validation, threshold selection and model selection. The same latent/nuisance identity
and intervention parameters can be rendered by either mechanism. A focal edit affects one eligible
beat; a persistent edit affects every eligible beat. A random sign and 0.04–0.30 mV peak are drawn
independently of location; width is 0.14–0.22 s, beginning 0.10 s after the known R center. Rates cover
45–110 bpm. This controlled timing does not assign a disease diagnosis to an arbitrary location.

Noise (0–0.025 mV SD), drift (0–0.12 mV amplitude, 0.05–0.40 Hz), phase, and shared gain (0.8–1.2)
are independent of edit choice and shared by both members of a pair. Independent primitive-coordinate
noise is coupled into limb leads. No-target/sham pairs preserve this same nuisance and have zero mask;
nuisance-only controls compare an observed nuisance-containing signal with its identical paired copy.
They are not compared with a different clean background to create intervention labels.

Scoring supports 500 Hz unchanged and 250 Hz whole-record `resample_full(...,1,2)` with its default
polyphase FIR. Both paired observations use the identical preprocessing. No filter is fit per example.
The exact support is **abs(edited.astype(float64)-baseline.astype(float64)) > 0.005 mV**, strictly,
after this preprocessing; the frozen threshold is never estimated from an example. Save the actual
postprocessing difference, not a nominal rectangular window. All lead-time cells are a valid scoring
domain. Also report the all-lead candidate domain within R+[0.06,0.39] s for every eligible beat,
which includes the finite FIR tails. Report positive prevalence both over all cells and conditionally
over this candidate domain, plus candidate-domain fraction. Outside support can contain subfloor
change; it must be no greater than 0.005 mV. Report subfloor changed-cell fraction and energy fraction,
maximum outside-candidate change and extension beyond the nominal edit windows.

## Committed design grid and decisions

The full run has seed IDs 0–63 for every profile A/B, focal/persistent mode and scoring rate 250/500 Hz:
512 positive pairs. It also has the same 64 seeds at both rates for each of sham (no target/no nuisance)
and nuisance-only (no target/shared nuisance): 256 negative pairs. This is a systematic design grid,
not independent patients or model evaluation. No seeds are retried or dropped. No GPU is used;
OPENBLAS/OMP/MKL/NUMEXPR threads are one and CUDA is hidden. Record wall time and source/dependency hashes.
Examples and their parameters stay local; the receipt contains aggregate diagnostics only.

All gates must pass; otherwise retain the failed receipt and report failure before a forward version.

- Deterministic replay matches all arrays and metadata exactly. Changing nuisance substream leaves
  latent/intervention parameters unchanged; changing edit substream leaves background/nuisance unchanged.
  Toggling the target leaves nuisance parameters and the baseline unchanged.
- Every positive has finite [12,5000] raw and [12,5000/2500] scoring arrays, matching actual support and
  nonempty support contained in the candidate domain. Every negative has identical pairs and zero support.
  Candidate domain is not an intervention-location marker supplied to the model.
- Limb algebra error is at most 1e-6 mV after either preprocessing. Absolute waveform magnitude is less
  than 10 mV; generated edit magnitude is at most 0.50 mV. This is engineering unit/scale accounting.
- Whole-record support occupancy is at most 0.18 for persistent and 0.03 for focal pairs. Conditional
  candidate occupancy is reported, not optimized. Positive-mask prevalence is the chance AP baseline.
- Profile endpoints are exactly zero with zero analytic endpoint derivative. At raw 500 Hz every
  affected lead's maximum discrete edit slope divided by its own peak is at most 80 per second;
  first/last nonzero transition slopes are at most 10 per second. Check every affected coupled lead.
  Apply the same maximum-slope bound at the scoring rate; FIR seams must not create abrupt transitions.
- At most 1% of change energy lies below the support floor. Maximum magnitude outside the candidate
  domain is at most 1e-6 mV. Report all finite-support resampling tails rather than ignoring them.
- Expected-invalid controls must be rejected: a whole-record mask fails occupancy/exact support;
  an empty mask fails exact support; a hard rectangular intervention fails the derivative bound;
  a corrupted derived limb lead fails the algebra bound; unpaired nuisance fails support fidelity.

The receipt pins this policy, the new generator/qualification sources and tests, shared resampler,
lead constants and path helpers, dependency versions, source commit, command and configuration.
The runner refuses to overwrite an existing receipt. Commit implementation and tests before the
full run. Write a results report directly from the receipt even if qualification fails. A pass
admits this engineering generator only; training still needs the other input/protocol/GPU gates.
