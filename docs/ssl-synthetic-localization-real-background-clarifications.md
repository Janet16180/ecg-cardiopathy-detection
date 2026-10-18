# Implementation clarifications before real-background qualification

18 October 2026. Addendum to the [prospective study](ssl-synthetic-localization-real-background-study.md),
before new real-background qualification or model scoring. These details make its already stated
conditional admission and rendering rules executable. The existing-map diagnostic has now completed;
its original pilot verdict remains failed, and no new real endpoint selected these rules.

Follow-up admission uses ONLY profile B, the all-mode subject macro, separately for R and S:
`average_precision_change.ci_low > 0` and `selectivity_contrast.ci_low > 0` for each arm. All four
conditions must hold. There is no selection between A/B or positive modes and no new advancement
rule for the original pilot.

The four perturbation patterns are equiprobable. Chest-only uses an equiprobable count of one, two
or three chest coordinates sampled uniformly without replacement. Every limb-active pattern adds
EXACTLY one uniformly selected chest coordinate. Magnitude is uniform in [0.04,0.30] mV, sign is an
independent equiprobable plus/minus, and all selected independent coordinates share that signed
coefficient. In the I=II pattern both are exactly equal. Width is uniform in [0.14,0.19] seconds.
These are engineering perturbation parameters, not disease territories or clinical severity.

For a float64 working copy x of the cached float32 background and authored coupled pulse delta,
let g be the common nuisance gain and n the common coupled native-grid additive noise/drift:

- observed baseline = float32(g*x + n);
- observed edited = float32(g*(x + delta) + n).

All arithmetic before these final casts is float64; each observation is rounded to float32 once.
With nuisance disabled g=1 and n=0. Gain multiplies the background AND intervention; noise/drift is
added afterward identically to both counterparts. The cache is never modified. Actual support and
delta-algebra checks use float64 subtraction of those final float32 observations. The analytic delta
g*delta is separately checked at float64 roundoff. No waveform clipping or normalization is added.

The continuous occupancy estimate is not a discrete guarantee. Every actual native mask must still
pass <=0.18 occupancy. Any failure remains a study-level qualification stop with no patient replacement
or parameter adjustment. Both original and additive policies are source-bound by the new receipts.
