# Saved-map diagnostic of the factorial mode tradeoff

18 October 2026. The user authorized continuing the recommended next steps. This protocol freezes
new descriptive calculations before opening prediction/mask payloads or calculating their new
metrics. The parent factorial remains complete with all twelve guards passed; this diagnostic has
no new advancement rule and cannot revise that decision. Cardiologist recruitment is unavailable.
No native clinical label definition changes. Keep PR67 in draft.

## Question and fixed population

Characterize the output changes accompanying the random-initialized paired model's focal AP gain
and repeated-edit AP decline. Distinguish lead allocation, temporal ranking and fixed-area coverage
from baseline phase-window alignment. These are descriptions of saved model outputs, not causal
claims about internal features. Retain all128 evaluation patients, both positive modes and allfour
arms RB,SB,RP,SP. No subgroup selection, tuning, model inference, fitting or new patients.

Use the parent explicit coordinates: global positive cases2256..2767 with variant0/1, map/mask row
=globalcase−2000, local evaluation patient0..127 and matching sham/nuisance counterpart determined
by the frozen metadata. Baseline is the matching control prediction against the SAME positive
support/candidate. Compute each positive map independently; the model never receives these new
projections or the ground-truth mask as an input.

## Frozen metrics

T is exact introduced support above0.005mV; C is the algorithmic all-lead candidate mask,
U=C minusT and O=outsideC. Each must have nonempty required regions, with T a subset of C. For each
baseline and edited map report the following fourteen fields, then edited minus baseline:

1. Full-grid AP againstT, reproducing the already audited parent value.
2. Lead AP over12 leads: each lead score is mean probability in that lead's fixed candidate cells;
   lead truth is anyT cell on that lead. Require both positive and negative lead targets.
3. Temporal AP over2500 samples: score is mean probability across all12 leads; truth is anyT cell
   at that sample. Require both positive and negative temporal targets.
4. Oracle-conditioned temporal AP: calculate full2500-time AP separately for each truth-positive
   lead, then average those leads equally. This diagnostic removes unaffected leads using truth;
   it is not an inference procedure or an additive decomposition of grid AP.
5–10. Full-grid precision and recall at fixed1%,5%,10% area. Use the parent's fractional boundary-tie
   convention, with expected equal credit across tied cells, never a fitted probability threshold.
11–13. Fractions of total probability mass inT,U,O. Sum native float32 values in float64 and divide
   by the full-map sum. Require positive total mass; stop on zero rather than adding an epsilon or
   dropping a case. Fractions describe score allocation, not calibrated confidence.
14. Candidate-minus-outside mean probability, mean_C(p) minus mean_O(p), in float64. Report the
   baseline value explicitly as an output phase-window alignment diagnostic, not expert boundaries.

Store support/candidate cell counts, support prevalence, positive lead/time counts, area budgets
K and attainable recall ceilings min(1,K/|T|). These show the structural ceiling on top-area recall,
especially for broad repeated edits. They are descriptive mask geometry, with no case filtering.
Baseline and edited fields share identical truth/mass definitions; their differences are computed
per case. Uncalibrated probability magnitudes across arms do not establish confidence comparisons.

## Aggregation and interpretation

Report allfour arms separately for focal and repeated edits. Each case corresponds to one patient
within each mode. For everyfield and baseline/edited/change phase, report equal-patient means and
95% whole-patient bootstrap intervals, 2000draws, seed20261028, shared interval helpers and exact
paired patient draws. Report P−B contrasts RP−RB and SP−SB under each mode. Retain all128 patients
and report zero/skipped draws explicitly. Do not search for a favorable metric/mode or promote these
many descriptive intervals to multiplicity-adjusted confirmations.

Grid AP is not an additive function of projected lead/time AP. Even consistent projection changes
only characterize output allocation and ranking. A direct AP drop accompanied by increased response
or coverage still remains a direct AP drop. Existing parent edited AP, response, frozen thresholds
and all twelve gates must remain unchanged. Do not retune thresholds, objective weights or steps
based on this diagnosis. Any follow-up fit needs a separate prospective protocol and actual profile.

## Admission, access and immutable bindings

SHA-anchor the passed parent independent audit74b02f60ac0d77bf071bc99e3ae6fcf460fd8ef0dfba21b9626f5d67646d6c04
and main evaluation3f4555614180442c1eb82c5e5b20ca4137d53ae35697421bfa4337cf4f588d28.
Freshly bind permitted source/metadata declarations and compare the complete independently
recomputed parent scientific summary, calibration thresholds and twelve gates with its live receipt
at absolute tolerance1e−12. Existing independent reproduction plus this fresh binding is sufficient;
do not rerun predecessor inference or its entire training-state audit.

The ONLY numerical array payloads allowed are the four evaluation_v2 probability banks and the
factorial preparation support/candidate masks. New calculations use evaluation rows256..767 only;
whole permitted arrays may be hashed for integrity. Parent cases/calibration/config/receipts and
source text are allowed. ALL raw/processed ECG waveform payloads and checkpoints, including hashing,
remain closed. Unused ancestor payload identities depend on explicit SHA-anchored passed-audit
catalogs, with every exception recorded. Missing/conflicting authority or a changed digest stops
admission. Check identities before and after calculation. No generic verifier may silently open a
forbidden payload. No clinical fields, ages, protected tests, downloads, GPU or external APIs.

## Resources and completion

New output directory is
`outputs/ssl_synthetic_localization/counterfactual_factorial_v1/mode_diagnostic_v1`.
Refuse overwrite and retain all failed attempts with exact costs. Main CPU AND monotonic wall limit
is120seconds with2seconds reserved for failure reporting. Independent audit has its own120seconds;
all main/verification attempts share240seconds and16MiB outputs. The original whole-stream4GiB and
cumulative14400CPU/7200GPU-wall ceilings remain. Current exact prior CPU745.8142945699988 and GPU
382.2241592870014 are derived from the SHA-bound parent audit; never compare rounded report literals.
CPU-only work uses one numerical thread and CUDA hidden. GPU-wall remains unchanged.

Freeze/root-integrate/review new source before the main run. Use artificial tests for real helper
signatures, fractional ties, projections, oracle conditioning, coordinates, paired aggregation,
forbidden reads, zero mass and truthful failure retention. Independently recompute the fourteen
fields, geometry and selected paired aggregates using separate formulas/shared bootstrap helpers;
bind exact verification source before one audit run. Scientific tolerance1e−12, accounting1e−9.
The runner writes an executed results report from local outputs, with both queue documents and
ranked follow-ups updated. Source and aggregate reports may enter Git; per-case rows/maps/identifiers
remain local. No new training or clinical claim follows automatically.
