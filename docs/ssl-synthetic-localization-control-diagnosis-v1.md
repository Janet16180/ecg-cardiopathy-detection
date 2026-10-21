# Saved-map diagnosis of background hotspots

19 October 2026. Prospective descriptive follow-up to the audited randomized-support learning study.
Its low absolute AP and frequent control maxima above 0.5 motivate examining where its largest
predictions occur. Freeze this protocol, companion config and implementation before computing new
diagnostic metrics. The earlier twelve-gate PASS and all historical failures remain unchanged.

## Evidence and access

Use only saved outputs below `outputs/ssl_synthetic_localization/randomized_localization_v1`.
The config pins the main summary/ledger, independent audit/ledger, `observed.npy`, `support.npy`,
`spatial_prior.npy`, and both arms' `evaluation_probabilities.npy` and `evaluation_metrics.json`.
Whole-file byte hashes of declared arrays are allowed for identity, including files containing
TRAIN rows. Numeric access to observed/support arrays is restricted to global indices 80–127.
The spatial prior is an existing TRAIN aggregate; do not reopen numeric TRAIN rows or refit it.
Read no checkpoints, native generator arrays, real ECGs, protected data or other output trees.
There is no model forward, optimizer update, download, GPU use or external inference call.

These 48 generator seeds are spent engineering development data, with supervised/permuted arms and
sham/nuisance/focal/repeated variants. Before any new diagnostic, exactly reproduce the stored
per-case AP, amplitude/prior AP, nuisance-site AP, paired responses, control summaries and full
primary/mode contrasts, bootstrap intervals and scientific gates. Production may reuse the frozen
learning metric helpers. Require exact stored numerical/Boolean agreement; a mismatch stops the
attempt with a failure receipt. Independent checking may use separately implemented arithmetic with
a declared 1e-12 absolute tolerance for floating metrics and exact integer/Boolean checks.

## Frozen hotspot and activity definitions

Each map has 60,000 cells in canonical lead order and C flatten order. Select exactly 600 hotspots
by descending stored probability, breaking ties by ascending flat cell index:
`np.argsort(-probability.ravel(order='C'), kind='stable')[:600]`. Do not tune the area or probability
cutoff. Record the cutoff value, counts strictly above/equal to it, and the number of selected tied
cells. Stable tie selection can create apparent lead/time preferences, especially for the spatial
prior, and must be disclosed. A hotspot is a top-ranked model cell, not an abnormality annotation.

For each saved observed variant, independently in all twelve canonical leads, compute float64 first
differences with the first sample prepended (initial difference zero), square, and convolve with
`ones(20)/20`, mode='same'. Activity is strictly greater than that lead's linear 90th percentile.
Retain per-lead occupancy; ties can make it less than 10%, including zero. This is a derivative-energy
proxy, not expert QRS/P/T delineation. Derived leads are included because the prediction grid includes
them; no cross-lead physiological-independence claim is made.

For global background index i, draw one shift using
`int(default_rng(2026103600+i).integers(500,4501))`. Roll the activity mask along time by this common
shift for all twelve leads. Use the identical shift for every variant and both arms of that seed.
This preserves within-mask cross-lead timing and per-lead occupancy while displacing it relative to
the saved prediction. It is one fixed alignment reference, not a null distribution or permutation
significance test. No alternative shifts are tried after seeing a result.

## Primary and secondary quantities

For each arm, seed and control variant c in {sham,nuisance}, define native overlap
`sum(H_c & A_c)/600` and shifted overlap `sum(H_c & roll(A_c,shift,axis=1))/600`.
The arm's per-seed primary value is the equal mean of the two control native-minus-shifted differences.
Report three estimates: mean supervised value, mean permuted value, and mean paired
supervised-minus-permuted value. Use 2,000 whole-seed resamples, default_rng(2026103701), with identical
48-seed index draws for all three estimates; use shared `patient_groups`/`patient_resample` helpers,
linear 2.5/97.5 percentile bounds, zero skipped draws. Label these seed-bootstrap intervals conditional
on the fixed fits, proxy and spent backgrounds. A negative or near-zero contrast remains informative.

Retain per-seed values and report secondary means by arm and variant, without additional intervals:

- Sham/nuisance hotspot persistence: `sum(H_sham & H_nuisance)/600`.
- Per-lead hotspot share: hotspot count in each lead divided by 600, for every variant.
- TRAIN-prior overlap: apply the same top-600 rule once to the saved spatial prior, giving P;
  report `sum(H & P)/600`, `sum(H & roll(P,shift,axis=1))/600`, and their difference.
- For focal and repeated separately, support precision `sum(H_positive & T)/600` and recall
  `sum(H_positive & T)/sum(T)`. Require nonempty saved T; do not manufacture zero-denominator values.
- Matched nuisance support precision `sum(H_nuisance & T)/600`, using that positive's same mask;
  precision response is positive precision minus this baseline quantity.
- Positive/nuisance hotspot persistence: `sum(H_positive & H_nuisance)/600`, separately by mode.

Activity occupancy, hotspot ties and fixed-cutoff control counts are descriptive context. The original
strict maximum>0.5 control rates must close exactly; 0.5 remains uncalibrated, with no clinical
false-positive interpretation. Prior overlap and lead allocation can reveal associations or
confounding, not causal explanations for network behavior. Support metrics concern introduced
smooth offsets only. Do not attach disease or healthy labels.

## Output, decision and accounting

Save the numeric per-seed rows, three primary intervals, secondary means, source/input/output hashes
and closure evidence. Figures are `fixed_control_hotspots.png`, showing global indices 80–87 without
selection (four columns: supervised sham/nuisance, permuted sham/nuisance; activity and hotspots
identified distinctly), and `diagnostic_summary.png`, showing primary values, persistence and lead
allocation from the frozen metrics. Do not replace examples after inspection. Raw arrays/maps are
referenced, not duplicated; small hotspot index arrays or masks may be retained within the allowance.

This study has no scientific PASS threshold, model selection, clinical recommendation or follow-on
fit. Integrity completion requires exact predecessor closure, verified identities, finite valid
geometry, exactly 600 hotspot cells per map and resource compliance. Report completion separately
from the sign or magnitude of diagnostic findings. No earlier verdict is reinterpreted.

New sibling destination: `outputs/ssl_synthetic_localization/generated_control_diagnosis_v1`;
its independent audit uses `independent_audit_v1` inside this new directory. Keep the entire old
learning tree read-only. Main and audit each have separate CPU 180 s, wall 240 s and output 32 MiB
allowances; GPU cost is zero. Use one numerical/intraop/interop thread and hide CUDA. Each stage
reserves 2 s and 1 MiB for failure/terminal receipts, refuses overwrite, enforces process CPU/wall
deadlines, and retains every failed attempt without automatic retry. Hash all consumed inputs and
source snapshots before/after calculation. The main runner is
`scripts/reports/diagnose_ssl_synthetic_controls.py`, with reusable logic in `ecg_experiment/`.

Start additive accounting from the hash-pinned previous audit terminal ledger, SHA-256
`383ab5707ecfeb28773d0617c7aaa8b8875cc50d77fa17bd82c1db299cf8f1a5`: cumulative CPU
32.489216225 s and GPU-lock wall 38.433311437256634 s. Add each new stage's own terminal cost exactly
once; do not add previous nested cumulative values again. Historical absent factorial costs remain
unverified. The executing coordinator writes a result report from actual outputs, obtains independent
saved-output verification, and updates both queues/backlog. No new fit or dataset admission follows
automatically from these descriptive results; PR #67 stays draft.
