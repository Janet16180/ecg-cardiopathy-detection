# Fixed pilot-domain generator admission policy

18 October 2026. Commit before executing this additional generator-design check. The v1 design-grid
receipt remains frozen. Its pass does not admit arbitrary new seeds. No model score is computed here;
including held-out B in an engineering truth check does not authorize learning or model selection on B.

The pilot uses 250 Hz ten-second observations. Set each latent identity to the unsigned big-endian
integer represented by the first eight SHA-256 bytes of UTF-8
`ssl_pilot_v1:synthetic:<role>:<index>`, with roles exactly `train`, `validation`, `evaluation`.
All related realizations inherit that same role. Check seed uniqueness and disjoint roles explicitly.
Never retry seeds, drop failures or alter counts to obtain a pass.

| Role / profile | Identities | Declared realizations |
| --- | ---: | ---: |
| train / A | indices 0–499 | 500 |
| validation / A | indices 0–127 | 512 |
| evaluation / A | indices 0–255 | 1,024 |
| evaluation / B | same evaluation identities | 1,024 |
| Total | 884 independent latent identities | 3,060 |

Training indices below 375 receive one positive: focal when index is even, persistent when odd,
with nuisance `bool(seed & 1)`. Indices 375–499 are zero-target controls with nuisance `bool(index % 2)`:
63 nuisance-only and 62 sham. Every validation/evaluation identity receives focal and persistent
positive examples with the same `bool(seed & 1)` nuisance, plus one no-nuisance sham and one
nuisance-only negative. The zero-target controls use focal mode and the cohort's profile; neither
control receives a positive support mask. Evaluation A is a secondary same-profile engineering
condition; evaluation B is the held-out edit-profile condition on the same background simulator.

Run the original `gate_failures` with the committed v1 thresholds on every planned pair. Repeat every
render and require exact array/metadata replay. Report full failures and metric quantiles per cohort;
preserve local per-case audit rows and source hashes, then report from the aggregate receipt. All
3,060 must pass; failures stop downstream training, including when only the same-profile secondary
fails. No numerical comparison with Experiment 058 is made. Keep one CPU numerical thread and no GPU.

If v1 fails, retain and report its failed admission receipt. Implement a new forward v2 generator,
leaving all receipt-pinned v1 sources untouched. The prospective v2 contingency narrows width from
0.14–0.22 to 0.14–0.19 seconds, preserving each draw's uniform fractional position in that range;
when a primitive limb coordinate is edited, retain exactly one of the originally selected chest
coordinates, selected uniformly by an independent edit substream. Chest-only edits retain their
original one-to-three chest leads. Thus at most six derived leads change for a limb edit and at most
three for a chest-only edit. With at most 17 beats this bounds the nominal (pre-resampling) continuous
persistent domain by 17×0.19×6/(12×10)=16.15%. Actual postprocessing support still needs every frozen
gate; this bound alone is not an admission claim. Preserve A/B implementations, gain/noise/drift,
background, signs, peak-amplitude range, target support floor and every original threshold. Apply the
same narrowing to all modes, roles, nuisance states and both profiles. Commit v2 implementation/tests
before its full 3,060-pair admission check. The generator version must appear explicitly in receipts
and the pilot protocol. No waveform data, diagnosis labels, GPU or model predictions are needed.
