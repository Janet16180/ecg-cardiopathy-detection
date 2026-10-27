# Generated quality v2 independent audit results

19 October 2026. The independent saved-output audit passed with zero mismatches. All 128 bank members,
all four variants per member, both intervention masks, every saved case metric, primary interval,
diversity summary and 15 operational gates agree with independent reconstruction. This establishes
numerical engineering integrity for generated backgrounds and controlled changes. Clinical fidelity,
physical calibration and disease localization remain unverified.

The independently recomputed mean native-minus-shifted activity Jaccard is 0.8094248365473182,
with 95% seed-bootstrap interval 0.8038931804338517 to 0.8148564292963815. There were 2,000 whole-seed draws,
seed 2026103301, and zero skipped draws. Native and shifted means from the verified saved rows are
0.870082503923729 and 0.06065766737641082. This measures coincident derivative-energy activity relative
to independent circular shifts. It does not validate waveform boundaries or clinical normality;
this descriptive contrast did not determine the operational PASS.

## Independent reconstruction

The checker regenerated all 128 CPU torch uniform-noise inputs from their fixed seeds, reconstructed
each float64 median/q95−q05 transformation, nuisance phase and lead-II Hann intervention, and required
exact bytes for all 512 canonical variants and 256 strict/thresholded mask pairs. All 128 raw output hashes
were distinct. The saved first-seed GPU replay equals the first native sample byte for byte.
No additional generator forward was performed by the checker.

All introduced strict changes remained inside the prescribed windows and five affected limb leads,
including aVL. Every thresholded support was nonempty and a subset of strict support. Per focal case,
thresholded/strict support contains 357/395 cells; repeated cases contain 2,856/3,160 cells. Against the
complete 60,000 cell grid, thresholded occupancies are 0.00595 and 0.0476. Minimum retained squared energy
was 0.9999650919962539 across both modes, above the frozen 0.99 minimum. The maximum below-threshold
residual was 0.00951528549194336 dimensionless units. Thresholded support therefore intentionally
omits small nonzero differences; strict support identifies complete numerical change.

Independent tie-block AP agrees with every saved amplitude and paired-difference score within 1e-12.
Verified mean absolute-amplitude AP is 0.011123282494581516 for focal and 0.07791658137693971 for repeated
fixtures. Paired-difference oracle AP is within 1e-12 of 1 for every case. The oracle uses the matched
baseline and is a construction check. Fixed lead/time masks and oracle information make these
fixtures unsuitable as a learned-localizer benchmark or evidence of native disease localization.

The audit checked 18,190 scalar values, including all case fields, primary/bootstrap metadata and
8,128 pairwise standardized RMS distances summarized by fixed percentiles. Spectral ratios use the
prospective half-open bands, with explicit zero-denominator handling. RMS/std, activity occupancy,
saturation, scales and diversity summaries match. The three fixed figure files have the expected
PNG dimensions and immutable hashes; their rendering semantics remain bound to reviewed source.
There was no subjective visual quality gate.

A total of 126 source/input/output binding checks passed before and after arithmetic. Separate
amendment bindings preserve the failed v1 receipts and four distinct prior own-cost ledgers. Model
identity, unchanged runtime state, CPU predecessor replay and architecture execution remain
source-bound claims supported by those receipts. The checker independently verified saved noise,
outputs, transformations, masks, metrics and GPU replay bytes; it did not load or rerun the model.

## Resources and retained failure

The v2 main terminal ledger records 7.165424007 CPU, 11.558977179 wall and 7.480909036 GPU-lock seconds.
The independent audit terminal ledger adds 2.445289515 CPU and 4.254422918 wall seconds, with zero GPU
cost. Its 180 CPU/wall-second allowance passed; audit peakRSS was 767,447,040 bytes. Main storage was
198,480,084 bytes, below 512 MiB. Final v2 study storage including the 25,678 byte audit is 198,505,762 bytes,
below the combined 528 MiB ceiling. Main and audit CPU for v2 total 9.610713522 seconds.

The four prior stages contribute 2.446404468 CPU seconds and 6.083721051 GPU-lock seconds: public
probe, probe audit, failed v1 main, and its metadata audit. Adding each own-cost ledger once gives
current-machine cumulative CPU 12.057117990 seconds and GPU-lock 13.564630087 seconds through this audit.
The frozen v1 resource STOP is preserved; neither its profile nor its absent bank was rerun by an
audit. Missing historical factorial costs are not treated as verified current-machine costs.

The checker was frozen in `a53291b` with pre-execution source-review correction `b0bce36`. A nested
output-host/source/parent/study binding regression test caught amendment-root mistakes prospectively.
Three v2 artificial tests and the independent arithmetic tests passed. The coordinator admitted and
executed one exact copied checker. No raw real ECG, patient data, closed set, external inference API,
model fitting or GPU computation occurred in this independent audit. Waveforms, weights and figures
remain local.

Evidence under `outputs/ssl_synthetic_localization/generated_quality_v2/`:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `0067dc16c609a7209cb626e8d31600f6b654d192e3b0803968ca3275db1660b2` |
| `independent_audit_v1/audit_script.py` | `d50982875a9f085e62fe7331c865f3d330d512072e3947c1adc3b041b203359a` |
| `independent_audit_v1/summary.json` | `5fc4ceb2e5fd2a405010e6527c555c1a675c56991e755531e02aabd67eaabb1f` |
| `independent_audit_v1/prospective_ledger.json` | `3b4e5880732b23b67a2d090c2391b0a4f9dea79c314ce20c79750a1ed1b65159` |
