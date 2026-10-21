# Released-generator quality and support results

19 October 2026. The released Pulse2Pulse model generated all 128 fixed-seed ECGs on the RTX 3090.
All15 operational integrity/resource gates passed. Independent verification reconstructed every
transformed signal and intervention mask, reproduced the primary comparison, and passed 18,190 scalar
checks and 126 binding checks with zero mismatches. This is actual generation and an engineering
support qualification. No localizer was fitted in this study, and no disease localization is established.

The primary cross-lead activity comparison was **+0.809425**, seed-bootstrap 95% interval
**[0.803893,0.814856]**. Sharp signal changes occurred at aligned times across independent leads much
more often than after independently shifting their activity masks. This is temporal structure under
the frozen derivative-energy rule; it does not validate P/QRS/T boundaries or physiological fidelity.

## Execution and the preserved resource failure

The [original protocol](ssl-synthetic-localization-generated-quality-v1.md) and all numerical choices
were committed before generation/scoring. The preceding public-generator probe and independent audit
passed. Its four CPU inputs reproduced raw, author-export and canonical output bytes exactly before
each GPU attempt. No historical localization arrays were recovered from the other checkout.

The first attempt stopped honestly after its GPU profile: conservative wall forecast 1,550.645226s
exceeded its 1,200s ceiling. It produced no bank or primary result. Its 4 CPU replays, 1 GPU profile,
failure, input noise and costs remain retained; see [failed-stage results](ssl-synthetic-localization-generated-quality-v1-results.md)
and [independent failure check](ssl-synthetic-localization-generated-quality-failure-audit-v1-results.md).

The separately committed [v2 resource amendment](ssl-synthetic-localization-generated-quality-v2.md)
raised only wall/GPU-lock allowance to 1,800s. It preserved 128 seeds, endpoints, controls, the 2×130
profile forecast, memory/storage limits and the decision rule. Its fresh actual full-path profile
cost 0.150091 CPU/0.165850 wall seconds and forecast 69.719488 CPU/74.105657 wall seconds, passing
before the bank. No warm-up discount or favorable seed/device selection was used. The failed first
attempt remains failed rather than being reclassified by this later successful attempt.

V2 completed 4 CPU predecessor replays, 1 disposable GPU profile, 128 bank forwards and 1 exact CUDA
replay. Fixed bank seeds are 2026103000–2026103127. The same released checkpoint and architecture
were used without optimizer updates, dependency changes, patient inputs or external inference calls.

## Actual signals and checks

Each native output is float32 eight-by-5,000, in I,II,V1–V6 order, at the published 500 Hz rate.
Per-lead float64 median and q95−q05 produce a dimensionless robust-scale representation, then the
four limb leads are derived. This changes amplitude ratios and does not establish mV calibration.
All128 raw hashes differ; every native/standardized lead varies and all arrays are finite. Exact
same-device replay and unchanged model tensor identity passed. Canonical equations held in all 512
stored variants. Zero raw values reached the frozen abs(raw)≥0.99 saturation threshold.

The 8,128 standardized independent-lead pairwise RMS distances had minimum 0.433901, median 0.658249
and maximum 0.896924. No near-duplicate cutoff was introduced, and these distances do not establish
absence of memorization or training-cohort overlap. Mean independent-lead power fractions were
0.355830 in [0.5,5)Hz,0.600174 in [5,25)Hz and0.043996 in [25,100)Hz, using [0.5,100)Hz denominator.
These describe the generated bank rather than acceptance as normal ECGs.

The primary used only the 8 independent leads, excluding algebraically derived channels. Mean
native activity Jaccard was 0.870083 versus 0.060658 after independent circular shifts. The paired
128-seed mean difference and 2,000 whole-seed bootstrap draws used seed 2026103301, with zero skipped
draws. This interval conditions on this checkpoint and these noise seeds; it is not a patient or
training-seed interval and has no clinical quality threshold.

## Introduced-change truth and localization limits

Each background has sham, nuisance, focal and repeated variants. Positive variants add the same
dimensionless Hann offset in independent leadII to the matched nuisance baseline; derived leads
propagate it to II,III,aVR,aVL,aVF. Strict and abs(final float64 difference)>0.01 masks were saved
for both modes. Sham/nuisance target masks are implicitly zero and are not stored as additional
positive-mode mask arrays. Nuisance remains a changed waveform without a target intervention label.

| Frozen descriptive endpoint | Focal | Repeated |
|---|---:|---:|
| Mean thresholded support fraction |0.005950|0.047600|
| Mean strict nonzero support fraction |0.006583|0.052667|
| Mean absolute observed-amplitude AP |0.011123|0.077917|
| Mean paired-difference oracle AP |1.000000|1.000000|
| Minimum retained squared-change energy |0.999965092|0.999965092|
| Maximum below-threshold absolute residual |0.009515286|0.009515286|
| Strict changes outside the intended hull |0|0|

Values summarize all 128 saved case rows per mode. Thresholded masks are subsets of strict support;
they retain more than99.9965% of change energy but are not the entire numerical support. Exact mask
reconstruction and all five affected limb channels passed. The oracle uses the unavailable matched
unedited recording, so its expected AP≈1 is construction integrity, not a learned localizer.
Absolute amplitude gives an engineering difficulty reference, not diagnostic or screening accuracy.

The fixtures have fixed times and a fixed independent lead. A position-only predictor could learn
them without reading the signal. They must not be used as a localization learning benchmark or
training set. A successor must prospectively vary locations and leads, separate generator seeds
between training/evaluation, include matched negative controls, and profile the actual learning path.

| Scientific question | What these outputs establish |
|---|---|
| ECG-shaped signals generated? |Yes: actual finite, varying native 8-lead outputs and derived 12-lead arrays.|
| Physiologically plausible signals? |Temporal/spectral proxies and fixed plots only; no physiological validation.|
| Disease-related findings controlled? |No. This released noise-to-ECG path is unconditional.|
| Introduced change located exactly? |Yes, for the separately constructed numerical intervention and its declared threshold.|
| Real disease localized? |No real disease masks, patient evaluation or clinical validation.|

## Resources, identities and local artifacts

Main terminal costs were 7.165424 CPU/11.558977 wall seconds, including reporting and bookkeeping
through the final main-receipt hash. GPU-lock occupancy was 7.480909 s, including profile and all
CUDA forwards. Peak CUDA allocated/reserved bytes were 45,372,928/46,137,344; peak process RSS was
1,277,452,288 bytes. CUDA0 was RTX 3090, driver 580.142, PyTorch 2.6.0+cu124; numerical/intraop/interop
threads were 1, deterministic operations enabled, and both TF32 paths disabled. Other GPU services
were left untouched; host escalation exposed the GPUs hidden by the restricted device namespace.

The main occupies 198,480,084 bytes. Main plus independent audit occupies 198,505,762 bytes, below
their separate 512 MiB and 16 MiB limits. All new-machine scientific stages, including the probe,
probe audit, failed profile, metadata audit, successful study and its audit, total 12.057118 CPU
seconds and13.564630 GPU-lock seconds. Each stage's own terminal cost is counted once. Historical
reported 745.814295 CPU/382.224159 GPU seconds remain unverified locally and were not silently added
or reset. Implementation, source review, artificial tests and document authoring are setup work.

Checkpoint: `deepsynthbody/deepfake_ecg`, revision
`262e0e67c0101cd4989d7c2f66f16d6d54756e70`, 42,375,017 bytes, SHA256
`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`.
The reviewed MIT architecture is Pulse2Pulse revision `f28e4b792e79c503b8b7e0912be9d304a3a7177b`;
its source SHA256 is `93273ea470c82cd90d7f705c1000e464bfa3e16f7d7733bb817c5bb6cbdbe78b`.
Exact named-tensor state before/after is
`bbe000ff6549d2c9591c8ae92e992a7735ce435e192a87fa4d8dab9eac1d5abc`.
The standalone assessment's license, physical calibration and exact checkpoint-cohort limitations
remain unresolved; neither weights nor downloaded architecture are redistributed in Git.

All bank paths below are local, under `outputs/ssl_synthetic_localization/generated_quality_v2/`:

- `noise.npy`: 128 retained noise inputs.
- `raw.npy`: 128 actual native 8-lead signals.
- `variants.npy`: 128×4 canonical 12-lead dimensionless observations.
- `strict_support.npy`, `support.npy`: 128×2 exact and thresholded change masks.
- `replay_raw.npy`, `predecessor_closure.json`, `profile.json`: replay/profile evidence.
- `cases.jsonl`:all 128 fixed case records and mode-specific metrics.
- `fixed_first_eight.png`: unselected native 8 and standardized canonical 12 traces.
- `known_support.png`: first 8 paired overlays and intervention supports.
- `quality_diagnostics.png`: bank-wide timing, scale and spectral distributions.
- `source_snapshot/`, `summary.json`, `prospective_ledger.json`: immutable identity and accounting.

Main summary SHA256: `0067dc16c609a7209cb626e8d31600f6b654d192e3b0803968ca3275db1660b2`.
Main terminal ledger SHA256: `4f78a069b021dae6b9d687415ca07a468cf7df3c40873c46cd7aec574ad10172`.

The [independent report](ssl-synthetic-localization-generated-quality-independent-audit-v2-results.md)
records 18,190 scalar checks,126 artifact bindings, exact full variant/mask reconstruction and primary
reproduction. Audit summary SHA256:
`5fc4ceb2e5fd2a405010e6527c555c1a675c56991e755531e02aabd67eaabb1f`.
It used zero model forwards or GPU operations. Runtime execution and CPU predecessor replays are
source/receipt-bound; independent saved-array reconstruction does not imply observing them anew.

This study delivers a reproducible released-generator bank and verified intervention mechanism.
It leaves physiological validation, meaningful varied-support learning, disease conditioning and
real disease localization open. Existing clinical labels and closed-test restrictions are preserved.
