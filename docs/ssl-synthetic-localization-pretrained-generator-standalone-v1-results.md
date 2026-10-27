# Standalone public Pulse2Pulse generation results

19 October 2026. The actual public pretrained Pulse2Pulse checkpoint loaded and generated four
fixed-seed ECG arrays on this machine. All seven prospective technical gates passed. The run completed
exactly six CPU forwards: one disposable profile, four fixed-seed generations and one exact replay.
This establishes bounded technical generation, not healthy ECGs, physiological realism, diagnostic
accuracy, physical mV calibration or regional disease control. No clinical labels or masks were attached.

The coordinator launched the source-reviewed standalone successor after its protocol and implementation
were committed. This report reads the executed local receipts and artifact metadata under
`outputs/ssl_synthetic_localization/pretrained_generator_standalone_v1/`.
The [protocol](ssl-synthetic-localization-pretrained-generator-standalone-v1.md) and
[release assessment](ssl-synthetic-localization-pretrained-generator-assessment.md) define the scope.
Historical localization payloads are absent on this machine; no predecessor reproduction is claimed.
The historical feasibility adapter/config/protocol were preserved and the new runner reused their
reviewed download, architecture, strict-loading, generation and numerical-check helpers.

## Executed technical checks

| Prospective gate | Executed result |
|---|---|
| All case checks | PASS: each raw output is finite float32, 8 by 5000, with temporal variation in every lead; exact author int32 export and canonical limb equations pass |
| Four fixed inputs | PASS: seeds 2026102901, 2026102902, 2026102903, 2026102904 all completed |
| Exact first-input replay | PASS: raw, integer export and derived twelve-lead arrays equal the first generation exactly |
| Distinct fixed-seed outputs | PASS: every pair of the four raw outputs differs |
| Strict finite tensor loading | PASS: weights-only CPU loading, exact architecture keys/shapes and finite float32 tensors |
| Architecture equivalence | PASS: both transpose classes exactly match; generator differs only by reviewed unused compatibility members |
| Model state unchanged | PASS: initial and final named tensor digests match |

The network has **10,590,997 parameters** and no tensor buffers. Four noise arrays were retained with
shape (4, 1, 8, 5000); every completed native output, author export, canonical twelve-lead output and actual
replay was saved. Published geometry is ten seconds at 500 Hz. Author export is float32 multiply 6000,
truncate toward zero to int32, then transpose 5000 by 8. Canonical twelve-lead values derive from these exported
integers in the same unverified numerical units, with absolute limb-equation tolerance 1e-5.
No conversion to physical mV or downstream localization was performed.

Actual downloaded sizes and SHA256 matched all four frozen declarations:

| Asset | Verified bytes | SHA256 |
|---|---:|---|
| Checkpoint |42,375,017|`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`|
| Pinned Hub source |11,269|`160fdee62b49e9960c09d2e669bb13c8069668930ae26019e63bd242c9f6a5b5`|
| Pinned MIT architecture |10,390|`93273ea470c82cd90d7f705c1000e464bfa3e16f7d7733bb817c5bb6cbdbe78b`|
| MIT license |1,074|`8f56e0a848cdb298c9b997c7d97188ebddb3d123c857684c660b6661abece712`|

Weights and Hub source use revision 262e0e67c0101cd4989d7c2f66f16d6d54756e70;
MIT architecture/license use revision f28e4b792e79c503b8b7e0912be9d304a3a7177b.
The exact checkpoint's patient membership and cohort overlap remain unverified. Conflicting release
license declarations described in the assessment remain unresolved; successful loading does not resolve
them. Downloaded remote architecture, checkpoint and generated arrays remain local and untracked.

## Measured resources and accounting

The actual profile took 0.028947509 CPU seconds and 0.130127469 wall seconds. Its conservative forecast,
including spent download/setup time and the reporting reserve, was 15.979493693 CPU seconds and
22.345079961 wall seconds, below the 180-second limits. Main summary elapsed time was 1.421000723 CPU and
9.147953223 wall seconds. The terminal ledger, measured after main receipt writing and hashing, charges
**1.424035600 CPU seconds and 9.162581691 wall seconds**. Its own final JSON serialization is the sole
unmeasured bookkeeping tail under the frozen reporting reserve.

Peak process RSS was 495,341,568 bytes. Main artifacts occupy 45,953,814 bytes, including all public assets,
noise, four cases, actual replay, snapshots and receipts, below the 96 MiB cap. No model fitting,
training updates, real waveform reads, external inference API calls or GPU use occurred. Main used
one numerical/PyTorch intraop/interop thread and retained immutable authored source snapshots.

This is a fresh `prospective_standalone_only` ledger, starting verified prior CPU/GPU at zero. The
historical 745.8142945699988 CPU and 382.2241592870014 GPU-wall numbers remain explicitly unverified
context because their local receipts are absent; they are not included in verified new totals.
The standalone main/audit reservation is 300 CPU seconds, zero GPU seconds and 112 MiB. This does not
reconstruct or reclassify the unavailable historical work.

## Independent saved-output verification

The separate committed checker executed once and passed with zero mismatches and 34 verified file
bindings. It independently regenerated every fixed noise input, checked all saved exports and limb
equations, compared the actual retained replay, checked seed-output distinction and reopened only
the public checkpoint with weights-only CPU loading to reproduce the tensor-state digest. It made
zero generator forwards, GPU calls or real ECG reads. Architecture execution, runtime immutability,
forward counts and no-training claims remain source/receipt-bound; saved replay and numerical
arithmetic were directly checked. This is not independent clinical validation.

The audit summary measured 0.245956461 CPU and 0.524094569 wall seconds, with 451,686,400 bytes peakRSS.
Its terminal additive ledger charges 0.247462913 CPU and 0.530772672 wall seconds. Verified standalone
main plus audit CPU is **1.671498513 seconds**, with zero GPU-wall, below the 300-second allowance.
The audit directory is 20,602 bytes; combined main/audit artifacts are 45,974,416 bytes, below 112 MiB.
Both terminal audit-budget and additive-budget flags passed. Receipts preserve earlier summary
measurements as well as the later terminal ledger charges; their final serialization tail is excluded
under the frozen reporting reserve.

## Receipt anchors

All paths below are relative to the standalone study directory.

| Artifact or binding | SHA256 |
|---|---|
| `summary.json` |`5c3ddf9aac3af4f7c6b38fc91558cd273e0b2221fe2dde09ebe777067e7209a5`|
| `prospective_ledger.json` |`76c3f16e7c2820cfc416af6edc65b4e123261132652f301dbd82615128a17657`|
| `profile.json` |`27be639b7ee8403279f4bcc056c4d2999a196375cd1000648e3a1cab8447dada`|
| `independent_audit_v1/summary.json` |`8fa5da061d689b3e4b1282d87e568f1553182f37025944480b3e13b2221b0b51`|
| `independent_audit_v1/prospective_ledger.json` |`9575bb804e073ec2fa6bc95eb847ca6b6103420798555cc878d3fe30b4abb791`|
| Authored standalone runner |`f9146d94e925f19672f4e9a49659bb3df83f0a8729e23f722064904c0821eb0f`|
| Initial and final named tensor state |`bbe000ff6549d2c9591c8ae92e992a7735ce435e192a87fa4d8dab9eac1d5abc`|
| Four raw arrays, concatenated array bytes |`eed5f45506bd4d101bf38e04f39c51ec26fa2b7931eab7dbc7d22117e6a1048c`|

The preserved checkpoint, complete input/output hashes and source snapshots bind this actual run.
A distinct prospective quality study may assess larger fixed-seed generation and exact-support
numerical interventions. This technical PASS does not authorize labels, a classifier comparison,
a real-waveform fit or a clinical claim.
