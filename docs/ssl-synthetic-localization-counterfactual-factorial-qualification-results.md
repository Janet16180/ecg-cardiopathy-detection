# Counterfactual factorial data qualification results

18 October 2026. The single authorized qualification passed for all 192 new validation backgrounds
and all 768 variants. No record was dropped, replaced or retried. This report qualifies data only;
no GPU, model inference or optimizer update occurred.

Protocol/configuration `8beb272`, implementation `26fee78` and final integrity gates `e44cf34` were
committed, root-integrated and independently reviewed before execution. The frozen runner executed
from `/tmp/ecg-wt-ssl-synthetic` with that worktree's `PYTHONPATH`, one numerical CPU thread and CUDA
hidden. All predecessor artifacts and pinned sources remain unchanged.

## Patient and source identity

The complete new 192 selection was saved before any selected raw header/data access. It excluded all
384 previously used validation patients, all 500 reused TRAIN patients and all 2,048 original SSL
patients. Each remaining original VALIDATION patient's lowest numeric ECG was eligible regardless
of quality flags; the frozen new namespace selected 64 calibration and 128 evaluation patients.
One original quality-exclusion flag and one review flag were retained. No clinical fields, native
pathology labels or ages entered selection or qualification.

The exact original 2,000 TRAIN case descriptors and their arrays remain referenced in the previous
study. Their canonical case-list digest is
`2eb37bc57c2f9650a4b9989c2b4791ce6e356cb7b7d8ec11ccffe2e42de64401`.
Preparation opened zero TRAIN waveform payloads and created no duplicate TRAIN bank. The model-stage
context checks each consumed TRAIN background row and every support/candidate/rendered case hash
before use. The old mixed waveform file's whole-file digest remains delegated to the passed-audit
catalog; its unrelated 192 old validation waveform rows are not scanned.

Passed parent training, metrics and latest diagnostic audits were freshly SHA-bound. All eight
parent scientific summary blocks, frozen thresholds and full nine-gate decision, plus all seven
saved diagnostic aggregate sections, reproduced at the unchanged absolute tolerance 1e-12. Both
predecessor full advancement decisions remain failed. Reused helper source identities were checked
against the anchored source catalogs; no old evaluation was rescored.

Each new raw pair freshly matched the official checksum table and original input-audit payload
hashes. The frozen readers verified explicit mV units, canonical leads and 500 Hz/12-lead/5,000-sample
geometry before one whole-record resampling to 250 Hz. The qualification verified 23,156,596 raw bytes.
Protected cohorts and engineering-test payloads remained closed.

## Executed gates

| Check | Result |
|---|---:|
| New backgrounds | 192: 64 calibration A, 128 evaluation B |
| New variants | 768: 256 calibration, 512 evaluation |
| Failed backgrounds/cases | 0/0 |
| Exact deterministic replays | 768/768 |
| Positive baselines matching their controls | 384/384 |
| Anchors per new background | 8–17 |
| Maximum actual support fraction | 0.1473333333, below 0.18 |
| Maximum observed added delta | 0.3388928771 mV, below 0.50 mV |
| Maximum added-limb algebra residual | 1.1920928955e-7 mV |
| Maximum quantization residual | 1.1809577228e-7mV |
| Maximum analytical added-limb residual | 0 |

Every case passed its individual signal-dependent rounding tolerance, finite geometry, exact
strict >0.005 mV support and support-inside-candidate gates. Controls were byte-identical to their
baselines. Native findings remain unknown: zero intervention support does not mean healthy, and
these digital pulses/algorithmic anchors are not clinical disease or expert wave-boundary labels.

New background rows 0–191 map explicitly to virtual subjects 500–691. New mask rows 0–767 map to global
cases 2000–2767; calibration is rows 0–255 and evaluation 256–767. The new namespace changes only new
validation case identities/parameters; native anchoring, edit profiles, nuisance and once-cast math
are reused unchanged. Only physical-mV backgrounds and bool support/candidate masks were saved;
there are no edited/baseline banks.

## Cost, storage and validation

Actual qualification cost was 2.988186446 CPU seconds and 3.021054257 wall seconds, including metadata
admission, selection, raw integrity/decoding, qualification/replay and final hashing. Exact cumulative
scientific CPU became 572.9219694699988 seconds; prior cumulative GPU-wall remains 226.41281207299835.
Runtime and storage gates passed. No separate post-verification run was performed.

Preparation occupies 71,659,665 bytes. The seven-array complete study forecast remains 437,760,000
payload bytes + 896 NPY-header bytes + 64 MiB shared metadata/checkpoint reserve = 504,869,760 new bytes.
After counting existing new files once, projected whole-stream storage is 3,984,290,138 bytes, below
4 GiB. Actual profile/recovery and deterministic BCE reproduction remain separate gates before new
calibration/evaluation.

Twenty-two artificial data/context tests passed and Ruff passed before execution. Coverage includes
all virtual/local split boundaries, frozen 384-patient exclusion and unfiltered selection, manifest
before payload, changed frozen helper/active TRAIN background/mask rejection, unused-waveform
authority deferral, source/input conflicts, no duplicate-cache forecast, overwrite refusal, measured
admission timeout retention and no further raw access after an anchor/case failure.

Local artifacts are under
`outputs/ssl_synthetic_localization/counterfactual_factorial_v1/preparation/`:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `498251dc05013ece766d3877755fe0a000db5b8a1db8dc2ffdcddad4bbefcdba` |
| `selection.json` | `0dfe6327530ebd6b8a4523d4dd9f31d251aaa59ad090116f2c194a320f7dccfc` |
| `cases.json` | `b493a303725c9c09ff31a8b6010c87b43ba70e141d46d6d317547802ddb5eafe` |
| `backgrounds.npy` | `c8c8d6226426285b3a4919aa1c90dedd16a2038b306eb87fb2ff1180cc2411f3` |
| `support.npy` | `05ab3d2a216914dc279b84886d372057cb53a78872f68d6342fa82ca6b4979a4` |
| `candidate.npy` | `a7b1f027e803f92205bdded1f31213aa798878d5e809963db2978bbec68b5651` |

Patient identities, parameters, waveforms and masks remain local and untracked. No remote action
occurred. Data qualification does not establish a loss benefit, clinical localization validity or
training admission.
