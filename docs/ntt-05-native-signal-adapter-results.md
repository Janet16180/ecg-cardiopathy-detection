# NTT-05 results: native signal reference contracts verified; learned encoder evaluation pending

19 October 2026 UTC (18 October locally). The native-signal reference adapter passed its engineering
contract checks and reproduced all 180 frozen NTT-04 physical-reference decisions exactly. It does not
use a rendered ECG, DINO or SAM. No pretrained native encoder was executed: its weights and historical
feature caches are absent in this checkout. There is no new clinical recognition/localization result.

The [prospective protocol](ntt-05-native-signal-adapter.md) was committed as `2e04fc6`. Pre-execution
clarifications and final source/tests were committed as `52a1bad`, after independent static review and
26 focused invariant checks. A single engineering execution completed in 0.259 seconds for the measured
verification stage, excluding Python/import startup. The integrated CPU suite passed **1,331 tests with
six skipped in 73.10 seconds**; repository-wide Ruff passed. Work remains local under the personal
identity. No waveform read, model inference, training, data/model download, GPU, push or PR was used.

## How the paper maps to signals

[No Time to Train](https://arxiv.org/html/2507.02798v4) pools frozen visual features inside reference masks,
aggregates instance/class prototypes, proposes target masks and matches their features to references.
The implemented successor keeps the reference comparison while changing its inputs and proposals:

| Component | Native implementation |
| --- | --- |
| Feature map | Already computed native encoder token vectors with explicit lead/time anchors |
| Candidate mask | Half-open physical-time interval and explicit lead set; label-free multiscale window generator |
| Mask pooling | Fractional temporal occupancy of compatible lead anchors, with complete coverage checks |
| Reference bank | Unit region vectors, equal instance aggregation within patient, then equal patient weights per finding |
| Matching | Cosine score, fixed tie order and caller-declared similarity/margin abstention thresholds |
| Redundancy handling | Same-finding, same-lead, same-record temporal score decay; recurring disjoint beats remain separate |

The adapter accepts dense native features; it does not create a pretrained model or infer an unknown
sample/token geometry. An explicit nonoverlapping lead-patch conversion is available for compatible
encoders. Another native model may provide its exact temporal anchors directly. A same-size vector
is insufficient compatibility: tokens, supports, reference banks and matches carry an embedding-space
contract identifier encompassing model/checkpoint, preprocessing and tokenization. Incompatible
identifiers are rejected. Annotation provenance remains separate from feature-space identity.

Coordinates refer to feature anchors, not proof that the underlying signal information is local.
Lead-anchored transformer tokens may carry information from other leads and times. All-lead tokens
cannot be sliced into single-lead evidence, and the adapter rejects that request. Matches also retain
patient and record/coordinate-frame identity, preventing suppression between separate ECGs or crops.
Missing required reference classes cause configuration errors; low or ambiguous matches can abstain
without assigning an implicit normal label. No clinical abstention threshold was calibrated.

Soft score decay is a declared one-dimensional adaptation, not the original mask code. Positive
lower-ranked detections decay only against same-finding, exact-lead-set candidates from the same frame,
using temporal intersection-over-self and nonnegative cosine similarity. Negative scores cannot be
boosted by multiplication toward zero. Abstentions cannot suppress candidates. Original regions and
adjusted scores are retained; no disjoint-beat union or unmarked-as-normal assumption is introduced.

## Actual native model availability

A separate agent audited the existing code interfaces and local artifact inventory without loading
waveforms or models. Its [portable audit](../reports/ntt05_native_encoder_inventory.json) records paths,
loader constraints, source evidence and absent artifacts.

| Encoder | Audited interface | Current limitation |
| --- | --- | --- |
| ECG-JEPA | Locally vendored architecture exposes 8 leads x 50 patches x 768 features, lead-major, 200 ms patches at 250 Hz | Official compatible source/checkpoint absent; dense tokens are contextual and omit four derived leads |
| xECG | Documented native temporal vectors mix all 12 leads; compatible architecture suggests 40 time tokens | Release/config/weights absent; cannot fabricate a lead axis or assume final time orientation without verification |
| Project CPC/SimDINO | Native frontend/context token sequences have explicit temporal structure and mix leads | Trained checkpoint and normalization missing; a fresh factory produces random weights, not pretrained features |

ECG-JEPA is the clearest audited candidate for a future lead/time integration. Source/API compatibility,
strict checkpoint loading, resampling, normalization, special tokens, context, effective resolution and
pretraining/evaluation membership need verification before scoring. Its 200 ms anchor cells cannot by
themselves establish fine pathological boundaries. A pooled whole-record cache would not replace dense
local token maps. No checkpoint was silently substituted or downloaded in this engineering study.

This is not the first native reference idea in the project. NTT-02 already compared physical masked
windows; NTT-04 compared physical and visual representations. PR61's study 064 already evaluated native
SimDINO normal-reference anomaly retrieval and reported a failed localization comparison. The present
finding-specific abnormal-reference adapter is distinct from that normal-only bank, but this study
supplies no performance evidence that it remedies the earlier result.

## Executed engineering gates

The runner verified frozen NTT-04 result, selection, feature, prediction and reservation-manifest hashes.
Before feature/finding access it confirmed every selected record/patient against the manifest's training
role. No broad label replay or reserved-role waveform was read. The existing 60 target patients remained
separate from all 27 support patients across three fixed banks. Replay used saved full-median-beat
physical vectors, fixed finding order and the original three references per finding/draw.

| Check | Executed result |
| --- | --- |
| Required synthetic invariant tests | 26 passed; zero failed, errored or skipped |
| Invariant source binding | All 10 source hashes and JUnit bytes match the frozen implementation |
| Saved patient-draw choices | 180 / 180 reproduced exactly |
| Existing correct patient-draw choices | 167 / 180, reproducing the predecessor only |
| Maximum cosine-score reconstruction error | 5.55e-16 |
| Target/support overlap | Zero |
| Invariant / replay / combined engineering gates | All passed |
| New learned-native recognition or expert localization score | Not executed |

The 167/180 value corresponds to the already reported 92.78% physical baseline. It is an integrity check,
not a new model result, independent evaluation, interval-localization score or additional 180 people.
Replay's region coordinates describe the saved median-beat frame only, not a new detected region in an
original recording. The full regional pooling/proposal/abstention/decay contracts are tested with synthetic
arithmetic examples; no synthetic diagnostic accuracy headline is reported.

Tests exercise exact physical patch mapping, fractional occupancy, lead compatibility, temporal gaps,
invalid vectors/coverage, patient-balanced prototypes, patient separation, equal-dimensional incompatible
feature spaces, abstention/ties, deterministic windows and same-record semantic decay. The executed
JUnit file is hash-bound to its source receipt; the runner requires that receipt and committed source
identity before replay. It enforces the 300-second ceiling at stage boundaries, preserves typed failure
reasons and excludes mutable status from immutable result binding. At import, Matplotlib used a temporary
cache because its default configuration directory was unwritable; execution succeeded without rerun.

An independent reviewer checked all ten sources against disk and commit, seven immutable input hashes,
the actual 26 JUnit cases, role separation, all 180 predecessor choices and score reconstruction, and all
engineering gates. The reviewer read no waveforms and loaded no model. The final result SHA256 is
`e554a50a695ec6bc96c6b066abee28658d79396a3f90e48899154a032e8d9fed`.
The [aggregate receipt](../reports/ntt05_native_signal_adapter.json) contains no patient identifiers or
per-record feature values. Local execution files are under `outputs/ntt05_native_signal_adapter/`, with
source-bound invariant receipts under `outputs/ntt05_native_signal_adapter_preflight/`.

## What remains

The engineering adaptation is viable and reviewable. Its clinical usefulness is untested. Recover a
verified compatible native checkpoint and normalization, profile actual CPU inference, then freeze a
separate scored comparison with physical and visual baselines, identical patients/references/proposals,
within-wave and absent-finding controls, coexisting annotations and support uncertainty. Automatic
proposal coverage, finding recognition and specialist evidence agreement need separate endpoints.
Independent acceptable-evidence annotations remain necessary for a clinical highlighting claim.

The current adapter does not establish disease etiology, prove that a pooled coordinate contains the
causal feature, or show that a new native model beats prior methods. Follow-ups enter the ranked backlog;
no new download, inference or clinical evaluation is automatically scheduled by this receipt.
