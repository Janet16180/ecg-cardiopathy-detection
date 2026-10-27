# Temporal diagnosis v1: audited range failure

19 October2026. The single narrow failure audit passed with zero mismatches. It confirms
the retained failure cause and accounting, not a completed lead/time diagnosis. The
full diagnostic checker remains unexecuted. Failure-audit protocol/checker/tests were
committed as `ab7fdba`; two artificial tests and Ruff passed before the copied check.

Live receipts under `generated_temporal_diagnosis_v1/independent_failure_audit_v1/`
confirmed1536 ordered unique fit/arm/background/mode rows and no nonfinite AP values.
Exactly one CASE endpoint exceeded the strict0–1 validation:

| Fit | Arm | Global index | Mode | Endpoint | Raw value |
|---|---|---:|---|---|---:|
| 1 | P | 2004 | Focal | Active-lead temporal AP | 1.0000000000000002 |

This is exactly one float64 ULP above1. The case's only active canonical lead was8,
and its saved AP had the same value. The strict validation in frozen executable
`191858915046faf1c1a3b00bd3901b76b412a43777ff71868c3224d215574736` rejected it.
The recorded last-current-case context (fit3/PK/2063/repeated) identifies the last row
written, not the range offender.

Five individual per-lead AP values exceeded1: fit1/P/2004/focal/lead8; fit1/P/2046/focal/
leads0,2,4; and fit2/PK/2028/focal/lead0. Four were1.0000000000000002; the lead2 value
was1.0000000000000004, two ULPs above1. All were inside the frozen eight-epsilon envelope
used by this cause check. Other case endpoints satisfied strict0–1. No value was clipped,
replaced or discarded. The saved-case file remained byte-identical.

All rows were saved before the failure. Reviewed source shows that transient B-arm arrays
and fit means were constructed before the P-arm guard raised. No completed descriptive
summary, PK−B contrast, bootstrap interval or figure was produced; the failed attempt is
not presented as a completed aggregate. The narrow auditor computed no AP, cross-case
endpoint mean, bootstrap or new descriptive contrast and opened no prediction/support,
waveform or checkpoint array. It checked42 source/snapshot bindings, raw row identity/
geometry/ranges, receipt identities, actual storage and additive costs.

| Terminal measurement | Failed main | Narrow audit |
|---|---:|---:|
| CPU seconds | 20.793865505 | 0.024243717 |
| Wall seconds | 36.820030403 | 0.087784491 |
| GPU seconds | 0 | 0 |
| Final artifact bytes | 715,891 | 19,419 |

The failed main remained within180CPU/240wall seconds and16MiB; the audit remained within
60CPU/60wall seconds and8MiB. Combined storage was735,310 bytes, including889-byte main
and791-byte audit terminal ledgers. Main plus audit CPU was20.818109222 seconds.
The once-only cumulative chain is now410.97305985599996 CPU and447.98139213304967 GPU-lock
seconds across16 measured stages. Earlier nested totals were not added again; unavailable
historical factorial costs remain unverified. Timing followed receipt/prior hashes and
byte enumeration as recorded; terminal ledger assembly/serialization is reserved unmeasured
bookkeeping, with its bytes included in the final size checks.

| Artifact | SHA-256 |
|---|---|
| Failed main receipt | `525c05ec36b261ae847885ef664d8547e2a2ed086c87713134ed114ba94c5116` |
| Failed main terminal ledger | `addd84989aba46a9c8a6263d9a4ae1597ffb648766915ccccba019d7db010710` |
| Raw saved cases | `a48bf0672cc4b8c77034d2ea6373483a7e907ab4bc1e371ab38a026099917cdf` |
| Narrow audit summary | `17ce0f66c51653a0c99d26e0c00c161b5c738657ef1d9f8cc7721acb59970827` |
| Narrow audit terminal ledger | `4ee0f57ef15ee12c63368ffd7d1eb35bd6cd4e106c6d272e965f776f9fed67a8` |
| Copied checker | `6d43065e46a681502b4c294bf51ba8dfd540fcaf47a45248f7e334454b3a6979` |

The separately committed v2 recovery keeps raw rows unchanged and fixes a symmetric
eight-float64-epsilon admission envelope before computing completed contrasts. It must
use its own destination, identity admission and independent audit; v1 stays failed and
fully charged. This cause audit supports no temporal-ranking or clinical conclusion.
