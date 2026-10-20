# Temporal/lead diagnosis v1 stopped at a strict floating-point range check

19 October 2026. This attempt is a retained technical failure, not a completed localization result.
All 1,536 fixed case rows were saved, but aggregation stopped at the strict AP range check. No complete
summary, PK-minus-B contrast, bootstrap or figure was produced. The independent narrow cause/cost
audit passed with zero mismatches across 42 source/snapshot bindings; it did not recalculate AP or
validate the new diagnostic endpoints against maps.

The [protocol](ssl-synthetic-localization-temporal-diagnosis-v1.md) was committed as `ec86255`
before the coordinator read the paired-suppression outcomes. Exact 29-file identity admission
`7642810` and reviewed source `ddea85a` preceded execution. Sixteen artificial tests passed.
Before new endpoint calculations, all 1,536 full-grid APs and the complete B aggregate reproduced
exactly. All final source/input bytes stayed unchanged through that recorded closure.

The strict aggregation guard encountered fit 1/P/index 2004/focal temporal AP
`1.0000000000000002` (`0x1.0000000000001p+0`), exactly one float64 ULP above 1. Its sole active
canonical lead 8 had the same raw AP. Saved probabilities were not out of range; this is a rounding
excursion in accumulated average precision. All 1,536 ordered identities were unique and all endpoint
values finite. No case was removed, replaced or clipped.

The narrow audit also reported all five individual-lead AP excursions: the same lead 8 above;
fit 1/P/index 2046/focal leads 0 and 4 at 1+1 ULP and lead 2 at 1+2 ULP; fit 2/PK/index 2028/focal lead 0
at 1+1 ULP. Only the first is also a case-endpoint violation. The fixed eight-epsilon recovery
admission includes every raw per-lead and endpoint value; it is not a one-outlier-only rule.

The failure receipt's last-current-case is fit 3/PK/index 2063/repeated, because every row was already
saved before aggregation. This context is not the actual offending row. V1 builds transient B-arm
means before reaching the P guard; those were not inspected, and no complete summary or paired
contrast was persisted. The [prospective clarification](ssl-synthetic-localization-temporal-diagnosis-v2-clarification.md)
records this precise execution boundary without changing the frozen source.

Main terminal CPU was 20.793865505 s, wall 36.820030403 s, GPU zero, final storage 715,891 bytes, all within
the original 180 CPU/240 wall seconds/16 MiB limits. The independently committed
[narrow audit](ssl-synthetic-localization-temporal-failure-audit-v1.md) used 0.024243717 CPU/
0.087784491 wall seconds and zero GPU, within 60/60 seconds/8 MiB. Main plus failure audit charged
20.818109222 CPU seconds once. Sixteen current-machine stages now total 410.97305985599996 CPU/
447.98139213304967 GPU-lock seconds. Unknown older costs stay unknown; terminal assembly/serialization
is the disclosed reserved bookkeeping tail after receipt hash/byte enumeration.

| Local evidence | SHA-256 |
|---|---|
|failure.json|`525c05ec36b261ae847885ef664d8547e2a2ed086c87713134ed114ba94c5116`|
|prospective_ledger.json|`addd84989aba46a9c8a6263d9a4ae1597ffb648766915ccccba019d7db010710`|
|cases.jsonl|`a48bf0672cc4b8c77034d2ea6373483a7e907ab4bc1e371ab38a026099917cdf`|
|predecessor_closure.json|`811a41b52b47341a51a7f2bc1295f9eddf20707950434eae70eb2106da406e61`|
|independent_failure_audit_v1/summary.json|`17ce0f66c51653a0c99d26e0c00c161b5c738657ef1d9f8cc7721acb59970827`|
|independent_failure_audit_v1/prospective_ledger.json|`4ee0f57ef15ee12c63368ffd7d1eb35bd6cd4e106c6d272e965f776f9fed67a8`|

All artifacts stay under `outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v1/`.
The independent full-v1 checker is preserved as committed, source-tested but unexecuted. Its
success-path analysis is not evidence for this failed attempt. No model forward, optimizer update,
GPU operation, new waveform read, download or external API call occurred in either attempted
analysis or narrow audit. The full parent B scientific FAIL remains unchanged.

The separately committed [v2 recovery](ssl-synthetic-localization-temporal-diagnosis-v2.md) retains
all raw rows byte-identically, uses unchanged comparisons/draws and only admits machine-roundoff
range excursions. It needs its own reviewed source, identity admission and independent map-derived
audit. This report makes no physiological, clinical or real-disease localization claim. PR #67 stays draft.
