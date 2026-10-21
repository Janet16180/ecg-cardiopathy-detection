# Identity admission for the unchanged-row diagnostic recovery

19 October 2026. V2 protocol/config were frozen as `611171d` before recovery aggregation. This
admission binds 29 explicit B parent files and six v1 recovery/failure-audit files, each freshly
hashed at its own path. It changes no endpoint, bootstrap, raw value, tolerance or resource limit.
The separate v2 clarification records the exact boundary of the failed v1 aggregation function.

The independent v1 failure audit passed with zero mismatches across 42 source bindings. Its summary
is `17ce0f66c51653a0c99d26e0c00c161b5c738657ef1d9f8cc7721acb59970827`; its terminal ledger is
`4ee0f57ef15ee12c63368ffd7d1eb35bd6cd4e106c6d272e965f776f9fed67a8`. That ledger's exact cumulative
base is 410.97305985599996 CPU seconds and 447.98139213304967 GPU-lock seconds. Add only new v2
main/audit costs to this base. V1's charged failure and failure audit remain in the chain once.

The audit confirmed 1,536 unique ordered rows, no nonfinite values and one case-endpoint excursion:
fit 1, P, index 2004, focal temporal AP is one ULP above one. It also identified five individual
lead AP occurrences above one, with a maximum of two ULPs: fit 1/P/2004/focal lead 8; fit 1/P/2046/
focal leads 0, 2 and 4; and fit 2/PK/2028/focal lead 0. These are distinct from the count of case
endpoint excursions. All already fall inside the frozen eight-epsilon envelope; no tolerance
change is needed or allowed. Every original value remains intact, including the five lead values.

The raw row artifact is bound to
`a48bf0672cc4b8c77034d2ea6373483a7e907ab4bc1e371ab38a026099917cdf` and must be copied byte-for-byte.
The failure audit performed no new AP, descriptive aggregate, model forward or optimizer update.
No recovery aggregate was computed for this admission. The B scientific FAIL and v1 technical
failure remain explicit; v2 has only descriptive endpoints and technical verification.

Final source review, exact predecessor closure, fixed limits and independent recovery audit remain
required. Unknown historical costs remain unverified. This admission authorizes no waveform,
checkpoint, other-panel or clinical-data access and no new inference or fitting.
