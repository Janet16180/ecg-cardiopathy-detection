# Clarification of the v1 diagnostic failure boundary

19 October 2026. This prospective clarification accompanies v2 protocol/config commit `611171d`
before any v2 execution or aggregate. The frozen files remain unchanged.

The v2 protocol's phrase "No v1 aggregate or contrast was computed" means no completed summary,
PK-minus-B contrast or bootstrap was computed or persisted. In the frozen v1 implementation,
`summarize` validates each arm's endpoint matrices and can build B-arm descriptive means in memory
before reaching P's out-of-range value and raising. Those transient means were not inspected by
the coordinator or scientific reviewer, and no summary or comparison artifact was produced.

The observed failure remains one saved temporal AP value one float64 ULP above one, appearing
both as its sole active-lead value and the case mean. All 1,536 raw rows remain unchanged. This
clarifies execution history only; it changes no endpoint, raw value, bootstrap, comparison,
admission tolerance, resource limit or planned recovery operation.
