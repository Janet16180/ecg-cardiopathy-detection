# Lead/time diagnostic recovery with explicit float64 roundoff admission

19 October 2026. This is a new prospective recovery after v1 stopped before aggregation. Preserve
all v1 source, protocol, output rows, failure receipt and costs. No v1 aggregate or contrast was
computed. The only changed numerical policy is admission of machine-roundoff excursions of AP
outside its mathematical range. Neither the endpoints nor their interpretation changes.

## Exact cause and unchanged question

V1 saved all 1,536 unique ordered rows, with no nonfinite values. Its strict range guard rejected
one scalar: fit 1, P, global index 2004, focal, active canonical lead 8. Its sole active-lead AP and
equal-lead temporal AP are `1.0000000000000002`, hexadecimal `0x1.0000000000001p+0`, exactly one
float64 ULP above one. Lead-presence AP on that row is 1.0. All other endpoint APs meet the strict
range. These range/identity facts were inspected without computing a new aggregate or contrast.

The v1 failure receipt is
`525c05ec36b261ae847885ef664d8547e2a2ed086c87713134ed114ba94c5116`; its terminal ledger is
`addd84989aba46a9c8a6263d9a4ae1597ffb648766915ccccba019d7db010710`.
V1 charged 20.793865505 CPU seconds and zero GPU. Its current cumulative CPU was 410.948816139
seconds and GPU-lock wall 447.98139213304967 seconds. A separate failure audit has fixed limits
of 60 CPU/60 elapsed seconds and 8 MiB; the recovery's identity-only admission must bind that
audited terminal ledger, including its additional actual cost, before execution.

Keep the original two endpoints, all four arms, three fits, 64 spent backgrounds, two positive
modes, equal-active-lead then equal-mode/background/fit aggregation, and six PK-minus-B contrasts.
Temporal AP is an oracle-conditioned scoring diagnostic on the known edited leads, not deployable
performance. Lead-presence AP uses twelve max-time scores. Derived canonical leads are dependent.
There is no new scientific PASS, shift reference, conditional interval or model selection.

## Bounded numerical change and inputs

Let epsilon be `numpy.finfo(numpy.float64).eps = 2.220446049250313e-16`. Admit finite AP scalars in
`[-8*epsilon, 1+8*epsilon]`, with tolerance `1.7763568394002505e-15`, including every saved individual
active-lead AP and both case endpoints. Reject larger excursions, missing values or nonfinite data.
Retain all raw values unchanged: no clipping, projection, dropped case, favorable seed replacement
or recomputation of the diagnostic endpoints. Copy v1 `cases.jsonl` byte-identically into the new
destination. Verify the complete fixed row order and identities, unique active-lead indices, exact
arithmetic mean of each row's active-lead APs, and the original one-ULP strict-range failure.

Only the saved v1 endpoint rows enter the new aggregation. Admit v1's closure/failure/main ledger
and the separate failure-audit receipts, plus the same 29 B files frozen in v1. In the recovery
main, B prediction/support files may be byte-hashed for identity but not numerically rescored.
Recompute B's complete aggregate from its bound saved metric JSON rows, requiring exact agreement
with both B's summary and the saved v1 predecessor-closure aggregate, including all fifteen gates
and 42 factorial secondaries. Preserve v1's source-bound exact 1,536-case full-grid AP closure
as inherited evidence. The independent recovery audit freshly reconstructs that full-grid closure
and every diagnostic row from the same B maps/support; no waveform, raw generator output, noise,
checkpoint, optimizer, other panel or clinical data is admitted.

The main's new summary uses the original crossed bootstrap unchanged: 2,000 draws with
`default_rng(2026105601)`, three fit rows followed by one shared 64-background vector, common to
all six contrasts. Use raw unmodified AP values, the same linear percentile endpoints and no
skipped draws. Retain every arm/fit/mode point and fit range. The fixed 2×2 figure and its 0–1 AP
axes are unchanged; any raw rounding excursion remains visible in saved numeric artifacts.
The parent B scientific FAIL and v1 technical failure remain explicit.

## Prospective limits, evidence and review

New destination is `outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v2/`. Main
limits are 90 CPU/120 elapsed seconds and 16 MiB, with one numerical thread and zero GPU work.
Reserve two seconds and 1 MiB for terminal/failure receipts. Independent recovery audit limits
remain 180 CPU/240 elapsed seconds and 16 MiB, one thread and no neural operations. No repeated
attempt, fallback or retroactive ceiling change is authorized.

Commit this protocol/config before any recovery aggregate. Then commit exact-path identity-only
admission after the separate failure audit, with the final audited cumulative ledger as the single
cost base. No scientific choice may change in that admission. Source review must verify raw-row
preservation, unchanged arithmetic/draws and no new neural or endpoint-scoring operation. Artificial
tests must match v1 exactly on ordinary in-range rows, admit the one-ULP case without altering it,
and reject values outside the eight-epsilon bound. Shared bootstrap/I/O helpers remain in use.

The independent audit must use separate endpoint arithmetic, check all saved rows against B's
maps/support, reproduce both predecessor closures and all six crossed contrasts, and verify limits,
bindings and additive costs. Exact identities/arrays remain exact; independent metric reductions
use the original 1e-12 comparison tolerance. The eight-epsilon change is solely range admission.
Write actual success or failure receipts and a coordinator results report; update both queues.
Unknown historical costs stay unknown. PR #67 remains draft, with no clinical or pipeline claim.
