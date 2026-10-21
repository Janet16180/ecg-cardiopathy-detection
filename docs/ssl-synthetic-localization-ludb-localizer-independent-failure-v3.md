# Independent localizer recovery v3: retained JSON-container failure

The single actual recovery33 failed before new localizer input/endpoint/CPU-replay reconstruction.
Its saved summary is local under `outputs/ssl_synthetic_localization/ludb_localizer_v1/independent_audit_v3/`.
It records `ValueError: audit metadata must be an object`, one mismatch. Summary SHA256:
`7956843af0d612792b67ffd1a648d33df05d7150c22fb728699a63f8a6b799c2`.
Terminal ledger SHA256: `d3bca8dca1056e0311b6c00190464f715104f858e0e99e42e6ce518b06831365`.

Actual CPU36.38738272 seconds, wall39.592332744039595 seconds, RSS757,784,576 bytes, GPU0;
all four resource gates passed. Actual verified33 totals are678.227157525 CPU seconds and
473.0495132012293 GPU-lock seconds. The ledger binds the preserved actual failed32 predecessor;
neither prior failure is removed, rewritten or charged twice.

The frozen object-only `synthetic_audit_io.read_json()` correctly rejects JSON arrays. The localizer
profile/numerical callers incorrectly used it for main's array-valued `input_metadata.json`,
`profile/input_metadata.json` and `record_ids.json`. The source call order places this failure in
profile metadata verification after the complete synthetic, timing and coherence closures. No
numerical disagreement with the provisional main31 result was established. This is a second audit
I/O contract failure and does not validate the main's newly reported negative landmark result.

Preserve all main/failure outputs and pinned sources. A separately committed narrow successor may
use an explicit array-only JSON reader at those exact paths, retain object-only readers elsewhere,
and verify every remaining actual metadata container/array header before new numerical work. It
must append actual33 once, retain unchanged endpoints/tolerances, and repeat no GPU inference.
