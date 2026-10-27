# Independent localizer audit v2: retained ordering failure

The single independent attempt at stage32 failed before evaluating any new LUDB localizer maps,
physical inputs or CPU checkpoint replays. The descriptive main31 result remains provisional.
The complete retained failure is local at
`outputs/ssl_synthetic_localization/ludb_localizer_v1/independent_audit_v2/`.

The saved summary reports `ValueError: Require all200 original records` (one mismatch). Its SHA256
is `db56f79efc8641dad71546821293a80febf92287b491cf71d02bd265f9a732b6`.
The actual ledger SHA256 is
`8e00e2e18182f38f81c4e7f9eb2112b6fe97f94b8b5e65c86bda8097a5c80218`.
Measured CPU was37.394049274000004 seconds, wall42.99549401784316 seconds, peak RSS752,832,512 bytes,
and GPU0. All four resource gates passed; technical verification failed. The verified actual32
cumulative totals are641.839774805 CPU seconds and473.0495132012293 GPU-lock seconds. Historical
costs remain unverified. The failure ledger preserves the genuine main31 predecessor identity.

The source-level cause is the call to the frozen independent timing `aggregate()` in
`ssl_synthetic_ludb_localizer_audit.predecessor_closures()`: it passes the original saved parser-first
row table directly. That aggregate requires record IDs exactly1..200 in ascending order. The saved
timing table writes the40 parser rows before the160 measurement rows. The main timing aggregate
sorts its input; earlier successful independent timing/coherence closure code also sorted in memory.
The new caller omitted that normalization. The immutable caller order places this failure after the
all-twelve synthetic metric/full-aggregate comparisons and before the new localizer numerical phase.
No disagreement with the new localizer numerical results was established.

Preserve the main output, every source/admission and this failed attempt. A prospectively admitted
runtime-only recovery will sort the200 saved record objects in memory before the same frozen timing
aggregate. It will not change event/N order or multiplicity, select records, rewrite files, change
scores/tolerances or repeat GPU inference. It must verify this actual failed32 cost and append its
own measured cost as33. No recovery has executed at this report's creation.
