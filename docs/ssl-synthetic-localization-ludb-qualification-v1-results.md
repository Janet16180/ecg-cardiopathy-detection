# LUDB input-admission failure before ECG reading

20 October2026. The first timing run stopped before any LUDB waveform/header/annotation was
decoded, predecessor contrast computed, or AP score produced. The coordinator's identity manifest
omitted one of27 required source dependencies:
`ecg_experiment/ssl_synthetic_pretrained_saved_audit.py`. The runtime correctly rejected it.
This is an input-admission failure, not a negative timing-proxy result.

Original scientific protocol/config8214287 and complete runtime source6e2d485/7a2fdfc preceded
manifestfa47d0d. Forty parser and160measurement records remain unspent. The unchanged v1output
contains only`failure.json`, its terminal ledger and an empty scores directory; no source snapshot,
predecessor closure, record rows, predictions or figures were produced. Source roster completeness
failed before the prior-cost verification step. All original files and their null fields remain.

The actual terminal cost is1.177479012CPU/2.386282444000244wall seconds, zeroGPU, peakRSS177,291,264
bytes. All four CPU/wall/RSS/storage gates passed. Failure SHA is
`86ebbc793ddcd650e7298536033d9872d83507016647cda2bc753c889d51b7eb`;
ledger SHA`559f4cc2ab0437cc38797d0afdd9a755c2fc39617ef3de779d42bb00c767710e`.
Because the run stopped before verifying its declared prior, its original ledger records
`prior_accounting_verified:false` and NULL cumulative values. These are not silently repaired.

A separately identified, independently reviewed setup metadata reconciliation links verified
acquisition22 to the measured failed run23:430.97339291699996+1.177479012 equals
432.15087192899995CPU seconds; GPU-lock remains447.98139213304967. It binds the original
acquisition ledger, failed ledger, failure and manifest, with SHA
`8ff30f48c3d92057be2c641dd3a168a2d009a9f23fc4206bd5ce24612fff2a24` at
`outputs/ssl_synthetic_localization/ludb_qualification_reconciliation_v1/reconciled_prior.json`.
This explicit accounting supplement represents the actual failed stage; it is not another
experiment or a fabricated replacement for the original null ledger. Its creation is setup;
the successor verifies it within measured runtime.

The separately frozen [runtime-v2 amendment](ssl-synthetic-localization-ludb-qualification-runtime-v2.md)
keeps every scientific choice and resource ceiling unchanged. It adds the missing dependency,
an executable source-only preflight and a distinct output path. No threshold, wave parser,
record role, seed, control or endpoint is retuned. Independent source/metadata review precedes
the successor's first ECG read, and the full numerical audit will retain this failure and cost
once. No clinical or closed-test evaluation occurred. PR67 remains draft.
