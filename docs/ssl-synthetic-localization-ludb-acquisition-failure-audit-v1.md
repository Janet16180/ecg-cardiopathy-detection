# Independent audit of the first LUDB acquisition failure

Prospective 20 October 2026 metadata-only audit, committed before execution. The first public
LUDB transfer stopped at its frozen wall deadline before extraction or scoring. Read only its
summary/terminal ledger, official checksum manifest as uninterpreted bytes, retained partial ZIP
as uninterpreted bytes, frozen acquisition source/config/protocol, and the prior audited ledger.
Do not open the ZIP container, decode its members, parse any waveform/header/annotation, compute
any AP, contact the network, or use a GPU. A failure is not a negative timing-proxy finding.

Admission identities: summary f59fc938a23930ecbd30043c15f28a49499a1a8135f7ac35ead68016526de78b;
ledger e505549b8110b2e4e7dadeca285209b115ffc6e50121d9c6088cbced054130f6;
partial ZIP 1bf23aa1ac52cd6ec1205baca4b819c9e2c90842ecd0314c80dd4a48c1b4158f;
manifest cccef1f3529519db8f26a333c97a6872a7d0e3e5c64a1448b76b161fd87fb75f.
Verify TimeoutError/Acquisition wall deadline, exactly two attempted requests, retained partial
6,291,456 bytes, manifest215,460 bytes, no extracted data, no final archive or success declaration,
and unchanged source hashes. Technical audit PASS records agreement with that retained failure.

Check receipt hash, stage19, additive CPU0.147010707 and unchanged GPU, predecessor SHA
a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf and cumulative CPU
429.64397165699995/GPU447.98139213304967. Compare actual main terminal CPU/wall against
60/600-second caps and summary clocks; finalization followed the598-second reserved deadline.
Do not label the transfer complete or verify unreceived file content. Preserve every byte.

Use one copied source-admitted checker once, with no overwrite. Independent caps30CPU/60wall
seconds,2GiB peak RSS,16MiB new artifacts; two-second/64KiB terminal reserve. Hide CUDA and
use one numerical thread even though only standard-library metadata/hash operations are needed.
Write source-bound success/failure and an additive stage20 terminal ledger below
`outputs/ssl_synthetic_localization/ludb_acquisition_v1/independent_failure_audit_v1/`.
The checker hash is supplied literally before execution. Measurement ends after receipt hashing
and retained-byte enumeration, before reserved ledger serialization. Report measured costs from
saved outputs, independently of development/test time. A later transport amendment must be
separately committed before using the retained prefix, and chain from this audit's ledger.
