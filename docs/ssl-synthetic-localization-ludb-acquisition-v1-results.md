# First LUDB acquisition stopped at its transport limit

20 October 2026. The committed public LUDB acquisition failed its reserved 598-second wall
deadline during archive transfer. No ECG waveform, annotation, diagnosis field or localization
score was decoded. This is a transport failure, not an unsuccessful QRS timing measurement.

The scientific protocol/config were committed as609e881/8214287 before acquisition; authored
acquisition source90e3c1a passed Ruff and three artificial archive-safety tests, with independent
source-review GO. The run issued the declared checksum-manifest request and one archive request,
without retry or fallback. It retained the complete215,460-byte official SHA256SUMS manifest
and6,291,456bytes of the unfinished14,474,484-byte archive. No extraction occurred. All files,
including partial bytes, remain under `outputs/ssl_synthetic_localization/ludb_acquisition_v1/`.

The received manifest SHA is`cccef1f3529519db8f26a333c97a6872a7d0e3e5c64a1448b76b161fd87fb75f`;
the incomplete archive-prefix SHA is
`1bf23aa1ac52cd6ec1205baca4b819c9e2c90842ecd0314c80dd4a48c1b4158f`. Neither establishes that
a complete release was acquired. The original source and scientific files remain unchanged.

The failure receipt records0.146305568CPU seconds,598.006871803198wall and1,235,140,608peakRSS
bytes. Its terminal ledger includes receipt hashing:0.147010707CPU and598.0104395500384wall,
zero GPU. These remain within60CPU/600wall/2GiBRSS caps; completion failed at the intentionally
earlier reserve deadline. The19current-machine stages now total429.64397165699995CPU and
447.98139213304967GPU-lock seconds, including this failure exactly once. Terminal ledger
assembly/serialization is the disclosed bookkeeping tail; development/tests remain separate.

Failure summary SHA:`f59fc938a23930ecbd30043c15f28a49499a1a8135f7ac35ead68016526de78b`.
Terminal ledger SHA:`e505549b8110b2e4e7dadeca285209b115ffc6e50121d9c6088cbced054130f6`.
The prior audited ledger remains`a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf`;
unknown historical costs remain unknown. Independent metadata-only cause verification is pending.

A separate prospective range-resume amendment is being prepared because the public server
advertises byte-range support and the measured transfer could not finish in ten minutes. It
will preserve this attempt, admit a longer fixed transfer budget, and require exact response and
per-file official checksums before any scientific execution. The200record roles, fixed proxy,
controls, bootstrap and decision rule do not change. No training or closed-data evaluation is
scheduled; PR67 remains draft.
