# LUDB range resume retained after an idle connection timeout

20 October2026. The separately committed range-resume attempt received the exact admitted
HTTP206 response but failed on a30-second socket read timeout. No extraction, ECG waveform or
annotation decoding, diagnosis parsing, QRS score or model operation occurred. Preserve this as
a second transport failure, independently of the unchanged scientific timing protocol.

Transport amendment0fe1123/945c6ab/2093384 and sourceb85e82b preceded identity admission3326727
and the one request. Both independent and scientific source reviews gave GO; sixteen artificial
transport tests and Ruff passed. The direct request disabled redirects and carried the exact
range plus quoted If-Range ETag. The response was206 with
`Content-Range: bytes 6291456-14474483/14474484`, `Content-Length:8183028`, and ETag
`"6006e8e6-dcdcf4"`. No fallback or second GET was made.

The receiver retained4,325,376new bytes after copying the immutable6,291,456-byte v1prefix.
The incomplete new prefix is10,616,832bytes, SHA
`95bb3ae62b52332b8e3ff9a6eb79eb4426f615fc20f324c80941823c8eac4956`.
It is not a verified complete archive. The failed summary records`TimeoutError: The read operation
timed out`; the3600-second whole-stage wall budget was not reached. All original v1bytes and
its separately executed independent failure audit remain untouched. Artifacts are local under
`outputs/ssl_synthetic_localization/ludb_acquisition_v2/`.

Summary clocks are0.193132089CPU/394.31859387783334wall seconds, peakRSS1,235,140,608bytes.
The terminal ledger charges0.19535622200000002CPU/394.3322474639863wall, zeroGPU; all five
CPU/wall/RSS/received-byte/retained-output gates passed. Retained bytes before the terminal ledger
were11,270,232. Receipt hashing and enumeration precede terminal timestamps; ledger serialization
is the disclosed two-second/16KiB bookkeeping tail. Setup/test/development time stays separate.

Summary SHA:`ffe2b782929f97d8ee61bd28988b5ea9570e36483bea1a66d8e8944ff351ab94`.
Ledger SHA:`9114b01b9e68d1ced6cf9c152014bbfcf4523d437dcff59fed53bcd11a38c833`.
The direct prior is the independent v1failure audit ledger
`2fe1c2a450d72c42223af87a4014fdde20064fd5b6474dbd215cb65fad59bb03`.
All21current-machine stages now total429.84679126699996CPU/447.98139213304967GPU-lock seconds,
including each retained failure/audit once. Missing historical costs remain unknown.

A new prospective v3transport amendment allows one request for the remaining3,857,652bytes
with a180-second socket idle timeout and the same3600whole-wall/60CPU caps. It preserves both
failed attempts and their accounting, keeps every scientific choice unchanged, and still requires
complete official asset checksums before any waveform evaluation. Independent review of these
metadata agrees with the failure; the complete acquired-asset chain will be independently audited
after successful acquisition. PR67 remains draft, with no training or closed-test access.
