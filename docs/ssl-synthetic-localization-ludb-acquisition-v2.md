# Bounded LUDB range-resume acquisition amendment

20 October2026. Commit this transport-only successor and its config before its one new request.
The original public-wave scientific protocol/config remain frozen and unchanged: same release,
200record IDs,40parser/160measurement roles, proxy, coverage/masks, controls, endpoints, seeds,
thresholds, source guards, independent verification and main/audit resource budgets. No LUDB
waveform or annotation content was read and no score exists from the failed acquisition.

## Preserved first attempt and new reason

Acquisitionv1 failed its reserved598-second wall deadline, within its600-second hard cap. Its
retained archive prefix is6,291,456bytes, not a complete release. Its summary records598.006871803198
wall seconds and0.146305568CPU; terminal ledger charges0.147010707CPU and598.0104395500384wall,
bringing current-machine totals to429.64397165699995CPU and447.98139213304967GPU-lock seconds
over19stages. It downloaded the complete215,460-byte official checksum manifest first. Keep every
v1byte and receipt unchanged. This is a preserved transport failure, not a negative QRS result.
No attempt to recover clinical data or another checkout's outputs is allowed.

A verified publicHEAD advertised byte ranges and a14,474,484-byte archive. The observed first
attempt's throughput makes a longer, separately bounded range request reasonable. This amendment
allows one such request; it does not silently retry or increase the old budget.

## Exact one-request plan

Preserve the original official archive URL as source provenance. Send the sole networkGET directly
to its previously observed frozen final URL `https://physionet.org/content/ludb/get-zip/1.0.1/`.
Disable automatic redirects and fail on any redirect response, so the oneGET budget counts actual
wire requests rather than a redirect chain. No newHEAD request. New output root is
`outputs/ssl_synthetic_localization/ludb_acquisition_v2/`, created exclusively. Before network access,
verify the complete v1summary, ledger, partial archive and official checksum-manifest bytes against
this config. Copy the v1prefix and checksum manifest byte-identically into the new root; read v1only.
Never append to, rename or remove v1files. Record and recheck their hashes after use.

The new HTTPGET sends `Range: bytes=6291456-` and `If-Range` with the declared archive ETag
`6006e8e6-dcdcf4` using proper HTTPquoted ETag syntax. The response must be206 and exactly
`Content-Range: bytes 6291456-14474483/14474484`. Require exactly8,183,028received archive payload
bytes and final file size14,474,484bytes. If Content-Length is present, it must equal8,183,028;
a missing header does not waive exact bounded body-length checking. A200response, incorrect range/total, short/overlong body,
changed source bytes, timeout or server error stops the attempt. No200full-download fallback, second
GET, new checksum request or automatic retry. Stream accepted response chunks to the copied prefix.
If the request fails, retain every v2partial byte and charge the stage; a future transport strategy
requires another prospective amendment.

The completed archive gets a recorded first-use SHA-256. It has no independently supplied prior
archive hash. Verify all2800admitted record files against the unchanged official versioned checksum
manifest; this file identity assurance is not a cryptographic publisher signature. Selective
extraction and safe member checks follow the frozen v1acquisition contract. No diagnosis CSV or
other unadmitted content is extracted or parsed. Source headers remain opaque during acquisition;
only their structural first13lines may later be read by the admitted qualification runner.

## Resource/accounting and continuation

One connection, oneGET, no retries. New ceilings are3600wall/60CPU seconds,2GiB peak RSS,64MiB
received network payload,64MiB selectively extracted files and128MiB total new retained outputs.
The copied prefix does not count as newly received network bytes; it does count toward retained
outputs. Reserve two seconds and16KiB for terminal receipt/ledger serialization; use live wall/CPU/
byte guards and report RSS. Hide CUDA, no GPU/training or numerical waveform work. Measure from
before prefix/manifest verification and copying through completed source/input hashes, retained-byte
enumeration and receipt hashing; disclose the bounded ledger-assembly tail. Preserve the exact
v1terminal ledger as fixed source predecessor. Before v2, a separately admitted narrow failure
audit appends its actual CPU/wall and zeroGPU cost once as stage20. The v2identity-only admission
then pins that latest verified audit terminal ledger, its SHA and actual additive totals. Require
its direct v1lineage and equality to the stage19base plus that audit cost; never reset costs to
v1or omit the audit. Append v2costs once to this verified stage20ledger. Acquisition fails on transport/identity/resource violation even if
some files have already been retained. Preserve all such files and failure receipts locally.

Freeze new acquisition source, tests and identity-only admission before this request. Independently
review the completed acquisition identities and predecessor ledger before scientific execution.
The main qualification's new identity-only admission may point its raw-data root and acquisition
receipt/ledger bindings to v2; this is the sole transport-source-pointer override to the original
scientific config. Its data remain the exact same public1.0.1release and its main/audit budgets do
not increase. Baseline may symlink the verified v2data directory beside its existing LUDB pointer;
never stage that link or modify shared raw files in the other checkout. No scores, record roles,
waveforms, diagnosis fields or endpoint choices change under this amendment.
