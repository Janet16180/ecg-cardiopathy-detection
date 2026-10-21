# Transparent split-phase LUDB transfer

20 October 2026. Prospective transport-only successor. Commit this protocol/config, new offline
source/tests and identity admission before execution. Scientific LUDB protocol/config, record roles,
proxy/coverage/controls, scores/seeds/decisions and main/audit budgets remain unchanged. V3's Python
download never started: automatic approval review timed out twice before process start. It produced
no attempt, payload or experimental cost stage. Preserve its unused design/source as history.
The two actually executed failures and the narrow audit remain charged once.

## Source and three phases

Immediate prior is the exact v2 failure stage21 ledger: CPU 429.84679126699996, GPU-lock
447.98139213304967 seconds. Reuse its immutable 10,616,832-byte prefix and existing official
checksum manifest, with all hashes and the failure19 → audit20 → failure21 lineage from config.
The only new GET requests the remaining 3,857,652 bytes of the same 14,474,484-byte public archive.

1. Offline preparation creates `outputs/ssl_synthetic_localization/ludb_acquisition_v4/` exclusively,
   verifies all source/prior/manifest/prefix identities and cumulative costs, and copies the opaque
   v2 prefix plus manifest into that new directory. Record preparation CPU/elapsed/RSS and source
   snapshots. Require curl output/header/resource-log paths absent. Record resolved curl/time binary
   identities and versions without network activity. Never write a previous attempt or parse ECGs.
2. Execute exactly one transparent public curl GET, under GNU time, using the command below. Disable
   curl configuration files; no redirect following, retries, HEAD, fallback or additional requests.
   Retain suffix, headers and time log even on failure. Approval waiting occurs between phases and
   is explicitly excluded from active runtime. An approval timeout before curl process start is
   not a network transfer; never claim it ran or invent a child runtime.
3. Offline finish verifies unchanged prepared/prior/source bytes, GNU time exit status zero and
   measured limits, final HTTP206 and exact range, optional exact Content-Length, and exact suffix
   size before assembly. Append only a valid suffix to the NEW v4 copied prefix, then rename it
   `release.zip`; never append to v2. Keep the suffix as evidence. Check final size/hash, safe archive
   whitelist and all 2,800 record files against the unchanged official manifest. No diagnosis CSV
   extraction or header-comment parsing. Any violation emits a retained FAIL and additive ledger;
   no second GET or rerun under this attempt directory. Handle interrupted byte counts/receipt work
   without losing the failure ledger or presenting an unknown byte count as exact.

The direct command, with repository-relative paths expanded only to this new output root, is:

```sh
LC_ALL=C /usr/bin/time -v -o outputs/ssl_synthetic_localization/ludb_acquisition_v4/curl_resources.txt \
  curl --disable --http1.1 --fail --silent --show-error --retry 0 \
  --connect-timeout 180 --max-time 3600 --speed-time 180 --speed-limit 1 \
  --max-filesize 3857652 --range 10616832- \
  --header 'If-Range: "6006e8e6-dcdcf4"' \
  --dump-header outputs/ssl_synthetic_localization/ludb_acquisition_v4/response_headers.txt \
  --output outputs/ssl_synthetic_localization/ludb_acquisition_v4/suffix.partial \
  https://physionet.org/content/ludb/get-zip/1.0.1/
```

No `-L` and no hidden downloader wrapper. The command's resource log supplies curl child user/system
CPU, elapsed time, exit status and peak RSS; optional stdout metrics do not replace that evidence.
Curl's connection and low-speed timeouts are distinct from an uninterrupted socket-idle guarantee.
Require final `Content-Range: bytes 10616832-14474483/14474484`, Content-Length 3,857,652 if present,
and exactly 3,857,652 suffix bytes. Parse the last final HTTP response block, allowing preceding
proxy CONNECT/informational blocks without counting them as another archive GET. HTTP200, redirects,
short/overlong responses or nonzero curl exit fail before concatenation/extraction. No new network
identity check is needed: archive identity is first-use hash plus every selected official file hash.

## Explicit active-phase budgets and accounting

Preparation plus finish share 60 CPU / 600 active wall seconds. The external curl has a separate
60 CPU / 3,600 wall-second acceptance ceiling, with its wall and low-speed flags active online.
CPU/RSS ceilings for curl are checked from actual GNU time measurements; do not claim live CPU/RSS
limits that this transparent command does not enforce. Total stage ceilings are 120 CPU / 4,200
summed active wall seconds, maximum phase RSS 2 GiB, 64 MiB selective extraction and 128 MiB retained
new outputs. Report CPU as preparation own CPU + curl child user/system + finish own CPU, once.
Report active wall as the sum of the three measured phases, not a single continuous elapsed clock.
Disclose excluded approval/interphase waiting; never describe 4,200 seconds as an uninterrupted
whole-wall cap. Peak RSS is the maximum across phases, not their sum. No GPU or model operation.

Use live offline CPU/wall/RSS checks, strict streamed byte counters and conservative storage guards.
After preparation, bound base bytes + remaining suffix file + remaining suffix appended to prefix +
64 MiB extraction + 1 MiB receipts below 128 MiB. Do not repeatedly scan the growing extracted tree.
Exactly enumerate retained bytes once at completion with offline timers active; disclose the final
2-second / 16 KiB ledger assembly reserve for each offline phase. Preparation measurement ends
after its preparation-receipt hash, before serializing `prepare_costs.json`; final measurement
ends after final receipt hashing, before the terminal ledger. Both small serialization tails are
explicitly disclosed rather than assigned invented CPU time. GNU time child clocks retain their
printed precision; they are not nanosecond measurements. Include source hashes, time-log/header verification,
assembly/extraction, file checks and receipt hashing in measured phase work. If preparation already
ran but transfer cannot start, retain that real work and finish with an explicit failure; never
reset or fabricate it. Missing child measurements after an uncertain launch remain unknown and fail
resource validation. Prior costs remain unchanged and any measured successor work is charged once.

V4 appends directly to stage21 and is stage22 when finalized; V3 contributes no fictitious stage.
Independent source/receipt review precedes qualification; the combined qualification audit may
verify the full acquisition and numerical chain afterward. Pin actual current ledgers by identity,
never reset to the historical 429 CPU baseline or omit earlier failed attempts. Raw access may use
untracked read-only-use `baseline/data/raw/ludb/1.0.1/data` →
`baseline/outputs/ssl_synthetic_localization/ludb_acquisition_v4/data`, with an equivalent worktree
alias. Preserve pointer directories, never stage links and never write through them. All new raw
bytes, suffixes, archives and receipts stay under baseline outputs. No write to the other checkout.
