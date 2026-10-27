# Scientific review of released-N timing qualification

20 October 2026. The prospectively defined engineering conjunction passed,
with independent numerical reconstruction and no mismatches. The fixed
40 ms derivative-energy proxy had mean empirical percentile rank
0.9636002546541803 at released N timestamps, versus 0.5130816149251329 for
the mean of six fixed phase controls. The primary paired gain was
0.45051863972904727, with whole-record bootstrap 95% interval
[0.4389598383639518, 0.4625890507482116]. This exceeds the frozen gain >= 0.05
and lower-bound >0 rules. It is a rank-concentration result, not 96.36%
detection accuracy, clinical sensitivity or learned-model transfer.

I inspected the saved main and successful independent-audit summaries,
terminal ledgers, aggregate statistics, all saved event-row metadata and the
two fixed figures. The measurement comprises 160 records, 1920 lead streams
and 17,539 released N events. All 200 records and 2400 streams remain present;
the 40 parser records have no numerical endpoint or control scores. The
bootstrap used 2000 whole-record draws, seed 2026106101, with no skipped draws.
Events/leads within a record were not treated as independent subjects.

The independent checker used separate byte-level annotation decoding and
unique-frequency midranks, rather than the main searchsorted rank code.
It reconstructed all raw event metadata, full measurement score arrays,
landmark/control vectors, aggregation and bootstrap draws, and verified exact
C2 and saved-record 95 closures. It compared independent byte events against
saved main WFDB event rows; it did not invoke WFDB a second time. Figure
verification checked saved signatures/layouts, not independently rendered
pixels. My visual review confirmed the fixed eight parser panels and the
explicit timing-only captions; those panels were not selected by outcome.

The complete saved rows contain 21,965 N events across all 200 records, no
duplicate N timestamps and no duplicate sample/code groups. There are 122
N events without an immediately preceding opening-boundary event. That flag
is descriptive source-order metadata, not an imputed boundary. In the
separately diagnosed record 95, exactly 35 QRS onsets were demonstrably absent;
that history remains unchanged. No missing onset was guessed, dropped or
turned into a negative label by the successful landmark study.

Reference spans range from 2199 to 4418 samples, with median 3644 samples across
all 2400 streams. Thus the typical reference spans 72.88% of a 5000-sample record;
this experiment does not establish timing behavior over every full-record
sample. Every reference sample is unlabeled. The six fixed shifted controls
can retain periodic alignment and are not a universal random null. No dense
QRS boundary, onset/offset error or false-positive rate is defined here.

Main stage 26 used 12.051356381 CPU seconds, 29.622269546147436 wall seconds and
635867136 bytes peak RSS. The first independent attempt 27 failed on a legacy
ledger schema before waveform or numerical reconstruction; its 1.717297662
CPU seconds and original null accounting remain preserved. An explicit setup
reconciliation counted that actual failure once, without inventing an old
receipt or creating another experiment. The new audit 28 used
4.4918088979999995 CPU seconds,11.832718590740114 wall seconds and 681332736
bytes peak RSS. Main and successful audit passed all resource gates, used no
GPU or training, and left cumulative current-machine CPU 454.04911085099997
and GPU-lock 447.98139213304967. Historical unavailable costs remain unverified.

Evidence is local under
`outputs/ssl_synthetic_localization/ludb_peak_timing_v1/`:
main summary SHA256 `2e48b4e06512f126bb0eddf12306b54c462da1e300fa2624c0effed1c1bc8c29`;
successful `independent_audit_v2/summary.json` SHA256
`d603ccd664a80f3c74ca6a657b6da38c7828d9b3ce69dee72309a55c313e1571`;
audit terminal ledger SHA256
`94afb161731e62568198f42ccf03d1e15e6a833f2b7c491172abb70688d1d400`.
The prior setup reconciliation is 27cdc3ceddc4a36e7be6ba2805caad98c094fcd5bb85e80d1d2c9b469d5b99dd.
The protocol/source identities and released-data hashes remain in the receipts.

The justified conclusion is that this fixed activity proxy concentrates near
released N landmarks in these public ECGs. This gives the existing synthetic
hotspot diagnostic a real annotation anchor for its timing ingredient. It
does not establish the physiological validity of every N timestamp, generated
ECG plausibility, disease conditioning, pathology masks, clinical localization,
SSL benefit or learned-model performance on real ECGs. The earlier strict
dense-span failures and possible prior rdann header-comment parsing deviation
remain documented; new annotation I/O used sanitized structural headers.

The separately committed coherence bridge may now use the actual successful
audit 28 prerequisite and its resource conjunction. Its real-versus-generated
activity-synchrony contrast remains descriptive and has no realism/quality
PASS threshold. That comparison can help interpret generator regularity and
possible shortcuts; this timing result does not itself validate coherence.
