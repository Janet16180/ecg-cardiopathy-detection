# Released QRS landmarks concentrate at high activity ranks in LUDB

20 October 2026. The fixed activity measure ranks highly at released QRS timestamps in real ECGs.
Main stage 26 and independent recovery audit 28 pass the prospective engineering conjunction.
Independent reconstruction found zero mismatches. This establishes concentration at released
landmarks; it does not establish disease localization or detection accuracy.

| Fixed primary measurement | Executed value |
|---|---:|
| Mean rank at released N landmarks | 0.963600255 |
| Mean of six fixed shifted ranks | 0.513081615 |
| Mean paired rank gain | 0.450518640 |
| Whole-record bootstrap 95% interval | [0.438959838, 0.462589051] |
| Bootstrap draws / skipped | 2000 / 0 |

The gain is 45.05 percentile points, exceeding the committed 0.05 rank-gain threshold with a
lower bound above zero. All 12 fixed lead mean gains and all 160 record gains are positive.
A mean rank around the 96th percentile is not 96% accuracy, sensitivity, AP or AUROC.

All 200 LUDB records and 2400 lead streams remain represented. The 40 parser records were
qualified before opening the 160 measurement records and have no rank or control metrics.
The measurement cohort contains 17,539 released N events; the parser cohort contains 4426.
All streams passed exact agreement between WFDB and independent raw annotation decoding.
There were no duplicate N timestamps. Reference-span sample quantiles across all streams are
[2199, 3475, 3644, 3857.25, 4418] of 5000 samples.

The new estimand was committed before any measurement record was opened. It retains every
explicit numeric-code-1 landmark and every other decoded event, without inferring boundaries.
There are 35 parser and 87 measurement N events without an immediately preceding onset token.
Those are descriptive syntax flags, not recovered physiological boundaries. The reserved
record-95 diagnosis established 35 genuinely absent explicit onset tokens. The original strict
dense-span qualification retains FAIL; this landmark study cannot rescue that interval contract.

The unchanged 20-sample derivative-energy proxy is computed on each full signal before restriction
to the first-through-last event reference span. Reference samples remain unlabeled. The six fixed
phase shifts can retain periodic alignment and are not a universal null. Events are averaged within
lead, 12 leads within record, then 160 records equally. Bootstrap seed 2026106101 resamples whole
records; correlated leads are not treated as independent subjects.

All 150 predecessor scalars across 1536 saved synthetic cases reproduced exactly. Before measurement,
the record-95 closure reproduced all 571 events, 107 N markers and 35 missing onset pairs across
12 streams. Protocol 55177cc, exact-prior config 583d317, main source 95df4e7, independent source
98a875f and complete admission 9ddfcf6 preceded scores. Source-only preflight verified 66 bindings.
Admission SHA: `58affcc86311b60956557dfeb9c3f5dccb974e579c886b6de3d6797caff6f6c0`.

Independent recovery reconstructed all 200 records, 2400 streams, 1920 measurement lead vectors,
17,539 landmarks, 160 full twelve-lead score arrays and 2000 draws. It passed 1,668,474 scalar
comparisons, 6742 identity checks and 253 predecessor checks with zero mismatches and identical
primary results. It uses independent physical decoding and unique-frequency ranks against the
main searchsorted arithmetic. Its MIT byte decoder is compared against saved main WFDB rows;
it does not call WFDB again. Figure verification checks signatures/layouts, not rerendered pixels.
The coordinator inspected the aggregate figure; scientific review also inspected the fixed panels.

| Actual stage | CPU seconds | Wall seconds | Peak RSS bytes |
|---|---:|---:|---:|
| Main 26 | 12.051356381 | 29.622269546147436 | 635867136 |
| Failed audit 27 | 1.717297662 | 3.421806860715151 | 498995200 |
| Recovery audit 28 | 4.4918088979999995 | 11.832718590740114 | 681332736 |

All resource gates pass, with zero GPU or training. Main retained 108,685,346 bytes before its
terminal ledger. Audit 27 failed on legacy accounting metadata before raw assets or numerical
reconstruction. Its original null accounting fields remain unchanged. The separately reviewed setup
reconciliation adds its measured CPU once; it is not another experiment. Recovery 28 changes only
metadata/schema handling and reuses the frozen independent mathematics. None of the 66 executed
main source bindings was rewritten. Cumulative numerical-stage costs through actual stage 28 are
454.04911085099997 CPU and 447.98139213304967 GPU-lock seconds, excluding development/setup.
Unavailable historical costs remain unknown.

Local outputs under `outputs/ssl_synthetic_localization/ludb_peak_timing_v1/` include 200 record
rows with 2400 event tables, 160 full score arrays, all controls and bootstrap draws, source/release
identities, closures and two figures: `lead_peak_rank_comparison.png` and
`fixed_N_marker_proxy_overlays.png`. Data, figures and receipts remain local.

| Artifact | SHA-256 |
|---|---|
| Main summary | `2e48b4e06512f126bb0eddf12306b54c462da1e300fa2624c0effed1c1bc8c29` |
| Main ledger | `8bb1bc5625ec53a1ab009c35a87c6098a438ddcca2db149efc6f00d251739494` |
| Recovery audit summary | `d603ccd664a80f3c74ca6a657b6da38c7828d9b3ce69dee72309a55c313e1571` |
| Recovery audit ledger | `94afb161731e62568198f42ccf03d1e15e6a833f2b7c491172abb70688d1d400` |
| Failed-audit setup reconciliation | `27cdc3ceddc4a36e7be6ba2805caad98c094fcd5bb85e80d1d2c9b469d5b99dd` |

The [independent report](ssl-synthetic-localization-ludb-landmark-independent-audit-v2-results.md)
and [scientific review](ssl-synthetic-localization-ludb-peak-timing-science-review-v1.md) provide the
verification scope. The separately prospective descriptive coherence bridge can now satisfy its
landmark prerequisite. It has not yet been executed. This work does not evaluate the learned
localizer on real ECGs, generated physiology, controlled disease or self-supervised benefit.
Clinical labels and closed tests remain unchanged; PR 67 stays draft.
