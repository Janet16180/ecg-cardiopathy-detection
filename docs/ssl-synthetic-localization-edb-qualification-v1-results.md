# EDB first annotation qualification results

Actual44 completed all90record calls from committed source938368e and literal184asset/
42source admission3dd2b48. All WFDB effective-view, asset/source and resource checks
pass. Engineering qualification FAILS:89recordings have3030unresolved annotation
issues. Independent reconstruction is pending. Technical execution completion does
not authorize downstream learning.

All90recordings/180channels/79published components and129600channel-windows remain.
Record issues force UNKNOWN with reason128alone. No record replacement, favorable
subset, unknown-to-negative conversion, waveform/model/GPU operation or fit occurred.

| Saved quantity | Count |
| --- | ---: |
| events | 802909 |
| selected_events | 11446 |
| issues | 3030 |
| episodes | 68 |
| base_episodes | 45 |
| extreme_episodes | 23 |
| axis_episodes | 23 |
| ST_positive_windows | 0 |
| reference_windows | 239 |
| unknown_windows | 129361 |

| Issue | Count |
| --- | ---: |
| unsupported_episode_candidate | 1659 |
| orphan_peak | 651 |
| unknown_quality_subtype | 677 |
| orphan_end | 13 |
| nested_same_key_begin | 4 |
| extreme_T_parent_count | 19 |
| unclosed_episode | 6 |
| missing_episode_peak | 1 |

ZeroSTpositive and239reference windows survive the frozen contract; reference does
not mean healthy. Most issues precede interval reconstruction, so this does not prove
the dataset lacks usable episodes. A saved unsupported-marker hash matches exactly
the documented `(ST1+` marker plus one terminal NUL byte. General WFDB quality-bit
semantics may also explain unrecognized subtypes. These are interpretation hypotheses;
any recovery needs new source review, prospective semantics and independent checks.
The original contract/FAIL/arrays remain immutable.

Local arrays labels.npy:uint8[90,2,720] and reasons.npy:uint16[90,2,720], complete record
index, selected events, candidates, issues, quality spans, signed/reason/quality
statistics and source snapshots remain under
`outputs/ssl_synthetic_localization/edb_annotation_qualification_v1/qualification_v1/`.
Detailed payloads/identifiers are not committed. No diagnosis or clinical validation
is established by annotation availability.

CPU:8.927878689s; wall:12.882999735s; reported peak RSS:
95645696bytes. All resource gates PASS. Addressed-file resource bytes count
read-only ATR alias targets conservatively, while payload catalogs exclude symlinks.
Cumulative44CPU:1284.291518832s;GPU-lock:
564.452248458s. Directprior43 preserves failed41/42
costs exactly once. Setup/tests remain separate; historical missing costs unverified.

Summary SHA:`2afa87395df76b7de980edb684d63deaa58adf42ffb46c98ca064cee12755343`.
Ledger SHA:`b8bd716df44281d03ed6d3c7f4e9e31318827c0736559feeb9665f3444e85f0e`.
Admission SHA:`7d49993d18b49243054dcc2f9cef9f6f3685a357d5ff19ae324f19649b28763c`.
