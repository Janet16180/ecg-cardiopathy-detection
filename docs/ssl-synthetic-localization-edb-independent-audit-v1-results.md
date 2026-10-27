# Independent EDB annotation audit v1 results

The independent audit passed with zero mismatches across all 90 released records,
129,600 label values and 129,600 reason values. It reproduced the completed main
qualification's engineering failure: combined annotation qualification remains
false. The root coordinator executed the committed independent runner locally;
the independent implementation agent reviewed the saved summary, ledger and all
90 completed comparison receipts to write this report. No further raw decode was
performed for this report.

## Executed evidence

The local directory is
`outputs/ssl_synthetic_localization/edb_annotation_qualification_v1/independent_audit_v1`.
The executed source admission SHA-256 is
`477df5d06500ce51d4394983e2ef3b77aae6fe15730294fb3ba250dd7e586549`.
The summary SHA-256 is
`de16a64e1271554307c7c99d850fc98e3805f2a8b9e2a5db3cb095b919e377e1`;
the terminal ledger SHA-256 is
`c20340a94f3cf4a2dccf56567b0baa028d3e14f4216745fce0cca6eb7b2caa00`.
The summary binds the original main summary
`2afa87395df76b7de980edb684d63deaa58adf42ffb46c98ca064cee12755343`
and its immediate actual ledger
`b8bd716df44281d03ed6d3c7f4e9e31318827c0736559feeb9665f3444e85f0e`.

All 90 attempted independent record decodes completed. Their 1,890 record/index
comparison flags passed, and every label and reason matched exactly. The retained
binary decoder errors total zero. The independent reconstruction reproduced all
802,909 raw events, 11,446 selected events, 3,030 annotation issues, 68 paired
episodes, 45 base episodes, 23 extreme episodes and 23 axis episodes. The episode
categories can overlap; these counts are not a partition.

The reconstructed availability contains 90 records, 79 published linkage
components and 180 observed channels. It has zero ST-positive windows, 239
reference windows and 129,361 UNKNOWN windows. The UNKNOWN values were never
counted as healthy negatives. All three role aggregates match the main outputs:

| Development role | Records | Components | ST-positive | Reference | UNKNOWN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train candidate | 55 | 47 | 0 | 0 | 79,200 |
| Validation candidate | 20 | 18 | 0 | 0 | 28,800 |
| Engineering candidate | 15 | 14 | 0 | 239 | 21,361 |

The identity receipt records 90 source-binding obligations, eight installed
runtime obligations, 406 main output hashes, 42 main source snapshots, ten
historical accounting obligations and 252 asset/acquisition obligations. These
are obligation counts, with repeated source checks retained. They are not counts
of distinct assets: the admitted acquisition contains 184 assets. Source,
original-asset and complete main-catalog identities were checked again after
reconstruction. The independent audit made no new WFDB call, waveform decode,
network request, model score or GPU operation.

## Resources and interpretation

The actual stage 45 consumed 2.894674603 CPU seconds and 4.719146547373384 active
wall seconds, with 49,684,480 reported peak RSS bytes. All CPU, wall, RSS, output
and network gates passed. Newly retained output before the terminal ledger was
951,510 bytes. The ledger adds this actual CPU cost once to stage 44, producing
1,287.186193435 cumulative current-machine CPU seconds; cumulative GPU wall time
stays 564.4522484582849 seconds. Earlier failed attempts remain in that lineage.
The ledger does not claim verified historical costs outside its declared chain.

`independent_audit_passed` is true, `main_engineering_passed` is false, and
`combined_annotation_qualification` is false. Agreement verifies that this frozen
v1 interpretation is reproducible from its admitted bytes. It does not make its
3,030 issues disappear, authorize downstream neural operations, establish clinical
pathology localization or turn its reference windows into verified healthy ECGs.
Any proposed semantic recovery needs a separate prospective source/output
admission; this audit's source and outputs remain frozen. All detailed artifacts
stay local, and all roles remain spent development data.
