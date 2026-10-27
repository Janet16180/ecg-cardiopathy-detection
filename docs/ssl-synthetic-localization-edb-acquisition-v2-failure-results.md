# EDB released-roster acquisition failure and correction

23 October 2026. Actual42 acquired the four permitted metadata files successfully, then stopped at the published-linkage membership guard before requesting any ATR or header. No record annotation, waveform or model was read. This report is written by the executing coordinator from its retained receipts and official released roster.

The source description's range `e0118–e0122` was initially expanded into five presumed released records. The actual official RECORDS file excludes `e0120`; that range has only four released members. Projecting the published links onto the 90 actual released IDs gives group sizes 4, 4, 2, 2, 2, 2, 2 and therefore 79 components, consistent with the reported 79 subjects. The apparent 79-versus-78 contradiction was our preliminary contiguous-range assumption, not a demonstrated contradiction in the released cohort. The frozen original protocol/review remains historical evidence of that mistake and guard failure; this report corrects the current interpretation.

The grouping is still based on published constraints, not independently verified biological identities. No released record is removed or replaced, and no clinical comments or annotation contents were used to reconcile the count. The fixed role namespace, seed and thresholds can remain unchanged.

| Actual42 measurement | Value |
|---|---:|
| Successful metadata GETs / received payload bytes | 4 / 35385 |
| ATR/header requests / annotation or waveform decodes | 0 / 0 |
| CPU seconds | 0.087513604 |
| Wall seconds | 6.7196471858769655 |
| Reported peak RSS bytes | 1235140608 |
| Resource gates | All pass under prospective 2 GiB cap |

The terminal failure is `Missing or overlapping published linkage constraint`. Actual41's failed RSS guard remains included once. Cumulative actual42 is 1274.499965065 CPU / 564.4522484582849 GPU-lock seconds. Current failure receipt SHA-256 is `86e17e4c410423ba225e9f3cba8bb333b374458b55c83f54a001c24543e80fca`; terminal ledger SHA-256 is `93adb153cde92fcb205be6d7fe1a6ad7e2f5ac0a83810b347dac861f0fd8d20b`.

All four files (SHA256SUMS.txt, RECORDS, ANNOTATORS and annotations.shtml) match their completed transfer receipts; the three non-table files also match official SHA declarations. Their total 35,385 bytes and exact transfer-receipt identities are separately pinned in the prospective v3 recovery config. Independent metadata-only correction checking passed 14 checks; its setup receipt SHA-256 is `035441e291ce4e74eff3e82b44dae08543f27e480471aec99df031b81d04f3e7`.

Reuse these exact local metadata bytes in a new acquisition version rather than repeating their GETs. Correct only the nonexistent range member and expected grouping count, retaining all 90 records and their published links. A separate source/admission must precede the 180 remaining ATR/header GETs. No annotation success, model result or final-goal completion follows from resolving this metadata issue.
