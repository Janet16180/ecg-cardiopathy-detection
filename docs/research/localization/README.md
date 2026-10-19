# ECG evidence localization

New localization code belongs in `ecg_experiment/localization/`, mirrored by `tests/localization/`.
Place each new study's protocol and results in its own directory here. Existing studies below remain at
their recorded paths, including the frozen implementations they reference.

## Current direction

- [Research plan](../../ecg-reference-localization-plan.md)
- [Executed findings](../../reference-localization-findings-2026-10-18.md)
- [Draft specialist evidence annotation guide](../../ecg-evidence-annotation-guide.md)

## Existing studies

| Study | Protocol | Results |
| --- | --- | --- |
| NTT-01: reference annotation audit | [Protocol](../../ntt-01-reference-label-audit.md) | [Results](../../ntt-01-reference-label-audit-results.md) |
| NTT-02: native candidate-window matching | [Protocol](../../ntt-02-native-reference-transfer.md) | [Results](../../ntt-02-native-reference-transfer-results.md) |
| NTT-03: compact visual matching and proposals | [Protocol](../../ntt-03-visual-reference-pilot.md) | [Results](../../ntt-03-visual-reference-pilot-results.md) |
| NTT-04: same-wave finding discrimination | [Protocol](../../ntt-04-same-wave-finding-discrimination.md) | [Results](../../ntt-04-same-wave-finding-discrimination-results.md) |
| NTT-05: native signal adapter engineering | [Protocol](../../ntt-05-native-signal-adapter.md) | [Results](../../ntt-05-native-signal-adapter-results.md) |

The [native adapter implementation](../../../ecg_experiment/native_reference_ntt05.py) remains frozen at
its original location. Its engineering checks do not establish learned-native recognition or clinical
highlighting. Read the results and live output receipts before reporting progress.

Follow the [shared directory guide](../README.md) for new code, commands, schemas and aggregate reports.

## Native pretrained signal execution

[NTT-06 protocol](ntt06-native-encoder/protocol.md) and [results](ntt06-native-encoder/results.md)
connect the native adapter to frozen ECG-JEPA and return scored original-time regions without image
inputs. Native QRS finding discrimination reaches 91.67% versus physical references 92.78%; the
primary gate fails. Its shifted-window control reaches 83.33%, while specialist localization agreement
remains unmeasured. Private dense/pooled vectors are available for a separate clustering protocol.

[CLUSTER-03](../clustering/cluster03-native-reference/results.md) now clusters both the raw native
vectors and No Time to Train affinity outputs on a broader matched cohort. Stable partitions depend
on representation/reference choices; the failed persistence gate and phenotype limits are retained.
