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
