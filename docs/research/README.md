# Research areas

Use these topic directories for new research. Existing frozen code, protocols, outputs and reports retain
their original paths. The topic indexes link those studies without duplicating or relocating them.

- [Localization](localization/README.md): reference matching, native signal regions and evidence annotations.
- [Clustering](clustering/README.md): physical morphology and learned representation geometry.
- [Explainability](explainability/README.md): grouped attribution, backgrounds and model dependence.

Execution state remains in the [queue](../experiment-queue.md) and its [JSON catalog](../experiment-queue.json).
Follow-up ranking remains in the [priorities](../experiment-priorities.md) and [backlog](../experiment-backlog.json).
A directory or index entry does not schedule a study.

## Layout for new work

| Content | Location |
| --- | --- |
| Reusable research code | `ecg_experiment/<area>/<module>.py` |
| Tests | `tests/<area>/test_<module>.py` |
| New experiment entry points | `scripts/experiments/<area>/<command>.py` |
| New report entry points | `scripts/reports/<area>/<command>.py` |
| Protocol and results | `docs/research/<area>/<study>/protocol.md` and `results.md` |
| Small aggregate artifacts | `reports/<area>/<study>/` |
| New schemas | `schemas/<area>/` |
| Private execution artifacts | Existing study-specific directories under `outputs/` |

Use `localization`, `clustering` or `explainability` for `<area>`. Create another descriptive area when the
work warrants it. Create a directory when it receives its first real file; empty placeholder directories
are unnecessary. Keep the existing shared bootstrap, paths and GPU helpers at their established imports.
Run nested commands from the repository root, for example
`uv run --no-sync python -m scripts.experiments.localization.run_ntt06` once that command exists.
That example names a future command, not a runnable experiment.

All new or edited file content must be in English, including comments, docstrings, documentation,
protocols, reports and user-facing text stored in the repository. Keep file names descriptive and English.
Before moving or editing an existing file, check the [frozen-source policy](../code-quality-review-2026-09-29.md).
New directories do not exempt code from prospective protocol, receipt, data or validation requirements.
