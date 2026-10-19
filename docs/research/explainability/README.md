# ECG model explainability

New attribution and model-dependence code belongs in `ecg_experiment/explainability/`, mirrored by
`tests/explainability/`. Place each new study's protocol and results in its own directory here.
Existing study paths are retained.

| Study | Protocol | Results |
| --- | --- | --- |
| SHAP-01: exact grouped linear explanations and background sensitivity | [Protocol](../../shap-01-linear-background-sensitivity.md) | [Results](../../shap-01-linear-background-sensitivity-results.md) |

Keep the explained model, output scale, feature grouping, background and replacement mechanism explicit.
Attribution agreement and clinical evidence agreement are separate questions. The
[localization area](../localization/README.md) contains the draft specialist evidence contract.

Follow the [shared directory guide](../README.md) for new code, commands and aggregate reports.
