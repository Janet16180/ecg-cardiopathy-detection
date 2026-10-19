# Experiment 041: forward cohort-validation correction

18 October 2026, before any new model fit or comparison score. The first attempt stopped in
cohort validation: the selected-feature cache had 19,229 labeled folds-1–9 records, while the
implementation expected 19,230. Its failure receipt and log remain unchanged in
`outputs/experiment041_cpu_screening_v1/` and the adjacent `.log` file. No integrity refit or
new model fit was reached.

The missing record has one exactly flat lead. The original `Avance2.8.ipynb` cell 24 explicitly
sets `eligible = quality.status != "derivacion_plana"` and `supervised = eligible & labeled`.
The prospective implementation overlooked that historical exclusion. The cached raw status
confirms the same rule. This is an input-validation correction, not a new quality policy or a
change to which rows the frozen comparison intended to consume.

Preserve the executed v1 module, runner and tests. Add a v2 module/runner that imports its
scientific helpers and verifies the selected cache against exactly the raw-metadata superclass
cohort after the notebook's pre-existing flat-lead exclusion. Verify raw quality identities and
consistency of flat-lead count/status; reject every additional missing/extra record, changed label,
patient mismatch or unsupported exclusion. Record the exclusion count and reason in the receipt.
Bind the notebook and raw quality fields as well as the original sources and both protocols.

All labels, roles, model settings, primary and secondary endpoints, bootstrap draws/seed,
threshold comparison and decision gates in [the frozen protocol](experiment-041-cpu-screening.md)
remain unchanged. No performance result has motivated this correction. Historical baseline
reproduction remains mandatory before the two new fits. Its original 1e-12 tolerances remain.

Use the exclusive new directory `outputs/experiment041_cpu_screening_v2/`, under the same
7,200-second launch timeout and four-thread cap. Stop on further integrity failure. Do not overwrite
or repair the failed attempt. Require artificial regression tests for the exact historical exclusion
and independent implementation review before launch. Experiment 042 binds the completed v2 041
artifacts as its predecessor; its extraction, model and scientific endpoint policies are unchanged.
