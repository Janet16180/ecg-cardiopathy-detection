# KEY-01 execution correction: numeric membership flags

3 November 2026, before new condition scores. The first attempt at source commit `ba53c05`
stopped in input qualification: converting SPH's literal `"0.0"` membership strings directly to
integers raised ValueError. Its `outputs/condition_keys_20261103/failure.json`, source binding
and run log remain unchanged. No keys, new predictions or performance metrics were produced.

The successor loader converts membership strings to floating-point numbers, requires every
value to be finite and exactly 0 or 1, then converts to integer. It preserves all record identities,
annotations and roles. Malformed or nonbinary values still stop the run. This changes parsing
only; the original prospective algorithm, primary comparison, controls, gates and cluster plan
remain unchanged. New files import the frozen scientific helpers; the first attempt's sources
are retained without editing.

Execute with `python -m scripts.experiments.run_condition_keys_v2`. Private output moves to
`outputs/condition_keys_20261103_v2/`; the aggregate report and notebook keep the original
KEY-01 identity and cite this correction. The successor independently records both protocols,
its committed sources, the exact predecessor replay and the same verified inputs.
