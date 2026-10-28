# Strict young and same-size general cohort protocol

This is prospective dataset construction from the completed, receipt-bound training candidates of
`outputs/cohort_age_label_priority_20261028/`. It creates new artifacts and preserves both existing 25,000-row
cohorts. No waveform, model score, Challenge final test or EchoNext test is opened by this stage.

The strict cohort includes every admitted candidate with an exact reported age from 18 through 25 inclusive.
Missing ages, censored ages, minors and older adults are excluded. There is no older-adult fallback. The number
of admitted strict-age records determines N; the code will neither weaken eligibility nor duplicate records
to reach 25,000. A metadata-only feasibility check found 770 such records before this protocol was written.

The general control takes N records from the complete admitted candidate pool, matching the strict cohort
exactly on source, label-evidence rank and binary target. Within each stratum the existing evidence-first
selector uses seed 42 with no age preference. The strict cohort also uses seed 42 for deterministic ordering.
The completed parent's human-label policy, proxy targets, original split assignments, quality decisions and
record/patient/hash exclusions are inherited unchanged. Both resulting cohorts must contain both classes.

The primary construction checks are exact strict-age membership, equal cohort sizes, unchanged parent targets,
unique record and waveform identities, and zero observed held-out identity/hash intersections. Any failed check
stops publication. Completion receipts bind the parent receipt and candidate-table hashes, this protocol,
implementation files, reference evidence and seed. No existing output is overwritten.

The general control matches sample size, source, evidence rank and class prevalence. It can include young
records, minors and unknown ages, and shared training records between arms are allowed and reported. It does
not match patient mix, disease severity or acquisition conditions; a later difference does not establish a
causal effect of age. These are ECG-annotation proxy cohorts, not diagnoses or a validated university
screening population. Challenge patient identities remain unknown. Any choice of a primary evaluation comparison
belongs to the separate evaluation protocol and the user's stated preference.
