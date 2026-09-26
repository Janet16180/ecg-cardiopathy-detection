# Do PTB-XL diagnostic groups form CPC neighborhoods, and do they transfer?

Protocol frozen before looking at outcomes, 26 September 2026. This is a
read-only exploratory analysis of the historical self-supervised CPC encoder.
It does not train a model, tune a classifier threshold, or inspect PTB-XL
calibration/test patients.

Use one seeded ECG per **PTB-XL training patient** from the frozen 40k CPC
cache. Join PTB-XL `scp_codes` to official `scp_statements.csv` using exact ECG
and patient IDs. A diagnostic code contributes a superclass only when its
released confidence is at least 50. Examine MI, CD, STTC and HYP separately;
labels can overlap. `NORM` is an ECG annotation, not patient health, and is
counted as a reference only when no other diagnostic superclass is present.
Rows with no qualifying diagnostic superclass remain in the geometry fit but
are excluded from each label-specific comparison.

Extract frozen 512-dimensional CPC pooled features. Fit PCA-16 **only on PTB
training patients**, normalize to unit length, and use Euclidean distance.
For each diagnosis, the primary statistic is the proportion of the ten nearest
other PTB training patients sharing that diagnosis among positive queries,
compared with its prevalence among all sampled PTB training patients.
Compare with 1,000 fixed-seed global label permutations; these exploratory
permutations retain each class count but cannot remove every acquisition or
clinical confound. Report the multi-label co-occurrence and exact cohort size.

Fit HDBSCAN on the same PTB vectors (`min_cluster_size=80`, `min_samples=20`,
excess-of-mass selection). Summarize every cluster and unassigned ECGs by
diagnostic superclass. No disease label enters PCA or HDBSCAN. A diagnosis
concentrated in a cluster is a post hoc enrichment, not a validated subtype.

**Cost amendment before any result:** The initial complete-cohort CPU command
was stopped after more than six minutes without a report. It produced no
outcome. Keep the full PTB patient cohort for PCA fitting and the primary
neighbor test, but perform HDBSCAN only on a uniform 4,000-patient subset
drawn without replacement with NumPy seed 42. This subset is exploratory and
its counts must not be presented as full-cohort cluster prevalence. The MIMIC
transfer comparison remains the original full one-patient-per-person sample.

For an out-of-source check, transform the frozen one-per-patient MIMIC sample
using the **PTB-fitted** PCA. Compare ten-nearest-neighbor homophily for the
exact cart-generated `Abnormal ECG` versus `Normal ECG` summaries, with 1,000
fixed-seed permutations within cart IDs. Also summarize those machine labels
in the original MIMIC clusters. These are broad machine interpretations, not
the four PTB diagnosis labels; this transfer check therefore tests general
ECG abnormality structure, **not** diagnosis-specific external validation.
No MIMIC or PTB label is used to fit the feature transform.

The development and test partitions remain closed. A stronger future test
would link independent cardiologist reports to MIMIC or use a separately
adjudicated cohort with the same diagnosis definitions. Neither data source
establishes a patient's complete clinical health from a ten-second ECG.
