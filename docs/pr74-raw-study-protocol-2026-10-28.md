# Corrective evaluation with fresh raw-derived dependencies

This successor implements the already registered corrective comparison using new administrative,
cohort, feature and baseline receipts. It changes no target, split, recipe, threshold rule or
decision. The primary remains strict ages 18–25 versus the equal-size general-age control matched
on source, evidence rank and target, using the fixed 50:50 HGB/xECG blend on reused SPH development.
Require at least two percentage points sensitivity gain and a positive lower paired-patient 95%
bound (2,000 draws, seed 20261028). Broad 25k cohorts, components and PTB comparisons are secondary.

The default path reads no historical SCREEN/041/042 role, feature, model or prediction artifact.
Administrative roles come from the raw bootstrap; new cohorts come from the cold cohort pipeline;
the PTB-only reference and PTB/SPH features come from the fresh baseline bootstrap. All input
receipts, source hashes and artifact bytes are checked before fitting and again before completion.
PTB fitting is folds 1–7, calibration fold 8 and evaluation fold 9. SPH is never training. Original
Challenge calibration/test groups and EchoNext test remain closed to model operations.

New Challenge features use only the admitted original-training candidate union. Before extraction,
profile up to 16 rows per source with raw checksum and canonical-window checks, 130 fixed measures
and frozen released xECG. Require at least 4 GiB free GPU memory, less than 3 GiB added peak memory,
two-times measured extrapolation plus 300 seconds below two hours, and new features below 1 GiB.
Acquire the shared GPU lock and use 16-record batches. No backbone parameters are fitted. Preserve
the existing PTB float64 arithmetic and six-significant-digit measurement serialization.

An explicit warm-reuse option may validate this successor using the just-completed corrective
Challenge extraction. It must verify that extraction's source/input/artifact hashes and exact
candidate IDs, targets, original training roles and canonical window hashes. This is recorded as
warm reuse rather than a new cold extraction; the default runnable cold path has no such dependency.
PTB features and all baseline heads in this successor must come from the fresh bootstrap.

Fit all four fixed cohort heads before any successor evaluation score. Fit preprocessing only on
training rows. Separately calibrate each model and site using only normal calibration scores at
the 5% normal referral budget, strictly score > threshold, retaining ties as unflagged. Export all
15 models, confusion counts and descriptive threshold sweeps; keep the frozen operating point
separate. Produce the same aggregate-only UI schema and local model bundle as the original run.
No cohort is promoted automatically. The patient bootstrap conditions on fitted models and the
calibration sample. These hospital ECGs do not establish performance or prevalence for Italian
students, and these model scores are uncalibrated annotation-proxy scores.

For verification, compare newly generated aggregate metrics to the published original corrective
aggregate. Require exact confusion matrices and absolute AUROC difference at most 1e-9; report
numerical differences or failures transparently, rather than forging historical receipt hashes.
All patient-level artifacts remain local. Only aggregate results and hashes enter Git.
