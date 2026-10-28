# Cold-start cohort data pipeline

Build the same prospective evidence-first cohorts from local raw releases without any historical SCREEN-01,
041 or 042 scientific completion receipt, probability file, fitted model or feature cache. Released encoder
weights are a later evaluation-stage input, not a data-stage prerequisite. All completed prior outputs remain
immutable. This successor uses a new `cold_start/` namespace and honest fresh administrative provenance.

Prerequisites are raw PTB-XL, SPH and Challenge releases with their checksum/download manifests. Bootstrap
PTB/SPH administrative roles under the separate committed raw-role protocol. Recover original PTB manifests
with seed 42 and the original Challenge split with seed 20260929, requiring the exact frozen split hash.
All real resource profiles precede full raw processing; at most four CPU workers with one numerical thread.

Fresh reference rows combine every Challenge held-out fixed-window reference, PTB folds 8–10 and the newly
raw-derived SPH normal-calibration/evaluation rows. No historical SPH qualifier is required. Training cleaning
uses the frozen view, quality and duplicate helpers and rejects every known held-out record/patient/window/full
hash match. Only original 10-second Challenge records with defined primary targets can supply supervised labels.

The new cohort loader admits PTB folds 1–7 according to the fresh administrative roles and verifies every
common stricter original target against the fresh superclass target. Label-evidence policy, known-patient
namespaces, age interpretation, evidence-first selection and strict source/evidence/target matched control are
unchanged. All selections use seed 42. Original 25k and strict 770 receipts remain distinct from this successor.

Optional caches may accelerate execution only after their tables, code and input bindings pass. They are
recorded as explicit cache inputs and are not prerequisites: absent caches cause raw decoding. Profiles always
exercise real raw paths. New completion receipts identify the fresh roles, references, policy, protocol and
source code actually consumed. Cold construction checks compare new membership and targets with known cohorts
where available; differences are reported rather than repaired by weakening admission. No model is fitted or
scored by the data pipeline and no Challenge/EchoNext final performance stage is opened.
