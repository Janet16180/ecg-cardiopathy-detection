# Fresh administrative role and raw-quality bootstrap

Create new administrative artifacts from local PTB-XL and SPH raw releases. No previous model, prediction,
handcrafted-feature cache or historical scientific completion receipt is a prerequisite. Existing qualification
rows may be compared afterward as an optional integrity check; identical rows do not confer the old receipt's
provenance, execution time or model replay status.

PTB uses the frozen diagnostic-superclass label, original folds 1–7 fitting, fold 8 calibration and fold 9
external-development roles. Fold 10 waveforms are not read by this stage. The historical baseline excludes
only records with a lead constant for the full recording, using the original float64 raw format16 decoder;
other technical quality does not remove a baseline record. Canonical float32 hashes are retained separately.
All decoded PTB raw pairs must match official per-file checksums before and after reading.

SPH uses all 25,770 metadata records, frozen primary AHA labels, the first 5,000 samples in float32 mV,
unchanged quality assessment and frozen exact-window duplicate resolution on raw AHA strings. Every unique/kept
record with a defined primary label remains eligible irrespective of descriptive QC failure. Patient assignment
is the original fixed hash `sha256('experiment042:20261018:'+patient_id).digest()[0] < 128` for calibration;
others enter evaluation. Normal calibration records precede evaluation records in the admitted table.
Calibration positives remain present in the full administrative evidence but are absent from the admitted
normal-calibration/evaluation table. Metadata/code files are verified against the release download manifest;
each HDF5 file is fingerprinted before and after decoding. The extracted record archive is not required.

A real deterministic resource profile precedes full execution. Four CPU processes and one numerical-library
thread per process are the maximum. No model is fitted or scored, no fixed presentation deadline is inherited,
and Challenge/EchoNext records are not read. New receipts explicitly identify administrative/raw-quality scope,
raw input hashes, source and protocol hashes, fixed role seeds, per-source artifacts, actual counts and resources.
Existing outputs are never overwritten. A cold pipeline may reuse completed bootstrap receipts only after their
input, source and artifact bindings pass. All row data remain local; repository reports contain aggregates.

On the current local release, optional comparison targets are 19,229 admitted PTB records (14,954 fitting,
2,129 calibration, 2,146 development) and 17,402 admitted SPH records (6,968 normal calibration, 10,434 evaluation).
Compare identities, ordering, targets and canonical hashes to current qualification rows where available. Do not
change exclusions or role rules to force these counts. Any disagreement stops the corresponding integrity claim.
