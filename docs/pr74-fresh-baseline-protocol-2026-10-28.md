# Fresh baseline bootstrap from local raw ECGs

This additive protocol removes historical SCREEN-01/041/042 cache prerequisites from scientific
reproduction. It creates new receipts with actual execution times and hashes. It never fabricates
historical scientific receipts or claims regenerated artifacts are recovered originals. The main
corrective execution retains its separate exact historical replay.

Inputs are the raw-derived administrative role builder's completed receipt and PTB/SPH row/pointer
tables, local raw PTB-XL 1.0.3 and SPH releases, and the released xECG checkpoint. There is no input
model, probability, embedding or historical measurement cache. The role builder enforces existing
labels, patient roles, duplicate resolution and the historical baseline admission: PTB folds 1–9,
excluding constant leads only; SPH keeps historical quality flags as descriptors. The stronger
quality policy for new supervised cohorts remains unchanged and separate from this legacy anchor.

All 19,229 admitted PTB records and 17,402 admitted SPH records receive the frozen 130 measurements
and released frozen xECG representation. Raw bytes and canonical float32 window hashes must match
administrative pointers before and after reading. PTB features retain original float64 integer/mV
arithmetic; SPH uses the first 5,000 samples widened exactly from float32. Serialization remains
%.6g. No fold 10, original Challenge held-out group or EchoNext test is scored.

Before full cold feature extraction, profile 32 PTB fitting records and 64 SPH admitted development
windows under the shared GPU lock. Require four GiB free, less than three GiB added peak memory,
less than two hours conservative two-times-plus-300-second forecast and less than one GiB new
feature storage. Default GPU0 is serialized with other project jobs. GPU1 may be used only with
its own explicit lock and the same real profile, without changing existing VLLM jobs.

Fit HGB and scaler/logistic xECG readouts solely on PTB folds 1–7, using the corrective study's exact
fixed configurations and seeds. Calibrate strict `score > threshold` using only each site's local
normal calibration scores at the five-percent normal referral budget. Preserve PTB fold 9 and SPH
evaluation identities. The fixed 50:50 blend is not retuned. This reproduces a known baseline on
reused development data; it does not add a new comparative claim or open a final test.

The scientific parity gate requires SPH HGB TP2395/FN1189/FP333/TN6517, xECG TP2407/FN1177/FP335/TN6515,
and blend TP2460/FN1124/FP346/TN6504, plus AUROCs 0.9079384857273202, 0.9131929011339939 and
0.9174945011079249 within 1e-9. Record actual numerical differences rather than assert identical
serialized model bytes. New artifacts, raw/feature/model hashes, environment and source/protocol
bindings remain local; only aggregate checks and reports enter Git. Scientific code is committed
before execution and rechecked before completion.

The fresh-input evaluator consumes these newly generated baseline features/models explicitly.
It does not require the old screened caches or the old deadline-based runner. New cohorts may reuse
this verified PTB feature table and extract admitted Challenge training features once. Document
which full cold stages were actually executed and distinguish representative raw-path tests from
a complete cold extraction. The user can rerun the complete cold path without patient artifacts
from this machine.
