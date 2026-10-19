# CLUSTER-03 CPU profile results

The corrected actual pretrained CPU profile passed all seven admission gates. The profile used the
exact permitted cohort of 1,993 candidates, a frozen synthetic input and the first four candidate ECGs
in numeric ID order. It did not compute disease or clustering scores.

The initial reader failed on the standard uppercase AVR/AVL/AVF spellings after one synthetic forward
and before encoding the first real ECG. Its typed failure was preserved after 8.835 seconds;
peak RSS was 1,433,366,528 bytes. The [execution amendment](execution-v2.md) was committed as
`10cddc1`; frozen original cache sources retain their paths and bytes. The successor canonical reader
and reusable execution implementation were committed as `61adaa0` before a fresh actual profile.

| Corrected profile measurement | Executed value |
| --- | ---: |
| Model load | 8.080 s |
| Entire measured stage | 11.921 s |
| Real-record operations, minimum / maximum | 0.582 / 0.659 s |
| Conservative projected extraction | 1978.036 s |
| Prospectively allowed extraction ceiling | 2,400 s |
| Peak RSS | 1,418,817,536 bytes |
| Synthetic / real forward counts | 1 / 4 |
| Model parameters frozen and finite native shape | Passed |

The projection is model-load time plus 1.5 times the slowest measured complete real-record operation
multiplied by 1,993. All four profile ECGs passed; none was replaced. The operation includes file hashes,
waveform reading, canonical ordering, automatic beat detection, actual native inference and pooling.
Each real operation was below the three-second gate, peak RSS below 6 GiB and profile time below 180 seconds.
This permits extraction under the new study budget; it does not admit model weight training or assert
that the whole extraction has already completed.

The completed profile receipt is `outputs/cluster03_native_profile_v2/result.json`, SHA256
`2a3b6ed09a7554de945f7d539eab925db488c7b5ccb2b8b2e740444afed2549c`.
Admission binds source/input/checkpoint/runtime identities, actual limits and final-status receipt.
The four profile ECGs are encoded again within the separate extraction stage; those extra profile
forwards are counted explicitly. No image entered the model, no GPU or closed data was used, and
all patient identities and waveforms remain local.

An independent read-only audit verified all 30 successor source hashes against disk and committed
bytes, all 30 immutable input hashes, the 26 unchanged first-failure source pins, the identity-only
1,999-patient source cohort and 27 reference exclusions, and the exact four profile records. It hashed
the checkpoint without deserializing it and independently reproduced every resource-gate calculation.
It accessed no waveform, target diagnosis or model. Extraction process environment was not separately
observable to that reviewer; the frozen admission code enforces the same declared runtime fingerprint.
The audit is local under `outputs/cluster03_native_reference_verification/profile_audit.json`.
