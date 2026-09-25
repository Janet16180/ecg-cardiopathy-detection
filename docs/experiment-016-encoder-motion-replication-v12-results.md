# Experiment 016 v12: final design stopped before executable admission

The [frozen v12 design](experiment-016-encoder-motion-replication-v12.md)
(SHA-256 `4404711d47f346a1352b7b1030407a37c53cca6ec0ef34e027f77fa3b1b142d2`)
did not become an executable experiment. No v12 manifest was frozen, no real-data
admission or CPU check was invoked, no GPU bridge or production update ran, and
no seed-47 development prediction was generated. Calibration and test remained
closed. V9's seed-46 result remains the only scientific encoder-motion outcome.

Implementation review exposed an unresolved conflict in the frozen checkpoint
cost rule. It charges every second of an incomplete production arm to `E`, while
`n` stays at zero until that complete pipeline finishes, so the same arm also
remains in the full two-pipeline `Pstar` forecast. V12 requires a gate before M,
at safe checkpoints, after M/before F, and before reporting. Discounting the
in-progress arm at checkpoints would change the frozen equation; leaving it
fully charged has no demonstrated complete-path forecast under the narrow
planning envelope. No pre-outcome rule in v12 specifies how to resolve that
overlap.

For scale, the initial planning projection is **7,096.074827 s** versus the
**7,200 s** ceiling. If the frozen **1,025 s** `Q` allowance is spent before M,
the literal checkpoint projection is **6,839.824827 s**, leaving only
**360.175173 s** of additional `E` while M is still incomplete. The frozen
[v9 M production receipt](../outputs/experiment016_encoder_motion_v9/production/M/complete.json)
records **885.241474 s** for one complete pipeline; carrying that historical
duration while `n=0` would project **7,725.066301 s** before F. This is an
illustration of the rule's sensitivity, not a measured v12 runtime or proof
that every possible bridge duration would fail. It also cannot establish a
passing checkpoint gate. The required materializer, launcher, complete CPU
check, compatibility map and measured overhead inventory were not completed,
so no real-data gate could pass regardless.

The v11 closure remains useful historical evidence: it verified 288 leaves,
1,051 hash obligations and 6,493,086,486 bytes in 172.721 s, with a slower
observed pass of 176.727 s. Review of that code confirmed it returns a receipt
rather than the data objects needed by v9/v10. A v12 materializer would need to
return the validated clean train/development rows, cache row index and mapped
waveforms, released train/development features, and the exact pinned nested v9
fingerprint from the same verified graph. V10's full loader cannot be reused
inside either v12 process because it recursively rehashes the same large inputs.
These are implementation findings only; no partial prototype was admitted as
an executable source.

This is a **design/implementation stop before real-data invocation**, not a
seed-47 nonreplication result. It consumed no v12 executable-study stage time;
engineering review is separate from the frozen `E` ledger. V10/v11's at least
approximately **1,666.763 s** spent remains separately recorded with its
uncertainty and omissions. The frozen v12 protocol and all older receipts were
left unchanged. There is no automatic v12 retry, production successor,
calibration/test promotion, or v13/reset fallback under this attempt.
