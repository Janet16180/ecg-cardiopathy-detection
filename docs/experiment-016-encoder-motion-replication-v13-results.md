# Experiment 016 v13: prospective seed-47 attempt closed before admission

The [frozen v13 protocol](experiment-016-encoder-motion-replication-v13.md)
did not admit a real-data bridge or any GPU work. Its required pre-GPU
phase-bound audit could not support the first mandatory 40-update cost gate.
There is no seed-47 replication result. The completed [v9 seed-46 result](experiment-016-encoder-motion-v9-results.md)
remains the only encoder-motion outcome; no calibration or test endpoint was
opened.

V13 corrected the incomplete-arm arithmetic in v12, but its initial
7,096.075-second envelope was only a design calculation. At the first M
checkpoint, 40 completed updates earn `f=40/480=1/12`. Assuming the entire
1,025-second preparation allowance is met exactly, and retaining the full
1,098.411-second forecast for unstarted F and 300 seconds for reporting, the
frozen equation allows the first M block **at most 117.504 seconds**. Any
preparation overrun lowers this limit. The completed v9 profile receipts report
whole-pipeline times of 1,098.411 seconds for M and 1,059.647 seconds for F;
production reported 885.241 and 1,070.049 seconds. Their per-update logs and
queue console logs have no block timestamps. The v9 runner also recorded
`complete_pipeline_seconds` before hashing final checkpoints and features,
so that timer does not cover those mandatory artifact reads.

The historical total duration cannot establish that the first 40 updates,
initialization, due diagnostic and checkpoint will fit 117.504 seconds. For
illustration only, assigning the slower complete-pipeline duration to the
unknown first block is consistent with the available phase timing evidence;
the frozen checkpoint formula would project 21,668.377 seconds. This is a
conservative planning scenario, **not a measured v13 checkpoint**, and it
does not prove every possible run would be that slow. No authenticated phase
data support a tighter bound across all mandatory 40-update and tail
milestones. The protocol explicitly requires a no-go before GPU when those
bounds cannot be supported. A short bridge might measure its own first three
updates, but it would not supply authenticated durations for 40-update blocks
or both full extraction/refit and replay tails; it cannot retroactively satisfy
the required pre-bridge audit.

The [stop receipt](../outputs/experiment016_encoder_motion_replication_v13/stop_receipt.json)
contains the arithmetic and a prospective executable inventory. That inventory
includes two fresh 6.493 GB historical closure passes (288 leaves and 1,051
hash obligations per pass), two approximately 684.6 MB bridge checkpoints,
two approximately 753 MB final arm artifact sets, their initial hashes and
report rereads, the complete CPU check, GPU bridge, bootstrap/report, completion
audit and safe-stop reserve. These are planned obligations and historical size
proxies, not v13 execution measurements. Neither a complete materializer nor a
compatibility receipt, executable manifest or CPU check was created after the
phase audit failed. The small new [cost helper](../ecg_experiment/xecg_encoder_motion_v13_cost.py)
implements the prospective remaining-work formula; four synthetic tests pass
and Ruff passes. The shell-recorded nonoverlapping executable check time is at least 9.394
seconds, with unmetered launch/edit overhead, so it is not a complete v13
elapsed-cost receipt or a measured admission gate.

No real-data admission invocation, GPU bridge update, production arm or
seed-47 development prediction occurred. V10 spent approximately 1,086.936
seconds and v11 at least 579.827 seconds in their separate failed attempts;
at least approximately 1,666.763 seconds preceded v13, with the recorded
uncertainty retained. Historical profile charge `H` is a planning term, not
new compute. V13 is closed without retry or resume; any changed budget or
cost policy requires a separately prospective decision and identity.
