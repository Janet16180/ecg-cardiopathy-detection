# Experiment 016 v10: seed-47 replication stopped before GPU work

The [frozen v10 protocol](experiment-016-encoder-motion-replication-v10.md)
asked whether v9's seed-46 encoder-update/readout-gap pattern recurs with a
seed-47 record order. **There is no v10 replication result.** The versioned
bridge failed during device verification, before either arm made a GPU update.
No seed-47 production pair, development predictions, bootstrap, calibration
evaluation or test evaluation was produced. The immutable [v9 result](experiment-016-encoder-motion-v9-results.md)
remains a one-optimization-seed development finding.

The real-data CPU check passed in 452.593 seconds. It rehashed the frozen v9
profile/production maps and completion artifacts, checked the same released
initial tensors and four training ECG logits, verified fresh optimizer state
and seed-47 record order, and confirmed the inherited training loop's AST
after the two allowed seed/identity and checkpoint cost guards. The
[CPU receipt](../outputs/experiment016_encoder_motion_replication_v10/check.json)
has SHA-256 `5279da4935c79b63d78b0247acf7e49e55c16ed1e81bfdec0174aca94b51f03a`;
the [compatibility receipt](../outputs/experiment016_encoder_motion_replication_v10/compatibility.json)
has SHA-256 `892a7d9a8ebeeba6f15213f4b2d3bfbce10d909053c5c93a5bde9f72dea9ac67`.
Four focused tests passed and Ruff passed on the frozen v10 sources. Those
checks did not exercise the CUDA device-query helper.

The [bridge-only manifest](../outputs/experiment_queue_016_encoder_motion_v10_bridge/queue.json)
and [source map](../outputs/experiment_queue_016_encoder_motion_v10_bridge/sources.json)
were frozen at SHA-256 `95e67edf19aa7329ee1106179ed291e432621627408007b0d4268cfd6acd60e9`
and `ba4c27e1608032aa8d12541a242f6c5b512cc9770519eb8154c5d50def1fb676`.
The coordinator's `--check` passed before launch. The exact launch command was:

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.coordination.run_priority_queue --manifest outputs/experiment_queue_016_encoder_motion_v10_bridge/queue.json --manifest-sha256 95e67edf19aa7329ee1106179ed291e432621627408007b0d4268cfd6acd60e9
```

The coordinator launched at about 17:52:40 UTC on 25 September 2026; the child
started at 17:54:53.197969 UTC and failed at 18:01:44.342708 UTC. The
[frozen log](../outputs/experiment_queue_016_encoder_motion_v10_bridge/job/priority_queue.log)
records `AttributeError: module 'torch.cuda' has no attribute 'get_device_count'`
in the new `_driver_receipt()`; PyTorch's function is `torch.cuda.device_count()`.
The [failed status](../outputs/experiment_queue_016_encoder_motion_v10_bridge/status.json)
has no completion marker. The log SHA-256 is
`d61fd01c508c372a656ff85dc541c7de9e94be255d7c6cf867cb1db083512a4c`;
status SHA-256 is `ed5d013b95d84d388a0ac322608675fe1aadbaa912eaf59f7059ded813af215c`.
Neither frozen source nor manifest was edited after failure.

The inherited two-hour gate leaves at most **1,385.175 seconds** for all new
compatibility, check, bridge and production preparation. CPU check (452.593 s),
the first source-manifest check (about 90 s), and the failed bridge coordinator
wall (about 544.343 s) already used about **1,086.936 s**, leaving about
**298.239 s**. These wall figures use the approximately recorded coordinator
launch time; the child timestamps are exact. Repeating just the observed
411.145-second pre-GPU child path would bring the conservative projected suite
to **7,312.905 s**, above 7,200 **before** any bridge updates, replay or fresh
production preparation. Warm-cache timing could change, so this is a measured
planning assessment rather than an observed completed-suite runtime. There is
no defensible passing gate for a corrected rerun under this protocol; none was
launched. The machine-readable [stop receipt](../outputs/experiment016_encoder_motion_replication_v10/stop_receipt.json)
has SHA-256 `82214c2975a15d51d2a22c33ce9766a20a53f566864fad2b9068fdcb245cdb0a`.

Future work would need a separately versioned bridge source/manifest fixing the
device query and a newly frozen cost decision that accounts for repeated full
input verification. The seed-47 outcome and v9 prediction remain unresolved;
no calibration/test gate has opened.
