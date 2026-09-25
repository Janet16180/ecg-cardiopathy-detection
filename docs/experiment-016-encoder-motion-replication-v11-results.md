# Experiment 016 v11: seed-47 replication stopped at the cost gate

The [frozen v11 protocol](experiment-016-encoder-motion-replication-v11.md)
did not reach GPU bridge updates. The cheap CPU preflight passed with the
installed PyTorch 2.6.0+cu124 CUDA API. A fresh transitive verification pass
then read **6,493,086,486 bytes** across **288 unique leaves**, resolving
**1,051 logical hash obligations** in **172.721 seconds**. It checked the v9
profile/full completion state, H/P cost evidence and vendored xLSTM source
tree. The reconstructed v9 input fingerprint matched the immutable v10 CPU
receipt exactly: 15,359 clean training ECGs, 1,306 development ECGs from
1,173 patients, the excluded ECG 12722, fixed split/label order, cache,
released weights and feature/probe relationships. This is provenance and
input evidence, not a seed-47 performance result.

The [persistent v11 cost ledger](../outputs/experiment016_encoder_motion_replication_v11/cost_ledger_v2.json)
charges the initial incomplete scan, failed expanded scan, semantic debug
rechecks, successful complete scan, two CPU M/F preparation probes, manifest
freeze and preflight. Its conservative **579.827-second lower bound** for new
v11 work is separate from v10's approximately 1,086.936 seconds of failed
work. The slower measured full leaf pass was **176.727 seconds**. Even assuming
a single-process bridge launcher and a single-process production successor,
the frozen rule requires one fresh pass for each distinct launch, the full
300-second bridge cap, and the 300-second reporting reserve:

```text
H + E + 1.25 * (2*P + Qremaining + R)
= 2693.797 + 579.827 + 1.25 * (2*1098.411 + 2*176.727 + 300 + 300)
= 7211.468 seconds > 7200 seconds.
```

This is already over the ceiling **before** the complete v11 CPU correctness
check, production-callable compatibility map, launcher/manifest checks, GPU
host/device preflight or artifact audits. A conventional coordinator with
another source scan would cost more. The [cost gate](../outputs/experiment016_encoder_motion_replication_v11/cost_gate_v2.json)
therefore failed, and the [stop receipt](../outputs/experiment016_encoder_motion_replication_v11/stop_receipt_v2.json)
records zero bridge/production updates and zero development predictions.
Calibration and test stayed closed. The v9 seed-46 development result remains
the only encoder-motion outcome; the seed-47 replication question is unresolved.

The reviewable [preflight-only manifest](../outputs/experiment016_encoder_motion_replication_v11/preflight_v2/manifest.json)
has SHA-256 `9aa92d5373121176309800a81e35f518c39ad49ea0ea4d8bfd641f98d89646da`
and source-map SHA-256
`be9546033054370055dca48077fa91f5addcf69a0dafbf8df54f2ff76c2d44c0`.
It ran from source revision `eedff79466c6fbafc7c4ece257f86520b4496659`
with:

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_xecg_encoder_motion_replication016_v11 --stage preflight --manifest outputs/experiment016_encoder_motion_replication_v11/preflight_v2/manifest.json --manifest-sha256 9aa92d5373121176309800a81e35f518c39ad49ea0ea4d8bfd641f98d89646da --device cpu
```

No bridge-only or production manifest was frozen. The CPU fake-CUDA tests
exercise the exact `torch.cuda.device_count()` helper, missing/malformed device
queries, wrong seed and failure before a historical large-file hash. A GPU
device receipt was not attempted; the sandbox reported CUDA unavailable to
PyTorch, and the failed cost gate forbade an escalated launch.

Before any future seed-47 attempt, freeze another versioned protocol with an
honest cost decision. Do not treat the verified byte closure, partial CPU
probes or preflight receipt as a completed replication or relax the v11
scientific controls after this stop.
