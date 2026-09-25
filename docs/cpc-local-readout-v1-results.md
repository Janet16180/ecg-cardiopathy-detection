# Compact CPC local readout v1: development result

**Status, 25 September 2026:** completed CPU-only frozen-feature screen. The
prespecified point-estimate screen passed at both label budgets. This is an
exploratory development result, not an independent replication or a calibration/
test result.

The two arms used the same historical 20-epoch seed-42 ordinary compact CPC
encoder, unchanged normalization and 009 cache. Both heads had 512 inputs and
513 trainable parameters. A used context mean and context max; B replaced only
the 256 context-max coordinates with normalized local-token maxima. Each
train-only scaler and unweighted L2 logistic head used fixed `C=0.01`, L-BFGS,
float64, `tol=1e-8`, at most 5,000 iterations and one CPU/BLAS thread. No
encoder update or waveform extraction occurred.

| Training labels | A AUROC | B AUROC | B−A AUROC | Paired patient 95% interval | A/B AP | A/B log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,518 | 0.91288 | 0.92199 | **+0.00911** | +0.00278 to +0.01540 | 0.95752 / 0.96183 | 0.36231 / 0.34428 |
| 15,359 | 0.92050 | 0.93299 | **+0.01250** | +0.00639 to +0.01900 | 0.96120 / 0.96694 | 0.33803 / 0.31565 |

The 2,000 paired bootstrap draws sampled the same 1,173 development patients
for both budgets and kept each patient's ECGs together. Seed was 250925; zero
draws were invalid because of a single class. The limited-label contrast
exceeded the prespecified +0.002 threshold and the full-label contrast exceeded
the −0.002 nonharm threshold. Both intervals exclude zero. This supports the
specific equal-width feature replacement for this frozen encoder and cohort.

The runner rehashed the historical 117.5 MB feature cache and its rows, checked
the recorded shape/dtype and branch order, then explicitly joined ECG and
patient IDs to the clean manifests. It selected only the 15,359 full training,
1,518 nested limited training and 1,306 development numeric feature rows.
Training and held-out patients were disjoint. Parameter serialization and
float64 logit replay were exact for all four heads. No calibration or test
numeric rows, labels or outcomes were evaluated.

Before development scoring, the measured gate included a full training-only
fit of each arm and a synthetic 2,000-draw paired-bootstrap timing run. It
projected **284.93 seconds** against the 7,200-second ceiling; observed total
was **42.95 seconds**, with peak RSS **422,784 KiB**. The projection included
120 seconds reserved for reporting and writing.
The process measured cache hashing and row loading, but an earlier input-shape
check may have warmed the operating system page cache; the 0.85-second combined
I/O observation is therefore not a guaranteed cold-disk measurement. Even the
conservative projection left 6,915 seconds of margin under the fixed ceiling.

The immutable local [manifest](../outputs/cpc_local_readout_v1/manifest.json)
has SHA-256 `f6a0873e698dd0420144be48b0cf94285e487fd79f25d5978397e2d4b1e7c4a2`.
It binds new source hashes, historical cache/row hashes, encoder checkpoint and
normalization identities, clean input receipts, software versions and fixed
analysis choices. The aggregate [report](../outputs/cpc_local_readout_v1/report.json)
has SHA-256 `f3c2bd9d0301f7a7ab69f39cc65515813f5b2b686d6cee9b874a02dc31c32c06`;
the [cost gate](../outputs/cpc_local_readout_v1/cost_gate.json) and
[completion](../outputs/cpc_local_readout_v1/completion.json) preserve timing,
identity and replay checks. Exact command:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_local_readout --manifest outputs/cpc_local_readout_v1/manifest.json --manifest-sha256 f6a0873e698dd0420144be48b0cf94285e487fd79f25d5978397e2d4b1e7c4a2
```

The encoder was trained historically on the original pool, before the clean
supervised exclusion. Development patients have been inspected in earlier
studies, and this is one frozen encoder seed; the bootstrap reflects patient
sampling uncertainty, not that repeated development exposure or retraining
variation. Passing the point screen supports discussing a separately frozen
replication. It does not open calibration/test or establish person-level health
or referral utility; the endpoint remains an ECG diagnostic annotation proxy.
