# Experiment 018: frozen development readout results

The new 115,359-record CPC continuation **did not improve** the fixed
development readout over its matched old-pool continuation. With all 15,359
training labels, new versus old AUROC was 0.91795 versus 0.91809: a difference
of -0.00014 (95% paired patient-bootstrap interval -0.00324 to 0.00288).
The interval includes zero. This is an exploratory development result for an
ECG diagnostic-annotation proxy, not a clinical outcome.

| Training labels | Encoder | Development AUROC | Average precision |
| ---: | --- | ---: | ---: |
| 15,359 | Unchanged starting CPC | 0.92079 | 0.96145 |
| 15,359 | Old-pool continuation | 0.91809 | 0.95998 |
| 15,359 | New-cohort continuation | 0.91795 | 0.95986 |
| 1,518 | Unchanged starting CPC | 0.91296 | 0.95762 |
| 1,518 | Old-pool continuation | 0.90564 | 0.95370 |
| 1,518 | New-cohort continuation | 0.90658 | 0.95398 |

At 1,518 labels, new minus old AUROC was +0.00094 (95% interval -0.00220 to
0.00434), also inconclusive. Against the unchanged starting encoder, the new
continuation differed by -0.00284 at full labels (interval -0.00794 to
0.00257) and -0.00638 at limited labels (interval -0.01167 to -0.00114).
These latter contrasts are descriptive and were not the primary comparison.
All intervals used 2,000 paired patient resamples with no invalid draws.

The result answers a narrow question: one ordinary CPC pass, matched for 902
optimizer updates and 115,359 ECG exposures, with this frozen pooled-feature
logistic readout, did not show a downstream benefit from the larger mixed
cohort. More data may need different sampling, objective or training duration;
this experiment does not distinguish those causes. The historical PTB
development set has been inspected in earlier studies, so these numbers do
not establish external generalization. Calibration and test patients were not
used. The six logistic heads were fit only on their specified PTB training
labels, using the unchanged train-only normalizer and fixed classifier
settings in the [frozen protocol](experiment-018-cpc-data-scaling-readout.md).

The full readout used the passed V100 profile (projected 1,635.60 seconds
under the 7,200-second gate). Frozen-encoder feature extraction took 38.46
seconds after production input verification. The exact production and audit
commands were:

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_scaling_readout --stage run
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -m scripts.validation.audit_cpc_scaling_readout
```

The local [result receipt](../outputs/experiment018_cpc_data_scaling_readout_v1/result.json)
has SHA-256 `969bfb76c91d914af9795b849801b966d3baa87a9fc7ed0c666fe631d5763ac0`.
The independent [artifact audit](../outputs/experiment018_cpc_data_scaling_readout_v1/readout_audit.json)
rechecked feature and prediction hashes, dimensions, patient counts, AUROC,
average precision and all saved contrasts. The corresponding local feature
and prediction arrays remain outside Git with the training data. The V100 is
idle; no follow-up or test evaluation is scheduled by this result.
