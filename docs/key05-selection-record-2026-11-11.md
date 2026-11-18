# KEY-05 selection record

11 November 2026. Written by `run_key05_condition_keys select` after every validation
analysis of [KEY-05](key05-condition-keys-protocol-2026-11-11.md) and before any test-role
label was read.

- Selected single key (K1 to K6), highest validation selection metric (within-source macro AUROC averaged over sources): **K6**.
- Selected K7: **K7-m4**.

Validation, v5 labels, 2,000 patient-cluster resamples (`outputs/key05_condition_keys_20261111/constructions.csv`):

| construction | selection metric (95% interval) | minus P0 | pooled macro AUROC (95% interval) | minus P0 |
|---|---|---|---|---|
| K1 | 0.9240 (0.9200, 0.9279) | -0.0458 | 0.9304 (0.9277, 0.9331) | -0.0478 |
| K2 | 0.9191 (0.9152, 0.9231) | -0.0507 | 0.9251 (0.9222, 0.9279) | -0.0532 |
| K3 | 0.8968 (0.8925, 0.9012) | -0.0729 | 0.8968 (0.8934, 0.9002) | -0.0815 |
| K4 | 0.9634 (0.9588, 0.9677) | -0.0064 | 0.9734 (0.9715, 0.9751) | -0.0049 |
| K5 | 0.9240 (0.9203, 0.9277) | -0.0458 | 0.9158 (0.9127, 0.9189) | -0.0625 |
| K6 | 0.9643 (0.9601, 0.9684) | -0.0054 | 0.9637 (0.9613, 0.9660) | -0.0145 |
| K7-m2 | 0.9602 (0.9557, 0.9644) | -0.0096 | 0.9690 (0.9671, 0.9708) | -0.0092 |
| K7-m4 | 0.9617 (0.9581, 0.9652) | -0.0080 | 0.9682 (0.9664, 0.9699) | -0.0101 |
| K7-m8 | 0.9542 (0.9503, 0.9578) | -0.0155 | 0.9619 (0.9601, 0.9636) | -0.0164 |
| N1 | 0.9249 (0.9196, 0.9299) | -0.0449 | 0.9348 (0.9315, 0.9381) | -0.0435 |
| P0 | 0.9698 (0.9664, 0.9728) | +0.0000 | 0.9783 (0.9769, 0.9795) | +0.0000 |

The selection metric is the within-source macro AUROC averaged over the six validation sources;
the pooled macro AUROC takes all validation records together. N1 and P0 are references, not
candidates.

Reading of the metric. The protocol's "validation macro AUROC" can mean either column. The selection
metric was named as the selection rule in a message to the orchestrator before any key was scored,
and kept after both columns were seen. The pooled column would select K4 and K7-m2 instead. No
reply had been received when this record was written. The validation outputs were produced at
commit `fad4ef5` (review sheets at `eb6f3db`); `validation_outputs_commit` below is the commit of
the `select` step.

```json
{
  "selected_single": "K6",
  "selected_k7": "K7-m4",
  "metric": "selection",
  "validation_outputs_commit": "519c745d31bfc100c502fd2da5246be13d981f55",
  "constructions_csv_sha256": "5611b6766c103a98119c25116b39cb36395c5cfd8f26a86b34ba5f560c5844a6"
}
```
