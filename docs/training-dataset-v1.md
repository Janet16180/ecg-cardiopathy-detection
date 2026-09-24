# Canonical merged training dataset, version 1

`data/processed/training_union_500hz_v1/` is a separate data-scaling cohort. It does not replace the frozen 56,875-record experiment input. It supplies original twelve-lead ECGs as float32 `[12,5000]` arrays, 500 Hz, ten seconds, physical mV, in lead order I, II, III, aVR, aVL, aVF, V1–V6. No filtering, clipping, normalization, resampling or fitted preprocessing is applied.

The candidate union is the frozen **17,418 PTB training records + 39,457 accepted MIMIC records + 19,789 curated Challenge records**. Challenge comprises 10,187 Georgia, 6,581 CPSC 2018, and 3,021 CPSC-Extra records. It includes the 945 clean Georgia pilot records once and the 18,844 previously identified novel signals. The three constant-lead Georgia pilot records never enter this candidate union. They were not part of the current 56,875 PTB/MIMIC cohort, so they must not be subtracted from that number. The 24 unique Challenge records with verified signed-16 rail hits are already absent from the curated candidates.

The builder rereads every candidate and rejects any newly found full-constant lead. It fails on unexpected duplicate identities or incompatible waveforms. Accepted rows are copied into new standalone shards, with original paths, source manifest, source ID, crop/view bounds, signal hashes and review flags retained in `train_manifest.csv`. `exclusions.csv` lists any newly rejected candidates, and `historical_georgia_quarantine.json` preserves the earlier three-record evidence. The final counts and label-budget changes are recorded in `metadata.json`.

The accepted union contains **76,598 training ECGs**:

| Source | Accepted training recordings | Newly excluded constant-lead recordings |
| --- | ---: | ---: |
| PTB-XL | 17,417 | 1 |
| MIMIC | 39,392 | 65 |
| Georgia | 10,187 | 0 |
| CPSC 2018 | 6,581 | 0 |
| CPSC-Extra | 3,021 | 0 |

The additional PTB exclusion is `ptbxl:12722`, patient `ptbxl:2818.0`, whose **V5 is constant across all 5,000 samples**. It belonged to the full-label selection and was absent from the limited-label selection. Consequently the clean full-label budget contains **15,359**, and the limited-label budget retains **1,518**. All 65 additional MIMIC exclusions affect SSL only. No removed record is replaced and no patient is reassigned. The earlier frozen arrays and label files remain unchanged.

Only PTB supplies the diagnostic-annotation proxy labels. `labels_fraction1.csv` and `labels_fraction0.1.csv` preserve the retained members of the original 15,360- and 1,518-label training selections; no labels are imputed or reassigned. Challenge annotations and MIMIC diagnoses are not imported. A missing target remains missing. The endpoint does not establish that a person is healthy or validate referral decisions.

PTB development, calibration and test records are absent from every training shard. `heldout_references.csv` preserves their original record/label identity and the existing patient partition: 1,306 development, 564 calibration, 1,896 test. Development/calibration reproduce `ecg_experiment.evaluation.partition_validation`, `GroupShuffleSplit(test_size=0.3, random_state=9001)`. These are **references only**, not a newly processed or evaluated held-out dataset. Fit any downstream normalization only on selected training data.

PTB and MIMIC patient identifiers are namespaced. Challenge patient identifiers are unknown and deliberately empty, rather than exposing the record ID as though it were a patient. Challenge is SSL-only; exact waveform deduplication does not establish cross-source patient independence or exclude transformed near-duplicates. Do not create supervised patient-independent Challenge evaluation splits from this union. CODE's unresolved native units/padding, incomplete Chapman processing, and the unfinished larger MIMIC cohort are excluded.

## Train with the loader

```python
from torch.utils.data import DataLoader
from ecg_experiment.training_dataset import TrainingECGDataset

directory = "data/processed/training_union_500hz_v1"
ssl = TrainingECGDataset(directory, purpose="ssl")
ssl_batches = DataLoader(ssl, batch_size=32, shuffle=True, num_workers=2)
batch = next(iter(ssl_batches))
signals = batch["signal"]  # float32 [batch,12,5000], physical mV
# In SSL, every target is -1 and target_available is False.

labeled = TrainingECGDataset(directory, purpose="supervised", label_budget="0.1")
supervised_batches = DataLoader(labeled, batch_size=32, shuffle=True, num_workers=2)
# Only retained PTB rows in the frozen 1,518-label subset can be returned here.
```

The loader verifies manifest/label hashes at initialization, a shard's full SHA256 on first access in each process, and the per-record hash and waveform contract on every read. It bounds open memory maps and returns copies, so training transforms cannot alter stored arrays. It does not fit normalization, consume held-out references, or schedule a model run.

## Reproduce and verify

```bash
.venv-pretrained/bin/python -m pytest -q tests/test_training_dataset.py
.venv-pretrained/bin/python -u -m scripts.build_training_dataset \
  --output-dir data/processed/training_union_500hz_v1 --workers 4
.venv-pretrained/bin/python -u -m scripts.build_training_dataset \
  --output-dir data/processed/training_union_500hz_v1 --verify-only
.venv-pretrained/bin/python -m scripts.smoke_training_dataset \
  --dataset-dir data/processed/training_union_500hz_v1
```

The builder refuses an existing destination, holds a destination-specific process lock, and publishes a new directory atomically only after a full independent reread. On failure it removes only its own staging directory. The immutable metadata binds input manifests/receipts, code hashes, source revision, command, tables and all new shards; `verification.json` binds the completed metadata. PTB raw pairs must match official per-file checksums immediately before and after decoding. MIMIC signals must match the frozen preparation audit's per-record canonical hashes; headers must explicitly state mV. Challenge pointers must match the published manifests, curated receipt and prior overlap gate, and every selected array must match its published canonical hash. Passing these checks verifies the data contract, not clinical signal quality.

The 24 September 2026 build passed a complete independent reread of **76,598 rows in 599 shards**, occupying 18,383,596,672 waveform bytes (18.38 GB). Construction plus verification took 1,468.1 seconds. The [verification receipt](../outputs/data_quality/training_union_v1/verification.json) binds dataset metadata SHA256 `1ebac6308fadcdba676f11e7ad09ddb7786bdba8bccdd0ee136be93c2a6f147e`. The [release receipt](../outputs/data_quality/training_union_v1/receipt.json) records commands, hashes and exact archived source files; the [build log](../outputs/data_quality/training_union_v1/build.log) and [exclusions](../outputs/data_quality/training_union_v1/exclusions.csv) are retained alongside it.

The combined loader/overlap/materialization suite passed **23 tests**. The separate [real-data CPU smoke](../outputs/data_quality/training_union_v1/loader_smoke.json) checked batches from all five sources, masked all SSL targets, selected only the frozen PTB supervised subset, confirmed train/development/calibration/test patient separation within PTB, and completed a finite forward/backward optimizer step on an actual `[8,12,5000]` batch. It created no trained model artifact and is not a predictive-performance result. No GPU was used.

Future performance comparisons must explicitly call this a changed data cohort, report source composition and exclusions, and keep a matched frozen-pool control. No model performance result or architecture scheduling change follows from this release.
