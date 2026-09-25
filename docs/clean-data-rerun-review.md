# Preprocessing review and clean rerun plan

25 September 2026. This work audits data and prepares manifests. Training and
scheduling remain paused while another session verifies the refactored runtime.
Historical datasets, model checkpoints, protocols and receipts remain unchanged.

## Cohorts for a fair comparison

The published canonical union is a data-scaling dataset, with **76,598** training
ECGs. Its PTB/MIMIC subset contains **56,809** records: **17,417 PTB-XL** and
**39,392 MIMIC**. This is the original 56,875-record cohort minus **66 records
with a full constant lead**. The additional **19,789 Challenge** ECGs should be
evaluated in a separate scaling comparison.

The clean full-label selection has **15,359** records; the fixed limited-label
selection still has **1,518**. No replacement labels or patient reassignment are
needed. Development, calibration and test retain their original **1,306 / 564 /
1,896** records. Waveform quality checks on those records must not silently
remove difficult evaluation cases or use outcomes to decide exclusions.

“Clean” here means passing the recorded integrity and waveform checks. It does
not establish freedom from all artifacts, adjudicated labels, clinical health,
or suitability for a university screening population.

## Processing decisions

- Preserve canonical lead order, physical mV and original 500 Hz waveforms.
  No new denoising, clipping, baseline removal or record-wise scaling is justified
  by this audit alone.
- Detect constant leads **before resampling**. Filtering boundary effects can
  make a constant nonzero input appear variable after downsampling. This was
  demonstrated synthetically; none of the 66 actual excluded ECGs showed that
  effect, so it does not explain their historical inclusion.
- Preserve each model's documented input transformation. CPC uses independent
  five-second halves resampled from 500 to 250 Hz with `resample_poly`; JEPA uses
  eight selected leads and whole-record Fourier resampling. They are different
  checkpoint contracts. A common raw source does not make them interchangeable.
- A clean training run that refits preprocessing must fit it only on retained
  training records. Reusing a pretrained checkpoint with its historical
  normalization is a separate, explicitly named transfer experiment.
- Cached features may be reused for retained records only when waveform,
  preprocessing, checkpoint and record/patient identities match. Refit any
  downstream scaler/classifier on the retained training selection. Filtering
  features does not erase excluded records from a historical encoder's SSL
  training or normalization.
- Exact waveform deduplication does not detect every shifted/resampled copy or
  establish cross-source patient independence. Challenge patient identity remains
  unknown; its annotations remain excluded from this binary supervised endpoint.

## Audit coverage

The canonical union builder originally checked patient separation. Its current
independent verifier does not independently enforce every held-out relationship,
label-budget relationship, or lead-order metadata field. A separate preflight
therefore binds the published metadata and checks those relationships against
the frozen manifests, rather than treating self-consistent file hashes alone as
proof of correct semantics.

The new preflight produces a manifest-only PTB/MIMIC subset referencing the
existing canonical shards; it does not copy the 18.38 GB union. It checks every
held-out raw waveform against official source checksums and records descriptive
quality flags and exact waveform overlap. It does not evaluate model predictions.

Independent CPC input checks compare a deterministic sample of retained
canonical waveforms against the historical 250 Hz cache. This is a sampled
transformation check, not exhaustive verification of all cached waveform bytes.

The current canonical loader hashes shards on first access in each worker and
keeps only four memory maps open by default. The matched PTB/MIMIC rows touch
444 shards containing about 13.64 GB. Two workers that each visit all those shards
can read about 27.28 GB for integrity hashing alone; restarting workers each epoch
can repeat that cost. This is an I/O estimate, not a timed benchmark. Verify
integrity once through a bound receipt, use a bounded contiguous view for small
pilots, and measure a full data pass before forecasting training time. A change
to verification caching needs explicit file-identity/invalidation checks.

## Proposed rerun order after runtime verification and resumption

| Order | Comparison | Cost and interpretation |
| --- | --- | --- |
| 1 | Refit cached JEPA and CPC linear probes on the clean label selections; optionally repeat the 014 fusion control | CPU only if cache provenance verifies. Cheapest diagnostic rerun; encoders retain their historical training exposure. The limited-label membership is unchanged. |
| 2 | Repeat 017 no-branch / convolution / template arms with identical clean inputs and initialization | Five epochs at both budgets, development only. First reproduce seed 42 under the changed cohort, then matched seed 43 plus the required artifact checks. Historical warm six-arm training was about four minutes; this is not a forecast for the refactored data path. |
| 3 | Repeat 015 matched classification / cached-teacher distillation control if cleanup changes conclusions or a reproducibility check is needed | Historically inexpensive after loading, but its original limited-label screen was negative; lower priority than 017 replication. |
| 4 | Clean SSL pretraining and then a separately controlled 76,598-record scaling study | Required to claim clean pretraining or benefit from more data. Profile a full epoch including I/O before choosing a budget; not automatically a cheap experiment. |

These are preparation priorities, not an executable queue. Every successor run
needs a new protocol/manifest, current source hashes, cohort and normalization
identity, and the appropriate runtime check. Calibration/test remain separate
from architecture selection. No existing frozen runner should simply be pointed
at the larger union and treated as an equivalent rerun.

## Validation evidence

The complete [clean-cohort preflight](../outputs/data_quality/clean_rerun_preflight_v1/receipt.json)
passed. It reread and officially checksum-verified **all 3,766 held-out ECGs**,
confirmed the original patient partitions and nested label identities, and found
**no exact held-out waveform collision** against the union's pinned training
hashes or other held-out recordings. There were **no constant held-out leads**;
seven recordings have an amplitude-over-10-mV review flag. All remain in their
original evaluation partitions. The flags are not diagnoses or automatic rejects.

The preflight published six CSV files (about **17.8 MB** total), including the
56,809-row matched training manifest, both clean label selections, unchanged
held-out references, exclusions and held-out audit. No waveform arrays were
copied. Relative `shard` pointers resolve against
`data/processed/training_union_500hz_v1/`, **not** the preflight output directory.
These are preparation artifacts; the directory is not directly loadable with
`TrainingECGDataset`, and no successor experiment runner has been scheduled.
The audit reused published training waveform hashes; it did not reread all
18.38 GB of union shards. Output and current source hashes were independently
rechecked after publication.

Completed commands (use a new output directory for any rerun):

```sh
UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES='' \
uv run --no-sync python -m scripts.validation.prepare_clean_rerun \
  --output-dir outputs/data_quality/clean_rerun_preflight_v1
UV_CACHE_DIR=/tmp/ecg-uv-cache CUDA_VISIBLE_DEVICES='' \
uv run --no-sync python -m scripts.validation.audit_clean_cpc_inputs \
  --output-dir outputs/data_quality/clean_cpc_input_audit_v1
```

Initial focused CPU suite: **25 tests passed** for the canonical loader, CPC
preparation, bounded waveform cache and baseline data processing:

```sh
UV_CACHE_DIR=/tmp/ecg-uv-cache CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-sync python -m pytest -q \
  tests/test_training_dataset.py tests/test_prepare_cpc_data.py \
  tests/test_bounded_waveform_cache.py tests/test_data.py
```

The [CPC input audit](../outputs/data_quality/clean_cpc_input_audit_v1/receipt.json)
checked **64 retained records (32 PTB, 32 MIMIC)** and **all 66 exclusions**.
Every transformed waveform matched its historical 250 Hz cache row bit for bit
(maximum absolute difference zero). All 66 raw exclusions reproduced the recorded
constant-lead evidence. Six focused audit tests and its Ruff check passed.
This does not establish exhaustive cache equivalence beyond the sampled retained
rows. No waveform was rewritten or model executed.

The independent [scientific review receipt](../outputs/data_quality/clean_rerun_scientific_review_v1/receipt.json)
freshly checked all four full-budget JEPA/CPC feature and identity files against
the historical Experiment 014 hashes; all matched. Twelve additional retained
waveforms (six per source) matched the historical CPC transform bit for bit.
The receipt also records the shard-byte accounting behind the I/O estimate.

The [template donor check](../outputs/data_quality/clean_rerun_scientific_review_v1/template_donor_receipt.json)
found no QC-excluded record among the 32 template-initialization donors recorded
in each of the six historical 017 configurations. This only checks initialization
membership; it does not rule out artifacts elsewhere or remove historical
encoder/normalization exposure.

The two new audit test files pass **15 tests** together (nine preflight checks and
six CPC input checks); Ruff passes for all six new Python files. Together with
the initial existing-code suite, **40 targeted tests passed**. The new tests
cover altered raw bytes, wrong units, duplicate signals across partitions,
patient overlap, conflicting nested labels, incorrect lead-order metadata,
protected output paths and resampling behavior.

```sh
UV_CACHE_DIR=/tmp/ecg-uv-cache CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-sync python -m pytest -q \
  tests/test_clean_rerun.py tests/test_cpc_input_audit.py
```
