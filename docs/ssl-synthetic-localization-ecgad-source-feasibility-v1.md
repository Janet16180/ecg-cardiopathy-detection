# ECGAD benchmark source feasibility

Documentation/source-only review,22October2026. ECGAD is a closer input-geometry
candidate for the final ten-second twelve-lead goal than two-channel ambulatory
ST episodes. Its existing project holdout is spent,and the current source review
does not establish a newly usable clinical benchmark. No benchmark array,mask,
waveform,record identifier or new model output was downloaded/read here. The first
EDB acquisition/qualification/audit continues unchanged.

## Primary evidence inspected

- [Original paper,arXiv2308.01639v1](https://arxiv.org/html/2308.01639),especially experiments and appendix.
- [Authors' official repository/README](https://github.com/MediaBrain-SJTU/ECGAD).
- [Data loader](https://raw.githubusercontent.com/MediaBrain-SJTU/ECGAD/main/dataloader.py).
- [Evaluation source](https://raw.githubusercontent.com/MediaBrain-SJTU/ECGAD/main/test.py).
- [Preprocessing source](https://raw.githubusercontent.com/MediaBrain-SJTU/ECGAD/main/data/preprocess.py).
- [Normalization helpers](https://raw.githubusercontent.com/MediaBrain-SJTU/ECGAD/main/utils.py).
- [Repository license](https://github.com/MediaBrain-SJTU/ECGAD/blob/main/LICENSE).

The README links a [public Benchmark Drive folder](https://drive.google.com/drive/folders/1STvUtzuOUovFT4YiQR0H-KTaZr0YnPZ-?usp=sharing)
and a Baidu alternative. Only its landing-page metadata was opened; no binary
request was made. Repository URLs above are mutable main-branch views,not executable
admission identities. Git revision lookup from this restricted shell failed DNS;
an immutable source revision and literal artifact identities must be obtained and
committed before any eventual scoring. An accessible README is not payload receipt.

## What can and cannot be established from these sources

| Item | Source-supported statement | Unverified issue requiring a bounded qualification |
| --- | --- | --- |
| Expert annotation | Paper reports400ECGs,22abnormality types and two experienced cardiologists providing signal-point labels | No per-case annotation process,reader agreement/adjudication measure,uncertain mask,exhaustiveness criterion or exact boundary tolerance was found in the inspected paper/README |
| Input geometry | Official code assumes twelve channels,500Hz and5000time samples; original source recordings are ten-second PTB-XL ECGs | Actual downloaded shapes,dtypes,channel order,sampling coordinates and label values remain unread; a400×5000×12array is expected,not locally verified |
| Artifact names | README/data loader name benchmark_data.npy and benchmark_label.npy | Current file identifiers,archive contents,byte counts,checksums and whether mapping/license files accompany them remain unknown |
| Source linkage | Published preprocessing uses PTB-XL1.0.1 and separates source fold10 from other folds | This script does not build the benchmark arrays or export their ECG/patient IDs; it cannot establish the benchmark's exact source membership or ordering |
| Rights/provenance | Repository LICENSE is MIT for software/documentation; PTB-XL is the underlying source | A separate mask-package license and any package-specific provenance/readme must be checked; software licensing alone does not identify data provenance |
| Historical status | Existing research plan marks ECGAD holdout spent; prior setup reports missing arrays/match table | Old absent files are not locally verified evidence and cannot be recovered from the other checkout; a new official acquisition gets new receipts,costs and identities |

The original paper's patient-level detection counts and point-label counts are
different quantities. Its PTB detection cohort sizes do not provide a400-case
patient linkage table. The later claim that190cases have nonempty abnormal masks
must not be substituted for an unperformed check of the actual release.

## Preprocessing and evaluation implications

The loader applies a separate per-record/per-lead min--max transform to the benchmark.
The helpers contain no flat-range guard. Evaluation then takes samples100:4900,
produces twelve-channel maps and calculates one AUROC after flattening records,
time and leads. This source-defined central4800-sample comparison is not the same
as a per-record localization endpoint. A method assigning generally higher scores
to entire abnormal records can benefit in a globally pooled statistic without
correctly ranking the supporting regions within each record.

The public preprocessing script applies a1Hz high-pass filter,a35Hz notch operation
and25Hz low-pass filtering via HeartPy; normalization and R-peak failures affect
its saved training/test arrays. The benchmark construction path is not present in
the inspected preprocessing code. Consequently,do not assume its released arrays
are raw physical-mV signals,that the same filters produced them,or that an identical
recipe can exactly reconstruct them from local PTB-XL1.0.3. Filters and independent
lead scaling also limit interpretation of clinical amplitudes. Source version,
operator order,lead identity and mask coordinates need literal verification.

The official train.py evaluates its named test set each epoch and retains a
checkpoint when that detection AUC improves. That selection recipe must not be
adopted as an untouched-test protocol. The current synthetic-localizer states are
already fixed; any future comparison must preserve their identities and declare
the whole existing ECGAD panel development data rather than selecting a favorable
fit,subset,crop or loss after seeing its masks.

## Prospective qualification needed before use

An eventual NEW bounded protocol could authorize official public artifact acquisition
and metadata-only NumPy header/file inventory first,with no pickle execution,
external data upload or source/record scoring. Pin the source revision,original
published link,license/readme bytes,current download identities and retained costs.
Actual package sizes must be bounded prospectively rather than inferred from the
expected array dimensions. Keep payloads/identifiers local under baseline outputs.
Do not invent an upstream author checksum if the link provides none: record
first-use hashes and commit the resulting asset admission before inspecting labels.

A separate committed label/source qualification then needs to establish:

1. Exact signal/mask shapes,dtypes,finite/binary domains,axis/lead/time correspondence,
   mask-support distribution and whether each sample has both target classes.
2. Original PTB ECG/patient identifiers through a released map or an independently
   verified unambiguous deterministic signal match. Approximate nearest-neighbor
   similarity alone is not a proven identity. Do not borrow the old missing match
   table or claim disjoint patients merely because a preprocessing script usesfold10.
3. Duplicate/repeated-patient groups and overlaps with all local real training,
   calibration/evaluation roles,including the fixed40mixed-training patients.
   Missing linkage remains a documented limitation and blocks a patient-disjoint
   fitting/clinical claim; it cannot be repaired by arbitrary row-based splitting.
4. Mask provenance,coverage and uncertain/unlabeled interpretation from existing
   released materials. Zero means unmarked under the supplied annotation,not
   universally healthy tissue. No model consensus or prediction can validate masks.
5. Fixed eligibility for all-zero/all-one masks,source artifacts and unannotated
   edges,plus exact coverage denominators. Report unusable cases and reasons rather
   than dropping them silently or selecting a visually convincing subset.

No new fit,training split or clinical threshold is admitted by this review. A
frozen-model diagnostic might eventually compare ALL prespecified generated-only
and mixed localizer states on the same qualified map coordinates,with absolute
per-record AP,matched time/lead controls and complete failures shown. Patient-
group intervals require actual patient linkage; otherwise report record-conditional
uncertainty with its limitation. Published pooled AUROC may be reproduced as a
separate source-integrity descriptor,but cannot replace within-record localization.

This is a route back to the original generator-supported localization question,
not evidence that ECGAD is a fresh clinical confirmation set or that Pulse2Pulse
provides disease truth. If qualified masks and source identities are available,
their actual short-record twelve-lead geometry may be a better next model endpoint
than EDB's reference-dependent episodes. If they are not available,retain the
specific blocker and continue justified source qualification; do not redefine a
record diagnosis or wave landmark as a dense pathology mask. Existing clinical
labels,closed-test restrictions,unknown generator membership and no-push rules persist.
