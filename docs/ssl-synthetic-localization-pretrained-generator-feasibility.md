# Pretrained ECG generator feasibility

18 October 2026. The user requested continued work using available resources, with no cardiologist
access and no requirement to use a GAN. This protocol freezes a small local runtime probe before
any model-weight download or generator inference. Compare architectures through public source and
release metadata; this is not a localization/classification comparison or evidence of clinical
fidelity. Keep PR67 in draft. No training, clinical label definition change or closed-test access.

## Candidate assessment and selected probe

Assess the original authors' public Pulse2Pulse/DeepFake ECG, SSSD-ECG, DiffuSETS and cNVAE-ECG
releases for accessible checkpoint size, inference dependencies, conditioning, regional mask
control, output geometry, units, training-cohort provenance and locally feasible execution. Record
primary URLs and distinguish declared availability from a downloaded/tested artifact. A whole-ECG
disease condition is not an exact regional abnormality label. No GAN/diffusion superiority claim
follows from publication date or architecture family.

The selected technical probe is the compact public Pulse2Pulse checkpoint at immutable HuggingFace
revision262e0e67c0101cd4989d7c2f66f16d6d54756e70:

- `https://huggingface.co/deepsynthbody/deepfake_ecg/resolve/262e0e67c0101cd4989d7c2f66f16d6d54756e70/pytorch_model.bin`
- Expected42,375,017bytes and declaredSHA256
  `e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`.
- Public model source `modeling_deepfake.py` at that same revision is11,269bytes, declaredSHA256
  `160fdee62b49e9960c09d2e669bb13c8069668930ae26019e63bd242c9f6a5b5`.

These are remote declarations until fresh local byte verification. Download ONLY explicitly listed
public files, under the new study directory, bounded by expected size, with partial transfers retained.
No auto-download cache elsewhere or recursively downloaded repository/model bundle. No raw real ECGs,
patient metadata/reports, external embedding API or telemetry. Reuse pinned existing dependencies;
no pyproject/lock edits or automatic installation. Do not use blind `trust_remote_code=True`.

The original paper describes training on MUSE-labelled-normal GESUS/Inter99 recordings. These labels
do not establish health truth for our project. Exact released-checkpoint patient/epoch provenance
and overlap with our cohorts are not independently established. The HF generic BSD tag and differing
license declarations in author resources require explicit attribution/limitations; do not redistribute
the checkpoint or copied remote architecture in Git. A public inference probe does not resolve those
release/provenance questions or make the samples eligible for a formal clean-cohort comparison.

## Reviewed runtime and fixed inputs

Inspect and pin the exact public model source before execution. Use an explicitly reviewed plain
PyTorch route preserving the released network/state dictionary, with no Transformers auto-loader,
unsafe checkpoint fallback or source-global monkeypatch. If a pinned source-derived class subset is
loaded locally, predeclare its exact class names/import globals, reject other AST structures and
record hashes; it stays a local third-party artifact. Prefer the released forward operations over
an unverified alternative implementation. Load checkpoint on CPU with `weights_only=True`, require
plain finite tensor state and exact keys/shapes under strict loading. Source/architecture/tests
must be committed, root-integrated and independently reviewed before any download/load/inference.

Use float32 CPU inference, one numerical/PyTorch thread, CUDA hidden, batch1. Generate the exact
8-by5000 uniform[-1,1] inputs using separate CPU torch.Generator instances seeded
2026102901,2026102902,2026102903,2026102904. The network receives only noise, not real records or
clinical information. Preserve/reuse these input bytes; reset the seed before each generation.
Retain raw generator float outputs and the author's numerical export convention multiply6000,
truncate toward zero to int32, transpose5000-by8. Repeat the first input once and require EXACT
native raw/export equality. Do not select seeds or regenerate rejected samples.

The published numerical multiply6000 alone does NOT prove microvolt units. Report raw values and
author-export integer units. Any division by1000 is an explicitly labelled numerical convention,
not verified mV truth; do not put these values through our frozen physical-mV localizer. Derive the
four additional limb channels by the standard linear equations from I/II and retain float32
canonical12-by5000 values in the same unverified numerical units. Verify equations with declared
float32 tolerance, not a physiology/health assertion. Do not attach clinical normal/abnormal labels,
introduced support masks or expert waveform-boundary labels.

## Technical gates and resource admission

Require all byte/shape/state checks, finite8-by5000 raw outputs and int32 exports, nonzero temporal
variation in every output lead, distinct outputs for all four distinct fixed inputs, exact replay,
finite canonical12lead geometry and declared limb algebra. Record each gate and their conjunction
as technical-generation feasibility only. No normality/fidelity/morphology gate, manual selection,
clinical classifier score, localization AP or patient evaluation is allowed.

Measure one disposable first-input forward as the actual CPU profile. Require its elapsed CPU/wall
cost times10 plus spent time plus15seconds reporting reserve to fit the same whole-probe180second
CPU AND monotonic wall limit. Forecast is conservative for four fixed generations plus one replay;
retain only the fixed inputs/outputs, not an extra sample bank. No GPU profile is required for this
CPU-only probe; no GPU lock or GPU job is used. If the profile fails, stop before the four generation
bank without tuning. Keep at least2seconds for a truthful failure receipt at every stage.

New study path is `outputs/ssl_synthetic_localization/pretrained_generator_feasibility_v1`.
Main output cap96MiB includes exact checkpoint, pinned public source, metadata, input/output banks,
partial transfers and receipts. No converted duplicate checkpoint or separate HuggingFace cache.
Enforce actual stream bytes plus the remaining96MiB forecast <=4GiB BEFORE download and at stage
boundaries. Do not delete predecessor artifacts to create room. Current cumulativeCPU745.81429457
and GPU-wall382.224159287 come from parent independent auditSHA
`74b02f60ac0d77bf071bc99e3ae6fcf460fd8ef0dfba21b9626f5d67646d6c04`.
Derive exact prior costs from that receipt; later parallel mode-diagnostic work must be charged
separately once to the stream ledger rather than double counted or overwritten. Cumulative CPU
ceiling14400, GPU-wall7200 and whole storage4GiB remain. New outputs reserve their own96MiB even
while the independent16MiB diagnostic is in progress. No competing full GPU run or new fit.

Refuse overwrite. Download/connectivity/load/profile/generation failures retain bytes, exact costs,
completed seeds and error with no automatic retry or unsafe fallback. The technical probe may finish
with a runtime PASS while physical units, disease fidelity and regional control remain UNVERIFIED.
The runner writes an executed report from actual outputs, separates assessed candidates from tested
ones, and updates both queues/backlog. A separate saved-output check verifies hashes, geometry,
export/replay/algebra and all technical gates without claiming independent clinical validation.
Weights, generated ECG arrays and remote architecture stay local; only authored adapters/tests,
public metadata, aggregate reports and attribution enter Git. No localization fit follows automatically.
