# NTT-06: frozen native ECG reference matching

## Question and primary decision

Does a pretrained native signal encoder add record-finding discrimination over physical references,
and does its apparent regional evidence depend on where tokens are pooled? This adapts
[No Time to Train](https://arxiv.org/html/2507.02798v4) to signals; it is not a replication of its image
benchmark. No ECG is rasterized for model input and no model weights are trained.

The primary endpoint is balanced record-finding accuracy of `native_qrs`, minus the strongest frozen
NTT-04 physical baseline (`raw`, `raw_full_context`, `duration`, `amplitude`, `energy`,
`combined_scalars`). Advance only if the exact point gain is at least 0.05 and the paired 95% patient
bootstrap lower bound exceeds zero. Recompute the strongest physical baseline in every bootstrap draw.
Use 2,000 draws, seed 20261018, shared patient bootstrap helpers, and count draws missing any finding.
Average correctness over the three fixed support draws within each patient before bootstrapping.
This conditional interval does not quantify source transfer or support-population uncertainty.

All endpoints below are frozen before any native clinical inference or score. A failed primary gate
cannot be rescued by selecting another representation, finding, control or endpoint afterward.

## Cohort and predecessor integrity

Reuse exactly NTT-04's 60 development targets (20 each strict project normal, CRBBB and CLBBB),
27 unique supports, three draws and three support patients per finding/draw. Use the saved selection;
no replacement, relabeling, subgroup selection or new patient sampling. These are previously explored
patients, not an independent confirmation. Coexisting source annotations remain unchanged.

Verify NTT-05 result SHA256 `e554a50a695ec6bc96c6b066abee28658d79396a3f90e48899154a032e8d9fed`
and reproduce its 180 predecessor physical choices and 167 correct choices. Also reconstruct NTT-04's
nine methods from saved vectors and reproduce choices, scores and summary within 1e-12.
Verify immutable predecessor selection/features/predictions/reservation hashes. Read reservation
identities first and require every selected ECG/patient to have PR67 training role; exclude any patient
touching PTB folds 9/10 through the original manifest. Do not reparse broad diagnostic metadata or read
reserved-role waveforms. Validate support/target patient separation before feature access.

## Encoder and eligibility

Use only [sehunfromdaegu/ECG_JEPA](https://github.com/sehunfromdaegu/ECG_JEPA) at
`d937ad2c2c8a1e22856ce7e4a23a30f84a71217c`, with the official multiblock epoch-100 checkpoint
(341,551,405 bytes; SHA256
`61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e`).
Verify source bytes before import and checkpoint bytes before CPU, weights-only deserialization;
load every encoder key strictly. Freeze parameters and set evaluation/inference mode.

The [original paper, section 5.1](https://arxiv.org/html/2410.08559v1) documents Chapman/Ningbo and
CODE15 pretraining, with PTB-XL used downstream. The pinned pretraining code agrees at source level.
These documented sources exclude PTB evaluation records. No released patient-level training manifest
or signed checkpoint-to-cohort receipt permits independent identity-level verification; record this
qualification and do not claim certified patient nonoverlap. A different similarly named JEPA release
that pretrained on PTB is inadmissible. No new pretraining is performed here.

Canonical physical-mV input is 12 leads by 5,000 samples at 500 Hz. Fourier-resample to 2,500 samples,
select I, II, V1-V6, and apply no amplitude normalization, following the official PTB path. Dense encoder
output is 400 by 768, ordered eight leads by 50 nonoverlapping 50-sample patches at 250 Hz. Anchors
are 200 ms, half-open seconds. No derived III/aVR/aVL/aVF features may be invented. All tokens have
full-record/eight-lead contextual information; FFT resampling also has global temporal support.

## Staged resource gates

After the protocol is committed, freeze and test the implementation before actual inference. Run a
synthetic, non-diagnostic CPU profile on the real pretrained encoder (two CPU threads), including load,
first forward, repeat forward, finite/shape checks and agreement of dense-token mean with the official
pooled representation (absolute tolerance 1e-5). Require deterministic repeat agreement within 1e-6,
profile wall time at most 300 seconds, per-forward time at most 20 seconds and peak RSS at most 6 GiB.
A successful profile permits at most 87 unique clinical records, one forward per record, and 1,800
seconds for the measured clinical stage, with deadline checks between records. No GPU or training.
Stop with a typed failure receipt if a gate, quality check or immutable input check fails. Do not
change scientific settings to rescue a stopped run; preserve failure artifacts and freeze any successor.

## Native references and controls

Detect beats using the existing label-free NTT-04 QRS detector, keeping only peaks whose full fixed
median-beat context [-0.30, +0.65) seconds fits in the record. Require at least three complete beats.
Use V1/V6 for the primary and matched temporal controls. Each QRS reference is [peak-0.15, peak+0.15).
Pool native tokens by exact fractional occupancy using NTT-05; normalize each beat, average equally
within patient, then equally across the three support patients per finding and normalize. Source
record labels provide weak finding-conditioned references; individual beats are not expert masks.

`native_qrs` compares patient-level averages of these QRS vectors. Freeze the following secondary
controls, with identical patients/supports and no learned transform:

- `native_late`: pool [peak+0.25, peak+0.55), with the same number/duration of regions, in both supports
  and targets. This is shifted-context evidence, not a guaranteed disease-free negative.
- `native_wrong_reference`: query QRS vectors against support late-window prototypes.
- `native_global_pair`: pool the full ten seconds of V1/V6 in supports and targets.
- `native_global_eight`: pool all eight available leads over ten seconds. Additional lead context makes
  this a descriptive whole-record control, not a matched primary alternative.
- `native_shuffled_labels`: cyclically rotate the QRS prototype labels by one place.

Report per-draw/per-finding accuracy, class-score margins, QRS-versus-late/global paired differences,
and target-vector cosine agreement. If late/global vectors perform similarly or agree closely, do not
interpret a QRS anchor as proof of localized abnormality. This native pipeline observes the original
10-second record; historical visual/physical methods used median beats. The comparison therefore tests
implemented pipelines and cannot isolate an intrinsic encoder-representation effect.

## End-to-end native region output

For each target and support draw, match every automatically detected complete QRS region against the
native QRS bank. Also match a label-free multiscale V1/V6 grid: widths 0.4, 0.8 and 1.6 seconds, stride
0.4 seconds, including right-edge windows, using the existing NTT-05 generator. Preserve all candidate
coordinates, class cosine scores, patient/record-frame identity, context declaration and embedding-space
identity in local JSON. Run NTT-05 same-finding/lead/record soft score decay and retain raw/adjusted
scores separately. Do not merge disjoint recurring beats or treat unselected windows as negative.

Similarity threshold -1 and margin threshold 0 deliberately retain all nearest-prototype assignments;
these are exploratory matches, not calibrated referral decisions or disease-presence probabilities.
Report candidate counts/finite scores and exact coordinate coverage. A fixed first target may receive a
local waveform plot with score overlays, explicitly labeled contextual prototype similarity. Plotting
is output only; no plot enters the encoder. No localization accuracy, absence sensitivity or expert
agreement is scored because independent acceptable-evidence annotations are unavailable.

## Artifacts and delivery

Reusable code goes in `ecg_experiment/localization/`, tests in `tests/localization/`, commands in
`scripts/experiments/localization/`, this protocol and results in this directory, safe aggregate reports
in `reports/localization/ntt06-native-encoder/`, and private artifacts in `outputs/ntt06_native_encoder/`.
Keep downloaded source/weights ignored under `third_party/ntt_native/`. Bind source, dependency,
checkpoint, predecessor and waveform hashes in local receipts, using repository-relative paths.
Public aggregate receipts must contain no patient/record identifiers, waveforms, per-record vectors,
checkpoint tensors or private predictions. Each executed stage gets a results report from its outputs;
update both queues and rank follow-up ideas. All files are English. Work remains local; PR #71 stays
draft and no push or merge is part of this implementation task.
