# Synthetic generation/localization handoff, 20 October 2026

Continue draft PR 67, branch `research/ssl-synthetic-localization`, against `main`. PR 70 screening
was complementary and already merged at `41871e35c3d04ca631021039f973a4bbccf2d393`; do not substitute
classification for synthetic-generation/localization. Read AGENTS.md and CONTRIBUTING.md, then the
current queue and live outputs. This handoff supersedes the 19 October recovery pointer without
rewriting frozen protocols, source or historical reports. No scientific job remains running.

## Actual completed results

- Released-N timing main26/recovery28 PASS: 160 measurement records/1920 leads/17,539 N landmarks,
  mean empirical rank 0.963600255 versus six-shift mean 0.513081615, paired gain 0.450518640,
  95% interval [0.438959838,0.462589051]. All 200 records/2400 streams remain; 40 parser records
  have no rank/control scores. Every record gain and all 12 lead mean gains are positive.
- Descriptive coherence main29/audit30: original128-seed generated excess activity Jaccard
  0.809424837 versus 0.745016012 in all200 LUDB records; real minus generated −0.064408824,
  interval [−0.079577463,−0.050221354], 2000 independent cohort draws, seed2026106401.
  Generated excess SD0.0321713 versus real0.0987096. Both cohorts are strongly aligned and overlap.
- Both independent numerical reconstructions pass with zero mismatches. Timing checks include
  1,668,474 scalars and all160 full score arrays; coherence checks include328 rows and13.12million
  mask cells. Exact C2, saved95 and original128-seed closures pass. Figures and complete arrays
  are saved locally; figures were also visually reviewed by the scientific reviewer.

Read the output-derived [timing report](ssl-synthetic-localization-ludb-peak-timing-v1-results.md),
[independent timing report](ssl-synthetic-localization-ludb-landmark-independent-audit-v2-results.md),
[coherence report](ssl-synthetic-localization-ludb-coherence-bridge-v2-results.md),
[independent coherence report](ssl-synthetic-localization-ludb-coherence-independent-audit-v2-results.md)
and separate scientific reviews. These are beat-landmark concentration and activity-synchrony
measurements. They do not establish real disease masks, generated physiological fidelity, disease
control, real-ECG transfer of the learned localizer or benefit from SSL. The 96th percentile is not
96% diagnostic accuracy. Reference samples are unlabeled. No clinician recruitment is required.

## Failures and frozen history

The strict dense-QRS-span qualification remains FAIL. Reserved parser95 contains35 explicit QRS
peak/end pairs without onsets across9leads; dual raw/WFDB decoding agrees. Never guess those
onsets, discard their uncertain intervals as negatives, or turn N markers into healthy labels.
Read the [annotation diagnosis](ssl-synthetic-localization-ludb-annotation-diagnosis-v1-results.md).
The landmark estimand was committed before any of the160 measurement records was opened.

Both acquisition transport failures remain. Acquisition v3 was never executed; v4 completed all
2800 official file hashes. The original main source-admission failure, later parser failure, and
independent audit27 legacy-schema failure remain preserved. Failed27 stopped before numerical
reconstruction and retained null prior/cumulative fields. Explicit setup reconciliation
`ludb_landmark_audit_reconciliation_v1/reconciled_prior.json`, SHA27cdc3ceddc4a36e7be6ba2805caad98c094fcd5bb85e80d1d2c9b469d5b99dd,
adds its1.717297662 CPU seconds once; it is not another experiment or a replacement receipt.
Recovery28 uses new files and unchanged numerical math. Legacy receipt18 lacks a stage-count
field, and acquisition19 gates reside in its bound summary; account for those schemas explicitly.

All30 current-machine numerical stages total460.93105389699997 CPU and447.98139213304967 GPU-lock
seconds, excluding development/setup. Unavailable 18 October artifacts/costs remain unknown;
never recover them from the other checkout. The original four-arm paired/suppression study still
FAILS at14/15 gates. Descriptive temporal/lead improvements and these new results do not rescue it.
The original AP-gated coherence bridgev1 stays unexecuted; v2 has a separately committed landmark gate.
The earlier WFDB original-header fallback deviation remains documented; all new annotation I/O
uses sanitized structural headers. No diagnosis fields were analyzed or labels changed.

## Local artifacts and identities

All paths below are relative to `/home/janetrivera/baseline/`:

- LUDB acquired records: `outputs/ssl_synthetic_localization/ludb_acquisition_v4/data/`.
- Read-only-use alias: `data/raw/ludb/1.0.1/data` points to that directory; verify with `ls -la`.
- Actual released ECG bank: `outputs/ssl_synthetic_localization/generated_quality_v2/raw.npy`,
  float32[128,8,5000], seeds2026103000–2026103127; original512 canonical variants also remain there.
- Released weights: `outputs/ssl_synthetic_localization/pretrained_generator_standalone_v1/pytorch_model.bin`,
  42,375,017 bytes, SHAe646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827.
  Hub revision262e0e67c0101cd4989d7c2f66f16d6d54756e70; architecturef28e4b792e79c503b8b7e0912be9d304a3a7177b.
  Checkpoint patient membership/overlap and physical calibration remain unverified.
- Timing results: `outputs/ssl_synthetic_localization/ludb_peak_timing_v1/`; successful audit is
  `independent_audit_v2/`, not the preserved failed `independent_audit_v1/`.
- Coherence results: `outputs/ssl_synthetic_localization/ludb_coherence_bridge_v2/`; audit is
  `independent_audit_v1/`. Four fixed figures, both mask banks and all328 rows are present.
- Latest audit30 summary SHAcfa7faed4ffef1ca68b4c5de6b1e16cd57bce3ce2d8fe051c725697a400eddfc;
  terminal ledger SHAa3acf4aae3f27b00d51b18e6e50921c9935267e0d8646433bf486ec3f274e11f.
- Admissions: timing `configs/ssl_synthetic_ludb_peak_timing_admission_v1.json` (66 sources),
  coherence `configs/ssl_synthetic_ludb_coherence_admission_v2.json` (93 sources). They include
  actual source/config/asset/prior identities. Their executed source bytes are frozen.

Earlier host checks confirmed two RTX3090s with actual PyTorch GPU operations under the shared
lock. New LUDB studies were CPU-only by protocol; a sandbox nvidia-smi error is not proof of no GPU.
Use `ecg_experiment/gpu.py`, an actual-path profile, full persistence/resource gates and limited CPU
threads before future GPU training. No new GPU training or generator forwards occurred here.

## Data location and ownership

Working checkout: `/home/janetrivera/baseline`.
The other agent owns `/home/janetrivera/ecg-cardiopathy-detection`; only permitted raw paths under
`/home/janetrivera/ecg-cardiopathy-detection/data/raw/` may be shared read-only. Never modify that
checkout, its Git/DVC state or its outputs. Known source subpaths include:
`ptb-xl/1.0.3/`, `challenge-2020/1.0.2/`, `challenge-2021/1.0.3/`, `sph/sph/`, `code-15pct/`,
`mimic-iv-ecg/`, `echonext/` (archive), `ludb/1.0.1/`, and
`ecg-reasoning-benchmark/90a53f1/ptbxl.jsonl`. Verify actual files; some sources may be incomplete.
LUDB was empty at initial inspection, so this study acquired its own public copy under outputs.

Use per-file/source-subfolder symlinks under baseline/data/raw beside tracked DVC pointers; never
replace a pointer-containing directory or stage raw-data aliases. PTB-XL aliases for database/scp/
checksums/records100/records500, SPH and the reasoning JSONL were previously created; check them
before use. They were not inputs to the new LUDB studies. All new data, weights, predictions,
checkpoints and receipts belong under baseline/outputs. MIMIC-IV-ECG and EchoNext are credentialed:
keep records/reports local and never send them to external APIs. All closed-test restrictions remain.

## Validation and next work

Final CPU suite:2079passed/6skipped,2existing all-NaN-fixture warnings,121.24seconds. Repository-wide
Ruff PASS. Offline source/wheel builds PASS; package inspection checks all56 new Python files in
the sdist and23 new package modules in the wheel, with no raw data/outputs/weights. The first
no-isolation build lacked hatchling in its selected environment; its failure log remains. The
successful isolated build reused the existing cached backend without dependency changes.
Local evidence: `outputs/ssl_synthetic_localization/ludb_repository_validation_v1/`.
Current-summary source preservation checks394 obligations across161 paths with zero mismatches;
this scope does not invent verification of missing historical receipts.

The coordinator used `/tmp/ecg-synthetic-autonomous-run`, backed by isolated metadata under/tmp;
all outputs point to baseline. Agents used separate isolated clones. Use explicit PYTHONPATH for
worktree code because the editable install points at baseline. Commits/pushes use
Janet16180 <clarajanet16180@outlook.com>; keep PR67 draft and never auto-merge.

Next proposed priority is a separately prospective frozen-localizer landmark/energy diagnostic,
using only actually present checkpoints and all prespecified fits/arms. Commit its endpoint,
controls, checkpoint admission and inference/persistence resource profile before scoring. A later
held-out-editor/background-source localization study needs matched controls, fixed partitions,
full actual-path profiling and GPU gates before training. These are backlog proposals, not results
or already-running work. Read both queues and rerun `python -m scripts.reports.rank_backlog`.
