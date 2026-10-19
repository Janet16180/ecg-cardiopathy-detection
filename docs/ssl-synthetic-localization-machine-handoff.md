# ECG localization handoff to the dual 3090 computer

18 October 2026. Start here when continuing draft [PR 67](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/67)
on another computer. Branch: `research/ssl-synthetic-localization`, base: `main`.
The user reports two RTX 3090 GPUs and local access only; a new Codex session will run there.
RAM, free disk, operating system, driver and actual GPU memory remain to be inventoried.
Keep the PR draft. This handoff does not launch a run or transfer data.

The strongest finding so far is that explicitly teaching the model to distinguish an edited ECG
from its matched unedited recording improves localization of introduced changes. The improvement
has a substantial tradeoff between edit types. We have not established localization of real heart
disease. First explain that tradeoff using saved maps, then replicate the experiment before
expanding training or replacing the generator.

## User objective and working agreement

The planned Italian student ECG data and cardiologist support are unavailable. The user wants to
investigate self-supervised learning on real ECGs followed by limited synthetic localization
supervision. The intended eventual output is a highlighted abnormal region, alongside a screening
prediction. No cardiologist recruitment is a feasible next step. GAN, diffusion, VAE and other
released models are eligible; GAN is not a requirement.

Another session owns No Time to Train. Keep this stream independent and do not alter its branch or
results. Coordinate agents in separate worktrees with a root coordinator, independent scientific
review and independent numerical verification. The user authorized progress and publishing to this
PR, then explicitly requested keeping it draft. Do not mark it ready or merge it.

Read `AGENTS.md` and `CONTRIBUTING.md`. Frozen sources and their receipts must be preserved; use new
files for successors. Every new experiment needs a committed protocol and decision rule before
scores, predecessor reproduction, an actual profile before training, an executed results report,
and updates to both queue files and the ranked backlog. The handoff describes this research stream;
other repository experiments retain their existing authorization and deferred states.

## What training on real backgrounds means

A background is an actual PTB-XL ECG. We introduce a controlled voltage perturbation into selected
lead-time cells and keep an exact support mask of the final numerical difference. A matched
unedited version provides the control. The model sees the observed ECG alone; the exact mask and
matched version are used for supervision or evaluation under the protocol.

These masks locate our intervention. They do not say which original regions are healthy or diseased.
A zero edit mask means no introduced change, not a healthy patient. Algorithmic timing anchors are
not specialist-confirmed wave boundaries. This distinction applies to every result below.

## Completed experiments and conclusions

R denotes random initialization; S denotes the real-ECG self-supervised initialization. B denotes
weighted binary cross entropy; P adds the frozen paired probability loss. AP is average precision
on the complete lead-time grid, not diagnostic accuracy. Intervals below are patient bootstrap
intervals, not variability across independent training seeds.

| Completed stage | Executed finding | Interpretation and report |
| --- | --- | --- |
| Input and partition audit | 15,657 permitted ECGs decoded; zero integrity failures; 11 training quality exclusions; 6,142 protected records unread | [Input audit](ssl-synthetic-localization-input-audit-results.md), [partition](ssl-synthetic-localization-partition.md) |
| Generator and runtime qualification | Initial generator occupancy checks and first deterministic GPU profile failed; prospectively amended successors passed | Failures are retained; [generator v2](ssl-synthetic-localization-generator-pilot-v2-results.md), [profile v2](ssl-synthetic-localization-profile-v2-results.md) |
| Original simulator pilot | AP R 0.637958, S 0.640600; S minus R +0.002642, interval −0.003550 to +0.008333 | Frozen advancement rule failed; [pilot results](ssl-synthetic-localization-pilot-results.md) |
| Frozen transfer to real backgrounds | AP R 0.232088, S 0.289397; gain +0.057309, interval +0.037000 to +0.077527 | SSL also aligned better before editing; higher edited AP does not establish stronger edit response; [transfer results](ssl-synthetic-localization-real-transfer-results.md) |
| Training with matched real backgrounds | AP R 0.346928, S 0.511194; AP-response gain +0.237969, interval +0.198587 to +0.278683 | Complete rule failed on selectivity and nuisance-control guards; [results](ssl-synthetic-localization-paired-background-results.md) |
| Saved-map control diagnosis | Seven of eight SSL nuisance alerts were already present on unchanged ECGs; one newly appeared | Does not establish noise invariance or clinical false-positive rates; [diagnostic results](ssl-synthetic-localization-paired-diagnostic-v2-results.md) |
| Four-arm counterfactual-loss factorial | Initialization-averaged direct edited-AP loss effect +0.060490, interval +0.045504 to +0.075278; all 12 prospective gates passed | Positive engineering result with mode tradeoff; [factorial results](ssl-synthetic-localization-counterfactual-factorial-results.md) |

Latest four-arm pooled AP: RB 0.368157, SB 0.547721, RP 0.469283, SP 0.567574.
All arms used the same 500 TRAIN backgrounds, 2,000 immutable inputs, 600 updates and 9,600
presentations. Both BCE arms reproduced all 600 predecessor losses and final tensor states exactly.
No new SSL pretraining or checkpoint search occurred. Evaluation used 128 fresh development patients,
with 64 separate calibration patients and all 384 previously spent validation patients excluded.

The paired-loss benefit for R was +0.101127 overall, but +0.277629 for focal edits and −0.075376
for repeated edits. For S it was +0.019853 overall. SP had the highest pooled AP. The negative
interaction means the paired loss helped R more; it does not show SSL is useless. Pooled success
does not imply success for every edit type. Only one fitted randomness and a previously studied
perturbation family were tested. Earlier failed verdicts remain failed.

The independent factorial audit passed 154,715 checks with zero mismatches. It checked case/response
metrics, aggregates, intervals, thresholds, all twelve gates and exact BCE training closure.
Waveform-derived absolute-voltage case scores were source-bound; their aggregate arithmetic was
independently checked, without reopening waveforms. See the [audit report](ssl-synthetic-localization-counterfactual-factorial-audit-results.md).

## Evidence anchors and current resources

The following local receipts were checked when writing this handoff. Preserve bytes and relative
paths; do not rewrite receipts to fit the new machine.

| Path below `outputs/ssl_synthetic_localization/counterfactual_factorial_v1/` | SHA-256 |
| --- | --- |
| `preparation/summary.json` | `498251dc05013ece766d3877755fe0a000db5b8a1db8dc2ffdcddad4bbefcdba` |
| `profile/summary.json` | `4007840a2363359567627025f37143c01119ce2e1668d99b154f9297907fd87f` |
| `training/summary.json` | `0a4ea42f74a79270c06b9dc43238ca203f8846a17557faee379e159619586b46` |
| `evaluation_v2/summary.json` | `3f4555614180442c1eb82c5e5b20ca4137d53ae35697421bfa4337cf4f588d28` |
| `independent_audit_v1/summary.json` | `74b02f60ac0d77bf071bc99e3ae6fcf460fd8ef0dfba21b9626f5d67646d6c04` |

The last completed audit records cumulative scientific CPU **745.8142945699988 seconds** and
GPU-wall **382.2241592870014 seconds**. The localization output tree currently has 948 files,
**3,927,557,715 bytes**. Its existing 4 GiB experimental allowance leaves **367,409,581 bytes**,
about 350 MiB. This allowance is distinct from physical free disk.

The existing stream ceilings are 14,400 scientific CPU seconds, 7,200 GPU-wall seconds and 4 GiB
outputs. The unrun mode diagnostic reserves 16 MiB; the unrun Pulse2Pulse probe reserves 96 MiB.
Charge each new stage, failure and audit once. Prefer mode main, then mode audit, then generator
probe with a prospective additive ledger. Never silently replace historical costs or count only
optimizer kernels. Setup and artificial tests are not scientific stage costs.

On the original computer, twelve verified redundant archives were deleted, freeing about 35 GiB;
physical free disk afterward was about 109 GiB. Extracted files were preserved. The CODE-15 archive
was retained because an extracted replacement was not found. The canonical EchoNext archive was
retained because closed test members still exist only inside it. These facts do not establish
availability on the new computer. Private cleanup receipt:
`outputs/dataset_setup_2026-10-18/archive_cleanup_2026-10-18.json` on the original machine.

The larger computer makes a broader prospective storage/compute plan possible. It does not change
an already frozen budget. The DiffuSETS assessment's storage rejection concerned this stream's
remaining allowance, not a proven inability of the old GPU to run it. No large generator training
experiment was silently run or omitted with a performance result.

## Pending work in priority order

### 1 Explain the focal versus repeated tradeoff using existing maps

Status: protocol committed; main implementation source-reviewed and artificially tested; independent
checker preserved and artificially tested. **Neither has executed on study arrays. No mode result exists.**

Read [protocol](ssl-synthetic-localization-factorial-mode-diagnostic.md) and
`configs/ssl_synthetic_factorial_mode_diagnostic_v1.json`.
Main: `ecg_experiment/ssl_synthetic_factorial_mode_diagnostic.py`, entry point
`scripts/reports/diagnose_ssl_synthetic_factorial_modes.py`.
Independent checker: `scripts/validation/audit_ssl_synthetic_factorial_modes.py`.

The frozen description covers lead ranking, temporal ranking, oracle-conditioned temporal AP,
fixed-area precision/recall, probability allocation and pre-edit phase alignment. Report all four
arms and both modes; do not select a favorable metric or invent a new advancement rule. Oracle
metrics are explanatory diagnostics, not an available inference method or an additive AP decomposition.

Only six numerical payloads may be opened or hashed: four
`evaluation_v2/probabilities_{RB,SB,RP,SP}.npy` maps and
`preparation/{support,candidate}.npy`. Use only the frozen evaluation rows for new metrics. The
runners also require exact permitted metadata/source bindings from the passed parent audit.
Raw/processed ECGs and checkpoints remain closed even for hashing in this diagnostic.

After transfer, integrity checks and final source review, execute main once with CUDA hidden and
all four numerical thread variables set to 1:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" \
.venv/bin/python -m scripts.reports.diagnose_ssl_synthetic_factorial_modes
```

Destination: `outputs/ssl_synthetic_localization/counterfactual_factorial_v1/mode_diagnostic_v1`.
It must not exist before a first attempt. Do not delete failed attempts to retry. Read actual outputs
and write the executed report. Freeze the independent script's exact bytes and the completed main
summary SHA, then run the checker with explicit `--root`, `--main-sha` and `--script-sha`.
Its intended destination is `mode_diagnostic_v1/independent_audit_v1`. Verify the checker contract
before execution; it has a separate 120-second CPU/wall allowance, as does main.

### 2 Check one released generator without claiming clinical validity

Status: public source/metadata assessment complete. Pulse2Pulse adapter and artificial tests are
preserved in the PR. **Weights have not been downloaded; no generator inference or technical PASS exists.**

Read [assessment](ssl-synthetic-localization-pretrained-generator-assessment.md),
[prospective feasibility protocol](ssl-synthetic-localization-pretrained-generator-feasibility.md),
and `configs/ssl_synthetic_pretrained_generator_feasibility_v1.json`.
The adapter is `ecg_experiment/ssl_synthetic_pretrained_generator.py`; its thin entry point is
`scripts/validation/probe_ssl_synthetic_pretrained_generator.py`.

Before any download, complete and commit the config's missing `public_assets` declarations and a
prospective protocol clarification for the pinned MIT architecture, license and class allowlist.
The current adapter requires those entries and is not execution-ready merely because it is committed.
The assessment contains exact immutable URLs, sizes and hashes. Allowed MIT classes are
`Transpose1dLayer`, `Transpose1dLayer_multi_input` and `WaveGANGenerator` at author revision
`f28e4b792e79c503b8b7e0912be9d304a3a7177b`. Reconcile latest scientific cost accounting and obtain
final independent source review before a single run.

The selected checkpoint is 42,375,017 bytes at Hub revision
`262e0e67c0101cd4989d7c2f66f16d6d54756e70`; declared SHA:
`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`.
Verify locally on download. Use CPU float32, batch one, strict `weights_only=True`, reviewed local
AST class loading, `eval()` and `inference_mode()`, with unchanged state hashes. No Transformers,
`trust_remote_code`, unsafe pickle fallback or package installation is needed. The prospective
probe contains six forwards: one profile, four fixed seeds and one exact replay. CPU/wall cap is
180 seconds, output cap 96 MiB including checkpoint and partial downloads. Stop honestly on failure.

Physical amplitude units, exact checkpoint cohort membership, clinical normality and regional disease
control are unverified. An unconditional generator can produce candidate backgrounds, not disease
masks. Do not feed its arbitrary export units into the physical-mV localizer or call synthetic
outputs healthy. Do not redistribute weights. Independent saved-output technical verification is
still needed after the run; no downstream localization fit follows automatically.

### 3 Replicate the paired-loss finding before expanding it

Status: proposed only; no seed replication protocol, profile or fits have executed.
Use the mode diagnosis to explain the present limitation, then freeze a matched replication plan
covering seed variability and both edit modes. Preserve the same primary contrast, controls,
patient exclusions and reporting of regressions; do not tune away an unfavorable mode using the
current evaluation patients. Any changed intervention family needs its own prospective qualification
and explicit development/confirmation policy. Additional training is not a consequence of technical
Pulse2Pulse success.

### 4 Assess larger generators and existing annotation resources

Status: source assessment only. SSSD-ECG has an author model link but a downloadable weight manifest
was not verified. DiffuSETS has a declared 1,485,588,360-byte inference archive; its standard demo
uses an external embedding API, while a separate example supports pre-extracted conditioning.
Inspect archive contents and provenance before downloading; do not copy private MIMIC latents or
send clinical reports to an external API. cNVAE has code but no verified released checkpoint in the
inspected source. Reassess under a new bounded resource plan rather than ranking by model novelty.

Public annotation options can be reviewed without cardiologist access: LUDB has expert wave
boundaries; QTDB has selected beat boundaries; European ST-T has episode annotations. These support
different tasks, not interchangeable twelve-lead disease masks. No new annotation download or
clinical label change was performed. First recover/audit existing intended ECGAD artifacts if
available; existing uncertain masks cannot be made true by model agreement. Compare with No Time
to Train only after agreeing on valid annotation semantics and a common permitted development split.

## Move the local evidence separately from Git

Pulling the PR supplies authored code, protocols, tests and small reports. It supplies **no** raw
ECGs, partition payload, probability maps, checkpoints or local receipts. DVC pointers alone do
not recover them; there is no approved data remote. A fresh clone can run artificial tests and
review sources before copying any data.

For full continuity of this stream, copy these two local directories by external drive or another
user-controlled local transfer, preserving repository-relative layout:

- `outputs/ssl_synthetic_localization/` — currently about 3.66 GiB; includes receipts, maps,
  generated/prepared banks, trained weights and retained failures.
- `data/processed/ssl_synthetic_localization_v1/` — currently just `partition.json`, 4,548,167 bytes.

For the saved-map diagnostic alone, a smaller transfer can contain its six allowlisted arrays,
parent audit/evaluation summaries and audit script, and all permitted source/text/JSON dependencies
in the parent binding catalog. The catalog includes 218 permitted text declarations; its forbidden
payload declarations remain authority-deferred. Build the manifest from that catalog and verify
scope before copying. Do not run a blanket payload hash verifier or an old full audit, which has a
broader access scope, merely to check transfer. A complete local stream copy is simpler for later
continuation, but does not expand the diagnostic's read permission.

Further fitting may additionally need permitted raw PTB-XL files under `data/raw/ptb-xl/1.0.3/`
and any exact upstream metadata declared by the selected protocol. Inventory these before copying
or downloading. The current mode diagnostic needs no waveform or checkpoint payloads.
Do not copy `.venv`, broken worktree symlinks or temporary absolute-path configuration as setup.
Do not publish arrays, patient IDs, generated examples or weights in Git/Git LFS or external APIs.
MIMIC and EchoNext credentialed data are unnecessary for the next two tasks; any later use must
respect dataset access terms on the new machine.

After transfer, check the five receipt SHAs above, then let the permitted successor verify its
allowlisted dependencies. Preserve original receipt bytes and scope exceptions. Missing or conflicting
bindings are a stop, not permission to regenerate a partition or rewrite historical paths. If old
hardware replay differs, retain that finding and prepare a successor equivalence contract instead
of changing an old receipt. GPU determinism across architectures is not assumed.

## Start the new computer safely

Use a fresh Linux x86-64 checkout, or establish a compatible Linux environment first. Commands below
assume GitHub CLI authentication and `uv` are already available. In an existing clone with local
changes, inspect them and use an isolated worktree rather than overwriting anything.

```sh
git clone https://github.com/Janet16180/ecg-cardiopathy-detection.git
cd ecg-cardiopathy-detection
gh pr checkout 67
git status --short
git log -5 --oneline
nvidia-smi
free -h
df -h .
uv sync --locked --group data --group tracking
```

Recreate Python 3.11 with the pinned PyTorch 2.6/CUDA 12.4 environment. Keep `pyproject.toml` and
`uv.lock` unchanged. Consult `environments/README.md`; do not replace locked dependencies to make a
new GPU work without a separately reviewed environment plan. The Pulse2Pulse probe uses the main
environment, not the separate historical pretrained-encoder environment.

Full CPU CI also requires the pinned reference source used in `.github/workflows/checks.yml`:

```sh
git clone https://github.com/dlaskalab/bench-xecg.git third_party/bench-xecg
git -C third_party/bench-xecg checkout 13e57523e418d8cddced73f81b27ba541413f4f3
uv run --locked ruff check ecg_experiment scripts tests conftest.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' uv run --locked --group data --group tracking python -m pytest -q
uv build --no-sources
```

Do not clone over an existing third-party tree. The original research worktree's `third_party`
symlink is broken; use the pinned checkout above, not that symlink. Run full tests in a clean
validation checkout with a regular local `outputs` directory; a science worktree symlink can fail
an intentional path-security test. `tests/conftest.py` already isolates a frozen auditor's import
side effect; do not edit that pinned auditor to fix thread-setting errors.

Inventory both GPUs, drivers, available memory, CPU/RAM and free SSD space. A 3090 normally provides
24 GiB per GPU; two cards do not automatically give one process a 48 GiB allocation. Start with one
GPU and the shared lock in `ecg_experiment/gpu.py`. Before concurrent jobs, review whether the lock
and receipt design supports per-device ownership; do not bypass a process-wide lock. Independent
replicates may later use separate GPUs under an explicit scheduling plan. Every new training path
needs an actual GPU profile, storage/runtime forecast and replay gate on this machine.

## Prompt for the next Codex session

> Continue draft PR 67 on branch research/ssl-synthetic-localization. Read AGENTS.md,
> CONTRIBUTING.md and docs/ssl-synthetic-localization-machine-handoff.md, then both experiment queues
> and the latest factorial results/audit reports. The Italian cohort and cardiologist are unavailable.
> Another session owns No Time to Train. Coordinate agents in isolated worktrees. Keep the PR draft.
> First inventory hardware and local artifacts without opening protected data; report missing files.
> Do not retrain completed studies, redownload datasets, change labels, or resume unrelated deferred
> experiments automatically. Once required local evidence and source checks pass, continue the frozen
> saved-map mode diagnostic and independent audit. Then finish the prospective Pulse2Pulse asset and
> accounting amendment, review it, and run its bounded technical feasibility probe. Those tasks are
> unexecuted, not results. Propose a prospective replication and larger-model budget based on the
> findings and measured two-3090 resources. Preserve all failed decisions and immutable receipts;
> no clinical claim follows from synthetic edit labels.

## Records to keep synchronized

`docs/experiment-queue.md` and `.json` are the live index; historical paragraphs lower in those files
may describe earlier local-only/no-push instructions that the user later superseded. This handoff
and the current top-level queue entry describe the present publication and pending-run state.
Write every executed report from actual local outputs. Add follow-ups to `docs/experiment-backlog.json`
and regenerate priorities with `python -m scripts.reports.rank_backlog` when proposals change.
Closed tests remain closed, including the stream's real engineering test, PTB-XL folds 9–10 for
this stream, Challenge test groups, EchoNext test and the final frozen test. No clinical label or
age-subgroup change and no deferred Experiment 008, 010 or 017 second seed is authorized here.
