# Project continuity

## Repository workflow after the refactor

- Read `CONTRIBUTING.md` and `docs/README.md` for setup and document locations.
- Use direct `uv` commands with the default `.venv`; see `CONTRIBUTING.md`.
  Root dependencies live in `pyproject.toml` and `uv.lock`. Preserve installed
  runtime dependencies while downloaders use them; use `uv sync --inexact`
  and inspect its dry run before syncing an active environment.
- Group commands by purpose under `scripts/`; reusable logic belongs in
  `ecg_experiment/`. Keep active download entry points and dependencies stable.
- Use Git history for maintenance changes; retain scientific protocols and evidence.
- DVC covers completed verified sources and fixed patient split manifests only.
  Never add active MIMIC/CODE download trees. See `docs/data-versioning.md`.
- The user wants data kept local while evaluating affordable storage. Do not
  upload training data to GitHub/Git LFS or configure/push a DVC remote without
  their choice. Git stores small DVC pointers, not the dataset bytes.
- MLflow is an optional index of aggregate results with explicit provenance;
  source receipts remain authoritative. See `docs/experiment-tracking.md`.
- CPU checks and data validation do not authorize training or scheduling.

## User-requested experiment pause

Experiments and scheduling are **paused for a repository refactor** as of 24 September 2026. Do not launch or resume training until the user requests it. This overrides earlier authorization to proceed automatically. The waiting legacy MIMIC scheduler was stopped with no training child; the GPU was idle. Downloaders remain active: preserve their current source/data paths while running. Refactoring is authorized; preserve historical sources using `outputs/refactor_pause/source_before_refactor.tar.gz` and `source_hashes.json`, retain data/checkpoints/results, and make new verified executable manifests after refactoring. Do not rewrite old hash receipts. See `outputs/refactor_pause/pause.json` and both queue documents for recovery.

On 26 September 2026 the user explicitly requested execution of the queued
NLP-inspired architecture Experiments 011–013 using Experiment 019's verified
25,000-record training subset. This request resumes those studies under the
new [common protocol](docs/nlp-inspired-25k-study.md), real V100 cost gates,
and sequential GPU lock; it does not resume deferred 008 or 010 or authorize
opening calibration/test before the prescribed development screens. The user
prefers the existing `.venv` and normal uv cache and permits `uv add` if a
library is genuinely needed. Check the live queue and process state before
assuming any stage has started or finished.

Later on 26 September, the user canceled the current NLP-inspired experiment
work to finish EDA first. The 011 profile-only queue was interrupted during
cache verification, before its V100 timing; its status is
`outputs/experiment_queue_nlp25k_011_profile_v1/status.json`. No 25k model
training or development readout ran. Do not launch 011–013 profiles, training,
or a successor queue until the user explicitly resumes this work. Preserve the
prepared code, manifests and receipts. EDA workers were resumed after the
profile stopped; do not stop or repurpose them for experiment work. The pause
receipt is `outputs/nlp25k_pause_2026-09-26/pause.json`.

On 27 September the user reported that EDA was finished and explicitly asked
to continue the 25k NLP-inspired experiments. This resumes 011–013 under their
profile and cost gates. The interrupted 011 v1 profile supplied no complete
cache hash, so `scripts/coordination/create_nlp25k_cache_seal.py` performs a
new one-time full SHA-256 verification before successor GPU profiles. Check
its live process and seal receipt before launching any dependent stage.

The fresh shared cache seal and all three 25k studies completed on 27
September. Real V100 cost gates passed before every full stage; the 011–013
development results and independent artifact audits are summarized in
`docs/nlp-inspired-25k-study-results.md`. The primary 011 CKDA–KDA and 013
Mamba-3–Mamba-2 patient intervals crossed zero; 012 mixed support trailed its
matched local control. No calibration/test data were opened and no NLP 25k GPU
queue is active. Do not automatically schedule a second seed or longer run;
use the queue documents for exact manifests, receipts and next decisions.

For experiment work, first read:

1. `docs/experiment-queue.md` — human-readable queue and recovery instructions.
2. `docs/experiment-queue.json` — authorized priorities, implementation status, dependencies and next actions.
3. The selected experiment's linked protocol and verification/results files.

The user authorized Experiments 011–013 (KDA/CKDA, a compact StripedHyena-inspired model, and Mamba-3) and all four Astra proposals, Experiments 014–017. They are authorized; do not ask again whether to add them. Experiments **011–013 are complete** as one-seed development screens. Experiment **014 is complete** (negative development screen at both label budgets). Experiment **015 is complete**, with a negative development screen. Corrected **017 v2 is complete** under `outputs/experiment_queue_017_v2/queue.json`; its limited-label development screen passed. Prepare a separately frozen second-seed replication and artifact checks before any calibration/test. This follow-up is not automatically scheduled. No experiment queue is currently active. Experiments 008 and 010 remain deferred; do not automatically resume them. The editable backlog is not a live scheduler.

## Running work and evidence

- Read live coordination files under `outputs/` before reporting progress or launching jobs. PIDs and Markdown status summaries are snapshots, not proof a process is alive.
- There is one V100 16 GB GPU. Use the shared GPU lock and coordinate the older MIMIC runner; it predates that lock. Keep the downloader running.
- `outputs/experiment_queue/queue.json` is the **historical frozen executable manifest** for 009 → 007 → 008 → 010. It stopped at the 008 profile after 009 and 007 completed. Its successor `outputs/experiment_queue_recovery_008/queue.json` passed the 008 profile and began training, then was interrupted when the user deferred the expensive 008 run. The first profile-only 010 manifest in `outputs/experiment_queue_profile_010/` failed a bitwise CUDA roundtrip check on a 1.49e-8 difference. The corrected profile-only manifest in `outputs/experiment_queue_profile_010_v2/` passed all three V100 arms. The full 010 run in `outputs/experiment_queue_full_010/` was then stopped after actual cache I/O made the projected suite exceed the adopted two-hour planning ceiling. It saved a verified native-arm epoch-1 checkpoint. The most recently completed NLP 25k manifest is `outputs/experiment_queue_nlp25k_013_full_v2/queue.json`, following complete 011 and 012 successors; check live status and handoff receipts before launching work. 008 and 010 are deferred for cost. The older MIMIC runner was subsequently stopped for the user-requested refactor pause; the downloader continues. The editable backlog under `docs/` does not automatically schedule a GPU process.
- Existing experiments verify code and input hashes. Implement new experiments in new files; do not alter frozen sources or a live executable manifest in place. Prepare a verified successor manifest when new runners are ready.
- Update both queue documents when a task changes state. Record actual commands, source revisions/hashes, verification receipts and result locations. Mark an experiment runnable only after implementation checks; require a real GPU profile before full training. Never present a proposed experiment or a smoke test as a performance result.
- Authorized jobs can proceed without another routine confirmation. Stop only the specific unsafe/incompatible action if a concrete conflict arises; continue independent work.

## Scientific constraints and preferences

- Do not trust MIMIC-IV-ECG labels. Its report statements and measurements come from the ECG cart's software, not from a physician (user preference, 28 September 2026). Use MIMIC for unlabeled self-supervised pretraining only; never as training targets, evaluation labels or evidence of accuracy. Prefer human-read sources (PTB-XL, SPH, Ningbo, Chapman, Georgia, CPSC) or echo-confirmed EchoNext labels.
- Raw twelve-lead ECGs; public-data experiments currently use 56,875 training recordings, with 15,360 full training labels or the fixed 1,518-label subset. Preserve patient splits and train-only fitting.
- The binary endpoint is an ECG diagnostic annotation proxy. It does not establish that a person is healthy or validate university referral decisions. Keep separate development, calibration and test patients.
- Compare new architectures using explicit controls for data, initialization, objective, model size and training budget. Published results in another domain do not establish ECG superiority.
- The user permits combining public data and learning from hidden public labels to simulate limited annotation. New experiments should use the frozen current pool for fair initial comparisons, with later data scaling recorded separately.
- The user authorizes subagents. When delegating, use Sol medium for simple research, Sol high for programming, and Astra for difficult scientific/design work.
