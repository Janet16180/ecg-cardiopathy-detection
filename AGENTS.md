# Agent guide

Master's project (Tecnológico de Monterrey): screening university students with 12-lead ECGs. A nurse
records the ECGs, a cardiologist reads them, and a model flags who needs follow-up. Keep this file short:
history belongs in Git and the queue documents, not here.

## Where things are

- Setup and layout: `CONTRIBUTING.md`, `docs/README.md`.
- State of the research: `docs/experiment-queue.md` and `.json` (what ran, where the outputs are),
  `docs/experiment-backlog.json` and `docs/experiment-priorities.md` (ranked next steps, regenerated with
  `python -m scripts.reports.rank_backlog`), and each experiment's `docs/experiment-NNN-*.md` protocol and
  `*-results.md` report.
- Candidate screening pipeline: `docs/pipeline-v2.md`, if present, otherwise Experiments 022b, 030, 033 and 035.
- Summaries: `docs/findings-2026-09-28.md` and the later results reports; the cardiologist meeting guide is
  `docs/cardiologist-meeting-prep.md`.
- Read the live files under `outputs/` before reporting progress. PIDs and Markdown summaries are snapshots.

## Git

- Work on a branch; open every PR against `main`, never stacked on another branch. Before handing a PR over,
  merge `origin/main` into it and check GitHub reports it mergeable.
- Stage files by path; never `git add -A`. Commit messages are short and plain, with no Co-Authored-By line.
- Parallel work uses worktrees (for example `~/ecg-wt-cpu`). Symlink `data`, `outputs`, `.venv` and
  `third_party` from the main checkout, run `git ls-files data outputs | xargs git update-index
  --skip-worktree`, and run code with `PYTHONPATH=<worktree>` because the editable install points at the
  main checkout.
- Use the default `.venv` with `uv`; `uv add` is allowed for a genuinely needed library, but check the dry
  run first (`uv sync --inexact`) and remember `pyproject.toml` and `uv.lock` are pinned.
- Another agent (Codex) may share the checkout. Never stage, stash or delete its uncommitted files.

## Frozen code and new experiments

- Experiment receipts pin about 330 files by path, SHA-256 or module name, including `pyproject.toml` and
  `uv.lock`. Never edit a pinned file. New work goes in new files. The method for finding pinned files is in
  `docs/code-quality-review-2026-09-29.md`.
- New code imports shared helpers instead of copying them: `ecg_experiment/intervals.py` for patient
  bootstraps and intervals, `ecg_experiment/paths.py` for paths written into receipts (repository-relative,
  never `resolve()`). Reusable logic lives in `ecg_experiment/`; scripts should not import other scripts.
- Every experiment:
  1. commits its protocol, with the primary comparison and decision rule, before any score;
  2. reproduces its predecessor's numbers exactly as an integrity check;
  3. ends with a results report written from the outputs by the agent that ran it;
  4. writes follow-up ideas into the backlog, where they are ranked. Run one early only if it needs no busy
     GPU, worktree or downloading data.
- Update both queue documents when an experiment changes state. Never present a smoke test or a proposal as a
  result. A real GPU profile gates any full training run.
- Use the shared GPU lock (`ecg_experiment/gpu.py`). Limit CPU threads when agents run in parallel.
- Closed data, never read until a user-approved final test: the PTB-XL test set, the Challenge test groups
  (`docs/challenge-splits-v1.md`) and the EchoNext test set. SPH has been read by Experiments 022-037 and now
  counts as development data. The one-time `final_frozen_test` waits for the user to freeze the pipeline.
- Deferred, do not resume without the user: Experiments 008 and 010, and the 017 second seed.

## Data

- Data stay local: no Git LFS, no DVC remote, no uploads, unless the user chooses otherwise. Git holds only
  small DVC pointers. MIMIC-IV-ECG and EchoNext are credentialed: only aggregate numbers leave the machine,
  and never through an external API.
- MIMIC labels come from the cart's software and are not trusted. MIMIC and CODE-15 are for unlabeled
  self-supervised pretraining only. Labels come from human-read sources (PTB-XL, SPH, Ningbo, Chapman, Georgia,
  CPSC) or echo-confirmed EchoNext.
- SPH is the external test hospital and never enters training. Chapman and Ningbo are one source family, as
  are CPSC and CPSC-Extra. The Challenge sources have no patient IDs, so their split is per record.
- Cohorts are quality-first: curated human-read sources, then CODE-15 from 150k, then MIMIC. See
  `docs/clean-cohorts-v2.md`, `docs/clean-ningbo-v1.md`, `docs/clean-code15-v1.md`.
- Known problems:
  - Ningbo JS13118 is corrupted upstream and excluded.
  - 4.2% of Ningbo ECGs have a lead that is zero for the whole recording.
  - 71 Chapman-Ningbo exact duplicates.
  - The CODE-15 amplitude unit is unresolved.

## Clinical scope

- The binary label (any MI, STTC, CD or HYP code against sinus rhythm alone) is an ECG-annotation proxy, not
  a diagnosis, and it disagrees with the 2017 international athlete criteria.
- Wait for the cardiologist before changing label definitions or adding age-subgroup analyses. Left
  ventricular high voltage alone stays undefined.
- Operating points come from a referral budget fitted on local normal ECGs (Experiment 030), not from a
  fixed sensitivity target.

## Style

- Follow the user's global CLAUDE.md: simple flat functions, type hints, NumPy docstrings, minimal comments,
  let errors raise, no emojis, plain English, little bold.
- EDA: plain Python modules first, notebook last. Results first, then text: every sentence must be backed by
  an executed output. Notebooks describe the data; project recommendations go in separate docs.
- Subagents are authorized. Give each one a complete brief, including the results report. For Codex: Sol
  medium for simple research, Sol high for programming, Astra for hard scientific or design work.
