# Cleanup inventory, 30 September 2026

This inventory checks which tracked files under `docs/`, `scripts/`, `notebooks/` and the repository
root could be removed. It excludes `data/`, `outputs/`, `tests/` and the `ecg_experiment/` library.
It is read-only: nothing was deleted, moved or edited. The branch base is
`origin/chore/commit-pending-nlp25k` (`c480c5e`), which is `main` plus the pending Experiment 011-013 files.

The short answer: the refactor journals you remember are already gone from the tree. They were
deleted on 24 September in `efc012f` and `bbfda41` (`docs/repository-refactor.md`,
`docs/archive/README-before-refactor.md`, `reports/repository-refactor.json`,
`scripts/verify_refactor.py`, `environments/archive/*`). Nearly everything left is protocol, result,
figure or tooling that a receipt or a frozen protocol depends on. This review found 5 delete
candidates and 3 merge candidates, about 96 KiB in total. What still carries refactor and pause
history is prose inside files that stay: `AGENTS.md`, `CONTRIBUTING.md`, `scripts/README.md` and the
queue documents (see "Refactor text that is still tracked").

## Method

- **Files.** `git ls-files` gave 402 files in scope: 205 in `docs/`, 172 in `scripts/`, 11 in
  `notebooks/` and 14 at the root.
- **Pins.** This review reran the scan in [the code quality review](code-quality-review-2026-09-29.md).
  It read 1,346 text receipts (`.json`, `.jsonl`, `.txt`, `.md`, `.yaml`, `.csv`, `.log`) under
  `outputs/` and `data/processed/` of the main checkout. It skipped 12 files over 20 MB in
  `data/processed/`; those are data tables, not receipts. Each file got one pin code:
  - `S` (pinned): an experiment receipt holds the file's repository-relative path, the SHA-256 of
    its current bytes, or its dotted module name. For root files only the hash counts, because a
    bare name such as `README.md` matches every other README.
  - `H` (historical): only the bare file name appears, in a receipt written before the refactor.
    The current bytes no longer match that receipt. The code quality review treats these as frozen too.
  - `W` (weak): the file appears only in `outputs/refactor_pause/source_hashes.json` or
    `outputs/repository_refactor/*`. Those lists record the pre-refactor tree, not an experiment's
    inputs.
  - `-`: not pinned anywhere.
- **Links.** Every tracked text file, including `tests/`, `ecg_experiment/` and the notebooks, was
  split into tokens. A file counts as linked when another file names it by path suffix or base
  name, or by module stem for Python files. "Links" is the number of files that link to it.
- **Sizes** are working-tree bytes. Git numbers come from `git count-objects`,
  `git cat-file --batch-check` and the GitHub API.

A link from a pinned document matters more than the link count suggests. A frozen protocol cannot be
edited to repair a broken link without changing its hash. So any file linked from an `S` document
was kept, whatever its own value. This rule is why `cpc-next-ideas.md` stays (linked from the pinned
Experiment 005 protocol), and also `xecg-next-experiments.md` and `jepa-notes.md` (both linked from
the pinned `astra-next-model-ideas.md`). It also keeps `cross-domain-architecture-candidates.md`,
which is linked from the pinned Experiment 012 protocol.

## Summary

| Category | Files | Size | Pinned (S/H) | Where |
| --- | ---: | ---: | ---: | --- |
| KEEP-ACTIVE | 89 | 1.29 MiB | 37 | root 14, docs 30, scripts 44, notebooks 1 |
| KEEP-EVIDENCE | 305 | 20.25 MiB | 170 | docs 131, docs/figures 39 (8.0 MiB), notebooks 10 (8.8 MiB), scripts 125 |
| MERGE-CANDIDATE | 3 | 16.3 KiB | 0 | docs 3 |
| DELETE-CANDIDATE | 5 | 78.0 KiB | 0 | docs 2, scripts 3 |
| Total | 402 | 21.6 MiB | 207 | |

Notes on the categories:

- **KEEP-EVIDENCE** includes 110 unpinned files. Most are `-results.md` reports, which are written
  after a run and so never appear in its receipts. The rest are figures and report scripts.
- **Unpinned audit scripts** count as evidence when their output exists under `outputs/`, for example
  `audit_cpc_scaling_v3.py`, `audit_nlp25k_successors_v2.py` and the MIMIC report wrappers. Without
  the script, that audit receipt cannot be regenerated.
- **KEEP-ACTIVE** covers the root configuration, the index and queue documents, the backlog and
  priorities, the current reviews, the data-source and cohort documents, and current tooling. That
  tooling is the downloaders, the queue coordinator, the NLP 25k seal and freeze commands,
  `scripts/data/*`, backlog ranking and MLflow import.

## Delete candidates

| File | Size | Reason | Linked from | Receipt pin | Superseded by |
| --- | ---: | --- | --- | --- | --- |
| `scripts/coordination/adopt_nlp25k_cache_seal.py` | 3,539 B | Adopts the 011 v1 profile's full hash as a shared cache seal. That profile was interrupted before the hash finished, and `experiment-queue.md` says the seal "cannot be adopted from this incomplete receipt". Never run. | none | none | `scripts/coordination/create_nlp25k_cache_seal.py` (pinned by `outputs/cache_sessions/nlp25k_v1/creation.json`) |
| `scripts/data/build_nlp25k_selected_raw.py` | 3,126 B | Opt-in builder for `data/processed/nlp25k_selected_raw_v1`. That directory does not exist, so it never ran. No queue or runner uses it. | none | none | The shared seal and in-runner staging (`ecg_experiment/selected_waveform_stage.py`) used by 011-013 |
| `scripts/coordination/handoff_priority_queue.py` | 23,623 B | One-off watcher that handed the frozen queue from Experiment 015 to its successor on 24 September. That handoff finished and the queue is historical. | `tests/test_handoff_priority_queue.py` (imports it); prose mention in `docs/experiment-queue.md:333` | weak only: name in `outputs/refactor_pause/source_hashes.json` and `outputs/repository_refactor/*` | Nothing. Current queues use `run_priority_queue.py` directly. |
| `docs/experiment-013-mamba3-plan.md` | 4,023 B | Planning note written before any runner existed ("No runner ... exists yet"). The frozen 25k protocols and the results replace it. | `docs/experiment-queue.json:573` (as 013's `protocol`), `docs/cross-domain-architecture-candidates.md`, `docs/important-papers.md` | weak only (refactor hash list) | `docs/experiment-013-mamba3-25k.md`, `docs/experiment-013-mamba3-25k-v2.md`, `docs/nlp-inspired-25k-study-results.md` |
| `docs/experiment001-results.pdf` | 45,545 B | Vector copy of the Experiment 001 comparison figure. The tracked PNG has the same content and is the one the reports embed. `scripts/reports/report_experiment.py` regenerates both. | `docs/model-findings-report.md:425` (prose, not a link) | none | `docs/experiment001-results.png` |

Side effects outside the inventory scope:

- Deleting `build_nlp25k_selected_raw.py` leaves `ecg_experiment/compact_selected_raw.py` and
  `tests/test_compact_selected_raw.py` without a caller.
- Deleting `adopt_nlp25k_cache_seal.py` leaves `adopt_external_profile_seal()` in
  `ecg_experiment/cache_session_seal.py`, and its tests, without a caller.
- Check the pins on those library files before removing them too.

## Merge candidates

| File | Size | Merge into | Reason | Linked from | Receipt pin |
| --- | ---: | --- | --- | --- | --- |
| `docs/cpc-improvement-investigation.md` | 4,361 B | `docs/cpc-improvement-research-2026-09-25.md` | A status stub with the question and a list of evidence pointers. The research note, the local-readout protocol and its results hold the substance. The question paragraph is the only text worth moving. | `docs/README.md:44`, `docs/experiment-queue.json:245` (`task`), and the research note itself | none |
| `docs/vision-to-ecg-architecture-candidates.md` | 6,716 B | `docs/cross-domain-architecture-candidates.md` | Vision proposals only; nothing was queued or run. Its partner document ("NLP, genomics and vision") already points to it as the vision shortlist, so one proposals document is enough. Or move its rows into `experiment-backlog.json`. | `docs/cross-domain-architecture-candidates.md:52`, `docs/experiment-queue.json:705` (`other_work.unselected_ideas`), `docs/important-papers.md:467` | weak only |
| `docs/reflection-papers-vs-results-2026-09-29.md` | 5,571 B | `docs/literature-review-2026-09-28.md` (closing section) | Compares that review's predictions with Experiments 022-028. It reads as the review's last chapter. | `docs/cardiologist-meeting-prep.md:815` | none |

After a merge, repoint each link listed above to the target file.

## Pinned files that look useless

Deletion is not recommended for these. Each row says what would be lost.

| File | Pin | Why it looks useless | Reproducibility cost |
| --- | :---: | --- | --- |
| `scripts/experiments/audit_xecg_droppath_rescue016_v6.py` (8.6 KiB) | S (14 receipts) | Replaced by `audit_xecg_droppath_rescue016_v6_v2.py` | The first v6 audit receipt could no longer be replayed from the current tree. `tests/test_xecg_droppath_rescue_v6_audit.py` would have to go with it. |
| `scripts/experiments/run_xecg_adaptation_mb4.py` (398 B) | H | Microbatch-4 wrapper for the deferred Experiment 008 recovery | The command in `outputs/experiment_queue_recovery_008/queue.json` would fail. A 008 resume needs a new manifest anyway. |
| `scripts/coordination/run_cpc_coordinated.py`, `run_xecg_coordinated.py`, `run_xecg_adaptation_coordinated.py` (14.3 KiB) | H | Coordinators for the 004-010 queues, from before the GPU lock | The bytes already differ from those receipts, so reruns already need the old commit. The cost is convenience. They form a chain (`run_xecg_coordinated` imports `run_cpc_coordinated`, which `tests/test_coordination_scripts.py` also imports), so they would have to go together. |
| `scripts/experiments/run_compact_suite.py`, `run_frozen_probes.py` (5.4 KiB) | H | Three-seed drivers for Experiment 001; nothing links to or imports them | Same as above: reruns already need the recorded commit. The cost is convenience. |

## Proposed deletion list

Safe: nothing links to these and no receipt pins them.

```bash
git rm scripts/coordination/adopt_nlp25k_cache_seal.py \
       scripts/data/build_nlp25k_selected_raw.py
```

Needs link fixes. Make the edits in the same commit.

```bash
git rm scripts/coordination/handoff_priority_queue.py tests/test_handoff_priority_queue.py
# docs/experiment-queue.md:333 names it in historical prose; leave the prose or add "(removed 30 Sep)".

git rm docs/experiment-013-mamba3-plan.md
# docs/experiment-queue.json:573: set 013's "protocol" to docs/experiment-013-mamba3-25k-v2.md.
#   That entry is also stale: it says prepared_awaiting_serial_gpu_profile and results null,
#   although 013 is complete.
# docs/cross-domain-architecture-candidates.md:36 and docs/important-papers.md:388:
#   repoint to docs/experiment-013-mamba3-25k.md.

git rm docs/experiment001-results.pdf
# docs/model-findings-report.md:425: drop the sentence about the PDF.

# Merges (after moving the text):
git rm docs/cpc-improvement-investigation.md
# docs/README.md:44 and docs/experiment-queue.json:245 -> docs/cpc-improvement-research-2026-09-25.md
git rm docs/vision-to-ecg-architecture-candidates.md
# docs/cross-domain-architecture-candidates.md:52, docs/experiment-queue.json:705,
# docs/important-papers.md:467 -> docs/cross-domain-architecture-candidates.md
git rm docs/reflection-papers-vs-results-2026-09-29.md
# docs/cardiologist-meeting-prep.md:815 -> docs/literature-review-2026-09-28.md
```

Pinned: not recommended. Listed only for completeness.

```bash
# git rm scripts/experiments/audit_xecg_droppath_rescue016_v6.py tests/test_xecg_droppath_rescue_v6_audit.py
# git rm scripts/experiments/run_xecg_adaptation_mb4.py
# git rm scripts/coordination/run_cpc_coordinated.py scripts/coordination/run_xecg_coordinated.py \
#        scripts/coordination/run_xecg_adaptation_coordinated.py
# git rm scripts/experiments/run_compact_suite.py scripts/experiments/run_frozen_probes.py
```

None of the linking files above is experiment-pinned. `cross-domain-architecture-candidates.md`,
`important-papers.md`, `model-findings-report.md` and `experiment-queue.json` carry only weak
pins. `README.md` and `cardiologist-meeting-prep.md` carry none. So every fix is an ordinary edit.

## Refactor text that is still tracked

The refactor and pause receipts (`outputs/refactor_pause/`, `outputs/repository_refactor/`) are not
in Git, because `outputs/` is ignored. What remains in Git is prose:

- `AGENTS.md` lines 22-69, "User-requested experiment pause", is 572 of its 1,282 words. It is a
  dated log of the 24 September pause, the 26 September cancel and the 27 September resume. It ends
  with "no experiment queue is currently active". The live facts would fit in about five lines:
  - no queue is active;
  - 011-017 are complete (017 as v2);
  - 008 and 010 are deferred;
  - calibration and test are still closed;
  - links to the two pause receipts.

  The rest is in Git history and `docs/experiment-queue.md`. This file holds project instructions,
  so the edit is your call.
- `CONTRIBUTING.md:35` ("Training and scheduling remain paused until the user requests a resume")
  and `scripts/README.md:8` ("training and scheduling remain paused") have been out of date since the
  26 and 27 September resumes.
- `docs/experiment-queue.json` `other_work.legacy_runner` still says "Stopped at user request for
  refactor".
- `docs/experiment-queue.md` (68 KiB) is mostly a dated running log. It could be cut to current state
  plus links, with the history left to Git. It is a queue document, so this review did not
  mark it for change.

## Git size

| Measure | Value |
| --- | ---: |
| GitHub repository size (API `size`) | 19,582 KiB (19.1 MiB) |
| Local object store (`.git/objects`, shared by all worktrees) | 34 MiB: 606 packed objects in 738 KiB, plus 2,154 loose objects (31.8 MiB) never packed by `git gc` |
| Objects reachable from all refs | 1,401 blobs, 44.4 MiB raw, 25.3 MiB zlib-compressed without deltas |
| Of which notebooks | 13.4 MiB (7.2 MiB are the old `eda/notebooks/` copies before the rename to `notebooks/`) |
| Of which PNG | 7.0 MiB |
| Tracked tree at HEAD | 23.6 MiB raw: notebooks 8.8 MiB, `docs/figures/` 8.0 MiB (39 PNGs; the 20 Experiment 024 audit strips are 4.6 MiB), `docs/experiment001-results.png` 0.43 MiB, `environments/pretrained/uv.lock` 0.22 MiB |

Largest tracked files at HEAD:

| File | Size |
| --- | ---: |
| `notebooks/01-jr-ptbxl.ipynb` | 2.69 MiB |
| `notebooks/06-jr-clean-cohorts.ipynb` | 1.23 MiB |
| `notebooks/04-jr-code15.ipynb` | 1.00 MiB |
| `notebooks/03-jr-challenge.ipynb` | 0.85 MiB |
| `notebooks/07-jr-sph.ipynb` | 0.75 MiB |
| `notebooks/09-jr-ningbo.ipynb` | 0.61 MiB |
| `notebooks/05-jr-cross-dataset.ipynb` | 0.58 MiB |
| `notebooks/02-jr-mimic.ipynb` | 0.45 MiB |
| `docs/experiment001-results.png` | 0.42 MiB |
| `notebooks/08-jr-echonext.ipynb` | 0.37 MiB |
| `docs/experiment001-results.pdf` and the Experiment 024 audit and reference PNGs | 0.04-0.27 MiB each |

History rewriting is not worth it:

- **The saving is small.** The most it could reclaim is about 7 MiB (the old notebook copies) out of
  a 19 MiB repository. Deleting files from the current tree reclaims nothing, because history
  keeps them.
- **The cost is high.** A rewrite changes every commit SHA. It needs a force-push, a rebase of every
  open PR and worktree, and coordination with the other agent working in this checkout.
- **Receipts pin commit SHAs from this repository.** Three do:
  - `b173a944da` in `outputs/experiment016_encoder_motion_replication_v14/final_receipt.json` and
    two 016 v14 queue manifests;
  - `eedff79466` in the 016 v11 cost ledgers and stop receipts;
  - `a2f81592f7` in `outputs/data_quality/training_union_v1/metadata.json`.

  After a rewrite those revisions would be unreachable.

Two cheaper steps do more:

- **Pack the local object store.** Run `git gc` in the main checkout at a quiet time; it shrinks
  the 31.8 MiB of loose objects locally. Do not run it while another agent is committing.
- **Commit re-executed notebooks sparingly.** Each full re-run of the ten notebooks adds up to
  about 9 MiB of new blobs. Notebook outputs are the main source of future growth.

## Full inventory

Pin codes are defined under Method. "Links" counts the tracked files that name this file.

### KEEP-ACTIVE (89 files, 1,322.4 KiB)

| File | KiB | Pin | Links |
| --- | ---: | :---: | ---: |
| `.dvcignore` | 0.3 | W | 0 |
| `.editorconfig` | 0.3 | - | 0 |
| `.env.example` | 0.3 | - | 0 |
| `.gitignore` | 1.4 | W | 0 |
| `.pre-commit-config.yaml` | 0.5 | - | 0 |
| `.python-version` | 0.0 | W | 0 |
| `AGENTS.md` | 9.2 | W | 2 |
| `CONTRIBUTING.md` | 4.9 | - | 5 |
| `LICENSE` | 11.2 | - | 2 |
| `README.md` | 4.2 | - | 15 |
| `docs/README.md` | 4.5 | - | 3 |
| `docs/cardiologist-meeting-prep.md` | 57.6 | - | 1 |
| `docs/challenge-label-mapping.md` | 14.9 | - | 10 |
| `docs/challenge-postprocessing.md` | 5.7 | W | 2 |
| `docs/challenge-splits-v1.md` | 6.2 | - | 10 |
| `docs/clean-code15-v1.md` | 8.2 | - | 4 |
| `docs/clean-cohorts-v1.md` | 6.3 | - | 5 |
| `docs/clean-cohorts-v2.md` | 11.0 | - | 4 |
| `docs/clean-data-rerun-review.md` | 10.2 | - | 4 |
| `docs/clean-ningbo-v1.md` | 7.7 | - | 8 |
| `docs/clean-sph-echonext-v1.md` | 2.7 | - | 1 |
| `docs/code-quality-review-2026-09-29.md` | 17.5 | - | 2 |
| `docs/code15-label-groups.md` | 1.5 | - | 1 |
| `docs/data-cleaning-practices-review.md` | 40.3 | - | 4 |
| `docs/data-processing-astra-review.md` | 15.3 | W | 2 |
| `docs/data-quality-assessment.md` | 21.0 | W | 6 |
| `docs/data-sources.md` | 6.4 | W | 3 |
| `docs/data-versioning.md` | 5.3 | - | 4 |
| `docs/eda-pipeline-review.md` | 9.0 | - | 6 |
| `docs/experiment-backlog.json` | 59.0 | - | 8 |
| `docs/experiment-priorities.md` | 13.9 | - | 6 |
| `docs/experiment-queue.json` | 103.4 | W | 7 |
| `docs/experiment-queue.md` | 66.9 | W | 11 |
| `docs/experiment-tracking.md` | 4.9 | - | 3 |
| `docs/important-papers.md` | 41.7 | W | 3 |
| `docs/literature-review-2026-09-28.md` | 8.6 | - | 8 |
| `docs/public-data-strategy.md` | 7.1 | W | 3 |
| `docs/sampled-100k-plus-labels-v1.md` | 6.6 | - | 4 |
| `docs/sph-echonext-eda-review.md` | 3.2 | - | 6 |
| `docs/training-dataset-v1.md` | 7.6 | W | 1 |
| `dvc.lock` | 3.8 | W | 0 |
| `dvc.yaml` | 1.6 | W | 1 |
| `notebooks/README.md` | 4.3 | - | 3 |
| `pyproject.toml` | 2.2 | S | 48 |
| `scripts/README.md` | 1.8 | - | 4 |
| `scripts/__init__.py` | 0.1 | - | 0 |
| `scripts/coordination/__init__.py` | 0.0 | S | 0 |
| `scripts/coordination/common.py` | 4.1 | - | 1 |
| `scripts/coordination/create_nlp25k_cache_seal.py` | 2.6 | S | 4 |
| `scripts/coordination/freeze_nlp25k_manifest.py` | 3.6 | S | 0 |
| `scripts/coordination/run_priority_queue.py` | 12.6 | H | 11 |
| `scripts/data/__init__.py` | 0.0 | S | 0 |
| `scripts/data/build_challenge_splits.py` | 8.5 | S | 1 |
| `scripts/data/build_clean_cohorts.py` | 11.2 | S | 4 |
| `scripts/data/build_clean_cohorts_v2.py` | 15.0 | S | 2 |
| `scripts/data/build_code15_clean.py` | 8.7 | S | 2 |
| `scripts/data/build_code15_cpc_cache.py` | 5.5 | - | 3 |
| `scripts/data/build_echonext_cache.py` | 5.2 | S | 4 |
| `scripts/data/build_ningbo_clean.py` | 13.9 | S | 2 |
| `scripts/data/build_sampled_training_dataset.py` | 11.4 | S | 2 |
| `scripts/data/build_sph_clean.py` | 3.6 | S | 3 |
| `scripts/data/build_training_dataset.py` | 28.7 | H | 2 |
| `scripts/data/cache_sampled_cpc.py` | 5.4 | S | 2 |
| `scripts/data/download_ptbxl_metadata.py` | 2.9 | H | 0 |
| `scripts/data/extract_challenge_features.py` | 21.6 | S | 1 |
| `scripts/data/materialize_challenge_ecg.py` | 27.3 | S | 3 |
| `scripts/data/prepare_beat_tokens.py` | 11.5 | H | 3 |
| `scripts/data/prepare_code15.py` | 15.3 | S | 3 |
| `scripts/data/prepare_cpc_data.py` | 23.2 | S | 4 |
| `scripts/data/prepare_georgia_ssl.py` | 9.0 | H | 1 |
| `scripts/data/prepare_ptbxl.py` | 13.5 | H | 5 |
| `scripts/data/prepare_public_ecg.py` | 14.9 | S | 8 |
| `scripts/data/prepare_xecg.py` | 7.8 | S | 7 |
| `scripts/data/prepare_xecg_ssl.py` | 11.1 | H | 3 |
| `scripts/download_missing_ecg.sh` | 0.5 | W | 2 |
| `scripts/download_ptbxl_waveforms.py` | 15.2 | S | 4 |
| `scripts/download_public_ecg.py` | 12.7 | S | 4 |
| `scripts/experiments/__init__.py` | 0.0 | S | 0 |
| `scripts/extract_pretrained.py` | 8.3 | S | 5 |
| `scripts/features/__init__.py` | 0.0 | S | 0 |
| `scripts/prepare_mimic_ssl.py` | 17.1 | S | 6 |
| `scripts/reports/__init__.py` | 0.0 | S | 0 |
| `scripts/reports/build_eda_caches.py` | 2.1 | - | 1 |
| `scripts/reports/rank_backlog.py` | 0.6 | - | 1 |
| `scripts/tracking/__init__.py` | 0.0 | S | 0 |
| `scripts/tracking/import_mlflow_history.py` | 2.6 | W | 3 |
| `scripts/validation/__init__.py` | 0.0 | S | 0 |
| `scripts/validation/validate_data_versioning.py` | 1.6 | W | 3 |
| `uv.lock` | 343.9 | S | 43 |

### KEEP-EVIDENCE (305 files, 20,738.5 KiB)

| File | KiB | Pin | Links |
| --- | ---: | :---: | ---: |
| `docs/astra-next-model-ideas.md` | 24.0 | S | 4 |
| `docs/clean-cached-probe-rerun-v1.md` | 1.9 | S | 2 |
| `docs/clean-probe-fusion-v1.md` | 1.6 | S | 2 |
| `docs/cpc-improvement-research-2026-09-25.md` | 9.1 | - | 5 |
| `docs/cpc-local-readout-v1-results.md` | 4.2 | - | 5 |
| `docs/cpc-local-readout-v1.md` | 5.5 | S | 5 |
| `docs/cpc-next-ideas.md` | 16.9 | W | 1 |
| `docs/cross-domain-architecture-candidates.md` | 11.4 | W | 6 |
| `docs/custom-architecture.md` | 7.7 | W | 5 |
| `docs/data-loading-runtime-fix.md` | 4.8 | W | 1 |
| `docs/experiment-001.md` | 7.9 | W | 2 |
| `docs/experiment-003-mimic.md` | 13.0 | W | 2 |
| `docs/experiment-004-cpc.md` | 15.0 | H | 3 |
| `docs/experiment-005-word2vec.md` | 5.0 | S | 2 |
| `docs/experiment-006-tokenization.md` | 18.4 | S | 2 |
| `docs/experiment-007-xecg.md` | 9.2 | S | 5 |
| `docs/experiment-008-vision-ssl.md` | 18.8 | H | 5 |
| `docs/experiment-009-mismatch.md` | 4.0 | W | 2 |
| `docs/experiment-010-crosslead.md` | 12.4 | W | 3 |
| `docs/experiment-011-delta-memory-25k-v2.md` | 5.9 | S | 2 |
| `docs/experiment-011-delta-memory-25k.md` | 4.8 | S | 2 |
| `docs/experiment-011-delta-memory.md` | 6.2 | S | 4 |
| `docs/experiment-012-temporal-hybrid-v3.md` | 3.7 | S | 5 |
| `docs/experiment-012-temporal-hybrid.md` | 5.0 | S | 3 |
| `docs/experiment-013-mamba3-25k-v2.md` | 3.6 | S | 2 |
| `docs/experiment-013-mamba3-25k.md` | 7.5 | S | 2 |
| `docs/experiment-014-fusion.md` | 5.4 | S | 3 |
| `docs/experiment-015-distillation.md` | 4.9 | S | 3 |
| `docs/experiment-016-droppath-rescue-profile-results.md` | 4.1 | S | 3 |
| `docs/experiment-016-droppath-rescue-v5.md` | 2.3 | S | 4 |
| `docs/experiment-016-droppath-rescue-v6-results.md` | 4.4 | - | 3 |
| `docs/experiment-016-droppath-rescue-v6.md` | 11.5 | S | 4 |
| `docs/experiment-016-droppath-rescue.md` | 10.9 | S | 5 |
| `docs/experiment-016-encoder-motion-replication-v10-results.md` | 4.2 | - | 7 |
| `docs/experiment-016-encoder-motion-replication-v10.md` | 15.1 | S | 8 |
| `docs/experiment-016-encoder-motion-replication-v11-results.md` | 3.7 | - | 4 |
| `docs/experiment-016-encoder-motion-replication-v11.md` | 19.1 | S | 6 |
| `docs/experiment-016-encoder-motion-replication-v12-results.md` | 3.2 | S | 3 |
| `docs/experiment-016-encoder-motion-replication-v12.md` | 13.7 | S | 3 |
| `docs/experiment-016-encoder-motion-replication-v13-results.md` | 3.9 | S | 2 |
| `docs/experiment-016-encoder-motion-replication-v13.md` | 21.5 | S | 3 |
| `docs/experiment-016-encoder-motion-replication-v14-results.md` | 7.4 | - | 4 |
| `docs/experiment-016-encoder-motion-replication-v14.md` | 13.6 | S | 4 |
| `docs/experiment-016-encoder-motion-v9-results.md` | 5.9 | - | 8 |
| `docs/experiment-016-encoder-motion-v9.md` | 13.1 | S | 7 |
| `docs/experiment-016-frozen-readout-audit-v7-results.md` | 4.9 | - | 3 |
| `docs/experiment-016-frozen-readout-audit-v7.md` | 9.6 | S | 4 |
| `docs/experiment-016-head-mechanism-v8-results.md` | 5.7 | - | 4 |
| `docs/experiment-016-head-mechanism-v8.md` | 12.9 | S | 4 |
| `docs/experiment-016-paper-investigation.md` | 6.1 | - | 4 |
| `docs/experiment-016-xecg-lpft.md` | 3.1 | S | 3 |
| `docs/experiment-017-clean-replication-results.md` | 4.4 | - | 5 |
| `docs/experiment-017-clean-replication-v2.md` | 1.3 | S | 4 |
| `docs/experiment-017-clean-replication.md` | 9.0 | S | 5 |
| `docs/experiment-017-morphology-v2.md` | 1.9 | S | 4 |
| `docs/experiment-017-morphology.md` | 6.7 | S | 2 |
| `docs/experiment-018-cpc-data-scaling-readout-results.md` | 3.1 | - | 4 |
| `docs/experiment-018-cpc-data-scaling-readout.md` | 3.6 | - | 4 |
| `docs/experiment-018-cpc-data-scaling-v2.md` | 3.4 | - | 3 |
| `docs/experiment-018-cpc-data-scaling-v3-results.md` | 2.2 | - | 4 |
| `docs/experiment-018-cpc-data-scaling-v3.md` | 3.2 | - | 4 |
| `docs/experiment-018-cpc-data-scaling.md` | 3.8 | - | 3 |
| `docs/experiment-019-cpc-25k-readout-results.md` | 3.7 | - | 5 |
| `docs/experiment-019-cpc-25k-readout.md` | 2.5 | - | 5 |
| `docs/experiment-019-cpc-25k-training-results.md` | 2.1 | - | 4 |
| `docs/experiment-019-cpc-25k.md` | 3.5 | - | 5 |
| `docs/experiment-020-full-development-readout-results.md` | 4.2 | - | 2 |
| `docs/experiment-020-full-development-readout.md` | 6.5 | S | 3 |
| `docs/experiment-021-code15-supervised.md` | 4.8 | - | 1 |
| `docs/experiment-022-sph-external-readout-results.md` | 6.5 | - | 7 |
| `docs/experiment-022-sph-external-readout.md` | 7.8 | S | 6 |
| `docs/experiment-022b-multisource-readout-results.md` | 16.2 | - | 5 |
| `docs/experiment-022b-multisource-readout.md` | 13.6 | S | 4 |
| `docs/experiment-023-echonext-readout-results.md` | 8.3 | - | 6 |
| `docs/experiment-023-echonext-readout.md` | 7.4 | S | 4 |
| `docs/experiment-024-clinician-review.md` | 12.9 | - | 4 |
| `docs/experiment-024-embedding-geometry-results.md` | 13.0 | - | 7 |
| `docs/experiment-024-embedding-geometry.md` | 14.7 | S | 5 |
| `docs/experiment-024b-multisource-geometry-results.md` | 15.2 | - | 6 |
| `docs/experiment-024b-multisource-geometry.md` | 14.7 | S | 4 |
| `docs/experiment-025-label-efficiency-results.md` | 6.8 | - | 9 |
| `docs/experiment-025-label-efficiency.md` | 6.7 | S | 6 |
| `docs/experiment-025b-label-efficiency-multisource-results.md` | 13.3 | - | 5 |
| `docs/experiment-025b-label-efficiency-multisource.md` | 12.5 | S | 4 |
| `docs/experiment-026-normal-manifold-results.md` | 12.7 | - | 7 |
| `docs/experiment-026-normal-manifold.md` | 9.6 | S | 5 |
| `docs/experiment-026b-multisource-normal-manifold-results.md` | 11.0 | - | 6 |
| `docs/experiment-026b-multisource-normal-manifold.md` | 10.5 | S | 4 |
| `docs/experiment-027-calibrated-threshold-results.md` | 9.0 | - | 6 |
| `docs/experiment-027-calibrated-threshold.md` | 7.7 | S | 5 |
| `docs/experiment-027b-multisource-calibration-results.md` | 11.8 | - | 5 |
| `docs/experiment-027b-multisource-calibration.md` | 10.0 | S | 5 |
| `docs/experiment-028-echo-label-efficiency-results.md` | 9.0 | - | 5 |
| `docs/experiment-028-echo-label-efficiency.md` | 7.0 | S | 4 |
| `docs/experiment-029-local-adaptation-results.md` | 14.6 | - | 6 |
| `docs/experiment-029-local-adaptation.md` | 14.0 | S | 4 |
| `docs/experiment-030-referral-budget-results.md` | 13.1 | - | 4 |
| `docs/experiment-030-referral-budget.md` | 11.3 | S | 3 |
| `docs/experiment-031-hybrid-screening-score-results.md` | 11.5 | - | 3 |
| `docs/experiment-031-hybrid-screening-score.md` | 12.2 | S | 3 |
| `docs/experiment-032-rhythm-findings-results.md` | 11.1 | - | 3 |
| `docs/experiment-032-rhythm-findings.md` | 15.4 | S | 3 |
| `docs/experiment-033-finding-heads-screen-results.md` | 10.9 | - | 3 |
| `docs/experiment-033-finding-heads-screen.md` | 13.5 | S | 3 |
| `docs/experiment-034-one-class-baselines-results.md` | 12.7 | - | 3 |
| `docs/experiment-034-one-class-baselines.md` | 13.2 | S | 3 |
| `docs/experiment-035-hard-subset-results.md` | 13.0 | - | 3 |
| `docs/experiment-035-hard-subset.md` | 14.4 | S | 3 |
| `docs/experiment001-metrics.json` | 62.3 | W | 3 |
| `docs/experiment001-results.md` | 11.0 | W | 7 |
| `docs/experiment001-results.png` | 432.6 | - | 4 |
| `docs/experiment002-metrics.json` | 5.5 | W | 3 |
| `docs/figures/experiment-024/audit/audit_01_2410.png` | 247.3 | - | 1 |
| `docs/figures/experiment-024/audit/audit_02_9669.png` | 262.6 | - | 1 |
| `docs/figures/experiment-024/audit/audit_03_12964.png` | 258.2 | - | 1 |
| `docs/figures/experiment-024/audit/audit_04_2415.png` | 247.2 | - | 1 |
| `docs/figures/experiment-024/audit/audit_05_7664.png` | 214.5 | - | 1 |
| `docs/figures/experiment-024/audit/audit_06_14915.png` | 247.3 | - | 1 |
| `docs/figures/experiment-024/audit/audit_07_17394.png` | 223.0 | - | 1 |
| `docs/figures/experiment-024/audit/audit_08_17624.png` | 258.1 | - | 1 |
| `docs/figures/experiment-024/audit/audit_09_17820.png` | 264.4 | - | 1 |
| `docs/figures/experiment-024/audit/audit_10_131.png` | 223.5 | - | 1 |
| `docs/figures/experiment-024/audit/audit_11_1636.png` | 209.3 | - | 1 |
| `docs/figures/experiment-024/audit/audit_12_4796.png` | 241.8 | - | 1 |
| `docs/figures/experiment-024/audit/audit_13_10521.png` | 235.4 | - | 1 |
| `docs/figures/experiment-024/audit/audit_14_10916.png` | 233.4 | - | 1 |
| `docs/figures/experiment-024/audit/audit_15_12377.png` | 268.4 | - | 1 |
| `docs/figures/experiment-024/audit/audit_16_17884.png` | 233.7 | - | 1 |
| `docs/figures/experiment-024/audit/audit_17_21366.png` | 202.0 | - | 1 |
| `docs/figures/experiment-024/audit/audit_18_2366.png` | 215.9 | - | 1 |
| `docs/figures/experiment-024/audit/audit_19_2862.png` | 238.4 | - | 1 |
| `docs/figures/experiment-024/audit/audit_20_4189.png` | 215.5 | - | 1 |
| `docs/figures/experiment-024/auroc_by_method_and_encoder.png` | 104.2 | S | 2 |
| `docs/figures/experiment-024/reference_cd_10133.png` | 252.3 | S | 1 |
| `docs/figures/experiment-024/reference_hyp_542.png` | 239.3 | S | 1 |
| `docs/figures/experiment-024/reference_mi_9600.png` | 225.5 | S | 1 |
| `docs/figures/experiment-024/reference_norm_3212.png` | 203.9 | S | 1 |
| `docs/figures/experiment-024/reference_sttc_18255.png` | 258.5 | S | 1 |
| `docs/figures/experiment-024b/ami_by_factor_and_k.png` | 119.8 | S | 2 |
| `docs/figures/experiment-024b/reference_cd_cpsc_2018_A2705.png` | 240.7 | S | 0 |
| `docs/figures/experiment-024b/reference_hyp_georgia_E01909.png` | 248.5 | S | 0 |
| `docs/figures/experiment-024b/reference_mi_ptbxl_5155.png` | 234.5 | S | 0 |
| `docs/figures/experiment-024b/reference_norm_ptbxl_9820.png` | 239.8 | S | 0 |
| `docs/figures/experiment-024b/reference_sttc_ningbo_JS17604.png` | 251.3 | S | 0 |
| `docs/figures/experiment-024b/sph_auroc_by_method.png` | 85.9 | S | 2 |
| `docs/figures/experiment-029/auroc_by_local_normals.png` | 79.1 | - | 2 |
| `docs/figures/experiment-029/sensitivity_by_local_ecgs.png` | 130.7 | - | 2 |
| `docs/figures/experiment-030/referral_budget.png` | 203.0 | - | 2 |
| `docs/figures/experiment-031/hybrid_score.png` | 123.2 | - | 2 |
| `docs/figures/experiment-033/finding_screen.png` | 107.1 | - | 2 |
| `docs/figures/experiment-034/one_class_baselines.png` | 95.4 | - | 2 |
| `docs/findings-2026-09-28.md` | 5.8 | - | 6 |
| `docs/jepa-notes.md` | 2.1 | W | 4 |
| `docs/mimic-cpc-cluster-label-profile.md` | 4.3 | S | 4 |
| `docs/mimic-cpc-clustering.md` | 3.3 | - | 3 |
| `docs/mimic-cpc-flag-audit.md` | 2.8 | - | 1 |
| `docs/mimic-cpc-machine-disagreement.md` | 7.1 | - | 3 |
| `docs/model-findings-report.md` | 52.6 | W | 6 |
| `docs/nlp-inspired-25k-study-results.md` | 4.3 | - | 4 |
| `docs/nlp-inspired-25k-study.md` | 5.1 | S | 8 |
| `docs/normal-manifold-novelty-search.md` | 14.4 | - | 3 |
| `docs/paired-adaptation-comparisons.json` | 1.3 | W | 3 |
| `docs/pretrained-notes.md` | 5.9 | W | 4 |
| `docs/processed-ecg-eda.md` | 11.7 | W | 2 |
| `docs/ptb-mimic-cpc-diagnosis-geometry-aggregate.json` | 6.1 | - | 1 |
| `docs/ptb-mimic-cpc-diagnosis-geometry-protocol-at-run.md` | 3.2 | S | 1 |
| `docs/ptb-mimic-cpc-diagnosis-geometry.md` | 6.2 | - | 5 |
| `docs/released-ecg-cpc.md` | 5.0 | S | 5 |
| `docs/vision-to-ecg-research.md` | 8.6 | W | 1 |
| `docs/xecg-next-experiments.md` | 13.1 | W | 3 |
| `notebooks/01-jr-ptbxl.ipynb` | 2,754.7 | - | 1 |
| `notebooks/02-jr-mimic.ipynb` | 462.9 | - | 1 |
| `notebooks/03-jr-challenge.ipynb` | 870.1 | - | 1 |
| `notebooks/04-jr-code15.ipynb` | 1,027.9 | - | 2 |
| `notebooks/05-jr-cross-dataset.ipynb` | 594.2 | - | 1 |
| `notebooks/06-jr-clean-cohorts.ipynb` | 1,260.0 | - | 3 |
| `notebooks/07-jr-sph.ipynb` | 767.4 | - | 4 |
| `notebooks/08-jr-echonext.ipynb` | 381.1 | - | 5 |
| `notebooks/09-jr-ningbo.ipynb` | 628.9 | - | 4 |
| `notebooks/10-jr-cohorts-v2.ipynb` | 265.6 | - | 3 |
| `scripts/coordination/run_cpc_coordinated.py` | 5.0 | H | 3 |
| `scripts/coordination/run_xecg_adaptation_coordinated.py` | 4.5 | H | 1 |
| `scripts/coordination/run_xecg_coordinated.py` | 4.8 | H | 3 |
| `scripts/experiments/adapt_ecgfm.py` | 30.1 | H | 3 |
| `scripts/experiments/audit_xecg_droppath_rescue016_v6.py` | 8.6 | S | 1 |
| `scripts/experiments/audit_xecg_droppath_rescue016_v6_v2.py` | 9.4 | S | 1 |
| `scripts/experiments/describe_hard_subset035.py` | 9.6 | S | 3 |
| `scripts/experiments/finetune_pretrained.py` | 32.1 | H | 4 |
| `scripts/experiments/probe_delta_gpu_staging011.py` | 3.5 | S | 0 |
| `scripts/experiments/probe_delta_memory011.py` | 4.0 | - | 1 |
| `scripts/experiments/probe_pretrained.py` | 6.7 | H | 3 |
| `scripts/experiments/profile_delta_memory011.py` | 7.0 | S | 2 |
| `scripts/experiments/profile_delta_memory_gpu011.py` | 6.5 | S | 2 |
| `scripts/experiments/refit_jepa_probe014.py` | 6.5 | H | 1 |
| `scripts/experiments/run_calibrated_threshold027.py` | 21.8 | S | 11 |
| `scripts/experiments/run_clean_cached_probes.py` | 0.6 | S | 1 |
| `scripts/experiments/run_clean_probe_fusion.py` | 0.7 | S | 0 |
| `scripts/experiments/run_code15_supervised021.py` | 14.6 | - | 2 |
| `scripts/experiments/run_compact_suite.py` | 3.6 | H | 0 |
| `scripts/experiments/run_cpc_25k.py` | 9.0 | S | 3 |
| `scripts/experiments/run_cpc_25k_readout.py` | 11.0 | S | 2 |
| `scripts/experiments/run_cpc_crosslead.py` | 31.9 | H | 3 |
| `scripts/experiments/run_cpc_crosslead_profile_v2.py` | 4.6 | H | 4 |
| `scripts/experiments/run_cpc_experiment.py` | 15.8 | S | 10 |
| `scripts/experiments/run_cpc_local_readout.py` | 0.6 | S | 2 |
| `scripts/experiments/run_cpc_morphology017.py` | 38.9 | H | 4 |
| `scripts/experiments/run_cpc_morphology017_clean.py` | 0.2 | S | 1 |
| `scripts/experiments/run_cpc_morphology017_clean_v2.py` | 0.5 | S | 0 |
| `scripts/experiments/run_cpc_morphology017_v2.py` | 0.9 | H | 5 |
| `scripts/experiments/run_cpc_prediction_mismatch.py` | 47.0 | H | 2 |
| `scripts/experiments/run_cpc_scaling_pilot.py` | 9.0 | S | 2 |
| `scripts/experiments/run_cpc_scaling_pilot_v2.py` | 10.2 | S | 3 |
| `scripts/experiments/run_cpc_scaling_pilot_v3.py` | 10.6 | S | 2 |
| `scripts/experiments/run_cpc_scaling_readout.py` | 7.6 | S | 2 |
| `scripts/experiments/run_cpc_temporal_hybrid012.py` | 18.7 | S | 1 |
| `scripts/experiments/run_cpc_temporal_hybrid012_readout.py` | 14.2 | S | 3 |
| `scripts/experiments/run_cpc_temporal_hybrid012_readout_v3.py` | 14.1 | S | 4 |
| `scripts/experiments/run_cpc_temporal_hybrid012_v3.py` | 18.7 | S | 2 |
| `scripts/experiments/run_cpc_tokenization.py` | 52.3 | H | 5 |
| `scripts/experiments/run_cpc_word2vec.py` | 21.1 | H | 4 |
| `scripts/experiments/run_delta_memory25k011.py` | 24.7 | S | 4 |
| `scripts/experiments/run_delta_memory25k011_v2.py` | 27.8 | S | 3 |
| `scripts/experiments/run_echo_label_efficiency028.py` | 16.9 | S | 3 |
| `scripts/experiments/run_echonext_readout023.py` | 31.6 | S | 2 |
| `scripts/experiments/run_embedding_geometry024.py` | 25.1 | S | 4 |
| `scripts/experiments/run_finding_screen033.py` | 23.8 | S | 1 |
| `scripts/experiments/run_frozen_probes.py` | 1.8 | H | 0 |
| `scripts/experiments/run_full_development_readout020.py` | 14.4 | S | 2 |
| `scripts/experiments/run_hard_subset035.py` | 33.8 | S | 2 |
| `scripts/experiments/run_hybrid_score031.py` | 31.2 | S | 1 |
| `scripts/experiments/run_jepa_cpc_distillation.py` | 40.1 | H | 6 |
| `scripts/experiments/run_jepa_cpc_fusion.py` | 33.3 | H | 4 |
| `scripts/experiments/run_label_efficiency025.py` | 16.0 | S | 12 |
| `scripts/experiments/run_label_efficiency025b.py` | 30.3 | S | 3 |
| `scripts/experiments/run_lead_innovation.py` | 19.6 | H | 1 |
| `scripts/experiments/run_local_adaptation029.py` | 28.2 | S | 4 |
| `scripts/experiments/run_mamba25k013.py` | 0.2 | S | 2 |
| `scripts/experiments/run_mamba25k013_v2.py` | 0.2 | S | 3 |
| `scripts/experiments/run_mimic_scale.py` | 29.1 | H | 4 |
| `scripts/experiments/run_multisource_calibration027b.py` | 20.8 | S | 12 |
| `scripts/experiments/run_multisource_geometry024b.py` | 46.0 | S | 3 |
| `scripts/experiments/run_multisource_manifold026b.py` | 23.7 | S | 11 |
| `scripts/experiments/run_multisource_readout022b.py` | 31.5 | S | 7 |
| `scripts/experiments/run_normal_manifold026.py` | 18.5 | S | 9 |
| `scripts/experiments/run_one_class_baselines034.py` | 24.2 | S | 2 |
| `scripts/experiments/run_referral_budget030.py` | 22.0 | S | 3 |
| `scripts/experiments/run_released_cpc.py` | 4.8 | H | 2 |
| `scripts/experiments/run_rhythm_findings032.py` | 34.9 | S | 1 |
| `scripts/experiments/run_sph_external022.py` | 29.5 | S | 12 |
| `scripts/experiments/run_xecg_adaptation.py` | 41.2 | H | 4 |
| `scripts/experiments/run_xecg_adaptation_mb4.py` | 0.4 | H | 2 |
| `scripts/experiments/run_xecg_droppath_rescue016.py` | 39.0 | S | 8 |
| `scripts/experiments/run_xecg_droppath_rescue016_v6.py` | 29.1 | S | 5 |
| `scripts/experiments/run_xecg_encoder_motion016_v9.py` | 36.0 | S | 6 |
| `scripts/experiments/run_xecg_encoder_motion_replication016_v10.py` | 25.3 | S | 3 |
| `scripts/experiments/run_xecg_encoder_motion_replication016_v11.py` | 0.9 | S | 3 |
| `scripts/experiments/run_xecg_encoder_motion_replication016_v14.py` | 37.0 | S | 2 |
| `scripts/experiments/run_xecg_finetune.py` | 47.1 | H | 3 |
| `scripts/experiments/run_xecg_frozen_readout016_v7.py` | 31.7 | S | 2 |
| `scripts/experiments/run_xecg_head_mechanism016_v8.py` | 28.6 | S | 2 |
| `scripts/experiments/run_xecg_probe_finetune016.py` | 67.4 | S | 13 |
| `scripts/features/extract_ecg_cpc.py` | 19.6 | H | 3 |
| `scripts/features/extract_jepa.py` | 9.3 | S | 4 |
| `scripts/features/merge_pretrained_features.py` | 6.6 | H | 1 |
| `scripts/features/prepare_feature_union.py` | 5.1 | H | 1 |
| `scripts/reports/audit_mimic_cpc_clusters.py` | 0.3 | - | 1 |
| `scripts/reports/audit_mimic_machine_disagreement.py` | 0.3 | - | 0 |
| `scripts/reports/cluster_mimic_cpc.py` | 0.3 | - | 2 |
| `scripts/reports/cluster_mimic_cpc_negatives.py` | 0.3 | - | 0 |
| `scripts/reports/compare_adaptation.py` | 8.3 | H | 3 |
| `scripts/reports/count_code15_label_groups.py` | 0.3 | - | 1 |
| `scripts/reports/count_mimic_cpc_flags.py` | 0.5 | - | 1 |
| `scripts/reports/diagnose_xecg_probe016.py` | 5.3 | S | 0 |
| `scripts/reports/eda_processed_ecg.py` | 49.3 | W | 3 |
| `scripts/reports/plot_finding_screen033.py` | 4.9 | - | 1 |
| `scripts/reports/plot_hybrid_score031.py` | 4.4 | - | 1 |
| `scripts/reports/plot_local_adaptation029.py` | 5.9 | - | 2 |
| `scripts/reports/plot_one_class_baselines034.py` | 3.3 | - | 1 |
| `scripts/reports/plot_referral_budget030.py` | 5.1 | - | 1 |
| `scripts/reports/prepare_mimic_priority_review.py` | 0.3 | - | 0 |
| `scripts/reports/profile_mimic_cpc_cluster_labels.py` | 0.3 | - | 2 |
| `scripts/reports/profile_ptb_mimic_cpc_diagnoses.py` | 0.4 | S | 4 |
| `scripts/reports/report_experiment.py` | 26.2 | H | 2 |
| `scripts/reports/report_xecg_adaptation.py` | 6.8 | H | 2 |
| `scripts/reports/report_xecg_probe016.py` | 10.0 | S | 1 |
| `scripts/reports/summarize_sampled_cohort.py` | 3.3 | S | 1 |
| `scripts/validation/audit_clean_cpc_inputs.py` | 3.4 | S | 1 |
| `scripts/validation/audit_code15_duplicate_annotations.py` | 8.8 | W | 1 |
| `scripts/validation/audit_code15_pilot.py` | 3.6 | W | 1 |
| `scripts/validation/audit_cpc_25k.py` | 3.1 | - | 2 |
| `scripts/validation/audit_cpc_25k_readout.py` | 3.8 | - | 3 |
| `scripts/validation/audit_cpc_scaling_readout.py` | 4.3 | - | 3 |
| `scripts/validation/audit_cpc_scaling_v3.py` | 4.1 | - | 0 |
| `scripts/validation/audit_dataset_quality.py` | 10.3 | W | 2 |
| `scripts/validation/audit_duplicate_labels.py` | 5.8 | W | 2 |
| `scripts/validation/audit_ecg_selection_bias.py` | 8.1 | W | 2 |
| `scripts/validation/audit_nlp25k.py` | 14.3 | - | 1 |
| `scripts/validation/audit_nlp25k_successors.py` | 23.4 | S | 1 |
| `scripts/validation/audit_nlp25k_successors_v2.py` | 23.6 | - | 2 |
| `scripts/validation/audit_public_pool_overlap.py` | 10.3 | H | 3 |
| `scripts/validation/prepare_clean_rerun.py` | 0.7 | S | 2 |
| `scripts/validation/smoke_training_dataset.py` | 6.2 | H | 2 |
| `scripts/validation/verify_code15_prepared.py` | 7.2 | W | 2 |
| `scripts/validation/verify_sampled_cpc_cache.py` | 4.3 | - | 1 |
| `scripts/validation/verify_sampled_training_dataset.py` | 5.9 | S | 2 |

### MERGE-CANDIDATE (3 files, 16.3 KiB)

| File | KiB | Pin | Links |
| --- | ---: | :---: | ---: |
| `docs/cpc-improvement-investigation.md` | 4.3 | - | 2 |
| `docs/reflection-papers-vs-results-2026-09-29.md` | 5.4 | - | 1 |
| `docs/vision-to-ecg-architecture-candidates.md` | 6.6 | W | 4 |

### DELETE-CANDIDATE (5 files, 78.0 KiB)

| File | KiB | Pin | Links |
| --- | ---: | :---: | ---: |
| `docs/experiment-013-mamba3-plan.md` | 3.9 | W | 3 |
| `docs/experiment001-results.pdf` | 44.5 | - | 3 |
| `scripts/coordination/adopt_nlp25k_cache_seal.py` | 3.5 | - | 0 |
| `scripts/coordination/handoff_priority_queue.py` | 23.1 | W | 2 |
| `scripts/data/build_nlp25k_selected_raw.py` | 3.1 | - | 0 |
