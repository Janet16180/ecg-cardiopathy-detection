# Synthetic localization handoff, 22 October 2026

The matched generated-only versus mixed real/generated localizer experiment is complete and independently verified. Its scientific advancement rule FAILS: 9 of 14 checks pass. This is a useful negative result about one training recipe, not completion of real disease localization. No scientific job remains running in this stream.

Read the [executed main report](ssl-synthetic-localization-mixed-background-v1-results.md), [independent recovery report](ssl-synthetic-localization-mixed-background-independent-audit-v2-results.md), [scientific review](ssl-synthetic-localization-mixed-background-output-science-review-v1.md), [preserved failed audit](ssl-synthetic-localization-mixed-background-independent-audit-v1-failure-results.md) and both experiment queues. The previous [21 October transfer handoff](ssl-synthetic-localization-local-transfer-handoff-2026-10-21.md) preserves the earlier positive numerical-transfer finding and its limits.

## Result and interpretation

Primary real-background/held-out-editor AP changes from 0.032520173 to 0.029844057: mixed minus generated-only -0.002676115, conditional 95% interval [-0.006867945, 0.001306168]. This fails the required +0.01 point gain and positive lower bound. Real sham/nuisance top600 output decreases about 0.156, while generated-editor preservation and the mixed model's amplitude-comparator lower bound fail. Quieter background output does not establish better localization or fewer clinical false positives. Existing findings in real training backgrounds are not automatically medically normal.

The study fixes architecture, objective, three initialization/schedule pairs, 400 updates per fit, geometry and presentation budget. The only intervention is replacing 40 of 80 generated backgrounds with 40 fixed PTB-XL training patients. All three original generated-only fits reproduce every loss/component row, final tensor and complete Adam state/options exactly before mixed evaluation. Training uses only the original Hann editor; evaluation reuses 64 generated and 64 LUDB development records and both original/held-out editors. No GAN fine-tuning, new generator forwards, changed clinical labels or closed tests occur.

Independent40 checks all 768 model-background rows, 51,144 metric/report scalars, all 2,000×348 bootstrap entries and 24 CPU model replays with zero mismatches. Maximum replay error is 3.8743019104003906e-7 under the frozen 1e-5 tolerance. The failed39 checker caller omitted a batch dimension; its receipt, sources and cost remain. Recovery adds only the singleton dimension and returned-record extraction, with unchanged numerical functions/rules, and checks both audit directories separately from the immutable main catalog.

This does not establish physiological validity, disease control, true disease masks or clinical localization. Exact introduced-change masks and LUDB wave landmarks retain their distinct meanings. The earlier paired-suppression 14/15 scientific failure remains a failure.

## Local state and accounting

All local payloads are under `/home/janetrivera/baseline/outputs/ssl_synthetic_localization/`:

- `mixed_background_v1/`: training banks, six fit/optimizer sets, exact baseline replays, actual training/resource profile, three mixed probability banks, all rows/draws and aggregate figure.
- `mixed_background_v1/independent_audit_v1/`: preserved failed39 and source snapshots.
- `mixed_background_v1/independent_audit_v2/`: successful independent40 and source snapshots.
- `mixed_background_cohort_v1/`: local patient manifest, independent metadata check and setup runtime.
- `mixed_background_setup_v1/`: actual locked CUDA device-access check.
- `mixed_background_repository_validation_v1/` and later validation versions: developer checks/builds; separate from scientific accounting.

Main38 admission is `configs/ssl_synthetic_mixed_background_admission_v1.json`, SHA-256 `7a0f54e24ba88af55f9927528381437868e84c07f37334d3fad6a86f444a1a24`, with 193 source and 47 input bindings. Original main summary is `c5c8c83de8e07225b4763a61a5f3e24727076c7daa53a120b5c6e62049b29dfa`; its ledger is `dcc6ee167ae3f8267a8f2e723285253d2230d580df8e5df81359c86cd3301c04`. Never edit admitted sources or overwrite these outputs.

Recovery40 admission is `configs/ssl_synthetic_mixed_background_audit_admission_v2.json`, SHA-256 `bd3b4c81a67adb7c82bb4221fda4d6cd63758776aa449e827f0296925e2673e4`, with 205 source bindings. Its summary is `4f2f5e4943c454f7aa45572cfcafc512781702a53fd25a1dc23a011ccbba483a`. The latest ledger is `mixed_background_v1/independent_audit_v2/prospective_ledger.json`, SHA-256 `0f7def28c1f5c5135e6096be38ca0d57826d5ab4d1d3e90acc7953a573a90feb`:

- Actual current-machine stages: 40.
- Cumulative scientific/presentation CPU seconds: 1274.365159998.
- Cumulative GPU-lock seconds: 564.4522484582849.
- Failed39 is included exactly once. Original unavailable historical costs remain unverified.
- Every main/recovery CPU, wall, GPU, RSS and output cap passes. Setup, coding, tests/builds and metadata preparation are separate.

The GPU run uses one RTX 3090 under the shared nonblocking lock, deterministic PyTorch 2.6.0+cu124, TF32 disabled and one CPU thread. Both physical RTX 3090s were verified; restricted-shell device visibility is not evidence of hardware unavailability. Do not bypass the shared lock or infer a full-training admission from inference smoke checks.

Developer QA passes: 2,151 full-suite CPU tests, six skips and two existing fixture warnings; final recovery/runtime/import checks pass 270 tests without warnings. Ruff and offline sdist-to-wheel builds pass, with all 18 new Python files / nine package modules included exactly and private payloads excluded. All 193 main and 205 recovery source bindings match current files and saved snapshots. Final complete validation receipt is `mixed_background_repository_validation_v3/final_complete_verification.json`, SHA-256 `6529def94c7661dfc913f8b15ac834180cabb477d6a5f70d74f6dd912b021c19`. Initial environment-dependent test failures and the source-unchanged isolated-fixture rerun remain documented; no scientific evidence is inferred from developer tests.

## Ownership and continuation

The user's latest instruction is **no push and no GitHub runners**. Commits remain local as Janet16180 / clarajanet16180@outlook.com. PR67 stays draft, against main, and unmerged. A read-only refresh confirms `origin/main` remains `41871e35c3d04ca631021039f973a4bbccf2d393`, already included. The advisor website is not republished by this work.

Coordinator source worktree: `/tmp/ecg-mixed-background-run-v1`, branch `research/ssl-synthetic-mixed-background`, shared writable Git metadata `/tmp/ecg-synthetic-autonomous.git`. Delivery targets the existing baseline branch `research/ssl-synthetic-localization` by local fast-forward; never push as part of that operation. Preserve the unrelated untracked `instruction.md`, SHA-256 `f3ca9162513dad1bbada237c19441cea2381bb8a17c1c7a2055d42dc24c52328`.

The new cohort is a bounded supervised numerical-target cohort, not v4 pretraining membership. It uses the exact official PTB metadata bytes and original partition hashing, quarantines any patient with a fold 8/9/10 record, and deterministically takes 40 of 10,550 eligible TRAIN patients without replacement. No clinical labels or demographic columns are interpreted. Unknown cross-source biological identity, unavailable old waveform catalogs and unknown GAN-training membership stay unknown.

Raw data targets are permitted only under `/home/janetrivera/ecg-cardiopathy-detection/data/raw/`, through read-only-use aliases under baseline `data/raw/`. Verify links beside tracked DVC pointers. Never modify the other checkout, its Git/DVC state or its outputs, and never stage raw-data aliases. Historical localization outputs absent from baseline remain absent. MIMIC-IV-ECG/EchoNext records and reports stay local; clinical labels and closed tests remain protected.

The next ranked synthetic-localization item is public expert annotation qualification for a real-abnormality endpoint. Start with a separately committed, bounded feasibility protocol covering annotation semantics, source identities, patient/evaluation roles, controls and usable lead/time resolution. No new model score, raw annotation scan, download, GAN tuning or training is scheduled by this handoff. Do not relabel wave landmarks, whole-record diagnoses or sparse option-conditioned answers as dense pathology masks, and do not substitute another screening/classification benchmark. Existing public evidence is the route forward; clinician recruitment is not a prerequisite.
