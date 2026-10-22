# Mixed real/generated localization training results

22 October 2026. Adding real ECG backgrounds to this small localizer's training did not improve its ability to locate unfamiliar added changes on real ECGs. The prespecified scientific rule fails: 9 of 14 checks pass. Technical execution and independent numerical reconstruction pass with zero mismatches. This report is written by the coordinator who executed the experiment, from the saved outputs.

The useful finding is that reducing background responses and improving localization are separate outcomes. Mixed training substantially reduces strong responses on real sham and nuisance controls, while the primary localization comparison does not improve. These controls concern an added numerical intervention. They do not identify medically normal ECGs or measure clinical false positives.

## Primary result

The [protocol](ssl-synthetic-localization-mixed-background-v1.md) and configuration were committed at `30300d2` before selecting the new training cohort or computing scores. The reviewed executable admission was committed at `d52dfa2` before execution. Primary is mixed minus generated-only direct average precision (AP) on real LUDB backgrounds with the held-out biphasic editor, equally averaging focal/repeated modes and three fixed fits.

| Primary endpoint | Result |
|---|---:|
| Generated-only direct AP | 0.032520173 |
| Mixed direct AP | 0.029844057 |
| Paired AP difference | -0.002676115 |
| Conditional 95% interval | [-0.006867945, 0.001306168] |
| Required mean gain | At least +0.010000000 |
| Required lower bound | Greater than zero |

Both primary checks fail. The interval includes zero and its upper bound is below the planned +0.01 improvement. This is evidence against the planned useful gain in this fixed comparison, without proving population equivalence, universal harm, or that every mixture would fail. AP ranks sparse lead-time target cells; it is not disease-detection accuracy or a percentage of diseases found.

All 2,000 draws use seed `2026106701` and shared whole geometry slots. Sources, editors, modes and the three paired fits stay together; fits are not resampled. No draws were skipped. These are conditional development intervals for 64 fixed slots and three fixed training runs, without a new-patient or training-population confidence claim. The original study used a different bootstrap seed; identical predecessor point estimates can consequently have slightly different intervals here.

## Controls and all source/editor comparisons

| Evaluation backgrounds | Editor | Generated-only AP | Mixed AP | Difference [conditional 95% interval] |
|---|---|---:|---:|---|
| Generated | Original Hann | 0.100072136 | 0.079718800 | -0.020353336 [-0.031713126, -0.010134269] |
| Generated | Held-out biphasic | 0.073252387 | 0.053169660 | -0.020082728 [-0.033130318, -0.008829110] |
| Real LUDB | Original Hann | 0.043713387 | 0.046452544 | +0.002739157 [-0.004210822, 0.009068879] |
| Real LUDB | Held-out biphasic | 0.032520173 | 0.029844057 | -0.002676115 [-0.006867945, 0.001306168] |

The generated-background preservation checks both fail their lower-bound requirement of greater than -0.01. Real held-out focal and repeated preservation checks pass separately, with difference lower bounds -0.007691521 and -0.009755507. These mode checks do not rescue the failed primary improvement rule.

| Real control, mean of highest 600 output cells | Generated-only | Mixed | Difference [conditional 95% interval] |
|---|---:|---:|---|
| Sham | 0.682759560 | 0.527175402 | -0.155584158 [-0.174723485, -0.134648453] |
| Nuisance | 0.684027060 | 0.527393804 | -0.156633257 [-0.175761733, -0.136014129] |

Both real-control preservation checks pass their upper-bound requirement of less than +0.02. The decreases are clear engineering changes, but their coexistence with failed primary improvement prevents describing mixed training as better localization. Existing disease findings may occur in these real backgrounds; zero targets mean only that no target intervention was added. The experiment does not determine whether suppressing an existing feature would be clinically appropriate.

Mixed training retains modest response to held-out added changes:

| Mixed model, real/held-out pooled endpoint | Mean [conditional 95% interval] |
|---|---|
| Edited AP minus nuisance-map AP | 0.018208823 [0.009460095, 0.029552968] |
| Edited AP minus absolute observed-input AP | 0.007852803 [-0.001569587, 0.019312250] |
| Edited AP minus fixed donor-mask AP | 0.015309703 [0.002938153, 0.029222653] |

Response and donor point/lower-bound checks pass. The amplitude point check passes but its lower bound fails. Together with the two primary and two generated-preservation failures, this gives five failures and nine passes. No arm, fit, mode or endpoint is selected after seeing scores. All 48 fit/mode point summaries and 348 named aggregate endpoints remain in `summary.json`; all 768 model-background rows and their 3,072 positive cases remain in `record_rows.jsonl`.

The saved `all_source_family_and_control_contrasts.png` displays every source/editor AP contrast and both prespecified real-control contrasts, including negative effects. It is a numerical comparison figure, without a clinical waveform-quality verdict.

## Matched training and cohort integrity

Both arms use the unchanged 35,660-parameter observed-only localizer, weighted BCE with positive weight 20, batch size eight, 400 updates, and the original initialization/schedule pairs. Initialization seeds are `2026103801`, `2026103802`, `2026103803`; schedule seeds are `2026103901`, `2026103902`, `2026103903`. Each fit receives 3,200 presentations. Generated-only uses the original 80 generated backgrounds; mixed keeps generated slots 0–39 and substitutes 40 fixed PTB-XL training patients in slots 40–79 with the identical slot geometry and editor. Only the original Hann target editor enters training. The Pulse2Pulse GAN is not fine-tuned or run again in this stage.

Before mixed evaluation, all three generated-only fits exactly reproduce every original 400-step loss/component row, final named tensor and complete Adam state/options. This preserves the actual predecessor behavior rather than comparing a new implementation against documentation. All three mixed subset/full prediction banks match exactly and fitted states remain unchanged by inference.

The new real cohort is explicitly a bounded supervised numerical-target PTB cohort, not a reconstructed v3/v4 pretraining cohort. Missing historical cohort artifacts remain missing. The committed metadata-only builder selects 40 patients from 10,550 eligible original-TRAIN patients, excluding any patient with a record in folds 8, 9 or 10. It selects one lowest-ID record per patient, then uses the fixed hashed order with no quality-driven replacement. No clinical labels or demographic fields are interpreted. An independent metadata-only reconstruction passes 141 checks with zero mismatches before waveform decoding.

The main run decodes exactly 40 selected raw records, checks their official byte hashes before/after decoding, validates structural headers, physical units and lead identities, and constructs canonical inputs using the original one-time robust scaling. No record is replaced. The evaluation reuses the saved 64 generated and 64 LUDB development backgrounds, all six variants and the fixed nonself donor assignment `(slot + 1) % 64`. No current LUDB evaluation record enters fitting. Metadata/source separation does not establish cross-hospital biological identity, historical waveform deduplication, or unknown GAN-training membership.

Only permitted PTB raw paths are read through local aliases beside tracked DVC pointers. The other checkout's Git state, derived files and outputs are not used. No new LUDB annotations, clinical labels, closed Challenge/EchoNext tests, SPH training data, downloads, remote inference or dependency changes are used.

## GPU execution and resource accounting

Actual host checks verify two RTX 3090s and a small locked PyTorch CUDA operation. Main execution uses CUDA 0 under the shared nonblocking lock, PyTorch 2.6.0+cu124, deterministic operations, disabled TF32, and one numerical/intra/inter-op thread. Restricted-shell visibility is not treated as GPU unavailability.

The actual-path profile performs ten disposable updates across both training banks, reloads persisted step-one model/Adam files and exactly replays steps two/three. It also performs 36 frozen-baseline inference batches and saved-map scoring. The prospective forecast, including persistence, hashing, CPU scoring and reserves, is 290.177866 CPU / 426.130727 wall / 176.450093 GPU-lock seconds and 900,554,693 output bytes; all admission gates pass before scientific training.

The scientific run performs 1,200 generated-replay plus 1,200 mixed updates, followed by 36 mixed-subset and 288 mixed-full inference batches. CUDA peak allocation/reservation is 103,817,216 / 125,829,120 bytes. Main receipt costs are 96.120735143 CPU / 128.290750127 wall / 40.252878704 GPU-lock seconds, peak RSS 1,719,930,880 bytes. Terminal ledger costs, including guarded receipt bookkeeping, are 96.135431559 CPU / 128.338468894 wall seconds. Retained output before terminal ledger is 820,392,950 bytes. Every CPU/wall/GPU/RSS/storage cap passes.

This is actual current-machine stage 38, with cumulative 1,172.911747755 CPU / 564.452248458 GPU-lock seconds, counting earlier failed stages once. Historical unavailable costs remain unverified. Metadata preparation, device setup, developer tests and builds are separate; they are not hidden scientific stages.

## Identities, saved artifacts and verification status

Local root: `outputs/ssl_synthetic_localization/mixed_background_v1/`. Real identifiers, individual signals/maps, model/optimizer files and receipts stay local and are not Git artifacts.

- Main admission: `configs/ssl_synthetic_mixed_background_admission_v1.json`, SHA-256 `7a0f54e24ba88af55f9927528381437868e84c07f37334d3fad6a86f444a1a24`; 193 sources and 47 inputs.
- Main summary: `c5c8c83de8e07225b4763a61a5f3e24727076c7daa53a120b5c6e62049b29dfa`.
- Cohort selection: `37ed98d9d1da0f4f521a52f947fbd597808a94e8851038c4bf9b858219a19104`.
- Observed training bank: `dd911f791be63a380bf2b76f3fa5965112054d7630c3ae1dc254fea3a303c970`.
- Mixed fit 1/2/3 full-map file hashes: `8c26860779571ce58d42407277c2b5ed264fef390d6e67f968017dc0297c142a`, `6c7746f9b18145157fc449f2bc0d17ef60e56779de5f8479ad3d96c64c241c08`, `3a448c317cb780da93872d6d0d94c071c2312cd97b622cf795128a9f4d0933eb`.
- Three mixed final tensor-state identities: `09c686603cb95c8cbbd1bbc60b6df86a25cba3086ebf316c45cac07de4f92922`, `3a61f0511b9777864619557c2e0e65aa5b77a273e311e4fb71f388fedf70ead5`, `57c34415e932990546c64a36cb39d9a4ca38ad69a6a072ed017fbcdf3b136d93`.
- Bootstrap matrix: `00251f51716406a1342aa0801fa384eb8a6e8aeec854abfca0094e935ea6cc98`.
- Figure: `d9bfad8e633928f27891da0d1ab47914c8c7c71e0137a5aaab6a18f29857103c`.

The unchanged released generator remains Pulse2Pulse, Hub `deepsynthbody/deepfake_ecg` revision `262e0e67c0101cd4989d7c2f66f16d6d54756e70`, weight SHA-256 `e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827` and reviewed MIT architecture revision `f28e4b792e79c503b8b7e0912be9d304a3a7177b`. Its prior generated bank is reused. The checkpoint's availability and engineering qualification do not establish clinical realism or controllable disease.

Independent attempt 39 stopped at `Require float32 batched actual native-rate model inputs`. The saved failure and sources remain at `independent_audit_v1/`; no main output is altered. It consumed 35.699429857 CPU / 38.490246259 wall seconds, zero GPU, peak RSS 980,168,704 bytes, with every resource gate passing. Cumulative actual39 is 1208.611177612 CPU / 564.452248458 GPU-lock seconds. The separately committed recovery is described below; the original failed receipt remains unchanged. Independent arithmetic/source authorship is distinct from independent physical observation of the original run.

## Successful independent recovery

Recovery protocol/config were committed at `bdf8f21`, successor source at `d402c6f`, and complete separate admission at `36fec77` before actual40. The correction adds the required singleton batch dimension and removes it from the returned map. An artificial test executes the real functional convolutions; an AST check confirms that the numerical dispatch changes only its replay callee. Main training, observations, masks, metrics, bootstrap draws, thresholds and verdict stay unchanged. Both audit directories are excluded from the original main catalog while failed39's entire source snapshot and cost lineage are verified separately. Complete source-bound file-format/header/history preflight passes before execution.

Independent40 reconstructs all 768 model-background rows / 3,072 positive cases, 51,144 metric/report scalars, 6,960 training-metadata scalars, 5,930 predecessor scalars and 3,072 profile scalars. It performs 1,336 byte-binding checks, reconstructs all 40 physical training records and exact target banks, checks six complete model/Adam states and three exact generated-baseline closures, and verifies all 2,000 × 348 bootstrap entries. Zero mismatches occur. Maximum bootstrap difference is 6.938893903907228e-17, below 1e-12. All 24 fixed mixed-state CPU replays agree within 1e-5; maximum difference is 3.8743019104003906e-7. The checker independently reproduces the scientific failure (9/14), rather than interpreting its own technical PASS as scientific success.

Independent40 uses 65.753982386 CPU / 71.582703368 wall seconds, zero GPU/training updates, peak RSS 991,080,448 bytes and 3,265,790 retained bytes before its ledger. Every resource cap passes. It starts directly from actual failed39 and counts that failed cost exactly once. Cumulative actual40 is 1274.365159998 CPU / 564.452248458 GPU-lock seconds. Historical costs remain unknown; setup and developer validation remain separate.

- Separate recovery admission: `configs/ssl_synthetic_mixed_background_audit_admission_v2.json`, SHA-256 `bd3b4c81a67adb7c82bb4221fda4d6cd63758776aa449e827f0296925e2673e4`, with 205 source bindings.
- Successful audit summary: `4f2f5e4943c454f7aa45572cfcafc512781702a53fd25a1dc23a011ccbba483a`.
- Successful audit ledger: `0f7def28c1f5c5135e6096be38ca0d57826d5ab4d1d3e90acc7953a573a90feb`.
- Preserved failed39 summary: `f5436ad157a203dfa310ef60191ec0f6130147fa53e8016830c95e52625df434`.
- Preserved failed39 ledger: `309ee61bd827362b26f247c60082f9f9ff375efaa656c8f6136c21248b6bd86a`.

Read the [independent recovery report](ssl-synthetic-localization-mixed-background-independent-audit-v2-results.md), [failed-attempt report](ssl-synthetic-localization-mixed-background-independent-audit-v1-failure-results.md) and [scientific interpretation](ssl-synthetic-localization-mixed-background-output-science-review-v1.md). The coordinator executes the separately authored checker on saved artifacts; independent physical observation of the original run is not claimed. Figure checks cover bytes, PNG signature and dimensions, not clinical interpretation or pixel rerendering.

## What this establishes and the next question

| Research requirement | Current evidence |
|---|---|
| Generate ECG-shaped signals | Released-generator signals are saved and independently verified in the earlier bank. |
| Physiologically plausible signals | Engineering descriptors and real-signal comparisons exist; clinical plausibility remains unvalidated. |
| Control disease findings | No disease-conditioning result. |
| Know the location of an introduced change | Exact numerical targets and modest learned transfer exist; mixed training fails this improvement rule. |
| Localize real disease | Not established; numerical edits and LUDB wave landmarks are not pathology masks. |

This bounded result does not justify scaling this mixture or claiming that GAN fine-tuning will solve the limitation. The next useful decision is whether existing public expert annotations provide a defensible real-abnormality localization endpoint with patient separation and appropriate controls. Any new annotation qualification, training or evaluation needs a separate committed protocol. It must preserve the missing-onset LUDB finding, current clinical labels, spent-development status and closed tests. Improving the generator could later be a controlled intervention, but cannot itself supply verified pathology boundaries.

The earlier paired-suppression 14/15 scientific failure and all retained failed attempts remain unchanged. PR 67 stays draft and unmerged. The user explicitly requests no push or GitHub runner use; changes remain local commits as Janet16180.

## Local developer validation

The complete CPU suite passes 2,151 tests, with six skips and two existing all-NaN fixture warnings (105.72 seconds). Two initial environment-dependent failures are preserved in the validation logs: one old test counted the entire shared output cache against its fixture budget, and another rejected the shared output symlink before its expected missing-file branch. The isolated validation worktree's output alias was replaced with its original tracked fixture directory, leaving the shared output target untouched; the complete suite then passed without source changes.

After the original independent runtime was added, 265 targeted/import tests pass without warnings (6.75 seconds). Its retained validation receipt is `outputs/ssl_synthetic_localization/mixed_background_repository_validation_v2/final_verification.json`, SHA-256 `9d637e7b525aaa303e778835d4e4390ae1dc7a730feb7da4782fa64eb9a67c85`.

Final recovery/runtime/import validation passes 270 tests without warnings (8.88 seconds). Repository Ruff and offline source-distribution-to-wheel builds pass. All 18 new Python files match source-distribution bytes, and all nine new package modules match wheel bytes; private data, outputs and weights are excluded. All 193 main source bindings match current source, main receipt and saved snapshots. All 205 recovery source bindings match current source and saved snapshots, and all 206 audit-summary bindings including its admission match. Final receipt: `outputs/ssl_synthetic_localization/mixed_background_repository_validation_v3/final_complete_verification.json`, SHA-256 `6529def94c7661dfc913f8b15ac834180cabb477d6a5f70d74f6dd912b021c19`. These are developer checks, not additional scientific evidence or GPU stages. No full-suite rerun was needed after the three recovery files; no dependency, raw-data or admitted-source changes occurred.
