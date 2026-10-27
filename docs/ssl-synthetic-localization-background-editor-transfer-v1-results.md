# Real-background and held-out-editor localization results

21 October2026. The fixed B localizer trained on released Pulse2Pulse backgrounds passes all six prospective numerical-transfer checks on64real LUDB backgrounds with a held-out biphasic editor. Independent reconstruction passes with zero mismatches. This establishes some transfer of introduced-change localization; it does not establish disease localization.

The result is modest: mean direct AP is0.032520173, compared with observed-amplitude AP0.021991254, nuisance-map AP0.011031472 and donor-mask AP0.013382811. AP is a ranking score over sparse lead-time targets, not diagnostic accuracy, sensitivity or the fraction of diseases found. The amplitude-gain lower bound is only0.000416452.

## Primary result and uncertainty

The [prospective protocol](ssl-synthetic-localization-background-editor-transfer-v1.md) and configuration were committed at6d6c65a before scores. Reviewed sources and171-source/80-input admission were committed at eff2c3f before execution. Primary is fixed B on real backgrounds with the held-out editor, equally averaging focal/repeated modes, three fixed fitted states and64records. All three mean gains and their lower95%bounds must exceed zero: six checks, all pass. There is no minimum useful-effect or clinical threshold.

| Prespecified primary comparison | Mean [conditional95%interval] |
|---|---|
| Edited-map AP minus matched nuisance-map AP | 0.021488701 [0.011647258, 0.034289499] |
| Edited-map AP minus absolute observed-input AP | 0.010528918 [0.000416452, 0.023331458] |
| Edited-map AP minus fixed donor-mask AP | 0.019137361 [0.006767303, 0.033732480] |

All2000 draws use seed2026106601 and shared whole geometry slots, keeping both sources/families/modes and the three fixed fits together. Fits are not resampled. Intervals condition on this checkpoint, these64real records/64generator seeds and the declared geometry/donor assignment. The fixed nonself donor is(slot+1)%64 without overlap filtering; it is an alignment control, not a universal null or a new-patient population test.

## All arms, sources and editor families

All cells remain visible; no winning arm is selected. Every table value is a record/mode/fixed-fit mean. P/K/PK are exploratory here. The original paired-suppression study still FAILS its15-gate rule with14passing; this successor cannot rescue that failure.

| Background | Editor | Arm | Direct AP | Nuisance AP gain | Amplitude AP gain | Donor AP gain |
|---|---|---|---:|---:|---:|---:|
| generated | hann | B | 0.100072 | 0.087355 | 0.077629 | 0.088420 |
| generated | hann | P | 0.202605 | 0.188672 | 0.180162 | 0.188898 |
| generated | hann | K | 0.086129 | 0.072722 | 0.063686 | 0.074010 |
| generated | hann | PK | 0.141980 | 0.129268 | 0.119537 | 0.129499 |
| generated | biphasic | B | 0.073252 | 0.061125 | 0.051580 | 0.062340 |
| generated | biphasic | P | 0.108126 | 0.094840 | 0.086454 | 0.095807 |
| generated | biphasic | K | 0.051369 | 0.038558 | 0.029698 | 0.039908 |
| generated | biphasic | PK | 0.071662 | 0.059536 | 0.049991 | 0.060251 |
| real | hann | B | 0.043713 | 0.032116 | 0.020387 | 0.029796 |
| real | hann | P | 0.074506 | 0.061708 | 0.051179 | 0.059962 |
| real | hann | K | 0.048676 | 0.035723 | 0.025350 | 0.032858 |
| real | hann | PK | 0.066598 | 0.054460 | 0.043271 | 0.052401 |
| real | biphasic | B | 0.032520 | 0.021489 | 0.010529 | 0.019137 |
| real | biphasic | P | 0.042486 | 0.030314 | 0.020494 | 0.028215 |
| real | biphasic | K | 0.028624 | 0.016268 | 0.006632 | 0.013644 |
| real | biphasic | PK | 0.036108 | 0.024551 | 0.014116 | 0.022422 |

For fixed B, real-minus-generated direct AP is -0.056358749 [-0.080967149, -0.035016252] for Hann and -0.040732215 [-0.062220444, -0.021884428] for biphasic. Original-Hann minus held-out-biphasic AP is 0.026819749 [0.015753034, 0.038730537] on generated backgrounds and 0.011193214 [0.005915710, 0.017424649] on real backgrounds. Both source and editor changes are harder descriptively. All four arms show these point losses; secondary intervals are exploratory.

These source comparisons pair operator geometry, not patients or clinical case mix. The biphasic kernel matches nominal pre-addition L2 energy but differs in peak amplitude, shape and final threshold-mask occupancy. Neither comparison isolates a causal generator-quality effect. A positive interaction does not mean better absolute performance on real ECGs.

## Every primary fixed-fit and mode mean

| Fit | Held-out mode | Direct AP | Nuisance AP gain | Amplitude AP gain | Donor AP gain |
|---|---|---:|---:|---:|---:|
| 1 | focal | 0.027547879 | 0.024520843 | 0.021829773 | 0.018394326 |
| 1 | repeated | 0.040002624 | 0.021440250 | 0.001738222 | 0.021529927 |
| 2 | focal | 0.023467905 | 0.020556514 | 0.017749799 | 0.017291431 |
| 2 | repeated | 0.043889220 | 0.024755439 | 0.005624817 | 0.024112549 |
| 3 | focal | 0.020089364 | 0.017433486 | 0.014371258 | 0.013767681 |
| 3 | repeated | 0.040124043 | 0.020225672 | 0.001859640 | 0.019728255 |

All six fit/mode point gains are positive against every declared comparator. Repeated-mode amplitude gains are only0.001738,0.005625 and0.001860; focal gains are larger. No separate statistical success claim is made for a mode or a new training-seed population. All96 source/arm/fit/mode point summaries and352 aggregate columns remain in the saved summary.

Control hotspots remain. Real B sham model-record maxima exceed0.5 in0.921875 of cases versus0.588542 on generated shams; averaged top600probability is0.682760 versus0.400629. These probabilities are uncalibrated, and real backgrounds may contain existing findings. These are engineering descriptors, not clinical false-positive rates or medically normal/diseased labels. P has higher peaks; K/PK suppress them with localization tradeoffs.

## Inputs, mathematical truth and preserved restrictions

All64generated seeds2026105000--2026105063 are held out from the localizers' original80-background training schedules, but already spent evaluation evidence. Unknown generator-training membership prevents a global patient-independence or memorization claim. Real records are the first64ascending LUDB measurement IDs not divisible by5, ending79; all160real records were already development material. The remaining96measurement/40parser IDs are not scored here.

Generated native signals are standardized once. Real saved canonical signals are already standardized and are not scaled again. Original generated sham/nuisance/focal/repeated arrays and masks reproduce their parent exactly; real sham reproduces the selected saved inputs exactly. There is no post-edit normalization.

Both sources use identical index2000--2063 geometry draws. The original Hann editor and held-out Hann-windowed sinusoid share lead/sign/width/center/amplitude declarations; the latter was absent from the target-editor training recipe, not necessarily absent from ECG waveforms. Strict masks use final canonical float64 edited-minus-nuisance !=0; targets use absolute difference>0.01. Their truth concerns the added numerical edit only. Models receive observed ECGs alone. No pathology boundaries, healthy labels, physiological validity or disease condition is inferred.

No new training, optimizer updates, generator forwards, raw ECG reads, diagnosis comments, closed-test reads, downloads, dependency changes or external inference calls occurred. The other checkout and its outputs were not used. Clinical label definitions and closed final tests remain unchanged.

| Scientific requirement | Evidence after this study |
|---|---|
| Generate ECG-shaped signals | Prior locally verified released-generator bank remains available. |
| Physiologically plausible outputs | Prior descriptors only; still not clinically validated. |
| Control disease-related findings | No disease conditioning or disease-control result. |
| Know where an introduced change occurs | Exact final-difference support and independently checked learned numerical transfer. |
| Validate localization of real disease | Not established; no true regional disease masks. |

## CUDA execution, independent verification and identities

Two RTX3090s were verified, including device permissions, PyTorch access and an actual locked CUDA matrix operation. Main inference used CUDA0 under the shared nonblocking lock, deterministic PyTorch2.6.0+cu124, disabled TF32 and one numerical/inter-op CPU thread. No CPU inference fallback or concurrent uncoordinated GPU use occurred.

All12states replay their parent CUDA batch within1e-6, and profile/full first-eight maps match exactly. Checkpoints retain their named-tensor identities. Actual profile includes both sources, six variants/all12states, transfer, float32 persistence/flush/hash and CPU scoring. CPU536.447744s, wall684.080559s, GPU-lock108.278145s and2882652949byte forecasts all pass before full inference. Peak CUDA allocated/reserved63441408/75497472bytes.

| Actual stage | CPU seconds | Wall seconds | GPU-lock seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|
| Main35 | 211.446087924 | 262.527044703 | 51.149856553 | 1426898944 |
| Independent36 | 99.844710860 | 103.043663225 | 0 | 987041792 |

Main retained2759086983bytes before its terminal ledger; independent36retained3027743bytes. Every CPU/wall/GPU/RSS/storage gate passes. Main counted12old batch4replays,144profile batch8forwards and1152full batch8forwards. Independent checking reconstructs all1536model-background rows/6144positive rows/192profile rows,815829new numerical scalars,15032geometry scalars,4314predecessor scalars and980bindings, zero mismatches. It compares2000×352bootstrap entries at1e-12 arithmetic tolerance; largest difference1.3877787807814457e-16. All96fixed CPU model-record replays agree within1e-5; maximum difference8.344650268554688e-7.

The coordinator ran the separate checker authored by the verification agent; independent arithmetic/source review is distinct from independent physical observation of the original execution. The [independent report](ssl-synthetic-localization-background-editor-transfer-independent-audit-v1-results.md) and [scientific interpretation](ssl-synthetic-localization-background-editor-transfer-output-science-review-v1.md) describe those limits.

Unused audit admissionv1 failed metadata-only preflight because the imported randomized_saved_audit parameter helper was omitted from the source union. V2 prospectively adds that one identity, with175sources, before the single successful audit36. No numerical audit ran under v1, no score/rule/source expression was changed, and no fictitious failed-stage cost was added. A separate, pre-execution plotting-wrapper metadata-key correction is likewise preserved in development commits, not reclassified as an executed scientific stage.

Through36, actual current-machine totals are1071.223942690CPU/524.199369754GPU-lock seconds, counting stages and failures once. Earlier unavailable historical costs remain unverified; development, tests/builds and setup are separate.

- Original main admission: configs/ssl_synthetic_background_editor_transfer_admission_v1.json, SHA74c60cbef919924f46922b12e7c2639fa4822262fa669337f1f8eddd26b6fb66.
- Main summary: c3ea28105ef99659b0f8854fd32430169ecb48f456b02b8b709fec306da93ba7.
- Main ledger:8cf80790df2ef3ce3a5c0d2cf539b1e6311697a01a4394e672c7f6b1d819e304.
- Successful separate audit admission:configs/ssl_synthetic_background_editor_transfer_audit_admission_v2.json, SHA2f2cb1a713d097e074dab9312d17568b23e8092e021e9db524a58fc04a81b84f.
- Independent summary:e4f60657ab5c19007bdf91a53b4d27b58408d51dd732ebbc7b7ac3aa9c03afd6.
- Independent ledger:d64d6ee087dd5547ae02f02a3adc58e141be0d53b9f1d2474a4f2f01aaef1b12.

All12localizer checkpoint paths/file hashes are explicit in the original admission; separate tensor digests are in model_states.json and both GPU phase receipts. The unchanged released generator identity remains Hub deepsynthbody/deepfake_ecg revision262e0e67c0101cd4989d7c2f66f16d6d54756e70, weight SHAe646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827 and reviewed MIT architecture revisionf28e4b792e79c503b8b7e0912be9d304a3a7177b. It was not run or fine-tuned again here.

## Local artifacts and next question

Local root:outputs/ssl_synthetic_localization/background_editor_transfer_v1/. inputs.npy holds2×64×6×12×5000float32 observations; strict_support.npy/support.npy hold2×64×4×12×5000Boolean masks. Twelve full float32 map banks are under probabilities/. geometry.jsonl,record_ids.json,record_rows.jsonl,bootstrap.npy,model_states.json,profile/ and source_snapshot/ retain the full numerical design, values and identities.

all_source_family_arm_results.png shows every source/family/arm. The original profile/fixed_examples.png shows untargeted leadII for slot0 and must be interpreted as a control-lead example. Separately admitted, fixed true-target-lead illustrations are recorded below after their own terminal render receipt; they cannot alter these scores. All images/signals/maps remain local.

The more direct next hypothesis is a prospectively matched comparison of the small localizer trained on generated-only versus real+generated backgrounds, with identical objective/updates/geometry and verified real TRAIN/EVAL patient separation. Generated arrays/seeds can be excluded from localizer fitting, while generator-cohort overlap remains unknown. Keep an editor family outside fitting, nuisance/donor controls, all fixed fits and a fresh actual-path GPU training profile. Do not train on current LUDB evaluation records or turn the real-background performance gap into proof that GAN fine-tuning will help.

Pulse2Pulse fine-tuning remains a possible later background-distribution intervention. It would not create disease masks or guarantee improved localization. This result motivates testing background composition in localization training first. No new training is scheduled by this report. Changes are committed locally as Janet16180; the user explicitly requests no push to avoid GitHub runner consumption. PR67 remains draft and unmerged.


## Fixed target-lead illustrations

Separately committed plotting protocol6917240 and admission164d34c precede render37. Two fixed-slot, all-editor/all-arm PNGs pass source/input before-and-after identity checks and dimension/hash checks;2750×1540each. They show the original metadata-declared edited lead and full target/map grids, with common probability scales and no favorable-example search.

- Local generated figure:outputs/ssl_synthetic_localization/background_editor_transfer_figures_v1/generated_slot0_all_editors_and_maps.png, SHA272151947be40d6905eabc2567010cae78a899fd137dfb53e47ed2236fe04918.
- Local real figure:outputs/ssl_synthetic_localization/background_editor_transfer_figures_v1/real_slot0_all_editors_and_maps.png, SHA5088a6b643ba489c59fbf15bc638a27f5cc2e73d518ed0f9f26245c7e80ff636.

Render37 uses5.552373506CPU/7.868500264wall seconds, zeroGPU/new scores/forwards,593379328byte RSS and2515275retained bytes before ledger; every cap passes. Summaryc4d2b4cd77bbdabbebfd8b7dfea261a1bb5cd595abb6aefa4af3509006744923, ledger73758dc5c2f8b20543a63549d61fc6ce669286cd988c2dd135c6453e1a024d4f. Cumulative37 is1076.776316196CPU/524.199369754GPU-lock seconds. Pixel/clinical interpretation is not an independent validation claim.

Final local developer QA: full-source CPU suite before later audit/plot additions passed2124tests/6skips/2existingfixturewarnings(115.61s). After those additions,277targeted scientific/import tests passed with zero skips/warnings(7.54s); repository-wideRuff and offline sdist→wheel builds pass. All13newPythonfiles and6newpackage modules match source bytes; no private artifacts packaged. Source files remain unchanged. V2 verification receipt SHAba3a4a42baa3662202549ee714a92d2a3298d9e8fcbfcda74b08cb8a3499041a. No GitHub runner was used.
