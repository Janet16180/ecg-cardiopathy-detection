# Experiment queue

**Updated:** 1 October 2026. This is the persistent project queue. The companion [JSON catalog](experiment-queue.json) records authorization, dependencies, protocols and next actions, including experiments that still need implementation. Read these two files first after a context reset; `AGENTS.md` points future sessions here.

**1 October, Experiments 046-050 completed, and a CPU timing measured:**
[Experiment 046](experiment-046-pipeline-v4-results.md) adopted [pipeline v4](pipeline-v4.md), v3 with the
unfitted ensemble readout of 045. It catches more athlete-criteria abnormal ECGs at 5% with 200 local
normals (composite +0.0121 [+0.0082, +0.0161] over v3), but it refers more "other" ECGs, such as sinus
bradycardia (27.5% against 24.9%). Three explanation experiments, on saved scores, explain a referred ECG with
042's beat-wave map `U_B` (rhythm) or the attention contributions (morphology).
[Experiment 047](experiment-047-explanation-rule-results.md) switched on `U_B` being red. It is negative by
its rule: it lost premature-beat hits (-0.136 [-0.232, -0.057]).
[Experiment 048](experiment-048-pvc-switch-results.md) switched on the xECG PVC head instead. It keeps every
`U_B` premature-beat hit and gains infarct-lead information (anterior contrast +0.189 [0.067, 0.307]). It is
the recommended explanation rule, although it sends half of the referred anterior infarcts to `U_B`.
[Experiment 049](experiment-049-focal-switch-results.md) is negative: a label-free focal-beat switch reached
only 24 of 49 referred PVC ECGs, because 12 NORM-only normals have one extreme beat. It confirmed the PVC
switch with the `combined_50` referral as the best configuration so far (93% of PVC ECGs explained on the
premature beat). [Experiment 050](experiment-050-attention-finding-heads-results.md) is negative: attention
finding heads on ECG-JEPA tokens are below the xECG PVC head at SPH (-0.0068 [-0.0128, -0.0015]), and the
PVC token map does not keep localization by the margin. The [CPU timing](inference-timing-cpu.md) of v4 plus
the explanation is about 0.33 s per ECG with 4 threads (0.83 s with one), with a 2.3 GiB peak, and the CPU
outputs equal the GPU path's. Outputs: `outputs/experiment046_pipeline_v4_v1/` to
`outputs/experiment050_attention_findings_v1/`, `outputs/inference_timing_cpu_v1/`.

**1 October, Experiments 043-045 completed; 046 running:** the user asked whether a neural network at the
end, an attention head or the tutor's CNN + transformer would do better, and asked for the work to continue
overnight. [Experiment 043](experiment-043-ann-heads-results.md) ran three stages against pipeline v2's
readout R. The logistic readout on concatenated xECG + JEPA features beats R (SPH +0.0017 [+0.0008, +0.0026],
full development +0.0041). MLP heads and the attention head on JEPA tokens match R (attention SPH +0.0017
[-0.0003, +0.0039]); the tutor's CNN + transformer from scratch is below R (SPH -0.0097 [-0.0119, -0.0075]).
No map improved on 042's `U_B`. [Experiment 044](experiment-044-pipeline-v3-results.md) adopted the
concatenated readout as [pipeline v3](pipeline-v3.md): composite sensitivity at 5% with 200 local normals
0.797 against 0.789 (+0.0080 [+0.0040, +0.0120]), at a higher normal referral rate (5.74% against 5.45%).
[Experiment 045](experiment-045-two-layer-map-results.md) found that a two-layer map (`U_B`, then
`attention_jepa`) loses premature-beat localization (-0.309 [-0.418, -0.202]), and that the mean-logit
ensemble of `logistic_concat` and `attention_jepa` beats pipeline v3's readout (SPH +0.0038 [0.0028, 0.0049],
full +0.0058). Experiment 046, pipeline v4 with the ensemble readout, is running in a separate worktree.
Outputs: `outputs/experiment043_ann_heads_v1/`, `outputs/experiment044_pipeline_v3_v1/`,
`outputs/experiment045_two_layer_map_v1/`.

**30 September, Experiment 042 completed:** the user asked for both per-lead approaches.
[Experiment 042](experiment-042-lead-wave-maps-results.md) compared ECG-JEPA patch tokens (the 041b fallback)
with beat-aligned P, QRS, ST and T pieces per lead. The beat-aligned map improves on 041 by its prespecified
rule: its top unit was on the premature beat in 93% of PVC ECGs (chance 20%), and it put red on 7.7% of benign
variants against 27%. ECG-JEPA did not improve on 041, and no map passed the infarct lead test. Notebook
`notebooks/12-jr-beat-wave-maps.ipynb` shows the beat-aligned map. Outputs:
`outputs/experiment042_lead_wave_maps_v1/`.

**30 September, Experiment 041 completed:** after the closeout, the user asked for the tutor's localization
idea and agreed its scope. [Experiment 041](experiment-041-fragment-localization-results.md) scored each
250 ms xECG section on PTB-XL development data. The unsupervised per-section distance put its top section on
a premature beat 82% of the time against 15% by chance (+0.669 [0.582, 0.758]), but as a worst-section
screen it reached AUROC 0.777 against 0.923 for the whole-ECG distance. The label-guided map reached 0.926
and still localized premature beats (77%). Neither map marked benign rhythm variants less (27% each). The
prespecified ECG-JEPA fallback (041b) is triggered and needs its own protocol; follow-ups are in the
backlog. Outputs: `outputs/experiment041_fragment_localization_v1/`.

**30 September day closed:** The user chose to close out today's completed work and merge it
into `main`. Experiment 039 finished all 36 fits and 18 audits; Experiment 040 completed its
resource profiles and six-package runtime diagnosis. Its 18 full Transformer/SimDINO fits remain
unrun after the original resource stop. No training, localization/clustering study or closed-test
stage is scheduled, and no process restarts automatically.

The completed code, protocols and reports are delivered through
[PR #58](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/58),
[PR #59](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/59) and
[PR #60](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/60), all targeting `main`.
GitHub records the final merge state. Original source identities, measurements and closed ledgers
remain unchanged; merge commits do not replace experimental provenance. Future full training or
fragment localization needs a prospective protocol and agreed scope.

**30 September runtime diagnosis completed:** The user-requested [runtime diagnosis](experiment-040-runtime-diagnostic-results.md)
completed six fresh 200-update probes in 350.77 seconds for CPC, SimDINO and their hybrid with
GRU/mLSTM. The independent audit passed all six packages, twelve checkpoints, exact recovery,
repeat feature hashes and unchanged parent/source identities. All 18 full fits remain unexecuted;
no development performance was scored. The new process has stopped.

The fixed full family projects 6,275.24 seconds of optimizer work (1.743 hours), or 9,500.79 seconds
(2.639 hours) including remaining preparation, profiles, checks, readouts and reporting. The
conservative scenario is 13,869.37 seconds (3.853 hours), with the correction reserve separate.
These are extrapolations from 200 updates and historical CPU proxies. Repeated nested source
validation measured 40.07 seconds per stage gate, projecting another 1,081.97 seconds. The analysis
also removes overlapping historical/new extraction and checkpoint reload costs while retaining
missing setup/validation work. It reproduces the original rejected forecast exactly.

Evidence lives in `outputs/experiment040_runtime_diagnostic/`; its ledger is separate from closed
039/040 accounting. The [runtime protocol](experiment-040-runtime-diagnostic.md), source `019c9d6`
and original receipts remain frozen. The conservative spent-plus-future scenario still exceeds
the original eight-hour shared ceiling after diagnostic work; no original admission was overridden.
Forward accounting/validation work enters the ranked backlog. [PR #60](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/60) targets `main`;
the user authorized its merge as part of today's closeout.

**30 September Transformer/SimDINO resource stop:** [Experiment 040](experiment-040-cpc-simdino-results.md)
passed all six real GPU profiles and exact normal/final-16-record checkpoint recovery. The all-or-none
time gate rejected the fixed 18-fit suite: 29,295.03 projected combined seconds versus the shared
28,800-second ceiling, including diagnostic and reporting reserves. All original outcomes remain
under `outputs/experiment040_cpc_simdino/`. No full fit, development score or primary-family decision
was produced; this resource stop supplies no performance evidence about the objectives.

Protocol `ac18dac`, frozen implementation `a675030` and the three matched seeds retain the complete
CPC/SimDINOv2-style/hybrid × GRU/mLSTM schedule. The compact local causal Transformer has 49-sample
raw support; historical 039 patch/CPC controls remain fixed. Experiment 039 closed at 10,685.80
charged seconds before the 040 profiles. The final 040 receipt audit verified 134 source/input hashes,
all profile/admission bindings and stage-ledger consistency. See the results report for measurements
and final accounting. The coordinator has stopped; no second process or automatic restart is scheduled.

Verification: 102 focused CPU tests passed in 11.91 seconds; Ruff passed. GitHub CI passed 1,167 tests
with six skipped in 114.16 seconds and built wheel/source distributions. Its setup fetches the exact
pinned upstream loss implementation. The earlier local worktree had one existing frozen v11 path-guard
failure from the shared outputs symlink; that isolated test passed in the main checkout.

The original backlog entry remains blocked by the resource gate. A separately authorized prospective
resource plan is ranked as follow-up. No coefficient/exposure/family change or correction cycle was
made; no closed-test evaluation is authorized, and 008 remains deferred. [PR #59](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/59) targets `main`;
the user authorized its merge as part of today's closeout.

**30 September encoder/context study completed:** [Experiment 039 results](experiment-039-cpc-encoder-context-results.md) report all 36 fresh fits: CNN, multiscale and
patch encoders crossed with GRU/xLSTM, three seeds and v4 25k/50k cohorts. Every fit completed
1,954 updates / 250,000 exposures and every cell passed audit. 2 of eight primary encoder comparisons met the frozen promising development rule.
The primary family uses shared whole-patient draws and a simultaneous band across eight
encoder-minus-CNN contrasts; context, scaling and interactions are exploratory. Read the
aggregate, interaction, collapse-gate and accounting receipts under
`outputs/experiment039_encoder_context/`. No closed test was scored. Separate user-authorized
Transformer/CPC+SimDINOv2 work belongs to Experiment 040 and has its own protocol and gates.

**30 September xLSTM result:** Experiment [038](experiment-038-cpc-xlstm-results.md) completed the
user-authorized 25k and conditional 50k comparisons in PR #56. All four full CPC arms trained fresh
for 1,954 updates / 250,000 exposures each; both development audits passed. Primary 1,518-label
AUROC was CPC+GRU 0.90384 versus CPC+xLSTM 0.89487 at 25k, and 0.90454 versus 0.89718 at 50k.
The xLSTM 50k-minus-25k gain was +0.00231 [−0.00707, +0.01136]. Neither architecture improvement
nor scaling met the frozen +0.005 / positive-lower-bound rule. Keep GRU as the baseline for this recipe;
independent-seed follow-up enters the ranked backlog. No further 038 run or closed-test stage is scheduled.

The original protocol was committed as `b39eb93` before new scores. Its real profile failed exact
cuDNN GRU dropout recovery, without full training or new scores. The committed native-GRU successor
(`30da31d`, runner `49400a7`) preserved architecture, dropout and initial weights; synthetic controls
and both real GPU profiles passed exact replay. The prospective 50k supplement (`85dea69`, runner
`80f761d`) binds the audited 25k reference. Actual charged work was 1,368.09 seconds at 25k, including
its cache and failed predecessor, and 1,448.29 seconds at 50k; both stayed below 7,200 seconds.
Read live receipts under `outputs/experiment038_cpc_xlstm_v2/25k/` and `50k/`, including results, audits,
checkpoints and stage ledgers. Ruff, 981 CPU tests, package build and implementation PR checks passed.

**27 September resume:** The user reported EDA finished and explicitly resumed
Experiments 011–013 on the fixed 25k subset. The V100 was idle and no study
runner was active at 04:44 UTC. Because the interrupted 011 profile never
finished its full-cache hash, a new one-time verification ran through
`scripts.coordination.create_nlp25k_cache_seal`, with the normal uv cache and
current `.venv`. It produced the shared seal before any
successor profile. Fresh executable manifests and real V100 cost gates still
precede training. The 26 September pause below is historical.

**27 September 011 profile:** The fresh full-cache seal completed in 1,175.11 s;
its [creation receipt](../outputs/cache_sessions/nlp25k_v1/creation.json) pins
the two historical waveform SHA-256 values and the local seal hash. CPU identity
checks passed for 011–013. The new [011 v2 profile-only manifest](../outputs/experiment_queue_nlp25k_011_profile_v2/queue.json)
passed coordinator `--check` and launched at 05:07:29 UTC. Its child was
observed on the V100 with 3,186 MiB allocated. Check the live
[status](../outputs/experiment_queue_nlp25k_011_profile_v2/status.json) and
profile receipt before reporting a cost gate or launching any training stage.

**27 September 011 full stage:** The v2 profile completed with 979.76 s
selected-row staging, finite 24-update trials for GRU/KDA/CKDA, and peak
allocation below 10 GB. Its conservative complete-study projection was
**6,194.37 s**, below the **7,200 s** gate. A separate
[training/readout manifest](../outputs/experiment_queue_nlp25k_011_full_v2/queue.json)
passed coordinator `--check` and launched its training stage at 05:26:20 UTC.
Inspect the live [status](../outputs/experiment_queue_nlp25k_011_full_v2/status.json)
and arm checkpoints before claiming updates or a development result. The
manifest contains only 011 train and development readout stages.
The training log subsequently showed GRU reach 800/902 updates with periodic
checkpoint writes; this is an observed training state, not a development result.

**27 September 011 completion:** The full queue completed at 05:38:44 UTC with
all three arms trained for 902 updates. The independent
[audit](../outputs/experiment011_delta_memory_25k_v2/audit.json) passed the
three checkpoint/optimizer/encoder movement checks and the fixed development
readout, including saved prediction hashes and metric replay. The
[result](../outputs/experiment011_delta_memory_25k_v2/result.json) used no
calibration or test data. CKDA minus KDA, the prespecified primary contrast,
was +0.00551 AUROC at 15,359 labels (paired patient 95% CI −0.00225 to
+0.01326) and +0.00479 at 1,518 labels (CI −0.00275 to +0.01304). Both
intervals cross zero; the single-seed result does not establish CKDA
superiority. CKDA minus GRU was +0.01046 (CI +0.00125 to +0.01954) at full
labels, a secondary contrast. Next: freeze and run the 012 v3 profile-only
successor, then apply its 7,200-second real V100 gate.

**27 September 012 profile:** The new
[profile-only manifest](../outputs/experiment_queue_nlp25k_012_profile_v3/queue.json)
passed coordinator `--check` and launched at 05:40:37 UTC under the shared V100
lock. It has no full-training stage. Check its live
[status](../outputs/experiment_queue_nlp25k_012_profile_v3/status.json) and
[profile receipt](../outputs/experiment012_temporal_hybrid_25k_v3/profile.json)
before deciding whether the 7,200-second complete-path gate passes.

The profile completed at 05:40:58 UTC. It staged the selected 3.0 GB in 3.55 s
with a warm filesystem cache, completed 24 real updates plus checkpoint replay
and training-only feature extraction for each arm, and projected **1,555.26 s**
for the complete study against the **7,200 s** ceiling. Peak allocated memory
was below 4.34 GB. A separate
[training manifest](../outputs/experiment_queue_nlp25k_012_train_v3/queue.json)
passed coordinator `--check` and launched at 05:41:51 UTC. The earlier cold
staging observed in 011 took 979.76 s; even two such cold staging passes leave
headroom against this gate. Inspect the live
[training status](../outputs/experiment_queue_nlp25k_012_train_v3/status.json)
and completion receipts. The separate development readout still needs its own
real V100 profile and cost gate.

**27 September 012 completion:** The three 902-update training arms completed
at 05:44:03 UTC and passed an independent checkpoint audit. The separate
[readout profile](../outputs/experiment012_temporal_hybrid_25k_v3_readout/profile.json)
projected **956.16 s** against 7,200 s. The verified
[development readout queue](../outputs/experiment_queue_nlp25k_012_readout_full_v3/queue.json)
completed at 05:46:34 UTC, and the independent
[audit](../outputs/experiment012_temporal_hybrid_25k_v3_readout/audit.json)
passed all saved features, predictions, identities and replayed scores. In the
[result](../outputs/experiment012_temporal_hybrid_25k_v3_readout/result.json),
mixed minus matched local support, the prespecified primary contrast, was
**−0.00996 AUROC** at 15,359 labels (paired patient 95% CI −0.01865 to
−0.00210), and **−0.00993** at 1,518 labels (CI −0.01914 to −0.00061).
This single-seed development screen disfavors the mixed support architecture;
no calibration or test data were opened. Next is a new 013 v2 profile-only
manifest and its real V100 complete-path cost gate.

**27 September 013 profile:** The new
[profile-only manifest](../outputs/experiment_queue_nlp25k_013_profile_v2/queue.json)
passed coordinator `--check` and launched at 05:48:39 UTC. Check the live
[status](../outputs/experiment_queue_nlp25k_013_profile_v2/status.json) and
cost receipt before admitting training. This manifest has no training stage.

The profile completed at 05:48:57 UTC. The selected 3.0 GB staged in 3.55 s
with a warm filesystem cache; all three 24-update trials, checkpoint replay and
training-only PTB feature passes completed. Peak allocated memory was below
5.48 GB. Its [receipt](../outputs/experiment013_mamba3_25k_v2/profile.json)
projected **1,374.09 s** for the full study, below the **7,200 s** gate. The
separate [full manifest](../outputs/experiment_queue_nlp25k_013_full_v2/queue.json)
passed coordinator `--check` and launched train then development readout at
05:50:10 UTC. Check its live
[status](../outputs/experiment_queue_nlp25k_013_full_v2/status.json) before
reporting any updates or score.

**27 September 013 completion:** The full queue completed at 05:54:30 UTC,
after all three arms reached 902 updates and development readout finished. The
original auditor omitted root-level dependency files from its path map and
stopped before writing a receipt. A separate versioned
[auditor v2](../scripts/validation/audit_nlp25k_successors_v2.py) corrected that
map without editing frozen experiment sources; its
[audit](../outputs/experiment013_mamba3_25k_v2/audit_v2.json) passed all
checkpoint, encoder movement, head, feature, prediction and metric checks.
The [result](../outputs/experiment013_mamba3_25k_v2/result.json) found the
prespecified Mamba-3 minus Mamba-2 contrast **−0.00818 AUROC** at 15,359 labels
(paired patient 95% CI −0.01722 to +0.00083) and **−0.00806** at 1,518 labels
(CI −0.01745 to +0.00178). These single-seed intervals cross zero, and the
point estimates do not favor Mamba-3. Mamba-2 minus GRU was +0.01250 at full
labels (CI +0.00375 to +0.02161), a secondary contrast. No calibration or test
data were opened. The three requested 25k development screens are complete;
no successor GPU work is automatically scheduled.
The cross-study [25k results summary](nlp-inspired-25k-study-results.md)
links every frozen queue, result and independent audit.

**26 September pause for EDA:** The user canceled the active NLP-inspired
experiment work to finish EDA first. The 011 profile-only coordinator and its
child were interrupted at 22:57:10 UTC during the large cache verification,
before V100 timing. Its [status](../outputs/experiment_queue_nlp25k_011_profile_v1/status.json)
is `interrupted`; no 25k training or development readout result exists. EDA
workers were resumed and the GPU was idle at the [pause receipt](../outputs/nlp25k_pause_2026-09-26/pause.json). Do not launch 011–013 profiles or
training until the user explicitly resumes experiments. Keep all prepared
versioned runners and historical manifests as drafts/evidence, and require a
new verified executable manifest and real GPU cost gate on resumption.

**26 September NLP-inspired 25k request:** The user requested execution of the
remaining NLP-inspired architecture studies on the verified Experiment 019
25,000-record subset. The [prospective common protocol](nlp-inspired-25k-study.md)
fixes that selection, a matched 115,359-exposure screening budget, and the
existing PTB training/development readout contract for 011–013. Implementation
and verification are in progress; this request authorizes the necessary
training after each real V100 profile and cost gate passes. It does not turn a
cost-only profile into a model result. No calibration or test evaluation is
scheduled by this initial development screen. Check the live lock and process
state before any launch. Source-hashed profile-only manifests for
[011](../outputs/experiment_queue_nlp25k_011_profile_v1/queue.json),
[012 v2](../outputs/experiment_queue_nlp25k_012_profile_v2/queue.json), and
[013](../outputs/experiment_queue_nlp25k_013_profile_v1/queue.json) passed
coordinator `--check`; their SHA-256 values are recorded in the JSON catalog.
They do not contain training stages.
The first 012 profile manifest is retained but superseded before launch: the
v2 profile includes the complete training and development-readout cost in its
admission gate. Those manifests were paused under the 26 September instruction;
the 27 September resume above requires newly verified successor manifests.

**011 profile launched at 22:23 UTC on 26 September:** The verified
[profile-only manifest](../outputs/experiment_queue_nlp25k_011_profile_v1/queue.json)
started its child after both redundant standalone CPU hash checks were stopped.
The child is first hashing the complete local caches; its CUDA profiling begins
only after that preflight passes. Its launch PID and this note are snapshots:
inspect the live [status](../outputs/experiment_queue_nlp25k_011_profile_v1/status.json),
child process and log before claiming an actual V100 measurement. No training
stage is present in the manifest.
It was interrupted at the user's request at 22:57:10 UTC before CUDA profiling
or a completed cache hash. The prepared shared cache seal cannot be adopted
from this incomplete receipt.

**PTB diagnosis geometry and MIMIC transfer completed:** The
[exploratory patient-balanced CPU study](ptb-mimic-cpc-diagnosis-geometry.md)
used 15,023 PTB training patients. Ten-neighbor sharing of MI, CD, STTC and HYP
annotations was 1.90×, 1.68×, 1.75× and 1.72× their sample prevalences. A
seeded 4,000-patient HDBSCAN screen found a `NORM`-enriched group and another
mixed abnormality group, not a pure cardiopathy subtype. Using only PTB-fitted
PCA, MIMIC machine-abnormal ECGs had 82.0% abnormal ten-nearest neighbors
versus 25.4% for machine-normal queries. The machine label cannot externally
validate the four PTB diagnoses. An initial full-cohort density run was stopped
for CPU cost before results; the protocol records the narrower density subset.
The completed command, hashes and counts are in the
[local receipt](../outputs/ptb_mimic_cpc_diagnosis_geometry/report.json).

**MIMIC cluster-label proximity audit completed:** The
[one-patient-per-ECG follow-up](mimic-cpc-cluster-label-profile.md) finds that
the largest cluster has 675/1,133 explicit machine summaries marked abnormal,
and CPC flags 674/675 of those but also 85/112 machine-normal ECGs. Among 4,134
patients with an exact abnormal/normal cart summary, abnormal queries have
82.0% abnormal ten-nearest neighbors versus 23.9% for normal queries in frozen
CPC feature space; the within-cart permutation result is exploratory. This
supports broad machine-abnormality structure, not verified cardiopathy or
clinical diagnostic accuracy. The CPU command and source hashes are in its
[local receipt](../outputs/mimic_cpc_cluster_label_profile/report.json).

**MIMIC missed-pattern audit completed:** The user asked whether clusters
contain ECGs with abnormalities that CPC fails to flag. The
[official-machine-report join and negative-only clustering](mimic-cpc-machine-disagreement.md)
found 119 CPC-negative/explicit-machine-abnormal ECGs from 100 patients, and a
modest exploratory concentration in one negative-only cluster (34/958 ECGs;
27 patients). A machine-text acute-MI alert appeared in 37 candidate ECGs,
spread across clusters. The patient-level evidence does not establish a
missed-disease subtype; prioritize independent review of the local candidate
IDs. The official machine metadata was checksum-verified and kept local. No
training, calibration/test evaluation, or scheduler was launched.

**Exploratory MIMIC clustering completed:** The user requested HDBSCAN on ECGs
after the binary CPC flag audit. The [read-only patient-balanced analysis](mimic-cpc-clustering.md)
found two dense groups among 7,892 seeded one-per-patient MIMIC ECGs: 1,497
and 1,042 records, with 95.8% and 69.3% CPC flag rates; 5,353 were unassigned.
This is not a diagnosis discovery or an independent validation of the CPC flags.
No training, calibration/test evaluation, or scheduler was launched.

**26 September 25k follow-up:** The user requested a smaller training pool after Experiment 018's negative scaling readout. [Experiment 019](experiment-019-cpc-25k.md) freezes 25,000 total source-stratified training ECGs from the verified 115,359-record cache, replayed over the same 902-update budget. Its primary comparison is 25k minus the completed 115k arm under the same frozen PTB development readout. A new full-cache hash and actual V100 cost profile must pass before this single arm trains; calibration/test remain closed.

**019 profile passed:** The 24-update full-path V100 profile measured 44.93 seconds, including the cached data loader. The full-cache preflight took 947.32 seconds; the conservative total projection was **4,230.55 seconds**, below the 7,200-second gate. The seeded 25k subset contains 15,170 MIMIC, 3,746 PTB-XL, 2,072 Chapman, 2,066 Georgia, 1,334 CPSC 2018, and 612 CPSC Extra ECGs. Its selected-index SHA-256 is `43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791` under the canonical JSON scheme. The training stage is authorized by this receipt; inspect the live process/checkpoint rather than treating this note as proof it is running.

**019 training completed:** The [training-only result](experiment-019-cpc-25k-training-results.md) confirms all 902 updates and 115,359 exposures completed in 319.30 training seconds after the full production cache hash. The independent audit passed checkpoint SHA-256 `1ac0035fe07aa421853056b99cf13300418546aa787cf8ded749c412cfd4d523`, optimizer/RNG state, and 24 changed finite encoder tensors. Mean CPC loss 2.42322 is not a downstream quality result. The separately frozen [development readout](experiment-019-cpc-25k-readout.md) is the next stage; its own training-only V100 profile/cost gate must pass first.

**019 readout profile passed:** The training-only 512-record V100 profile and full historical PTB cache hash passed. Its projected readout cost was **1,668.85 seconds** against 7,200. The full development-only stage has launched under the shared GPU lock; do not interpret a launch as a result. Its runner binds the 25k checkpoint, previous 115k predictions, exact PTB cache and fixed label identities. Calibration/test stay closed.

**019 development readout completed:** The [frozen result](experiment-019-cpc-25k-readout-results.md) gives 25k/115k full-label AUROC 0.91960/0.91795, difference +0.00164 with paired patient 95% interval -0.00073 to +0.00396. At 1,518 labels, the difference was +0.00208 with interval -0.00034 to +0.00443. Both include zero; the unchanged starting encoder remains highest by point AUROC. The independent saved-artifact audit passed and no calibration/test patient was used. This single subset is not a paper result by itself; the paper-directed cross-encoder/task question remains the priority.

**Paper-directed priority after 019:** The user asked to focus on experiments capable of yielding a defensible research contribution. The updated [readout-gap investigation](experiment-016-paper-investigation.md) now includes the completed two-seed xECG development evidence and current prior work. The next scientific decision is a controlled cross-encoder or cross-task replication of the representation-versus-readout question, with independent patients; another CPC subset sweep alone is lower priority. This is a research direction, not a live GPU job or permission to open calibration/test.

**26 September data-scaling training request:** The user requested training on the newly verified 115,359-record cohort. [Experiment 018](experiment-018-cpc-data-scaling.md) freezes a one-pass compact CPC continuation against a matched-update old-pool control. A real-data V100 profile and 7,200-second cost gate precede any training; neither a profile nor SSL loss is a downstream performance result. The original experiment pause no longer blocks this user-requested study. Keep experiments sequential and preserve all older receipts.

**018 v1 cost stop and v2 successor:** Direct new-cohort loading took 352.46 seconds for 24 profiled updates, versus 51.41 seconds for the old cache. The v1 full-study projection was 19,565.44 seconds, above the gate, so no training began. [V2](experiment-018-cpc-data-scaling-v2.md) keeps the same scientific arms and builds a local 250 Hz cache before a new full-path V100 profile. The cache is preparation, not a result. No architecture queue item was resumed.

**018 v2 cost stop and v3 successor:** The complete local cache passed independent replay and full-file hash checks. Its one-time build took 4,410.00 measured process seconds. The v2 V100 profile measured 37.96 seconds for 24 old-arm updates and 4.70 seconds for 24 new-arm updates, but its prespecified full-study projection was 8,561.00 seconds, above 7,200; no full training arm started. [V3](experiment-018-cpc-data-scaling-v3.md) treats the completed verified cache as an existing data asset and prospectively gates all new v3 work, with historical preparation disclosed separately. V3 still requires its own measured profile before training.

**018 v3 live launch:** Its fresh full-hash and real-data V100 profile passed the frozen gate at **3,654.68 projected seconds** versus 7,200. The old/new 24-batch timings were 32.73/1.30 seconds. The full matched run launched on 26 September at 03:14:33 UTC, passed its production cache hash, and began the old-control arm. Check live process and `outputs/experiment018_cpc_data_scaling_v3/{old,new}/latest.pt`; a launch is not a completed training result.

At the first observed old-control checkpoint, **100 of 902 actual updates** completed in 93.87 training seconds. The checkpoint was present and SHA-256 `34bfe7a4226bfa053167e8e2c9922e3ce3c33ec852d4bf18f0b25493ef5778d6` at that observation; later checkpoints replaced this resumable file. The live process held the V100, with the new-data arm then waiting.

**018 v3 completed:** Both matched arms finished 902 updates and 115,359 exposures. Old/new training time was 361.21/347.26 seconds, with mean CPC loss 2.35513/2.43679. The [training-only result and artifact audit](experiment-018-cpc-data-scaling-v3-results.md) confirm both checkpoint hashes, all optimizer steps, and actual encoder changes. The GPU is idle. Different-pool SSL losses do not rank representation quality; a separately frozen development readout is the next step. Calibration/test remain closed.

**018 fixed readout preflight:** The [development-only protocol](experiment-018-cpc-data-scaling-readout.md) is frozen before new predictions. It compares unchanged, old-continuation and new-continuation encoders with identical 512-feature extraction and fixed train-only logistic heads at 15,359 and 1,518 labels. Its CPU identity check found exactly 1,306 development ECGs from 1,173 patients and no train/development patient overlap. A full-path V100 profile and cost gate precede extraction. Calibration/test are not scored.

The readout's training-only V100 profile passed at **1,635.60 projected seconds** against 7,200, including both full-cache preflights and a 900-second CPU-fit/report reserve. The full stage then completed. The [development-only results](experiment-018-cpc-data-scaling-readout-results.md) show no benefit from new-cohort CPC: at 15,359 labels, new/old AUROC was 0.91795/0.91809 (difference -0.00014, paired patient 95% interval -0.00324 to 0.00288). At 1,518 labels, new/old AUROC was 0.90658/0.90564 (difference +0.00094, interval -0.00220 to 0.00434). The unchanged starting encoder scored 0.92079/0.91296 at the two label budgets. The artifact audit passed; calibration/test remain closed. Experiment 019 is the separately frozen 25k follow-up.

**Historical pause (2026-09-24T19:40:47.068565+00:00):** The user paused experiment training for the repository refactor. On 25 September, the user requested continuing experiments after the refactor commit. Experiment 011 implementation has resumed; 008 and 010 remain deferred for cost, and the legacy MIMIC scheduler has not been restarted. Downloads continue, so preserve their source/data paths.

**Refactor handoff:** Completed results and checkpoints remain in place. `outputs/refactor_pause/source_before_refactor.tar.gz` and `source_hashes.json` preserve and verify the pre-refactor source tree; `pause.json` records the stopped scheduler and archive hash. Preserve experiment output/data identities and archived source evidence. After refactoring, verify equivalent preprocessing, patient splits, model initialization and checkpoint loading, then create new source maps/manifests before resuming. Do not silently rewrite old provenance receipts to match refactored code.

**25 September data preflight:** [Preprocessing audit and cheap rerun plan](clean-data-rerun-review.md) completed with 40 targeted tests. The separate [manifest-only clean cohort](../outputs/data_quality/clean_rerun_preflight_v1/receipt.json) retains 56,809 original-pool PTB/MIMIC records, 15,359 full labels and the unchanged 1,518-label subset; all 3,766 held-out records retain their partitions. The 76,598-record union is a separate scaling cohort. The preflight itself was preparation only and did not schedule training. Its priorities do not change the architecture backlog.

**25 September scoped rerun:** The [clean cached-probe and fusion repeat](../outputs/clean_cached_probe_rerun_v1/report.md) completed on CPU with new clean training labels. Full-label development AUROC moved from 0.959719 to **0.959686** for JEPA and 0.936230 to **0.936192** for ordinary CPC; limited-label results were unchanged. The 014 fusion development gate remained negative at both budgets. No calibration/test outcomes were evaluated. This measures removal of one faulty labeled PTB ECG, not added Challenge/Chapman data or clean encoder pretraining. [Experiment 016](../outputs/experiment016_xecg_probe_finetune/report.md) also completed its scoped development screen; its probe-initialized fine-tune underperformed the random-head control.

**25 September new scoped experiment:** The user requested one promising experiment at a time, with Astra designing and Sol executing. [017 clean transfer and second-seed replication](experiment-017-clean-replication.md) completed as a development-only study. Its CPU input check passed: all 15,359 clean labeled training ECGs matched the canonical 500 Hz to historical CPC 250 Hz transform bit for bit; all 32 original seed-42 template donors remain in training. The first [profile-only manifest](../outputs/experiment_queue_017_clean_profile_v1/queue.json) stopped after a full no-branch epoch because its checkpoint comparison used CUDA tensors against CPU-loaded saved tensors. It produced no performance result or training arm. [Version 2](experiment-017-clean-replication-v2.md) pins that comparison to CPU, passed a fresh CPU check and V100 profile, then completed all 12 training arms and its artifact audit. Calibration/test remain untouched. No other architecture or scaling study is running.

**016 two-epoch DropPath rescue stopped at the cost gate.** Astra froze the [mechanistic protocol](experiment-016-droppath-rescue.md) to investigate the original xECG fine-tuning decline. The clean CPU check verified the exact 15,359-record cohort, released feature/cache/weight identities, and a fixed-C=0.01 probe at **0.9619404 development AUROC**. The V100 diagnostic showed a smaller mean train/eval logit shift for Residual than Legacy (4.10 versus 7.03) but substantial mismatch in both; Off matched eval exactly. Earlier versioned v2–v4 profiles isolated checkpoint metadata, memory, and tiny CUDA replay-difference issues without launching training. The [v5 verification addendum](experiment-016-droppath-rescue-v5.md) froze the corrected replay criterion. The [v5 profile-only manifest](../outputs/experiment_queue_016_rescue_profile_v5/queue.json), SHA-256 `5e4d7bb56241eb406d9edde4052775086551504908c26ece5e195be85d05d348`, passed its source check, diagnostic, and all three complete V100 profile correctness checks. Its frozen projection was **7,624.25 seconds**, above the **7,200-second ceiling**. The coordinator records `failed` because the cost gate intentionally exits nonzero; no two-epoch study arm or full training manifest was created. See the [aggregate profile report](experiment-016-droppath-rescue-profile-results.md). The separately frozen one-epoch successor below does not change this v5 decision.

**016 one-epoch successor completed:** The [seed-43 v6 protocol](experiment-016-droppath-rescue-v6.md) used the same clean cohort, frozen probe and original schedule's first 240 updates, with Legacy, Residual and Off arms. Its inherited full-path cost gate passed after a bounded V100 bridge. All three arms completed and the conservative final projection was **4,528.85 seconds**, below 7,200. At update 240, Residual beat Legacy by **+0.03552 development AUROC** (paired patient 95% interval **+0.02646 to +0.04470**), supporting a partial mechanism explanation. Residual remained **0.01372 below the probe**; Off was similarly below, so **neither met the practical rescue gate**. The original full manifest stopped only at a gradient-audit `None` handling error after all training; a [verified audit/report-only successor](../outputs/experiment_queue_016_rescue_v6_audit_v2/queue.json) completed without retraining. The [results and limitations](experiment-016-droppath-rescue-v6-results.md) include artifact hashes, sampling uncertainty and actual launch provenance. This repeatedly inspected development cohort cannot independently confirm the effect; calibration/test remain untouched.

**016 frozen-readout audit completed:** Astra froze the [v7 diagnostic protocol](experiment-016-frozen-readout-audit-v7.md) after v6. No encoder updates were allowed. The complete Off V100 extraction and released-control fit passed the [measured gate](../outputs/experiment016_frozen_readout_audit_v7/cost_gate.json) at **1,345.73 projected seconds** versus the 7,200-second ceiling; the [verified full successor](../outputs/experiment_queue_016_readout_v7_full/queue.json) then finished Residual/Legacy extraction, fixed C=0.01 train-only refits, and the prespecified development report. Off refit exceeded its saved joint head by **+0.01513 AUROC** (paired patient 95% interval **+0.00925 to +0.02087**) and reached **0.96293**, within the frozen near-complete-recovery rule relative to the released probe's **0.96194**. Its +0.00099 probe-relative AUROC did **not** meet the +0.002 useful-adaptation rule. This postmortem, single-seed result supports a recoverable linear readout component without establishing why the joint head lagged or that adaptation helps on new patients. [Results and receipts](experiment-016-frozen-readout-audit-v7-results.md); no calibration/test or follow-up training.

**016 matched-objective head mechanism completed:** The [frozen v8 CPU protocol](experiment-016-head-mechanism-v8.md) compared head-only A raw AdamW, B raw probe-objective Adam and C standardized probe-objective Adam on unchanged v7 Off features. A full-path seed-16080 profile with two-start D reference, exact checkpoint replay and source verification passed the [cost gate](../outputs/experiment016_head_mechanism_v8/cost_gate.json) at **516.72 projected seconds** versus 7,200. The [verified production successor](../outputs/experiment_queue_016_head_mechanism_v8_full/queue.json) completed all six 10-epoch arm/seed runs. At the prespecified 240-update endpoint, C−B AUROC was **−0.00084 / −0.00060** for head seeds 44/45, and the two-seed mean paired patient interval included zero: the conditioning screen failed. A recovered to **0.96249 / 0.96185** from the saved joint head's 0.94781 and passed its separate stationary-feature sufficiency rule; B−A was negligible, so the regularization-package rule failed. The [results and limitations](experiment-016-head-mechanism-v8-results.md) condition on one frozen encoder and a repeatedly inspected development cohort. Keep the released probe; no GPU, encoder updates, calibration/test or automatic next experiment.

**016 matched encoder-update intervention completed:** The separately frozen [v9 protocol](experiment-016-encoder-motion-v9.md) compared a moving xECG encoder with a zero-encoder-learning-rate control that still computed all encoder gradients and used the same full-model clipping. Both arms used the same clean 15,359-record cohort, released weights/probe head, seed-46 order, one warmup epoch and 240 updates. The [full V100 profile](../outputs/experiment_queue_016_encoder_motion_v9_profile/queue.json) passed replay, bitwise frozen-encoder identity and released-feature controls, and projected **5,814.82 seconds** against the 7,200-second gate. The [separate verified production successor](../outputs/experiment_queue_016_encoder_motion_v9_full/queue.json) and [development report](../outputs/experiment016_encoder_motion_v9/report.json) completed. F joint-head AUROC **0.962486** exceeded M joint-head **0.956419** by **+0.006067** (2,000 paired-patient draw interval **+0.001696 to +0.010901**), passing the prespecified harm screen. M's fixed-C refit reached **0.964098**, versus F/refrozen released probe **0.961940**; the readout-gap difference was **+0.008224** (interval **+0.004030 to +0.012802**), passing the narrower readout-gap interpretation. This is one optimization seed on repeatedly inspected development patients, with realized clipping allowed to differ downstream; it does not prove a universal encoder-harm mechanism or held-out clinical benefit. [Results, checks and hashes](experiment-016-encoder-motion-v9-results.md). Retain the released probe; calibration/test remain closed.

**016 seed-47 encoder-motion replication stopped before GPU work:** The separately frozen [v10 protocol](experiment-016-encoder-motion-replication-v10.md) passed its 452.59-second real-data CPU compatibility check. Its verified [bridge-only manifest](../outputs/experiment_queue_016_encoder_motion_v10_bridge/queue.json) then failed during device verification because a new helper called nonexistent `torch.cuda.get_device_count()`. No bridge update, production arm or development prediction ran. The failed launch consumed about 544.34 seconds after about 90 seconds of manifest checking; only about 298.24 seconds of the frozen new-work allowance remained. Repeating the observed 411.14-second pre-GPU child path would exceed the 7,200-second planning ceiling before training, so no corrected bridge or production successor was launched. [Failure evidence, cost accounting and limits](experiment-016-encoder-motion-replication-v10-results.md). The v9 one-seed result was unchanged at that stop; the later v14 replication is recorded below.

**016 seed-47 v11 stopped at cost gate before GPU work:** The [frozen v11 protocol](experiment-016-encoder-motion-replication-v11.md) preserved v10's matched M/F arms and seed-47 endpoint. Its new cheap API preflight passed on CPU; a corrected complete transitive pass verified **288 unique leaves, 1,051 hash obligations and 6.493 GB**, plus exact equality to v10's pinned nested v9 input fingerprint. The persistent ledger charges all complete and failed scans, semantic debug rechecks and CPU preparation probes. Under the most favorable single-process bridge/production launch assumptions, the conservative projected total is **7,211.468 seconds**, already above the **7,200-second** ceiling before the complete CPU compatibility check or GPU bridge. The [cost gate and stop report](experiment-016-encoder-motion-replication-v11-results.md) preserve all evidence; no bridge/production update, seed-47 development prediction, calibration or test evaluation occurred. V11 is closed without a replication result, and no experiment queue is active.

**016 seed-47 v12 attempt closed before GPU work:** The [separate frozen v12 protocol](experiment-016-encoder-motion-replication-v12.md) preserved v10's science and proposed a two-process verification/bridge then production path. Implementation review found that its checkpoint equation counts elapsed work in an incomplete arm in `E` while still forecasting two full pipelines until `n` increases. Under the planned 1,025-second nonpipeline allowance, the literal rule leaves **360.175 seconds** for incomplete M work at a checkpoint; historical v9 M production took **885.241 seconds** in total. This comparison is planning evidence, not a measured v12 gate or a proof that every possible run fails. No unambiguous complete-path checkpoint forecast, full materializer/launcher, CPU compatibility pass, or measured admission was established, so no executable manifest or real-data invocation was made. The [stop report](experiment-016-encoder-motion-replication-v12-results.md) and [receipt](../outputs/experiment016_encoder_motion_replication_v12/stop_receipt.json) record zero seed-47 updates and predictions. V10/v11 spent at least approximately **1,666.763 seconds** separately, with uncertainty retained. V12 is closed without a replication result; calibration/test remain closed.

**016 prospective seed-47 v13 closed before real-data admission:** The [v13 protocol](experiment-016-encoder-motion-replication-v13.md) preserved the requested 15,359-label seed-47 M/F comparison and corrected incomplete-arm accounting. Its pre-GPU phase-bound audit found that the first mandatory 40-update block would need to finish within **117.504 seconds** even if all 1,025 seconds of preparation met their floor exactly. The authenticated v9 evidence records only whole-pipeline times; it has no 40-update or extraction/replay phase durations to support that bound. The initial **7,096.075-second** envelope was a design calculation, not a measured admission gate. V13's own rule requires a stop when conservative phase bounds cannot be supported, so there was no bridge-only manifest, real-data invocation, GPU update, production arm or seed-47 development prediction. [Results, synthetic gate checks and inventory](experiment-016-encoder-motion-replication-v13-results.md); [stop receipt](../outputs/experiment016_encoder_motion_replication_v13/stop_receipt.json). V9–v12 evidence is unchanged, and the earlier v10/v11 spending remains separately disclosed. V13 is closed without retry; calibration/test remain closed.

**016 seed-47 v14 development replication completed:** The separately frozen [v14 protocol](experiment-016-encoder-motion-replication-v14.md) retained v10's 15,359-label M/F science and completed one fresh seed-47 pair after a passed CPU check, bounded bridge, production gates and independent artifact audit. The frozen-encoder joint head scored **0.961254** AUROC versus **0.953001** for the moving encoder: F−M **+0.008252**, with a 2,000-draw paired-patient interval **+0.002745 to +0.014180**. The prespecified primary screen passed. After a train-only fixed-C readout refit, M scored **0.962963** versus F **0.961940**; its excess readout gap **+0.009275** also passed the narrower point screen. The original seed-46 joint-head gap was **+0.006067**, making the two-seed mean **+0.007160**. The [results and limitations](experiment-016-encoder-motion-replication-v14-results.md) and [final receipt](../outputs/experiment016_encoder_motion_replication_v14/final_receipt.json) preserve all controls, hashes and bootstrap intervals. Reconciled new v14 work was **4,675.5 seconds** against the prospective 7,200-second incremental gate; historical v9–v13 spending was at least approximately **6,406.509 seconds separately**, so combined research spending exceeded two hours. This remains a repeatedly inspected development-cohort result on one released encoder, not independent-patient clinical validation or proof of an exclusive mechanism. Retain the released clean probe; no automatic follow-up, calibration or test evaluation.

**017 profile gate passed:** The corrected V100 profile completed one full-data epoch for every arm and exact model/optimizer/RNG checkpoint checks. The conservative 12-run plus audit projection is **2,081.77 seconds**, below the 7,200-second planning gate; peak allocated GPU memory was **1.02 GB**. The [separate full successor manifest](../outputs/experiment_queue_017_clean_full_v2/queue.json) verified source hashes and ran only this experiment's twelve training arms followed by its development artifact audit. Profile epoch scores are timing evidence, not selected performance results.

**017 result:** The limited-label template arm passed its numeric gate at both optimization seeds, beating matched convolution by **+0.00293** and **+0.00352** development AUROC. Artifact flags and single-patient deletions did not reverse the gain; no single template dominated it. Both paired patient bootstrap intervals include zero, so the effect remains uncertain. Full details and limitations are in the [clean replication results](experiment-017-clean-replication-results.md). This does not test clean SSL pretraining or the larger 76,598-record union.

**Historical 017 finding:** The first morphology-template screen at 1,518 labels had AUROC **0.9487**, versus **0.9462** for matched convolution and **0.9431** for no-branch CPC. At full labels, convolution and template tied near **0.9620**, versus **0.9571** for no branch. That one-seed result led to the completed clean two-seed follow-up above; neither study used calibration or test predictions. [Original 017 results](../outputs/experiment017_morphology_templates_v2/report.md). Experiments 014 and 015 had negative screens.

**CPC equal-width local readout completed:** The [ranked evidence note](cpc-improvement-research-2026-09-25.md) led to the separately authorized [CPU-only protocol](cpc-local-readout-v1.md). A new immutable [manifest](../outputs/cpc_local_readout_v1/manifest.json) bound the historical 009 compact-CPC cache, clean split inputs, encoder/normalization identities and new source hashes. The two 512-feature fixed-C heads used 1,518 and 15,359 clean labels. The pre-outcome cost gate projected **284.93 seconds** against 7,200; actual total was **42.95 seconds**, peak RSS **422,784 KiB**. Local max replacement improved development AUROC by **+0.00911** at 1,518 labels (2,000 paired-patient draw 95% interval **+0.00278 to +0.01540**) and **+0.01250** at 15,359 labels (**+0.00639 to +0.01900**). The prespecified point screen passed. [Results, command, hashes and limitations](cpc-local-readout-v1-results.md). This one historical encoder seed and repeatedly inspected development cohort warrant a separately frozen replication discussion; no calibration/test, GPU work or deferred 010 restart occurred. Priorities 011–013 remain unchanged.

## Pending priority: cheapest likely first

The user authorized **all four Astra proposals**, now Experiments **014–017**. They take priority over the larger architecture implementations. The order below is an implementation priority, not a dependency chain: an unpromising early screen does not block an independent later experiment.

| Priority | Experiment | Status | Expected pilot cost / next work |
| --- | --- | --- | --- |
| 1 | [011: KDA / CKDA](experiment-011-delta-memory-25k-v2.md) | Complete development screen | Three 902-update arms and independent artifact audit passed; primary CKDA–KDA patient interval crosses zero at both label budgets. No calibration/test. |
| 2 | [012: StripedHyena-inspired](experiment-012-temporal-hybrid-v3.md) | Complete negative development screen | Mixed support trailed matched local support by about 0.010 AUROC at both label budgets; independent audit passed. No calibration/test. |
| 3 | [013: Mamba-3](experiment-013-mamba3-25k-v2.md) | Complete development screen | Three 902-update arms and versioned independent artifact audit passed; Mamba-3 point estimate trailed Mamba-2 at both budgets, with patient intervals crossing zero. No calibration/test. |

The [Astra research note](astra-next-model-ideas.md) supplies the four new designs, controls and primary sources; the JSON catalog links each proposal's exact section. **These ranges are unmeasured planning estimates, not promised completion times**, and exclude implementation work. Only 014 is clearly the cheapest starting point. The remaining ranges overlap; actual complete-pass timing may change their order. A short GPU compute profile alone is insufficient, as Experiment 010 demonstrated. Include cold/warm loading, every comparison arm, evaluation and checkpoint writes in the working two-hour GPU-pilot planning gate.

**The 011–013 development screens are complete; no NLP 25k GPU queue is active.** Recheck live status before any future GPU stage. The 016 v14 seed-47 development replication completed; no follow-up is scheduled. The editable backlog does not schedule a process. Use development patients for early screening, then freeze choices before calibration/test. GPU work remains sequential.

## Deferred and completed studies

| Experiment | Status | Evidence / recovery |
| --- | --- | --- |
| [016 v14 seed-47 encoder-motion replication](experiment-016-encoder-motion-replication-v14.md) | Complete; both frozen development screens passed at seed 47 | [Results and limitations](experiment-016-encoder-motion-replication-v14-results.md); [final receipt](../outputs/experiment016_encoder_motion_replication_v14/final_receipt.json); no calibration/test |
| [016 v10 seed-47 encoder-motion replication](experiment-016-encoder-motion-replication-v10.md) | Stopped before GPU updates; bridge helper failure and insufficient measured cost headroom for an unchanged-path retry; no replication result | [Failure and cost report](experiment-016-encoder-motion-replication-v10-results.md); [failed immutable bridge manifest](../outputs/experiment_queue_016_encoder_motion_v10_bridge/queue.json); no calibration/test |
| [016 v8 matched-objective head optimization](experiment-016-head-mechanism-v8.md) | Complete CPU-only development diagnosis; conditioning screen failed, stationary-head AdamW sufficiency screen passed | [Results](experiment-016-head-mechanism-v8-results.md); [verified production successor](../outputs/experiment_queue_016_head_mechanism_v8_full/queue.json); no calibration/test |
| [016 v7 frozen-backbone readout audit](experiment-016-frozen-readout-audit-v7.md) | Complete development-only diagnostic; near-complete fixed-readout recovery, no useful-adaptation gate | [Results](experiment-016-frozen-readout-audit-v7-results.md); [verified full successor](../outputs/experiment_queue_016_readout_v7_full/queue.json); no calibration/test |
| [016 v6 one-epoch DropPath screen](experiment-016-droppath-rescue-v6.md) | Complete development screen; no practical rescue | [Results](experiment-016-droppath-rescue-v6-results.md); [verified audit/report successor](../outputs/experiment_queue_016_rescue_v6_audit_v2/queue.json); no calibration/test |
| [016 v5 two-epoch DropPath rescue](experiment-016-droppath-rescue.md) | Profile complete; cost gate failed | [Aggregate diagnostic and cost report](experiment-016-droppath-rescue-profile-results.md); all three profile correctness checks passed, projected 7,624.25 seconds exceeds 7,200-second ceiling; no two-epoch training result |
| [017 clean transfer and second seed](experiment-017-clean-replication.md) | Complete development screen | [Results and limitations](experiment-017-clean-replication-results.md); [verified full manifest](../outputs/experiment_queue_017_clean_full_v2/queue.json); no calibration/test |
| [016: xECG probe-initialized fine-tuning](experiment-016-xecg-lpft.md) | Complete | [Negative development screen](../outputs/experiment016_xecg_probe_finetune/report.md); frozen probe 0.96195 AUROC, random-head fine-tune 0.94513, probe-head fine-tune 0.93138; verified [successor manifest](../outputs/experiment_queue_016/queue.json) |
| [017: morphology templates](experiment-017-morphology-v2.md) | Complete | [Positive limited-label development screen](../outputs/experiment017_morphology_templates_v2/report.md); requires replication |
| [015: JEPA-to-CPC distillation](experiment-015-distillation.md) | Complete | [Negative development screen](../outputs/experiment015_jepa_cpc_distillation/report.md); small full-label gain, limited-label loss |
| [014: JEPA + CPC fusion](experiment-014-fusion.md) | Complete | [Negative development screen](../outputs/experiment014_jepa_cpc_fusion/report.md); JEPA alone selected at both budgets |
| [008: xECG adaptation](experiment-008-vision-ssl.md) | Deferred for cost | Approximately 27.7-hour profile projection; no resumable arm-A checkpoint |
| [010: cross-lead CPC](experiment-010-crosslead.md) | Deferred for cost | Actual native epoch 1 took 15.6 minutes; verified epoch-1 checkpoint preserved |
| [009: prediction mismatch](experiment-009-mismatch.md) | Complete | [Results](../outputs/experiment009_cpc_prediction_mismatch/report.md) |
| [007: released xECG](experiment-007-xecg.md) | Complete | [Results](../outputs/experiment007_xecg/report.md): AUROC 0.9374 full labels, 0.9283 at 10% |
| [006: tokenization](experiment-006-tokenization.md) | Complete | [Results](../outputs/experiment006_cpc_tokenization/report.md) |

008 and 010 are excluded from the active priority list and will not automatically restart.

## Coordination

The coordinator is `scripts/coordination/run_priority_queue.py`. Historical commands
and source hashes below refer to the original checkout; use its recorded revision
or the pause snapshot for recovery. New execution requires a freshly verified manifest. The original frozen manifest, `outputs/experiment_queue/queue.json`, contained 009 → 007 → 008 → 010 and stopped at the 008 profile on a CUDA out-of-memory error at 04:56 UTC. Its status, launch receipt and failure log remain in that directory. Its successor, `outputs/experiment_queue_recovery_008/queue.json`, profiled 008 successfully and started adaptation, then was interrupted at the user's request. The first 010 profile-only manifest failed a bitwise CUDA save/restore check; its versioned successor passed the profile. The full 010 manifest, `outputs/experiment_queue_full_010/queue.json`, was also interrupted after real data I/O exceeded the cost envelope. **The corrected 017 v2 queue, the 017 clean replication full queue, the 016 v6 audit/report successor, the 016 v7 frozen-readout audit, and the 016 v8 CPU head-mechanism study are complete; no experiment queue is active. The original 016 v6 full queue stopped only at the post-training audit and was recovered without retraining. The earlier 015 queue completed 015 and was then interrupted only at its original 017 stage.** The executable manifests differ from the editable `docs/experiment-queue.json` backlog. Individual runners retain their existing GPU locks; the coordinator itself owns no CUDA context. Jobs use source-map hashes, source hashes and completion-artifact checks.

The old coordinator exited after 008's failed profile and resumed the legacy Experiment 003 orchestrator. The successor launch verified that old coordinator was absent, the downloader remained active and the legacy orchestrator had no child. It paused the legacy orchestrator at 14:50:59 UTC and started the 008 profile. The downloader continues; the legacy orchestrator resumes when the new queue completes, fails or handles an interruption.

At 15:06:30 UTC, the successor 008 coordinator received SIGTERM to honor the user's lower-cost preference. It terminated its GPU child and resumed the legacy runner. The 008 log records arm A updates through 30; no `resume.pt` or completed encoder exists under `outputs/experiment008_vision_ssl/pretrain/`, because periodic checkpoints begin at update 100. Those observed updates are not resumable and would be recomputed if 008 is later selected. All files and the four-arm GPU profile receipt were preserved. A separate verified profile-only manifest, `outputs/experiment_queue_profile_010/queue.json` (SHA-256 `2313405e360737130a8f7b9aceb46e5a81dd36317472dee937ae819e653057b8`), launched at 15:07:28 UTC. It runs only 010's `profile` command and requires the three profile JSON receipts. Its job log and completion marker are isolated under `outputs/experiment_queue_profile_010/job/`, leaving a future full 010 queue's completion identity free. The legacy runner was paused again; the downloader continues.

The initial 010 profile reached a CUDA resume comparison but stopped because strict bitwise equality found a maximum absolute parameter difference of `1.4901161193847656e-08` in 94 of 3,840 elements checked in the first failing tensor. No variant profile JSON was saved. The historical failure remains under `outputs/experiment_queue_profile_010/`. A new file, `scripts/run_cpc_crosslead_profile_v2.py`, reuses the frozen objective/profile code and checks both model and AdamW states after save/restore using FP32 `atol=1e-7`, `rtol=1e-5`, logging maximum absolute differences. Its source SHA-256 is `ca81c8c536b29d814be6c210adfdb4b6fdc0264524f4979d14269f75c62f54d7`. The versioned source map SHA-256 is `d25e53b4b89278d169f42914fa253d26c391dab8606fa3fa7645906f23cbc61d`. A new immutable profile-only manifest, `outputs/experiment_queue_profile_010_v2/queue.json` (SHA-256 `946210121f1a8d38bd00cb891b341fb62d171ef1f47d77e53d294a4555cc2855`), passed `--check` and launched at 15:13:33 UTC. Its job log/completion marker are isolated under `outputs/experiment_queue_profile_010_v2/job/`. No full 010 training is scheduled by this manifest.

The corrected profile completed at 15:17:33 UTC. Its five measured updates per arm estimated 65.06, 62.50 and 63.46 seconds per SSL epoch for native, withinlead and crosslead respectively, with about 3.04 GB peak allocated GPU memory. Multiplying those short-profile figures by ten epochs of three arms gave an initial 31.84-minute SSL estimate. Extrapolating Experiment 004's cumulative supervised histories to six transfers added about 4.29 minutes; that initial estimate omitted the full workload's sustained random cache I/O, as observed below. These are runtime measurements and estimates, not performance results. Profile JSON SHA-256 values are `87f58d33e8f24374b61ef6788470927a1e1c3c15d3cf3fa9bfaf4649ec4784be` (native), `918ed663896af9f831068422ebde7720158ba5b9c8ef8a8da5fe12ee8c9f8b8c` (withinlead), and `dbb5d9917b47cca8811a505dd766fd8c16bb019ab5b753c5eee03bf3887b6eeb` (crosslead). They and their resume checks were verified before launch. A new immutable full manifest, `outputs/experiment_queue_full_010/queue.json` (SHA-256 `3cea5a83ca96cba1c6e392f43fb1b15c611f7bf85b14e90d90207278974738e9`), passed `--check` and launched at 15:18:32 UTC. Its job log and completion marker are isolated under `outputs/experiment_queue_full_010/job/`. It uses the original frozen 010 runner and source map. 008 is excluded.

The short profile underestimated sustained random cache I/O. The full 010 runner passed its profile fingerprint gate and completed native SSL epoch 1 in **934.59 seconds (15.58 minutes)** for 445 updates and 56,875 record exposures, saving `outputs/experiment010_cpc_crosslead/native_ssl/{epoch_state.pt,history.json}` at 15:38:23 UTC. Epoch 2 remained incomplete after four additional minutes. If that observed warm-epoch pace continued, 29 remaining SSL epochs alone would require more than 116 minutes before transfers and other overhead, exceeding the adopted two-hour planning limit. The full coordinator received SIGTERM at 15:42:23 UTC and terminated its GPU child; the legacy MIMIC runner resumed, and the downloader continued. The epoch-1 checkpoint was loaded and verified to contain model, optimizer, RNG, fingerprint and one history row. Its SHA-256 is `808363dbcbfabecfe42a49f2b5e42c4a10adfb27dc833d2e7d61f4ca590652d8` and the history JSON SHA-256 is `bd3e0039384916414477a24d94ff845b41867e22c5c1a23906f089e3e7ac6e46`. Epoch 2's unsaved work would be recomputed if resumed. No 010 classification result exists. The short-profile runtime estimate above is superseded by this actual full-epoch observation.

The queue refuses changed source fingerprints, PID reuse, unsuccessful predecessor completion, or missing expected outputs. Running this coordinator requires the host process namespace, as did the previous coordinators. Its launch receipt records the exact command and process identities. Do not start another copy while it is active.

The successor manifest SHA-256 is `fc900ab16934bc6904be68f11b14743baa254987d768a97238e47486145d8e9a`; its new 008 source map SHA-256 is `d7dc9a47408c0843a5bfbcb2e9af3a937645fde596a3dbc79cd3041f56e5d663`. It verifies the original frozen implementation plus the new `scripts/run_xecg_adaptation_mb4.py` entry point (SHA-256 `d24b074f93c06ce475787cb4f9349a0931a47bc82472011bd5f452209ed6544a`). The original coordinator source is reused unchanged. The successor launched at 14:50:59 UTC with `.venv-pretrained/bin/python -u -m scripts.run_priority_queue --manifest outputs/experiment_queue_recovery_008/queue.json --manifest-sha256 fc900ab16934bc6904be68f11b14743baa254987d768a97238e47486145d8e9a`. The manifest verification command with `--check` passed; the 008 runner's `--stage check --device cpu --threads 1` confirmed the audited 56,875-record pool and microbatch 4 before launch. All four V100 profile arms passed, with peak allocations of 9.16–9.19 GB and checkpoint roundtrips; the [profile receipt](../outputs/experiment008_vision_ssl/profile_cuda.json) SHA-256 is `2de0d0497c2a37178153197f7af45efd23bbefb46d76952077360afb393048ef`. The coordinator launched the 008 `all` stage at 14:54:47 UTC. Profile extrapolation estimates about 22.0 hours for the four SSL arms and 5.65 more hours for supervised transfers; this is an estimate, not a performance finding.

## Current corrected queue and recovery

The most recently completed immutable manifest is now `outputs/experiment_queue_016/queue.json`, SHA-256 `a6d4d2aa4e1a3fcff0a67b11b4889ce9de4759600ddc92bab2e56f16586c8c32`. Its `--check` passed before launch, and [status](../outputs/experiment_queue_016/status.json) ended `complete` with return code zero at 04:26:41 UTC on 25 September. The source map SHA-256 is `faec5e00f589f2d557365a03e88c3899f596f40f98ab9cf0ee43012c032e5f98`. Its predecessor was the completed 017_v2 manifest, SHA-256 `91c19b6243629b63af8ce901ba5d1c0d4d3b7a3acb063aae0e5c2fc36c87f8eb`; that source map SHA-256 is `cf4daac96656191c2af9f177d6306b075c467402c5fd6f004a4e1f962013d8db`. The v2 CPU receipt SHA-256 is `b10749d8612565ee515869b105cb58b527155c8491e51fabf01bb4b589e698bb`.

`handoff_priority_queue.py` (removed 30 September 2026) passed six focused tests, verified 015's completion artifacts, and signaled the old coordinator only after it switched to 017. The old 017 child was stopped before its profile produced any output. The original sources and manifests remain unchanged. Handoff helper SHA-256 observed at launch: `b94e64293a6af84627a582696caec214d3e92f8b760332f975f965bfa3b66db4`. The exact launch-time helper is archived as `outputs/experiment_queue_017_v2/handoff_source_at_launch.py`, matching the recorded hash. The current helper received subsequent guard improvements; these did not alter experiment sources. The exact detached commands and PID identities are in `outputs/experiment_queue_017_v2/{handoff_watch.json,launch.json}`; the log confirms successor launch. Do not repeat the watcher or launcher while a coordinator is alive.

Historical 017 recovery command (completed; do not relaunch):

```bash
.venv-pretrained/bin/python -u -m scripts.run_priority_queue --manifest outputs/experiment_queue_017_v2/queue.json --manifest-sha256 91c19b6243629b63af8ce901ba5d1c0d4d3b7a3acb063aae0e5c2fc36c87f8eb
```

The 017 profile and train stages used the historical `scripts.run_cpc_morphology017_v2` entry point. Each five-epoch comparison was resumable. The completed 016 successor used the current `scripts.coordination.run_priority_queue` entry point and default `.venv`; its exact command is in the JSON catalog and its completion receipt. Check live status and the shared GPU lock before any future launch.

## Public readouts and screening studies, 28-29 September 2026

Complete. Each study had a protocol frozen before any score, ran once, used no PTB-XL test ECG and no EchoNext
test ECG, and has its own results report. Combined summary: [findings of 28 September](findings-2026-09-28.md).
Candidate ranking and follow-ups: [experiment priorities](experiment-priorities.md) and
[backlog](experiment-backlog.json).

| Experiment | Protocol | Results | Outputs |
| --- | --- | --- | --- |
| 022 SPH external readout (v3) | [protocol](experiment-022-sph-external-readout.md) | [results](experiment-022-sph-external-readout-results.md) | `outputs/experiment022_sph_external_v3/` |
| 023 EchoNext structural heart disease (v3) | [protocol](experiment-023-echonext-readout.md) | [results](experiment-023-echonext-readout-results.md) | `outputs/experiment023_echonext_v3/` |
| 024 Embedding geometry and prototypes | [protocol](experiment-024-embedding-geometry.md) | [results](experiment-024-embedding-geometry-results.md) | `outputs/experiment024_embedding_geometry_v1/` |
| 025 Label efficiency of frozen encoders | [protocol](experiment-025-label-efficiency.md) | [results](experiment-025-label-efficiency-results.md) | `outputs/experiment025_label_efficiency_v1/` |
| 026 One-class normal-manifold screening | [protocol](experiment-026-normal-manifold.md) | [results](experiment-026-normal-manifold-results.md) | `outputs/experiment026_normal_manifold_v1/` |
| 027 Calibrated 95%-sensitivity threshold on SPH | [protocol](experiment-027-calibrated-threshold.md) | [results](experiment-027-calibrated-threshold-results.md) | `outputs/experiment027_calibrated_threshold_v1/` |
| 028 Echo label efficiency | [protocol](experiment-028-echo-label-efficiency.md) | [results](experiment-028-echo-label-efficiency-results.md) | `outputs/experiment028_echo_label_efficiency_v1/` |

Headline: frozen ECG-JEPA and xECG beat our CPC everywhere (PTB-XL 0.959 and 0.962 against 0.921; SPH 0.911
and 0.915 against 0.876; EchoNext 0.823 and 0.838 against 0.812), but a PTB-XL 95%-sensitivity threshold
reached only about 92% at SPH.

Next, per the user (29 September 2026): after the Ningbo EDA, cleaning, Challenge label mapping and cohorts v2
(branch `eda/ningbo-v1`), extract Ningbo features once on the GPU and rerun the experiments Ningbo could
affect. These are 022, 024, 025, 026 and 027 (`rerun_of` in the backlog), then the multi-source
leave-source-out study. 023 and 028 use echo labels and are not rerun. New ideas are written into the backlog
and run early only when they can run in parallel without contention.

## Multi-hospital reruns and local adaptation, 29 September 2026

Complete. Ningbo, Chapman, Georgia and CPSC features were extracted once
(`outputs/features_challenge_v1/`, [extraction code](../scripts/data/extract_challenge_features.py)) and split
by record with a frozen seed ([Challenge split v1](challenge-splits-v1.md)). Each study froze its protocol
before any score and reproduced its predecessor exactly. None read the PTB-XL test set or the Challenge test
groups.

| Experiment | Question | Result | Outputs |
| --- | --- | --- | --- |
| 027b ([protocol](experiment-027b-multisource-calibration.md), [results](experiment-027b-multisource-calibration-results.md)) | Does calibrating on several hospitals make the 95% threshold transfer? | No; hospitals miscalibrate in opposite directions | `outputs/experiment027b_multisource_calibration_v1/` |
| 026b ([protocol](experiment-026b-multisource-normal-manifold.md), [results](experiment-026b-multisource-normal-manifold-results.md)) | Normal reference fitted on several hospitals | Adopted; xECG SPH 0.858 to 0.878 | `outputs/experiment026b_multisource_manifold_v1/` |
| 022b ([protocol](experiment-022b-multisource-readout.md), [results](experiment-022b-multisource-readout-results.md)) | Readout trained on PTB-XL plus Challenge labels; leave one source out | Adopted; xECG SPH 0.915 to 0.939; Ningbo helps | `outputs/experiment022b_multisource_readout_v1/` |
| 025b ([protocol](experiment-025b-label-efficiency-multisource.md), [results](experiment-025b-label-efficiency-multisource-results.md)) | Label efficiency with pooled labels | Mixed for xECG; 1,000 pooled labels about equal all PTB-XL labels | `outputs/experiment025b_label_efficiency_multisource_v1/` |
| 024b ([protocol](experiment-024b-multisource-geometry.md), [results](experiment-024b-multisource-geometry-results.md)) | Do clusters follow hospital or diagnosis? | Diagnosis for JEPA and xECG; hospital still readable (AUROC 0.93-0.95) | `outputs/experiment024b_multisource_geometry_v1/` |
| 029 ([protocol](experiment-029-local-adaptation.md), [results](experiment-029-local-adaptation-results.md)) | Local ECGs needed at a new site (SPH simulated) | 95% sensitivity needs at least 45 local positives and then refers about half or more of normals; local normals add little | `outputs/experiment029_local_adaptation_v1/` |
| 030 ([protocol](experiment-030-referral-budget.md), [results](experiment-030-referral-budget-results.md)) | Threshold set by referral budget from local normals | 5% budget with 200 local normals catches 0.775; conduction findings most misses | `outputs/experiment030_referral_budget_v1/` |
| 031 ([protocol](experiment-031-hybrid-screening-score.md), [results](experiment-031-hybrid-screening-score-results.md)) | Readout plus distance-from-normal hybrid | Negative (-0.028 at 5%); helps on held-out conditions | `outputs/experiment031_hybrid_score_v1/` |
| 032 ([protocol](experiment-032-rhythm-findings.md), [results](experiment-032-rhythm-findings-results.md)) | Detectors for athlete-criteria rhythm findings | Usable at SPH (PVC 0.990, WPW 0.992, AF 0.9999, AV block 0.9999, long QT 0.934) | `outputs/experiment032_rhythm_findings_v1/` |
| 033 ([protocol](experiment-033-finding-heads-screen.md), [results](experiment-033-finding-heads-screen-results.md)) | Binary readout plus PVC/WPW heads under one budget | Adopted; composite +0.037 at 5%, binary cost 0.003 | `outputs/experiment033_finding_heads_screen_v1/` |
| 034 ([protocol](experiment-034-one-class-baselines.md), [results](experiment-034-one-class-baselines-results.md)) | Other one-class scores on the same embeddings | Mahalanobis stays; best alternatives within 0.004 | `outputs/experiment034_one_class_baselines_v1/` |
| 035 ([protocol](experiment-035-hard-subset.md), [results](experiment-035-hard-subset-results.md)) | Why pooled readouts lose on the PTB-XL hard subset | Normal-label mismatch; reweighting recovers 81% with no SPH cost | `outputs/experiment035_hard_subset_v1/` |
| 036 ([protocol](experiment-036-random-encoder.md), [results](experiment-036-random-encoder-results.md)) | Does the normal reference need pretraining? | Yes: pretrained 0.878 against random 0.686 at SPH | `outputs/experiment036_random_encoder_v1/` |
| 037 ([protocol](experiment-037-pipeline-v2.md), [results](experiment-037-pipeline-v2-results.md)) | Pipeline v2: 035 readout with 030 and 033 operating points | Adopted by rule (-0.0048 composite, fails strict non-inferiority); spec in [pipeline-v2](pipeline-v2.md) | `outputs/experiment037_pipeline_v2_v1/` |
| 041 ([protocol](experiment-041-fragment-localization.md), [results](experiment-041-fragment-localization-results.md)) | Where in the ECG is the abnormality? Per-section maps | Unsupervised map finds premature beats (+0.669 over chance) but worst section detects at 0.777; label-guided 0.926; JEPA fallback triggered | `outputs/experiment041_fragment_localization_v1/` |
| 042 ([protocol](experiment-042-lead-wave-maps.md), [results](experiment-042-lead-wave-maps-results.md)) | Per-lead maps: ECG-JEPA patches and beat-aligned waves | Beat-aligned improves on 041 (premature beats 93% against 20%; benign red 7.7% against 27%); JEPA does not; lead test not passed | `outputs/experiment042_lead_wave_maps_v1/` |
| 043 ([protocol](experiment-043-ann-heads.md), [results](experiment-043-ann-heads-results.md)) | ANN heads, attention head on JEPA tokens, tutor's CNN + transformer | xECG + JEPA logistic beats R (SPH +0.0017); MLPs and attention match; CNN + transformer below (-0.0097); no map improves on `U_B` | `outputs/experiment043_ann_heads_v1/` |
| 044 ([protocol](experiment-044-pipeline-v3.md), [results](experiment-044-pipeline-v3-results.md)) | Pipeline v3: concatenated-feature readout with 030 and 033 operating points | Adopted; composite +0.0080 [+0.0040, +0.0120] at 5%, normals referred 5.74% against 5.45%; spec in [pipeline-v3](pipeline-v3.md) | `outputs/experiment044_pipeline_v3_v1/` |
| 045 ([protocol](experiment-045-two-layer-map.md), [results](experiment-045-two-layer-map-results.md)) | Two-layer explanation map and a detection ensemble, from saved scores | Two-layer map loses premature-beat localization; ensemble beats v3's readout (SPH +0.0038, full +0.0058) | `outputs/experiment045_two_layer_map_v1/` |
| 046 ([protocol](experiment-046-pipeline-v4.md), [results](experiment-046-pipeline-v4-results.md)) | Pipeline v4: v3 with the 045 ensemble readout | Adopted; composite +0.0121 [+0.0082, +0.0161] at 5%; "other" referred 27.5% against 24.9%; spec in [pipeline-v4](pipeline-v4.md) | `outputs/experiment046_pipeline_v4_v1/` |
| 047 ([protocol](experiment-047-explanation-rule.md), [results](experiment-047-explanation-rule-results.md)) | Explain referred ECGs with `U_B` when it is red, else attention | Negative: loses premature-beat hits (-0.136 [-0.232, -0.057]); keeps lead information | `outputs/experiment047_explanation_rule_v1/` |
| 048 ([protocol](experiment-048-pvc-switch.md), [results](experiment-048-pvc-switch-results.md)) | Switch the explanation on the xECG PVC head | Improves on `U_B`: keeps every premature-beat hit, anterior contrast +0.189 [0.067, 0.307]; sends 68 of 136 referred anterior infarcts to `U_B`; recommended rule | `outputs/experiment048_pvc_switch_v1/` |
| 049 ([protocol](experiment-049-focal-switch.md), [results](experiment-049-focal-switch-results.md)) | Label-free focal-versus-diffuse switch | Negative (-0.405 against the PVC switch); normals' threshold too high; PVC switch with `combined_50` referral best so far | `outputs/experiment049_focal_switch_v1/` |
| 050 ([protocol](experiment-050-attention-finding-heads.md), [results](experiment-050-attention-finding-heads-results.md)) | Attention PVC and WPW heads on ECG-JEPA tokens | Negative: PVC below the xECG head at SPH (-0.0068); PVC map does not keep localization (-0.045 [-0.148, +0.054]) | `outputs/experiment050_attention_findings_v1/` |
| CPU timing ([note](inference-timing-cpu.md)) | Pipeline v4 plus the explanation without a GPU | 0.33 s per ECG at 4 threads, 0.83 s at 1; 2.3 GiB peak; outputs equal the GPU path | `outputs/inference_timing_cpu_v1/` |

Next, ranked in the [backlog](experiment-backlog.json) and [priorities](experiment-priorities.md):
`referral_budget_operating_point`, then `hybrid_screening_score` and `rhythm_findings_detector`, with the
novelty search for the normal-reference finding. `final_frozen_test` runs once, after the operating-point rule is
chosen. `student_criteria_label` and `young_subgroup_readout` wait for the cardiologist meeting;
`enriched_positive_sensitivity` waits for clinic ECGs. SPH has now been read by 022-029 and serves as
development data. An [independent audit](audit-2026-09-30.md) on 30 September 2026 found every protocol committed before its run
and every recomputed headline number correct. It also found that the PTB-XL test set was already read by 001-008,
and that cohorts v2 contain Challenge test-group records.

## Research branches

The [NLP/genomics architecture shortlist](cross-domain-architecture-candidates.md) supplies the rationale for authorized Experiments 011–013. These are independent model families, not xECG modifications. They have no measured project results yet. The additional [vision architectures](cross-domain-architecture-candidates.md) and [xECG modifications](xecg-next-experiments.md) remain research candidates outside this accepted queue.

## Resume after losing conversation context

1. Read `AGENTS.md`, this file, the JSON catalog and the selected experiment's linked plan. User authorization for 011–017 is already recorded; 011 is the next architecture implementation. The subsequent 25 September request to continue experiments resumed 011 implementation after the refactor pause; it did not resume deferred 008 or 010.
2. Read live coordination files and inspect the actual process identities/GPU in the host namespace. An old PID or Markdown status does not prove that training is active. The launch receipt contains the exact current command.
3. Preserve the downloader, legacy runner and frozen sources. Start the next unfinished implementation in new modules. The current pool is 56,875 train ECGs; supervised budgets are the fixed 15,360 and 1,518 labels, with separate development/calibration/test patients.
4. Before making a new job runnable, freeze its protocol and source revisions, test recurrence/gradient correctness and resume behavior, and create a verified successor executable manifest with the correct predecessor. Keep a real GPU profile as a gate before full training. Do not append to or rewrite a live frozen manifest in place.
5. Update this file and the JSON catalog as implementation, verification, launch and results arrive. Record concrete commands and artifact paths; missing commands currently mean an implementation task, not a runnable experiment.

### Recovery pointers

- Completed tokenization: `outputs/experiment006_cpc_tokenization/coordination.json` and `runner.log`.
- Interrupted full 010 queue and epoch-1 checkpoint: `outputs/experiment_queue_full_010/{queue.json,launch.json,status.json,runner.log}` and `outputs/experiment010_cpc_crosslead/native_ssl/`. Completed 010 profile: `outputs/experiment_queue_profile_010_v2/`. First failed 010 profile: `outputs/experiment_queue_profile_010/`. Interrupted 008 recovery: `outputs/experiment_queue_recovery_008/`. Historical failed queue: `outputs/experiment_queue/`.
- Per-job logs for earlier 009/007/008 stages: `<output_dir>/priority_queue.log`; interrupted full 010 log: `outputs/experiment_queue_full_010/job/priority_queue.log`.
- Completed MIMIC 200k preparation / older ECG-FM suite: `data/processed/mimic_ssl_200k/metadata.json` and `outputs/experiment003_mimic/{status.json,prepare_mimic_200k.log}`. The legacy experiment runner remains stopped after the refactor pause; do not restart it as a side effect of a new queue.
- Environment: default `.venv` via direct `uv` commands; one V100 16 GB; shared lock `/tmp/ecg_project_gpu.lock`. Historical launch receipts retain `.venv-pretrained` paths.
- Existing completed comparisons: `outputs/experiment004_cpc_40k/report.md` and `outputs/experiment005_cpc_word2vec/report.md`. Interpret them as exploratory PTB-XL proxy results, not validation of healthy status or student referral decisions.

## Unsupervised localization follow-ups, 2 October 2026

The user authorized autonomous localization research with subagents on the existing PR #61. Experiments
052 and 053 committed their prospective protocols before scoring. They run on CPU in isolated worktrees, fit no pathology
labels, and use local training/development ECGs. Each must reproduce the previous `U_B` results, compare
fixed-size marks, and write its results report from live outputs. The closed test sets stay closed.

| Experiment | Question | State |
| --- | --- | --- |
| 051 ([protocol](experiment-051-beat-sum-map.md), [results](experiment-051-beat-sum-map-results.md)) | Beat-score sums | Complete; neither map improved on U_B |
| 052 | Does comparison with the other beats in the same raw ECG improve focal localization? | Implementation; protocol committed |
| 053 | Do training-normal calibrated lead/wave scores improve persistent lead localization? | Implementation; protocol committed |

The existing PVC overlap target is an automatic proxy. Fixed wave windows can attribute an early beat to
the preceding beat's T window; prospective common-support metrics and known-location waveform alterations
will check localization without accepting a gain from wider highlights. Clinical region correctness still
needs independent cardiologist marks or expert beat annotations.
