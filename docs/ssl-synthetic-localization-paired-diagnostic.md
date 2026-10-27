# Saved map diagnostic of control flags and phase alignment

18 October 2026. The user authorized continuing after the matched-control training result. Freeze this protocol before computing new diagnostic aggregates. The parent study completed and its independent audits passed, but its complete advancement rule failed on R selectivity and S minus R nuisance-control flags. Those thresholds, models, numbers and both failed gates remain unchanged.

## Questions and scope

How many SSL nuisance-control flags already occur on the same patient's unchanged ECG, and how many appear or disappear with the fixed nuisance bundle? How much does R's apparent localization reflect an output pattern aligned with algorithmic heartbeat windows already present before the introduction? Do the saved optimization losses describe distinct training trajectories?

This is a descriptive saved-output diagnostic, with no new advancement criterion. It cannot determine internal causal feature reliance, isolate individual nuisance components, label native pathology or turn the failed parent study into a pass. No fitting, new inference, checkpoint deserialization, waveform access, patient selection, threshold changes or unused test access occurs. All work stays local without pushes or PR updates.

## Admission and identities

Require the unchanged complete parent evaluation, independent training audit and independent metrics audit, with explicit SHA-256 anchors in the configuration. Both audits must pass with zero mismatches and bind the exact evaluation summary. Freshly compare all eight independently recomputed parent summary blocks, thresholds and all nine decision gates against the parent evaluation at absolute tolerance1e-12. Preserve the full failed conjunction and original simulator failure. Bind new source, protocol, configuration and every consumed metadata/mask/map/loss input before and after execution.

Read only saved R/S float32 native probabilities shaped (768,12,2500), bool support/candidate masks shaped (2768,12,2500), frozen case/selection/qualification metadata and saved training losses/progress. Keep raw headers/data, processed background/signal/baseline/edited waveforms, clinical annotation payloads, model checkpoints and protected cohorts closed. Check unused waveform/checkpoint digest declarations through the existing source-bound receipts, explicitly marking these scope exceptions; never open those payloads merely to hash them.

Calibration global case indices2000–2255 map to saved rows0–255; evaluation indices2256–2767 map to rows256–767. Cases are ordered focal, repeated, unchanged(sham), nuisance, four per patient. Use within-role subject IDs separately from global indices; assert all64/128 patients, complete variants, correct roles, no duplication and native mask/map geometry. Positive baseline/control equality is checked through their frozen input hashes and nuisance metadata; this diagnostic does not independently regenerate waveform truth.

## Fixed control analyses

For each arm and role, retain the original arm-specific frozen threshold. A flag is maximum native probability strictly greater than that threshold. Reproduce every existing calibration/evaluation flag, calibration-map slice digest and calibration threshold; do not refit a new threshold.

Report the paired unchanged-versus-nuisance2x2 table: neither flagged, unchanged-only, nuisance-only, both. Include all patients, flagged counts/rates, persistent and both flip counts, and paired nuisance minus unchanged flag-rate difference with whole-patient interval. Report the analogous R-versus-S2x2 table separately for unchanged and nuisance controls, including S-only/R-only/both/neither and S minus R rate difference. Calibration rows are descriptive fitted-data checks, not independent error estimates.

For all control patients and each within-arm transition category, describe the native maximum, signed threshold margin (maximum minus threshold), and nuisance-minus-unchanged maximum change. Report mean/median/min/max and 5th/95th percentiles with NumPy linear quantiles, retaining category counts and explicit null summaries for empty categories. No post hoc score thresholds or patient deletion are introduced.

The observed parent S8 nuisance flags versus S7 unchanged flags give a net difference of one patient, whereas the failed parent veto compares S8 with R1 nuisance flags, a difference of seven. Report both comparisons and both directions of paired flips. Do not interpret a net difference or the cross-arm veto as proof that noise alone caused each flag: the authored nuisance bundle combines gain, drift and noise, and native findings remain unknown.

## Fixed probability level and ranking analyses

For each positive evaluation case and arm, let T be the exact introduced support, U=candidate minus T, and O=outside candidate. Use its matched same-patient control map as the baseline, selected by the frozen nuisance flag and exact global control index. Report baseline and edited mean probability in T/U/O, their respective changes, baseline and edited T-minus-U contrast, and change of that contrast. Also report full-map mean absolute probability change and maximum absolute probability change. Include baseline-site AP, edited AP and AP change using the original full-grid tied-score convention, checking the parent case metrics rather than changing their definition.

For unchanged/nuisance controls, report mean probability inside all algorithmic candidate windows C and outside C, and C-minus-outside contrast, for each arm and role. Compare these absolute levels with the positive-case AP and probability changes. A positive phase-window output contrast is consistent with phase alignment; it cannot establish which internal feature produced it. High baseline AP can coexist with low absolute probabilities and little response to the added intervention.

Keep all cases. Report each positive mode separately and patient-macro averages with focal/repeated equal within each patient. For probability levels/changes/contrasts and AP comparisons use shared whole-patient interval helpers, 2,000 draws from default_rng(20261026), percentile2.5/97.5 and skipped-draw counts. R/S contrasts use paired patient draws; within-arm descriptive mean intervals compare with zero. No cell/beat bootstrap, subgroup search or sign/significance filter. Control phase summaries and paired flag/max changes use patient-level draws separately within each role. Store patient-level diagnostic rows only locally; Git receives aggregate reports.

## Training losses and resources

Reconcile all600 finite losses per arm with the saved progress log and training receipt. Report six fixed blocks of100 updates, each block's mean/std(ddof0)/min/max, and first/last10-update means. Compare R/S on the matched schedule descriptively, without treating dependent steps as independent patients or selecting another checkpoint. Loss alone does not establish localization or identify an optimizer mechanism.

Run on one numerical CPU thread with CUDA hidden. Limit the main diagnostic to120 seconds CPU AND monotonic wall time, the independent check to120 seconds, combined240 seconds, with a2-second failure-output reserve. Count prior cumulative545.144970413 CPU seconds and prior paired-stage111.012039341 seconds from the complete metrics audit; preserve the parent's1800-stage/14400-global CPU ceilings and4GiB stream cap. Keep combined new diagnostic/audit outputs within16MiB. Enforce live deadlines and actual storage at stage boundaries; refuse overwrite, retain failures and their measured costs. No libraries/downloads or duplicate map/waveform banks are required.

Commit all source/protocol/configuration before execution, integrate into the coordinator worktree and obtain source review before running once. Freeze the independent audit script before its execution; verify with its own arithmetic rather than production diagnostic helpers, retaining waveform/hash scope exceptions. The running agent writes a results report from outputs and the coordinator updates both queues/backlog/ranking. Follow-ups are hypotheses ranked from this diagnostic; no new training or changed parent decision is launched by this protocol.
