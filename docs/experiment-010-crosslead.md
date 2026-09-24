# Experiment 010: future prediction across disjoint ECG lead groups

**Protocol date:** 24 September 2026. **Status:** deferred for cost after an interrupted full run. All three V100 profile arms passed at 15:17:33 UTC; a separate full-run manifest launched at 15:18:32 UTC under coordinator PID 80086. The first profile-only run had stopped at a bitwise CUDA save/restore equality check: the maximum difference in the first failing tensor was `1.49e-8`. The successful profile used a new entry point that retained the frozen objective/profile code and verified model and AdamW states with `atol=1e-7`, `rtol=1e-5`, logging maximum absolute differences. The full run used the original frozen runner. The short profile missed sustained random cache I/O: native SSL epoch 1 took 15.58 minutes, and epoch 2 remained incomplete after four more minutes. Continuing all 30 SSL epochs and six transfers was projected to exceed the user's two-hour cost limit, so the coordinator was stopped at 15:42:23 UTC. The native epoch-1 model/optimizer/RNG checkpoint is preserved; no 010 classification result exists. Fourteen focused CPC tests, strict real-cache/bootstrap checks and a real two-record CPU optimizer update passed previously. See the [verification receipt](../outputs/experiment010_cpc_crosslead/provenance/verification.json), [profile completion](../outputs/experiment_queue_profile_010_v2/job/priority_queue_completion.json), [full-run launch](../outputs/experiment_queue_full_010/launch.json) and [interruption status](../outputs/experiment_queue_full_010/status.json).

The V100 profile estimated 65.06, 62.50 and 63.46 seconds per SSL epoch for native, withinlead and crosslead, with about 3.04 GB peak allocated GPU memory and numerical model/AdamW save/restore agreement. Ten epochs across all three arms initially projected about 31.84 minutes of SSL work; six supervised transfers added roughly 4.29 minutes by extrapolating Experiment 004's cumulative histories. Those short-profile figures were superseded by the actual full-epoch I/O measurement above. The native epoch-1 checkpoint is `outputs/experiment010_cpc_crosslead/native_ssl/epoch_state.pt` (SHA-256 `808363dbcbfabecfe42a49f2b5e42c4a10adfb27dc833d2e7d61f4ca590652d8`). It was loaded and checked to contain the epoch, model, optimizer, RNG, fingerprint and matching history. Epoch 2 had not saved and would be recomputed on resume. These are runtime and progress records, not clinical or classification results. The profile receipts are `outputs/experiment010_cpc_crosslead/profile_{native,withinlead,crosslead}.json`.

## Hypothesis and fixed comparisons

Does asking a compact CPC context to predict future features from a different lead group improve transfer beyond learning from the same masked lead inputs? Shared electrical dynamics could provide useful supervision. This is a hypothesis about representation learning, not a claim that partial lead sets preserve every diagnosis or that cross-lead prediction is methodologically novel.

Keep the existing 1,041,024-parameter encoder from [Experiment 004](experiment-004-cpc.md): four causal convolution blocks, 256-wide features, and two causal GRU layers of width 256. All three arms continue ordinary full-twelve-lead CPC. A shared encoder also processes two fixed partial-lead views; the primary comparison changes only which view supplies auxiliary future targets.

| Arm | Total SSL loss | Comparison |
| --- | --- | --- |
| `native` | `L_full + 0 * L_within` | Matched continuation and execution reference |
| `withinlead` | `L_full + 0.1 * L_within` | Effect of the masked-view auxiliary package |
| `crosslead` | `L_full + 0.1 * L_cross` | Effect of changing auxiliary target lead group |

The primary comparison at each label budget is **crosslead minus withinlead**. Report withinlead minus native and crosslead minus native as secondary comparisons. An improvement over native alone cannot identify a cross-lead effect.

## Inputs, lead groups and initialization

Use exactly the frozen 004 pool of **56,875 training ECGs**: 39,457 MIMIC and 17,418 PTB records, with existing source-qualified patient identities, exclusions and manifest hashes. Other rows in the full cache are held out and must not enter SSL. Inputs retain canonical lead order, 250 Hz sampling, and independently preprocessed five-second halves.

Load the exact fixed training-only per-lead mean and standard deviation used in 004. Normalize the full twelve-lead waveform first, then multiply excluded channels by zero. Do not zero raw physical signals before normalization. Zero in a partial view denotes an omitted channel; the fixed channel positions determine the group without new mask embeddings.

| View | Present leads | Canonical zero-based indices |
| --- | --- | --- |
| Full | All twelve | `0..11` |
| A | I, V1, V3, V5 | `0, 6, 8, 10` |
| B | II, V2, V4, V6 | `1, 7, 9, 11` |

The partial views omit III, aVR, aVL and aVF; these algebraically derived limb leads could otherwise provide direct relationships across groups. The full view retains them. Each partial view has one independent limb lead and three interleaved chest leads. The split is fixed before results; this first experiment does not establish robustness to every possible lead partition. Cross-lead correlations, shared artifacts and easy rhythm/phase cues can still contribute to the objective.

Every arm strictly loads the **completed final epoch-20 ordinary CPC state from Experiment 004**, including its encoder and three trained prediction heads. Verify the checkpoint's completed budget, configuration, full SHA256 and encoder agreement with `encoder.pt`. Do not choose a bootstrap state using downstream test results, and do not substitute an Experiment 005 or 006 winner.

Create separate A-query and B-query auxiliary head sets, each containing three bias-free `Linear(256,256)` heads. Initialize both sets by copying the original three trained horizon-specific CPC heads. All arms contain the same 1,630,848 pretraining parameters; the six extra heads account for 393,216. They are removed for downstream transfer.

## Exact objective and prediction integrity

Execute three encoder passes in the same order in every arm: full, A, B. Denote their token/context pairs by `(z,c)`, `(z_A,c_A)`, `(z_B,c_B)`. The same encoder weights are shared across all passes, but recurrent states reset for each half and each pass. With the existing `cpc_loss(tokens, contexts, heads)`:

```text
L_full = CPC(z, c, heads_full)
L_within = (CPC(z_A, c_A, heads_A) + CPC(z_B, c_B, heads_B)) / 2
L_cross = (CPC(z_B, c_A, heads_A) + CPC(z_A, c_B, heads_B)) / 2
```

Keep horizons 4/8/12, temperature 0.1, normalized dot products, queries beginning at token index 3, and the original same-half candidate mask. Only the actual future position is positive. Other positions in the same target half are negatives except those within three tokens of the positive. Neither a simultaneous token match nor a pooled same-record agreement loss is introduced. Gradients flow through target tokens, as in ordinary CPC; there is no stop-gradient target, EMA teacher or extra reconstruction objective.

Changing the target lead group does not alter temporal support. The CNN stride is 16 resampled samples, receptive field 33, and the resampling filter contributes ten resampled samples of future support. At horizon four, the query's last raw-support position is `16*t + 10` and the target's first is `16*t + 22`, leaving 12 resampled-sample intervals (48 ms) of separation. Constant lead masks introduce no future-dependent decisions.

The native arm executes both auxiliary losses with a scalar weight of zero, including their backward graph. Its auxiliary gradients are zero; AdamW may still decay auxiliary-head weights because zero gradient tensors exist. Those heads neither contribute to the native encoder objective nor survive transfer. Identical pass order also keeps dropout RNG consumption comparable across arms. Report measured compute: executing the same graph does not guarantee exactly equal wall time.

## Fixed training and transfer budget

- Continue SSL for **ten epochs per arm**, batch 128, seed 42, using the same shuffled record order. Reset optimizer states at continuation start. Use AdamW, base learning rate `1e-3`, weight decay `0.01`, two-epoch warmup and the existing cosine schedule with floor factor 0.1. Clip gradient norm to 1.0. Preserve the epoch-indexed schedule used by the implementation rather than silently adopting a different step-level scheduler.
- Profile all three arms on the real cache before comparative training, including backward, Adam states and deterministic save/restore. Use the V100 16 GB sequentially under the shared GPU lock. Profile states must not initialize comparative runs. Batch changes require a documented common protocol update before training.
- Finish all three SSL arms before downstream test evaluation. Select the final-budget SSL encoder in each case; do not search SSL epochs using downstream development or test performance.
- Discard every CPC/auxiliary head. Transfer the encoder into the original `CPCClassifier`, whose per-half mean/max contexts are concatenated and averaged across halves before its linear binary head. **Every supervised and inference input uses all twelve leads**, with no lead masking.
- Fine-tune each encoder on the existing **1,518-label and 15,360-label** manifests: **six supervised transfers**. Retain the 004 optimizer policy, 40-epoch ceiling, patience 8, development-AUROC selection, shared classifier-head initialization, and original calibration/test partitions. The classifier architecture is identical for all arms and has 1,041,537 parameters.

This suite is substantially more expensive than the frozen mismatch readout: three ten-epoch SSL runs, each with three encoder passes, then six transfers. Matched record exposure counts a recording once per optimizer example; also report the three input-view exposures explicitly. Experiment 006's continuation skips 24 query tokens, so its state or metrics cannot replace this suite's native control.

## Verification, diagnostics and reporting

Before queue execution, verify exact bootstrap/normalization identities, group membership and post-normalization masking, strict head transfer, loss formulas, finite gradients, and gradients from each auxiliary direction. In evaluation mode, the native total must equal ordinary full-view CPC. Altering the auxiliary target routing must not alter input masks, parameter count, pass order, data order or original full-view loss for the same model state. Test that later input changes cannot affect earlier contexts and confirm exact model/optimizer/RNG continuation with dropout enabled.

Log full and directional auxiliary losses, token variance in all three views, gradient norms/clipping, finite-parameter checks, runtime, updates, record/view exposures and memory. Adjacent-token cosine similarity measures temporal similarity and is not by itself a random-record collapse test. Report source composition and inspect a fixed training-only diagnostic set; lower cross-lead loss is not proof of improved screening performance. Save configuration, source snapshots, package versions, input/checkpoint hashes and resumable RNG/optimizer state.

Fit Platt calibration and the threshold attaining at least 95% sensitivity using calibration patients only, then freeze both for test evaluation. Report observed test sensitivity/specificity, AUROC, AP, Brier score and the existing 500-resample paired patient bootstrap. Present paired crosslead-minus-withinlead differences directly, with all arm outcomes retained. Achieving the calibration sensitivity target does not guarantee the same test sensitivity.

The already examined PTB test cohort and one training seed support exploratory comparisons only. These labels are an ECG abnormality proxy: they do not establish overall health, disease absence, clinical referral performance or generalization to university students. The full-view loss preserves a learning path for lead-specific information, but it does not guarantee that the auxiliary objective will leave every relevant feature intact. Negative findings are scientifically useful and must be reported.

The completed profile command was `.venv-pretrained/bin/python -u -m scripts.run_cpc_crosslead_profile_v2 --stage profile --variant all --labels all --device cuda --threads 1 --ssl-epochs 10`. The interrupted full-stage command was `.venv-pretrained/bin/python -u -m scripts.run_cpc_crosslead --stage all --variant all --labels all --device cuda --threads 1 --ssl-epochs 10`. Both manifests and their logs remain frozen for recovery. No 010 run is active.
