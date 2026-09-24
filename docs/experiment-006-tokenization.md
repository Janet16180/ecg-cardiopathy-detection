# Experiment 006: learned targets and causal ECG chunks

**Protocol date:** 24 September 2026. **Status:** complete at 03:18 UTC. All five SSL arms and ten supervised transfers finished. Learned chunks beat fixed chunks at both label budgets, but ordinary native-grid continuation remained stronger. See the [results](../outputs/experiment006_cpc_tokenization/report.md) and [paired comparisons](../outputs/experiment006_cpc_tokenization/paired_comparisons.json).

**Prelaunch verification:** nine tokenizer/preparation tests and five model tests passed. Synthetic execution covered all five arms, both label budgets, the epoch-five teacher/codebook refresh, calibration/evaluation, and completed-run reuse. A real-cache preflight verified waveform/beat identities and the complete pretrained CPC checkpoint. Beat metadata covers 60,641 recordings; 56,875 belong to the training pool. Source snapshots, package versions, and the coding agent's verification receipt are saved under `outputs/experiment006_cpc_tokenization/provenance/`. Synthetic metrics are not study results.

## Question and comparisons

Can a small change to the units modeled by compact CPC improve transfer, while retaining its causal CNN and two-layer 256-wide GRU?

| Arm | Change | Primary control |
| --- | --- | --- |
| `continuation` | Ten more epochs of ordinary native-grid CPC | Common reference |
| `clusteraux` | CPC plus prediction of learned future cluster IDs | `continuation` |
| `fixedchunk` | Mean CNN features within fixed 16-token chunks; GRU updates at chunk ends | Chunk-family reference |
| `beatchunk` | Chunk ends follow a causal heartbeat detector, with bounded fallback | `fixedchunk` |
| `learnedchunk` | Signal-dependent, learned chunk ends using straight-through gradients | `fixedchunk` |

All arms begin from the **final fixed-budget Experiment 004 CPC encoder**, chosen by protocol, without consulting its downstream test ranking. Load the epoch-20 `epoch_state.pt["model"]` CNN, GRU and all three trained CPC prediction heads into every arm; record the full checkpoint SHA256. Verify that its encoder tensors match the final `encoder.pt`, and record that file's hash too. Transfer weights strictly, mapping the two GRU layers to the chunk GRU cells where needed. Initialize only the new duration, boundary and auxiliary-cluster parameters. The source checkpoint is fixed by training budget, not downstream performance.

These five runs test three intervention families within a bounded budget. The cluster comparison evaluates its auxiliary-learning package; without a separate static-codebook arm, it cannot isolate the effect of refreshing a codebook. Chunk-versus-native comparisons also change temporal compression; the fixed-chunk arm is the primary control for beat and learned boundaries. All five arms use the same conservative query warmup.

## Data, initialization and budget

- Use the frozen Experiment 004 cache: **56,875 training recordings**, comprising 39,457 MIMIC and 17,418 PTB recordings. Retain the existing patient splits and training-only per-lead normalization.
- Preserve independent five-second halves at 250 Hz. CNN features have width 256, stride 16 samples and receptive field 33 samples. Each half has 79 native feature positions.
- Continue self-supervision for **10 epochs per arm**, batch 128, seed 42. Keep AdamW, weight decay 0.01 and the existing two-epoch warmup/cosine schedule at base learning rate 1e-3. Reset the optimizer for each continuation run. Keep the CNN trainable so its representations can respond to the new objective.
- Use final-budget SSL states, without development or test selection between SSL epochs. Complete every SSL arm before beginning downstream test evaluation.
- Fine-tune every encoder at both existing label budgets: **15,360** labeled training ECGs and the fixed **1,518-label** subset. Retain the 40-epoch ceiling, patience 8, development-AUROC checkpoint selection, separate calibration patients, and existing paired patient bootstrap evaluation. This makes ten downstream runs.
- Initialize the downstream binary classifier identically across arms using an independent seed/captured state; different encoder constructors consume different random draws. Downstream training uses binary cross-entropy, without the cluster or boundary-count SSL losses. Boundary behavior may therefore change during fine-tuning and should be inspected in the selected checkpoint.
- Profile real-cache execution before launch; report actual wall time, parameter counts, optimizer updates and record exposures. The sequential chunk implementation can cost more than the native GRU even with fewer state updates. Equal epochs and input exposure do not mean equal compute.

## Family 1: iterative learned targets

Predict a discrete description of a future local ECG feature, alongside the existing continuous CPC objective. Offline clustering followed by prediction and refreshed clustering is motivated by [HuBERT](https://arxiv.org/abs/2106.07447). This ECG prototype uses causal future prediction, not HuBERT's bidirectional masked speech objective.

Use **64 clusters** and a deterministic bounded sample of normalized CNN features drawn only from training recordings. Freeze the sampled recording IDs, half IDs and token indices before fitting. The inexpensive initial implementation uses 2,000 training recordings and 100 sampled tokens per recording, at most 200,000 features. Fit `MiniBatchKMeans` with batch size 4,096, `n_init=3`, `max_iter=100`, and random seed 42. Assignment uses Euclidean distance to the fitted centers; do not silently renormalize centers into a different spherical-clustering objective.

The intended two rounds are an initial codebook/teacher from the bootstrap encoder for epochs 1–5, then a refreshed codebook/teacher from the student's epoch-five CNN for epochs 6–10. Freeze the teacher CNN or cache its code IDs within each round. Assigning IDs from the continuously changing student CNN would create moving labels within the round and must be reported as a different design. Persist the teacher, centers, fit-sample identities and refresh state atomically with the epoch checkpoint so resuming cannot repeat or skip a refresh. Keep all clustering and refresh inputs within training patients.

Use a separate 256-to-64 linear classifier for each of the three future horizons, or explicitly condition a shared classifier on the horizon. An unconditioned shared classifier would receive the same context but three different desired labels. The loss is:

```text
L = L_CPC + 0.1 * mean_horizon,query CE(cluster_head_h(context_t), teacher_code_(t+h))
```

Keep horizons 4/8/12, query positions starting at index 24 in all five arms, and the original same-half candidate exclusions for CPC. No cluster labels enter the encoder as inputs; they are detached prediction targets. Target-codebook class numbers are arbitrary. At the round-two refresh, reset the three auxiliary classifiers and remove their AdamW optimizer moments; retain the encoder/CPC-head optimizer state. Use a dedicated deterministic initialization context that restores the main training RNG afterwards. Save the reset seed and completed refresh state in provenance.

For each fitted codebook, record the fit-sample label histogram, occupied-cluster count, normalized assignment entropy and largest-cluster fraction. During training, record cluster cross-entropy, CPC loss, token variance, representation cosine similarity, gradient norms and clipping frequency. Common normal morphology may dominate a small codebook; rare ectopy or subtle morphology may be merged. Better code prediction is not itself evidence of improved downstream detection.

## Families 2 and 3: actual causal chunks

The CNN remains the local waveform feature extractor. Chunking changes when the GRU updates. Let `z_t` be the original CNN token, and let `g_t` equal one when the current chunk ends **after including `z_t`**. Initialize accumulators and both recurrent states separately for each half:

```text
a_t = (1 - g_(t-1)) * a_(t-1) + z_t
n_t = (1 - g_(t-1)) * n_(t-1) + 1
u_t = a_t / n_t + W_duration * log(n_t / 16)
h_candidate = two_layer_GRU_update(u_t, h_previous)
h_t = g_t * h_candidate + (1 - g_t) * h_previous
```

Use a shared-form `Linear(1,256,bias=False)` duration projection in all three chunk arms, initialized to zero. It makes chunk duration available without warping each beat to a common time length. Copy the bootstrap two-layer GRU weights exactly into equivalent GRU cells and preserve interlayer dropout 0.1. The update/hold rule applies to **both** recurrent layers.

At forward evaluation, the GRU therefore has one state update per actual chunk. Hold the most recently completed context on the native time grid between emissions. A vectorized implementation may still calculate discarded candidate states at every grid position to support straight-through training. Report this explicitly: fewer logical states do not establish a speedup.

For classification and pooled diagnostics, pool **actual emitted chunk states**, including the final remainder: compute a mean weighted by emission gates and a maximum over hard-emission positions, concatenate these 256-wide statistics, then average the two half representations. This excludes pre-emission zero states and preserves a maximum from an occasional chunk. Mean pooling weights chunks equally; duration is already an explicit GRU input. Learned-gate mean pooling retains straight-through gradients, while the maximum uses a hard emission mask. Use this same readout for all three chunk arms. The native/cluster arms retain their original native-grid pooling, so comparisons across these families also include this readout change.

Use the **original unpooled CNN tokens as future CPC targets** in every chunk arm. This avoids introducing a target chunk that contains part of the query's past. Keep horizons 4/8/12, temperature 0.1 and the original same-half negatives. Skip the first **24** query positions in **all five arms**, so a first completed chunk is available and the prediction head does not normalize an all-zero initial context. This common valid-query mask keeps predictive supervision exposure matched. It is a deliberately longer warmup than Experiment 004; compare against the new continuation control.

A held context can serve several consecutive native-grid queries with different future targets. This introduces variable context age between emissions, a deliberate consequence of the compressed sequence. A negative chunk result may reflect this forecasting/compression tradeoff rather than an inherently bad boundary policy. The fixed-chunk control shares this mechanism; inspect time since the last emission alongside chunk lengths when interpreting results.

Every chunk arm forces an emission at the half's final position. The beat and learned arms also cap chunks at 24 tokens (approximately 1.536 seconds), making a missed boundary a bounded loss of timing resolution. Record forced boundaries separately from detected/learned boundaries. No accumulator, hidden state or boundary-detector state crosses the half boundary.

### Fixed chunks

End a chunk after every 16 CNN tokens, plus the final remainder. The 79-token half therefore has five chunks. This is the architectural control for replacing native-grid context updates with chunk summaries; it is not a presumed physiological beat length.

### Heartbeat-aligned chunks

Use the implementation's fixed heuristic detector: canonical leads II and V2; causal second-order 5–18 Hz Butterworth bandpass at 250 Hz; envelope given by the larger absolute filtered amplitude at each sample; threshold `max(0.045 mV, 2 * preceding_EWMA)` with EWMA update coefficient 0.01; local-maximum confirmation 16 samples later (64 ms); and a 75-sample (300 ms) refractory period. The local maximum may inspect 16 samples on either side of its candidate because the event is emitted only after the confirmation delay. All constants are frozen before downstream evaluation. Its decision at native grid position `t` may use resampled waveform samples only up to that position's endpoint. Use only past samples for threshold adaptation. If a local maximum requires confirmation samples, emit at the **confirmation time**, never retroactively at the earlier peak time. Map to the first grid endpoint at or after that confirmation. Detector code, lead choice, refractory period, warmup, threshold constants and fallback counts must be recorded in the implementation snapshot.

Do not use full-half percentile/median thresholds, globally ranked peaks, full-record lead selection, whole-record RR normalization, or a detector that backdates outputs after inspecting future samples. These can encode future waveform information in an apparently causal query. Boundaries are heuristic beat alignments, not clinical annotations. Low-quality recordings, inverted/low-amplitude signals and ectopy can cause errors; report per-source detection and forced-fallback rates. The bounded causal timeout remains part of the arm, rather than excluding difficult recordings after looking at results. Never switch an entire half to fixed boundaries because its eventual detection count is low: that would make early boundaries depend on future events. A final low-detection flag is diagnostic only and must not alter past gates.

Require at least four native tokens between emitted boundaries, except the forced final remainder. Earlier confirmed events are ignored rather than retroactively moving an existing boundary. Intervals between accepted confirmed events define variable-duration chunks. Their means may discard narrow within-beat morphology, so keep the local CNN and preserve original-token future targets. This tests whether physiological chunk boundaries help the context model; it does not establish that an averaged beat is a complete clinical representation.

### Learned boundaries

Use a small gate conditioned on the current causal CNN feature and elapsed chunk length:

```text
p_t = sigmoid(w dot z_t + b + (n_t - 16) / 4)
g_hard = 1[p_t >= 0.5]
g_ST = g_hard + p_t - stop_gradient(p_t)
```

Initialize `w` and `b` to zero: hard forward boundaries then match the fixed-16 control. Force no emission before four tokens, and force emission at 24 tokens or the half end. Forced decisions are constants rather than learned gate decisions. Use hard thresholds during both training and inference; no stochastic training/inference routing mismatch is introduced. The gate can move a boundary in response to waveform features while the elapsed-length prior discourages pathological lengths.

Add the small count regularizer below only to this arm, where `N` is the sum of straight-through gates in one half, including its forced final emission:

```text
L = L_CPC + 0.1 * mean_halves(((N - 5) / 5) ** 2)
```

The initial fixed-16 route has five emissions and zero count penalty. Straight-through gradients are a biased estimator through discrete decisions; this is a practical prototype, not an exact optimizer or a reproduction of [H-Net](https://arxiv.org/abs/2507.07955). Log mean hard counts, mean chunk lengths, forced ends, and boundary disagreement with the fixed route, together with global gradient norms and clipping frequency. Gate-probability and per-parameter gradient inspection are follow-up diagnostics if these reveal instability or degenerate routing. A model whose boundaries remain fixed is a valid negative outcome. A model emitting only at forced limits has not demonstrated useful learned segmentation.

## Raw-support and leakage checks

For current preprocessing, a native CNN target spans 33 resampled samples. The anti-alias filter has future support equivalent to ten resampled samples on either side. Expressing raw-signal support in 250 Hz sample units, a query at index `t` can use samples through `16*t + 10`; the earliest support of the horizon-four target is `16*(t+4) - 32 - 10 = 16*t + 22`. There is therefore a positive separation of 12 resampled-sample intervals (48 ms) at the shortest horizon. Chunking can extend the query's **past** support without closing this future gap, provided its boundary decisions never inspect future waveform samples.

Required implementation checks are finite loss/backward behavior for all five arms; strict CNN/GRU weight transfer; fixed and initially learned boundary agreement with dropout disabled; both recurrent states held between emissions; forced maximum/end boundaries; and input-perturbation checks establishing that later waveform samples cannot change earlier gates or contexts. Treat confirmation latency and preprocessing support as part of this check. Never construct a query from a completed future beat and label it with the beat's earlier start time.

Verify that cluster fitting and any refresh read training identities only; future teacher targets are supervision, never query inputs. Checkpoint tests must cover the codebook round, auxiliary head state, optimizer, RNG states and refresh completion as well as ordinary epoch resumption. Logging and fitting should preserve training dropout/data-loader random streams.

## Reporting and interpretation

The primary downstream differences are `clusteraux - continuation`, `beatchunk - fixedchunk`, and `learnedchunk - fixedchunk` at each label budget. Include the native continuation in the report to show whether a chunk family helps overall. Compare calibrated AUROC, average precision, sensitivity/specificity at the frozen calibration threshold, Brier score and available patient-bootstrap intervals. Show runtime and memory beside predictive metrics.

Use development patients for the already specified supervised selection; calibration patients fit calibration and threshold; test patients provide the final exploratory comparison. This test cohort has been examined in earlier work, and one training seed does not quantify retraining variability. No result here establishes referral performance among university students. No novelty, full H-Net reproduction, validated physiological segmentation, or compute advantage is assumed.

The scientific protocol and implementation have been reconciled by static review. Source/data hashes and the coordinator's execution tests provide the launch record; any subsequent change to these constants, refresh rules or pooling details requires an explicit protocol update.

The intended study command is `.venv-pretrained/bin/python -m scripts.run_cpc_tokenization --stage all --variant all --labels all --device cuda --threads 1 --ssl-epochs 10`. Outputs are under `outputs/experiment006_cpc_tokenization/`; beat metadata is under `data/processed/cpc_beats_40k/`. The root coordinator owns profiling, execution and launch receipts. This research/design task does not itself launch GPU work.
