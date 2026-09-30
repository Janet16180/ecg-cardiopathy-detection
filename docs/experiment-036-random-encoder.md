# Experiment 036: the normal reference on a random-weight xECG

Frozen 29 September 2026, before any feature of a random or fine-tuned encoder is extracted and before any
score of this experiment is computed. This covers the backlog item `random_encoder_control`, which the
[novelty search](normal-manifold-novelty-search.md) lists as a control a reviewer would ask for. Before this
freeze only aggregate 022b, 026b and 034 results, their receipts and metadata (record counts, fit-set
membership, window starts, checkpoint hashes) were read. A randomly initialized xECG was built on the CPU
twice from one seed to confirm that the seeded construction is deterministic; no ECG passed through it.

**SPH is development data.** Experiments 022 to 034 have read it. It is unseen by every encoder here, but it
is not a final test.

## Question

034 found that the density placed on the frozen embedding barely matters: nine alternatives to the Gaussian
did not beat it. What matters is the embedding and which normals are in the reference. The next question
follows: does the embedding need self-supervised pretraining? A randomly initialized network with the same
architecture, forward pass and pooling also maps ECGs to 1,024 numbers, and random features are known to be
a strong baseline for some signals. If a random xECG ranks SPH ECGs nearly as well, the "foundation
embedding" part of the claim is not earned.

## Encoders (arms)

All arms use the released xECG architecture (`third_party/checkpoints/xecg`, `config.json` unchanged, vanilla
sLSTM backend, `drop_path_prob` 0.5 as in `external_encoders.load_xecg_backbone`), in evaluation mode, with
the same input path and the same output: the model's own average pooling over patches (`cls_type` "avg"),
1,024 dimensions, the last block's output. Only the weights differ.

| Arm | Weights | Role |
| --- | --- | --- |
| `pretrained` | the released self-supervised weights | reference; its frozen features of 022, 026b and the Challenge cache |
| `random_36001`, `random_36002`, `random_36003` | the architecture's own initialization, seeded | **primary comparison** (mean over the three) |
| `finetuned016` | the released weights fine-tuned end to end on PTB-XL training labels in Experiment 016 | secondary, supervised |

- **Random initialization.** `torch.manual_seed(seed)` immediately before the official `xECG` class is built
  on the CPU with the released configuration; nothing is loaded into it. The seeds 36001, 36002 and 36003 are
  fixed here. The initialization is xLSTM 2.0.4's own `reset_parameters` scheme (for example, sLSTM
  recurrent kernels start at zero and the gate biases follow its power-law rule) and PyTorch's default for
  the patch convolution: the state pretraining starts from. The runner builds every random model twice and
  requires identical state-dict hashes, which it records.
- **Fine-tuned encoder.** `outputs/experiment016_xecg_probe_finetune/random_head/model.pt` (hash checked
  against its `complete.json`): the released xECG with a new linear head, trained for one epoch on the 15,360
  labeled PTB-XL training ECGs; the epoch was selected on PTB-XL development AUROC. Only its backbone is used
  here, with the same pooling. Its head must reproduce 016's saved development logits
  (`development_logits.npz`) from the extracted pooled features to within 1e-4, or the arm is dropped. It is
  the only supervised encoder already available; nothing new is trained. Because its epoch was chosen on
  PTB-XL development and it was trained on PTB-XL training labels, its PTB-XL development numbers are
  optimistic; its SPH numbers are not.
- **Pretrained arm.** Its analysis uses the existing frozen features, loaded as 022b and 034 load them, each
  checked against its receipt. As a sanity check, the runner re-extracts it on the checked samples below and
  requires bit-identical features.

## Records

Only these records pass through the encoders, in this order:

| Set | Rows | Source of the rows |
| --- | ---: | --- |
| `ptbxl_train` | 17,083 (9,840 abnormal) | PTB-XL training folds with a standard label, 022's order |
| `ptbxl_development` | 1,572 (884) | PTB-XL development fold with a standard label, 022's order |
| `sph` | 21,008 (7,190 positive) | SPH `use_evaluation` rows with a primary label, 026's order |
| `challenge_train` | 22,494 (17,520 positive) | 022b's kept Challenge training rows (`training_rows.csv`, hash checked against 022b's receipt), in that file's order |

Inputs are built exactly as for the frozen features: PTB-XL with `external_encoders.ptb_xecg_input`, SPH with
022's `read_checked` window (hash checked against the manifest) and `external_encoders.xecg_input`, and the
Challenge records with `challenge_features.read_verified` (official checksums verified), `canonical_window`
at 022b's saved `window_start` and `xecg_input` of the float64 window. Ningbo windows must match the clean
manifest's `window_sha256`. Features come from `external_encoders.xecg_features` (microbatches of 16).

No PTB-XL calibration or test record and no Challenge calibration or test record is read.

## Scores

Every arm gets the same two scores, fitted on its own features:

1. **Normal reference (primary).** `normal_manifold.fit_mahalanobis` and `mahalanobis_scores`, unchanged
   (StandardScaler, PCA-64, Ledoit-Wolf, squared Mahalanobis distance), fitted on 026b's `pooled` fit set:
   the rows of `outputs/experiment026b_multisource_manifold_v1/fit_sets.csv` with `pooled` true, in that
   file's order (5,872 PTB-XL NORM training ECGs, then 4,974 Challenge training normals; 10,846 in all).
   For the pretrained arm this must reproduce 026b's saved `xecg_pooled` SPH and development scores to within
   1e-9 relative, or the run stops.
2. **Supervised readout (reference point).** 022b's `pooled` readout: `multisource_readout.fit_readout`
   unweighted (train-only StandardScaler, L2 logistic regression, `C=0.01`, lbfgs, `tol=1e-8`,
   `max_iter=5000`, float64) on the 39,577 rows of `ptbxl_train` followed by `challenge_train`, with the
   standard label. For the pretrained arm this must reproduce 022b's saved `xecg` `pooled` SPH and
   development probabilities to within 1e-9, or the run stops. If the solver does not converge for another
   arm, that arm's readout is reported as not converged and left out; the readout carries no decision.

Nothing is tuned. Each score is fitted once per arm.

## Evaluation

The standard label throughout: SPH's primary label, PTB-XL's superclass label.

1. **SPH (primary):** 21,008 ECGs, 20,364 patients, 7,190 positive. AUROC of both scores for every arm;
   average precision beside it.
2. **PTB-XL development (secondary):** 026's 1,306 original development ECGs (1,173 patients, 843
   abnormal), both scores.
3. **Per condition (secondary):** on SPH, each superclass (MI, STTC, CD, HYP) against the SPH primary
   negatives (13,818 ECGs), as 022 read them; both scores.

### Intervals

All from the shared `ecg_experiment/intervals.py`, 2,000 whole-patient draws, seed 40040, the same draws for
every arm and score of a set:

- **Per seed:** `intervals.paired_auroc_difference` (pretrained minus `random_s`, and `finetuned016` minus
  `pretrained`).
- **Mean over seeds:** pretrained AUROC minus the mean of the three random AUROCs, computed within each draw
  on the draws of `intervals.patient_groups` and `intervals.two_class_resamples` from a fresh
  `numpy.random.default_rng(40040)`, so they are the same draws as `paired_auroc_difference` uses. The
  interval is the 2.5 and 97.5 percentiles of the valid draws; single-class draws are skipped and counted.
- **Share of the gap that is the embedding:** within each draw, the mean-over-seeds gap of the normal
  reference minus the same gap of the supervised readout, with its interval. A positive value means the
  label-free score depends on pretraining more than the supervised readout does.

## Primary comparison and decision rule

- **Primary comparison:** SPH, the normal reference's AUROC, `pretrained` minus the mean of the three random
  seeds, with its whole-patient interval. The three per-seed contrasts are reported beside it.
- **Decision:**
  - **Pretraining is needed** if the mean-over-seeds interval lies at or above +0.02 (lower bound ≥ 0.02) and
    every per-seed interval lies above 0.
  - **Pretraining is not needed** if the mean-over-seeds interval lies below +0.02 (upper bound < 0.02): a
    random xECG is within 0.02 AUROC of the pretrained one at SPH. If the whole interval is below 0, the
    random encoder is better, which is reported as such.
  - Otherwise the result is **inconclusive** at this margin.
- 0.02 is the size the paper would call a material gain; it is the 026b gain of adding hospitals to the fit
  set (+0.020) and the order of 022b's pooling gain, and four times the 0.005 "negligible" level used since
  022b.
- PTB-XL development, the readout, the gap share, the fine-tuned arm, average precision and the per-condition
  pattern are described by where their intervals lie (above 0, below 0, includes 0). No decision attaches to
  them.

## GPU stage and gates

One process on the shared V100 under the project's GPU lock (`ecg_experiment.gpu.gpu_lock`, non-blocking;
the run stops if another job holds it), with at most three CPU threads (the main thread and two record
readers; BLAS and PyTorch limited to one thread).

1. **Profile.** The first 128 records of each set (512 in all) through all five arms, timing reading and
   each arm. Projected total = model loading + per-record read time × all records + per-record model time ×
   (all records × four arms + the check samples). The full extraction runs only if the projection is at most
   90 minutes; otherwise the run stops and is reported.
2. **Pretrained reproduction.** The pretrained arm is re-extracted on the profile records and on 512 evenly
   spaced records of each set; every feature must equal the frozen feature exactly (largest absolute
   difference 0.0). Any difference stops the run.
3. **Determinism.** After the full pass, the first 128 records of each set are re-extracted with every
   random and fine-tuned arm; every feature must be identical to the saved one.
4. All features must be finite; a nonfinite feature stops the run.

Features go to `outputs/experiment036_random_encoder_v1/` (not committed).

## Closed data and exclusions

- PTB-XL calibration and test ECGs and the Challenge calibration and test groups stay closed.
- No age or other subgroup analysis.
- No setting is changed after a score is seen. A bug found after scoring is fixed only if it violates this
  protocol, and the fix and the discarded run are reported.

## Caveats written into the results

- xECG was pretrained without labels on data that includes Chapman and Ningbo waveforms, which are in the
  fit set and the readout's training rows. SPH is unseen by every arm.
- Three random seeds measure the spread of one initialization scheme, not of every possible random encoder.
  A random network's features depend on its initialization scale; this is the architecture's own scheme,
  not a tuned one.
- The fine-tuned arm is one run of one recipe (one epoch, chosen on PTB-XL development), trained on PTB-XL
  labels only, so it is a supervised encoder of one kind, not a trained-from-scratch one.
- One fit per score and arm. The bootstrap holds the fits fixed.
- The label is an ECG annotation proxy: a normal ECG is not proof of health, and none of these normals are
  young adults.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_random_encoder036 --stage profile
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_random_encoder036 --stage extract
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_random_encoder036 --stage analyse
```

New files only: `ecg_experiment/random_encoder.py` and `scripts/experiments/run_random_encoder036.py`. The
frozen `xecg.py`, `external_encoders.py`, `challenge_features.py`, `normal_manifold.py`,
`multisource_readout.py`, `intervals.py` and the 022, 022b, 026, 026b and 034 runners are imported, not
changed. The runner hashes every input, source and this protocol into its receipts, performs the checks
above, refuses to overwrite a finished stage, and writes `profile.json`, `features.npz` and
`extraction.json`, then `result.json`, `scores.npz` and `run.log`. Results go to
`docs/experiment-036-random-encoder-results.md`.
