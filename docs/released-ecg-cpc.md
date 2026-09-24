# Released ECG-CPC checkpoint comparison

This comparison uses the authors' released S4 ECG-CPC weights. It is separate from the matched compact CNN/GRU CPC versus CPC+CMSC experiment. Our frozen encoder and regularized logistic regression evaluation do not reproduce the paper's complete benchmark, supervised fine-tuning, or frozen query-attention evaluation.

Sources: [benchmark and code](https://github.com/AI4HealthUOL/ecg-fm-benchmarking), [paper](https://arxiv.org/abs/2509.25095), [released checkpoint](https://figshare.com/articles/dataset/ECG-CPC_Checkpoint/30192604). The checkpoint release is licensed CC BY 4.0. The reported pretraining source is HEEDB, approximately 10.7 million ECGs. No PTB-XL source is reported in that corpus; cross-source patient or waveform overlap has not been independently audited here.

The archive's MD5 matched Figshare's `7cdd0d54786b4d98248afc6dce63beea`. The extracted `last_11597276.ckpt` SHA256 is `bc253edc6ac279ce2caec86d3c3a21b740e43735ca03be1dc7c50e630b0ad4cf`. The extractor requires that exact checkpoint hash, records the repository revision, and strictly loads all 76 backbone tensors. Only the SSL prediction-head weight is omitted. Stored training metadata reports epoch 1 and global step 230,000; the example configuration's epoch ceiling is not evidence of completed checkpoint training duration.

The official encoder uses four 512-channel convolutions, kernels 3/1/1/1, strides 2/1/1/1, batch normalization in evaluation mode, and four causal S4 blocks with model dimension 512 and state dimension 8. Serialized encoder hyperparameter metadata contains stale kernel/stride defaults; the tensor shapes and packaged inference YAML agree on the architecture used here. The instantiated backbone has 2,932,736 parameter elements; this observed count differs from the paper's approximately 3.8-million model summary and is reported explicitly.

**Numerical compatibility:** this machine lacks the Python development headers needed to compile PyKeOps's optional CUDA binder. The extractor uses the unmodified official encoder and S4 modules, replacing only their Cauchy reduction backend with the mathematically identical native PyTorch complex conjugate-pair sum. A float64 test checks that sum against the official real-component formula to `1e-14` tolerance. Strict checkpoint assignment handles expanded S4 buffers without partial loading. The saved S4 Fourier buffers and finite-length correction in C correspond to 1,200 tokens; that internal capacity is retained when extracting shorter 300-token sequences. A real PTB-XL ECG produced finite CPU/GPU crop embeddings with maximum absolute difference `3.73e-8` and mean absolute difference `2.68e-9`.

Preprocessing follows the packaged inference configuration: raw canonical 12-lead mV, four nonoverlapping 2.5-second crops, each resampled from 500 to 240 Hz with resampy's default filter, without per-record normalization. Each crop produces 300 S4 tokens. Average tokens within each crop and average the four 512-dimensional features. Averaging features is equivalent to averaging logits under a fixed linear classifier; the published benchmark's averaging of probabilities is a different evaluation choice.

Use the existing seed-42 PTB manifests: all 15,360 eligible training labels primarily, and 1,518 labels secondarily. StandardScaler and logistic regression fit only their labeled training partition; the existing development split selects regularization; calibration patients fit Platt scaling and the threshold; the test set is evaluated afterward. As in the ongoing experiments, the test cohort has already been examined, so this comparison is exploratory. The extractor reads only waveform identifiers and paths into the network.

## Completed frozen-feature results

Both probes completed on 23 September 2026, after extracting all 19,126 required ECGs. These are real PTB test results from the released encoder, not results from the compact local CPC or word2vec-style experiments.

| Training labels | Test AUROC | Average precision | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| 15,360 (all eligible) | 0.93794 | 0.96734 | 0.93640 | 0.66476 |
| 1,518 (10% manifest) | 0.92347 | 0.95952 | 0.93138 | 0.59058 |

Thresholds were selected on calibration patients for at least 95% calibration sensitivity; the achieved test sensitivities are lower, as shown. Full metrics and patient-bootstrap intervals are in `outputs/experiment004_cpc_40k/released_full/ecg-cpc_released_linear_seed42/metrics.json` and the analogous `released_10pct` directory. These frozen mean-pooling results do not establish the best performance obtainable by fine-tuning or by using the paper's other readouts.

Run `.venv-pretrained/bin/python -m scripts.run_released_cpc` to extract features and run both probes sequentially. The extractor shares `/tmp/ecg_project_gpu.lock` with the compact experiment, checkpoints extraction progress after each batch, and validates identities on resume. Outputs and current status are under `outputs/experiment004_cpc_40k/`; download or extraction completion is not a model evaluation result.
