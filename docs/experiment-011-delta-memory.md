# Experiment 011: compact ECG KDA versus CKDA

**Implementation in progress, 25 September 2026. No GPU profile or model result yet.**

The primary comparison is CKDA minus KDA on the frozen 56,875-record CPC
training pool. A freshly initialized two-layer GRU-CPC is the practical
reference. All arms use the same causal 250 Hz convolutional stem, independent
five-second halves, 79 positions per half, three CPC prediction horizons, and
mean/max classification readout. Only the context mixer changes. The initial
stem and prediction heads are copied from one fresh draw to all three arms;
KDA and CKDA also share context projection weights at initialization.

The compact reference in `ecg_experiment/cpc_delta_memory.py` uses short causal
depthwise Q/K/V projections, an output gate, and the diagonal-decay,
rank-one delta recurrence from the [CKDA paper](https://arxiv.org/html/2609.24797)
and [authors' artifact](https://github.com/OpenEuroLLM/ComplexKDA). Its KDA
arm uses sigmoid channel gates and sigmoid write rates. CKDA changes the ranges
to tanh channel gates in `(-1, 1)` and doubled-sigmoid write rates in `(0, 2)`.
This is an ECG adaptation of the core recurrence, not a reproduction of the
authors' full language-model block or optimized kernels. Both ECG arms have
identical parameter counts. The plain PyTorch scan is the correctness reference;
its training speed on the V100 has not been measured.

The first four CPU tests compare forward values and gradients against the
paper's explicit transition matrix, check future-token causality, verify
independent state across ECG halves, and confirm shared initial parameters.
These checks do not establish a performance result or justify a full run.

A locked, synthetic V100 feasibility probe at batch 128 passed finite-gradient
checks for all three arms. The [local aggregate receipt](../outputs/experiment011_delta_memory/implementation_compute_probe.json)
has SHA-256 `16849049b304d79b8461130bf2b2caa3e32c44fe6b08a8ae42e0466c126ce8e7`
and records the reference source SHA-256. After one warmup pass, two measured
forward/backward passes took 0.044 seconds for GRU, 0.310–0.322 seconds for
KDA, and 0.311–0.312 seconds for CKDA. Peak allocated memory was 1.04 GB,
6.88 GB, and 6.89 GB respectively. KDA and CKDA each have 1,245,832
parameters; the GRU reference has 1,237,632. These are compute-only numbers:
they exclude cache I/O, optimizer updates, checkpoint writing, evaluation, and
full-pass effects. They are not a real-data profile or a training result.

Before any full training, finish a resumable three-arm runner, pin exact input
and code hashes, freeze the optimizer and label budgets, create a verified
successor executable manifest, and profile all three arms with real cache I/O,
evaluation, checkpoint writing, and recovery on the V100. A proposed 20 SSL
epochs remains provisional until the complete-pass cost fits the planning gate.
Development patients are for model screening; calibration and test remain
separate.
