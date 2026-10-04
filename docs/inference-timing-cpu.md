# CPU-only inference timing of pipeline v4 and its explanation

Measured 1 October 2026 for a student clinic without a GPU. This is a timing and correctness measurement,
not an experiment: it computes no score of any evaluation set and reads no development, calibration or test
ECG. Script: `scripts/reports/time_inference_cpu.py`. Output: `outputs/inference_timing_cpu_v1/result.json`
(run from commit `1f1faf0` with this script before it was committed, 121 seconds).

## Method

**What is timed.** The full per-ECG path of [pipeline v4](pipeline-v4.md) plus the explanation of
Experiment 048 (the PVC switch: `U_B` as the rhythm layer, the attention contributions as the morphology
layer), in seven steps:

1. **Read and preprocess:** read one PTB-XL WFDB record (`external_encoders.read_ptb_float64`), build the
   xECG input (`xecg_input`, 100 Hz) and the ECG-JEPA input (`jepa_input`, 8 leads, Fourier resampling to
   2,500 samples).
2. **xECG features:** the released backbone, `xecg.load_xecg(device="cpu", backend="vanilla")`, one ECG per
   forward pass.
3. **ECG-JEPA tokens:** the 400 output tokens (`lead_wave_maps.jepa_tokens`, batch 1). The official loader
   (`external_encoders.load_jepa`) calls `torch.load` without a map location and moves the encoder to CUDA.
   A new helper, `cpu_inference.load_jepa_cpu`, builds the same encoder with the same parameters and loads the
   same checkpoint mapped to the CPU. No pinned file was edited.
4. **Attention heads:** the three saved networks of Experiment 046 (`attention_jepa_weights.npz`) on the
   tokens after the float16 storage round trip, as in pipeline v4. The step gives the mean logit and the mean
   per-token contributions.
5. **Readouts and finding heads:** R3 (linear part on the 1,792 xECG + pooled JEPA features; pooled JEPA is
   the token mean), E = (R3 + A) / 2, z_PVC, z_WPW and F, from `pipeline_v3_heads.npz`.
6. **Beat-wave map `U_B`:** 042's beat cutting (`beat_pieces`) and the squared Mahalanobis distance of every
   beat, lead and wave from its reference (`wave_scores`, `beat_unit_map`).
7. **Explanation:** the attention unit map, 048's switch (`pvc_switch.switch_explain`, z_PVC above 048's
   1.654) and the marks (`explanation_rule.rule_marks`, red thresholds 1,158.8 and 0.275 as 048 saved them).

Not timed: comparing E and F with the site thresholds (two comparisons per ECG) and drawing the figure.

**Rows.** 50 PTB-XL training ECGs, a seeded sample (seed 2026) of 042's labeled training pool restricted to
ECGs in the 043 token cache. The script refuses to run if a sampled ECG is a development row. For the
SPH check below, 50 SPH ECGs from the 043 token cache (seed 2026); only their saved tokens and features are
read, not their waveforms.

**One-time cost.** The 48 wave references of `U_B` (12 leads by 4 waves, Ledoit-Wolf) are fitted once from
the 5,872 fit-set ECGs (042's NORM-only training ECGs), as 042 and notebook 12 do: read and cut every beat
(042's parallel reader), then fit. This runs in a child process with 4 threads, so its memory is reported
separately from the per-ECG peak. A clinic would do this once and store the 0.8 MiB of references.

**Threads and timing.** `CUDA_VISIBLE_DEVICES=` (the script stops if a GPU is visible) and
`OPENBLAS_CORETYPE=Haswell`. The timing is run twice, with 1 and with 4 CPU threads, set with
`torch.set_num_threads` and `threadpoolctl.threadpool_limits`. Each run scores one untimed warm-up ECG
first, then the 50 ECGs one at a time. Each step is timed with `time.perf_counter`, and the total is the
sum of the seven steps. The report gives the median and 90th percentile per ECG. Model load times are
measured once, before the timing runs. Peak memory is the process's maximum resident set size
(`getrusage`).

**Correctness checks.** The CPU outputs are compared with the saved GPU-path values:

- xECG features against the cached features (tolerance 1e-4).
- Pooled JEPA (token mean) against the cached JEPA features (1e-4).
- Tokens against 043's float16 token cache.
- The attention logit on CPU tokens against the same heads on the cached tokens.
- On 50 SPH ECGs, the CPU attention heads on cached tokens, R3 on saved features, and E, against 046's
  saved SPH logits (1e-5).
- The fit-set beat count must equal 042's (63,479).

**Machine.** AMD Ryzen Threadripper 7960X (24 cores, 48 threads), 125 GiB RAM, Python 3.11.13, PyTorch
2.6.0, NumPy 2.4.6. Other agents may have been using the machine during the run. Record reads come from an
NFS home directory.

## Results

**Correctness.** Every check passed:

| Check | Largest difference | Tolerance |
| --- | ---: | ---: |
| xECG features against the cache (50 ECGs) | 2.7e-5 | 1e-4 |
| Pooled JEPA against the cache | 2.4e-6 | 1e-4 |
| Tokens against 043's float16 cache | 2.0e-3 (99.3% of values equal after float16 rounding) | float16 step at this scale 3.8e-3 |
| Attention logit, CPU tokens against cached tokens | 3.7e-5 | |
| SPH: attention heads against 046 | 6.4e-7 | 1e-5 |
| SPH: R3 against 046 | 3.6e-15 | 1e-5 |
| SPH: E against 046 | 3.2e-7 | 1e-5 |
| Fit-set beats | 63,479, equal to 042 | exact |

**Per ECG** (50 ECGs, milliseconds):

| Step | 1 thread, median | 1 thread, 90th pct | 4 threads, median | 4 threads, 90th pct |
| --- | ---: | ---: | ---: | ---: |
| Read and preprocess | 24.7 | 54.3 | 5.2 | 5.9 |
| xECG features | 101.5 | 108.2 | 57.1 | 61.8 |
| ECG-JEPA tokens | 682.1 | 705.3 | 250.5 | 268.0 |
| Three attention heads | 4.1 | 4.6 | 2.9 | 4.1 |
| Readouts and finding heads | 0.18 | 0.26 | 0.18 | 0.27 |
| Beat-wave map `U_B` | 10.3 | 11.1 | 10.6 | 11.2 |
| Explanation rule | 0.05 | 0.07 | 0.05 | 0.07 |
| **Total** | **826** | **864** | **328** | **351** |

The read step is not a thread effect. The 1-thread run came first and read the 50 records from NFS for the
first time; the 4-thread run read them again from the file cache. With a cold cache, reading costs about
20-50 ms per record on this NFS mount; a laptop's local disk should be faster.

**Model loads** (once per session; files already in the page cache from the earlier smoke run): xECG 1.6 s,
ECG-JEPA 3.3 s, the three attention heads 0.02 s, the readout and finding heads 0.002 s. In the first, cold
smoke run they took 2.1 s and 7.6 s.

**Memory.** Peak resident memory of the timing process:

| Point | Peak RSS |
| --- | ---: |
| Before loading models (Python, PyTorch and the PTB-XL tables the script uses to pick rows) | 683 MiB |
| After loading every model | 1,922 MiB |
| After both timing runs | 2,261 MiB |

The models add about 1.2 GiB. On disk they take 218 MB (xECG), 326 MB (ECG-JEPA), 1.3 MB (attention heads)
and 0.2 MB (readout and finding heads).

**One-time `U_B` references.** 54 s in all: 51.6 s to read and cut the 5,872 records (042's parallel
reader), 2.3 s to fit the 48 references. The fitting process peaked at 2,538 MiB. The fitted references take
0.8 MiB.

In this sample, 12 of the 50 training ECGs had z_PVC above the switch, so `U_B` would explain them if they
were referred. The referral itself was not applied, because it needs the site's local normals.

## What this means for a clinic laptop

- **Speed is not a problem.** On this CPU, one ECG takes about 0.33 s with 4 threads and 0.83 s with one.
  ECG-JEPA is three quarters of that; xECG most of the rest. The explanation adds about 11 ms (the beat map),
  and the attention heads about 3-4 ms. A laptop core is slower than a Threadripper core. Even at two to
  three times slower, which is an assumption and not a measurement, a 4-thread laptop would score one ECG in
  about 1 s. A nurse records one ECG in several minutes, so the screening keeps up with recording.
- **Batch work is also manageable.** Fitting a site's thresholds needs every local normal ECG scored: about
  5.5 minutes for 1,000 normals at 4 threads here. Fitting the `U_B` references once took under a minute.
  They can be shipped with the models instead (0.8 MiB).
- **Memory fits an 8 GB laptop.** The scoring process peaked at 2.3 GiB, including tables a clinic would not
  load. About 1.2 GiB of that is the two encoders. Experiment 043's 38 GB token cache is needed only for
  training, not for inference.
- **What still has to be checked on the real laptop:** the timing on its CPU and disk (run this script with
  `--threads` set to its core count), whether its PyTorch build uses the same CPU kernels, and the ECG-JEPA
  tokens against this reference within float16 resolution. The CPU path gives the same outputs as the GPU
  path within the tolerances above, so no model needs to change for a CPU-only site.
