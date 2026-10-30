# Full xECG: bounded single-pass successor

Written after the first resource profile and before any new development outcome. The original
four-/two-epoch schedule-selection study did not run: its FP32 profile measured 4.873 s per
effective 32-record update and its smallest forecast was 19,472 s. Preserve that negative cost
result; do not relabel its failed gate as passed.

The user requested trying full-model fine-tuning after delivery and authorized GPU use. The first
notebook is complete. This successor asks the narrower practical question: does one fixed complete
pass over the exact 25,000 training ECGs improve the confusion matrix at the original normal budget?
It performs no learning-rate search or epoch selection. All 25,000 records fit the final model once;
no validation record chooses its settings. The learning rate is fixed prospectively at 3e-6 for
the entire backbone and 3e-5 for the head; AdamW .01 decay, BCE, clip norm 1, no augmentation,
stochastic depth zero, seed 42, effective batch 32, microbatch 8. All parameters remain trainable.

Initialize from the verified released backbone and the exact all-training frozen logistic recipe
from Avance 5 (training-selected C=.1). Reuse parent checksum-verified waveform readers and
head conversion. First reproduce the frozen FP32 representations and head outputs on 64 real
training inputs. Profile BF16 autocast on the RTX 3090: require finite outputs/loss/gradients,
head score difference <=.02 from FP32 before any update, actual input-projection changes,
tensor-only checkpoint save/reload equality, and allocated memory below the initially free memory
minus .5 GiB. Profile two warm-up updates and four timed updates. Candidate weights are discarded.
If BF16 fails numerical or resource checks, stop this successor and design a separately recorded
FP32 single-pass fallback rather than silently changing precision.

Forecast 1.5 times one full 25k pass plus fixed PTB/SPH development inference, plus 600 seconds
for preparation; require <=7,200 seconds. Capture original and successor source/manifest hashes.
Only after a passing profile, prepare a local 100 Hz cache of the exact 25k training inputs and
run one fresh seeded pass from the initial backbone/head. Save final weights as safetensors.
No original frozen source, receipt or pretrained weights are changed.

Use the same PTB folds 8/9 and SPH calibration/evaluation roles as Avance 5. No PTB fold 10,
original Challenge reserved group or EchoNext test is scored. Final evaluation is FP32 for both
fine-tuned model and controls. Calibrate each candidate only with the existing local reference
normals, strict score > cutoff, 5% budget. Report the fine-tuned component and its fixed 50:50
mixture with the unchanged measurement-boosting probabilities. Compare each to its exact Avance 5
control using confusion counts, AUROC, AP and paired whole-patient sensitivity intervals, 2,000
draws, seed 20261029. A practical improvement still requires >=2 sensitivity points, positive
95% lower bound, and <=.5 points more normal referrals. Never change a model after the scores.
All results remain exploratory, conditional on these fits/calibration samples and hospital proxies.

Keep raw inputs, record identities, probabilities and weights local. Preserve the completed v1
notebook and report negative as well as positive results. BF16 training is a numerical successor,
not an exact replication of the original FP32 candidate schedules.
