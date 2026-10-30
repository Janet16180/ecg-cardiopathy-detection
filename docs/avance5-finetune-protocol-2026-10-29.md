# Full xECG fine-tuning after the first executed delivery

Authorized by the user's follow-up: try updating the entire model on the available GPUs to improve
the confusion matrix. Execute this study only after Avance5.8 has completed sequentially.
All results are exploratory development evidence; closed Challenge/EchoNext tests and PTB fold 10
remain closed. SPH is never used for classifier training or epoch/hyperparameter selection.

Use the exact all-age 25,000-record supervised manifest and released xECG backbone from Avance 5.
Reproduce its frozen embeddings on 16 real training records before optimizer profiling. Use existing
checksum-verified `read_training_signal`, physical mV, canonical 12 leads and the unchanged official
FFT 500-to-100 Hz transform. Cache only these training inputs. No clinical labels are redefined.

Seed 42. Five stratified patient-group folds inside the training cohort: fold 0 chooses epochs,
fold 1 supplies calibration normals only, folds 2–4 fit candidate models. Unknown patient identities
use their record IDs. No independence claim is possible for unobserved cross-record patient identity.
Initialize each candidate from the same released backbone and a frozen logistic readout fitted
only on folds 2–4. Use the logistic C selected by Avance 5's training-only search. Convert scaler
and logistic coefficients algebraically into a one-logit head and verify prediction equivalence.
The unused mask token can have no gradient; every backbone parameter otherwise remains trainable.

Candidate schedules: backbone learning rates 1e-5 and 3e-6; head rate ten times the backbone rate;
AdamW weight decay .01; FP32; stochastic depth 0; no waveform augmentation; BCEWithLogitsLoss;
gradient norm clipping 1; effective batch 32; four epochs per schedule. Seed, data ordering and
initial head are matched. Choose the candidate epoch by standardized partial AUROC at max_fpr=.05
on training fold 0, tie-breaking by lower learning rate and then earlier epoch. Report sensitivity
at a cutoff fitted on fold-1 normals, but do not select a threshold from fold-0 outcomes.

Under the shared GPU lock, profile two complete optimizer updates with batch 8 (batch 4 only if
batch 8 has a measured OOM). Record finite loss/gradients, actual parameter changes, GPU memory,
forward inference time and save/reload equality through safetensors. Require enough free GPU
memory and a conservative remaining forecast <=7,200 seconds. Forecast all fits, validation
inference, worst-case final refit and development inference, times 1.5 plus 600 seconds for input
preparation. If the two-schedule/four-epoch profile exceeds this gate, use one 1e-5 schedule/four
epochs; if still over, use one 1e-5 schedule/two epochs. These cost-only choices happen before
candidate outcomes. Stop rather than silently weaken the gate if the smallest schedule fails.

After choosing the schedule and epoch within training, discard candidate weights and refit from
the same released backbone on all 25,000 ECGs for the selected number of epochs. Initialize the
head with the all-training logistic fit, preserving the exact original C. Always report the best
fine-tuning attempt, even if it loses to the frozen control; do not conceal negative results.

Evaluate the final fine-tuned score once on reused PTB fold 9 and SPH development, each with the
same normal-only calibration roles and strict score > cutoff five-percent budget. Also report
the fixed equal mixture of the unchanged Avance 5 measurement boosting and the fine-tuned score.
Controls are the exact frozen xECG and blend from the executed notebook, trained on the same 25k.
Save all TP/FN/FP/TN counts, AUROC, AP, costs and paired patient-bootstrap sensitivity differences
(2,000 draws, seed 20261029), conditional on fitted models and calibration samples. A practical
improvement requires >=2 sensitivity percentage points, positive 95% paired lower bound, and
no more than .5 percentage point higher normal referral on SPH. This is a development decision,
not clinical validation; no post-score retraining, threshold optimization or source selection.

Store raw inputs, identifiers, predictions, optimizer data and weights only in ignored local outputs.
Publish only an aggregate report and optional notebook addendum. Bind committed protocol/source
and input digests before profiling. Keep the first completed notebook usable while this runs.
