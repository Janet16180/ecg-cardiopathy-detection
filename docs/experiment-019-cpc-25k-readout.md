# Experiment 019: fixed development readout for the 25k arm

**Frozen 26 September 2026 before new 25k development features or scores.**
This study compares the completed 25k CPC continuation with the existing
115,359-record continuation at the same 902-update/115,359-exposure budget.
The primary endpoint is 25k minus 115k AUROC with all 15,359 clean PTB
training labels. The 1,518-label subset and 25k minus unchanged starting
encoder are descriptive secondary analyses. The old-pool continuation is a
historical reference, not the primary comparator. All outcomes are ECG
diagnostic-annotation proxies, not clinical health or referral decisions.

Reuse the exact historical experiment-004 250 Hz PTB cache, its original
train-only normalization, the clean fixed PTB training labels and exactly
1,306 development ECGs from 1,173 patients. Take the same 512 CPC GRU
mean/max pooled features as Experiment 018. Fit a separate StandardScaler and
L2 logistic head on PTB training patients only for each label budget, with
`C=0.01`, unweighted classes, intercept, L-BFGS, `max_iter=5000`, `tol=1e-8`,
seed 42 and one BLAS thread. No fine-tuning, checkpoint selection, threshold
choice or hyperparameter selection uses development patients.

The completed 115k and starting-encoder prediction arrays are reused only
after checking their SHA-256 and exact record, patient and target order against
the new readout. Verify the 25k checkpoint against its completion receipt and
audit, and rehash the historical PTB cache. Compare probabilities with 2,000
paired patient-bootstrap draws, seed 18045, keeping every ECG from a sampled
patient together and counting invalid single-class draws. Report AUROC,
average precision, differences and 95% intervals. The existing development
patients have been inspected by prior experiments; any finding is exploratory
and requires independent replication. Calibration and test patients are never
featurized or scored.

A training-only real-data V100 profile measures the actual loader and 25k
encoder on 512 ECGs, then projects full extraction using a 1.5 margin and a
900-second reserve for two CPU fits, bootstrap, hashes and reporting. Include
both full historical-cache hash preflights, and require the projection to fit
7,200 seconds. Use the shared GPU lock. Store local features and predictions
outside Git; record aggregate receipts and commands in the result report.

The completed [development-only result](experiment-019-cpc-25k-readout-results.md)
records the frozen comparison and artifact audit.
