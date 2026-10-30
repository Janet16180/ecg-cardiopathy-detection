# Avance 5: delivery protocol

User authorization: prepare and execute Avance5.8, then attempt full-backbone GPU fine-tuning.
This notebook is a development comparison, never the one-time final frozen clinical test.

Use the verified all-age 25,000-record supervised cohort, with unchanged annotation proxies.
Read its verified measurement and xECG caches; fit imputers/scalers only inside training.
Use three deterministic StratifiedGroupKFold training folds (seed 42); known patients remain
together and unknown patients use record identifiers. This cannot prove patient independence
where identity is unavailable. Optimize histogram boosting (three configurations), random forest
(two configurations), and xECG logistic C (.001, .01, .1) within training only.

Compare individual SVM and MLP recipes from Avance 4 adapted to the 130 raw measurements,
logistic regression, frozen xECG with logistic regression, and five ensembles: random forest,
histogram boosting, equal SVM/MLP voting, SVM/MLP stacking with genuinely out-of-fold training
predictions, and equal measurement-boosting/xECG blending. Include the original 111-feature
Avance 4 reference separately, with its original population and metrics explicitly labeled.
Choose the new recipe by training OOF AUROC, then report all candidates on reused PTB fold 9
and SPH development. Calibrate each cutoff on that hospital's existing normal-only calibration
partition, using the existing strict score > cutoff rule and 5% normal-referral budget.
No original Challenge reserved group, PTB fold 10, or EchoNext test is scored.

Compare released JEPA, CPC and xECG using their existing PTB embeddings and the same
folds 1–7 fitting / fold 8 normal calibration / fold 9 development roles, fixed logistic C=.01.
Their original pretraining populations and costs differ; no universal architecture claim follows.
Reproduce the saved 25k all-age blend matrix before claiming cache integrity. Preserve the
published age comparison and its uncertainty rather than inventing an improvement or null effect.
Generate ROC, precision–recall, confusion matrices and threshold tradeoffs with numeric
interpretations. Published US athlete prevalence informs conditional arithmetic scenarios only;
it is not our ECG-label prevalence, Italian prevalence or a validated disease prediction.

Keep patient predictions, weights and embeddings local. Export aggregate reports, figures and the
executed notebook. Record timings separately for fitting, tuning and historical feature extraction.
Attribute AI assistance accurately. Individual responsibilities carried from Avance 4 are a proposed
division for review, never an assertion of newly verified work.

After the notebook has executed successfully, freeze a separate full-backbone fine-tuning protocol
and source hashes. Profile actual forward/backward/optimizer steps under the shared GPU lock
before choosing a feasible run. Fit and choose epochs using training-only partitions; do not use
SPH or PTB fold 9 to choose epochs, learning rates, losses or cutoffs. Compare against the frozen
xECG control at the same normal-referral budget, including actual confusion counts and timings.
