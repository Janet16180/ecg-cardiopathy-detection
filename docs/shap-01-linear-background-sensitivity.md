# SHAP-01: exact grouped linear-logit explanations under two backgrounds

Prospective bounded CPU protocol, 18 October 2026. The question is how much the explanation of a fixed
model changes when its reference population changes, even though its predictions stay fixed. This is
a model-dependence baseline on previously examined physical measurements, not independent evidence of
pathology, localization quality or etiology. No new clinical label definition, age/sex subgroup analysis,
waveform read, GPU, new dependency, download, push or PR is authorized.

## Inputs and integrity before labels

Reuse CLUSTER-01's 1,999 analyzed training patients and 53 physical morphology features. Before parsing
diagnosis payloads or fitting: verify its original result SHA256
`74f0c82918f158a61368d308acdab5d5e98fe759852c966de76867e5c99bc8e7`, declared metadata/feature/identity
hashes, the exact 1,999 aligned unique-patient rows, 53 feature names and finite measurements. Reproduce
the predecessor's 17,418/15,023 folds-1–8 identity counts, its 13,938/12,078 PR67 train identities, and
its prospective 2,000-patient hash sample with exactly one non-replaced exclusion. Compare the full set
of training ECG identities with the retained PR67 reservation manifest before any diagnosis parse.
Do not reread the predecessor's waveforms or access any reserved patient's clinical content.

Keep only rows with a defined existing diagnostic-superclass binary label (`superclass_label`): any
MI/STTC/CD/HYP annotation versus NORM alone. Preserve and count undefined rows; use the existing
strict project label (`project_label`) only to choose the clean-normal background. Neither proxy is
a clinical diagnosis. Clinical payload parsing occurs only after record/patient reservation checks.

## Fixed internal split and model

Within the permitted cohort assign each canonical integer patient using SHA256 of
`shap01_split_v1:20261018:{patient_id}`, integer digest modulo 10,000: below 8,000 internal training,
otherwise internal evaluation. One ECG per patient means both sets are patient-disjoint. This new
internal evaluation is exploratory because the cohort/measurements were already seen in CLUSTER-01.
It is not PR67 validation or engineering test.

Fit a fresh StandardScaler on defined-label internal training rows only. Fit exactly one L2 logistic
regression on all 53 standardized features: C=1, balanced class weights, lbfgs, max_iter=2000, tol=1e-8,
seed 20261018. No PCA, residualizer, winsorizer or transform fitted on the full predecessor cohort is
reused. No feature/model selection, threshold tuning or second fitted model. Require convergence and
both target classes in training/evaluation. Secondary predictive context is evaluation AUROC with
2,000 shared whole-patient bootstrap draws, 2.5/97.5 percentiles, seed 20261018 and skipped-draw count.

## Exact explanation and backgrounds

For the fitted decision function `f(z) = intercept + beta @ z`, compute exact interventional Shapley
values directly, without a SHAP dependency:

`phi_j = beta_j * (z_j - mean_background_j)`

`base = intercept + beta @ mean_background`, so `f(z) = base + sum(phi_j)`.

This is the independent-intervention linear-logit game, not conditional SHAP for correlated leads or
an additive explanation in probability units. The formula follows the
[official LinearExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html)
and the additive explanation framework of [Lundberg and Lee](https://arxiv.org/abs/1705.07874).
For this additive model, summing the feature values within a declared group also gives its exact
group-coalition Shapley value under the same intervention game. Standardization is fixed across games.

Compare two complete, nonempty backgrounds using the identical model: all model-training patients,
and the subset of model-training patients whose existing strict project label is exactly zero.
No internal evaluation row or undefined-target row enters either background. Do not select backgrounds
by explanation quality, predictive score, cluster or evaluation annotation. Use all their rows, no draw.

Group R/S/ST/T amplitudes into each of the 12 canonical leads. Put QRS duration, QT duration, RR variation
and axis sine/cosine into one global duration/rhythm/axis group. Every feature belongs exactly once.
These groups are dependent: limb leads are algebraically related and axis uses lead measurements.
Independent interventions can create physiologically implausible feature combinations. Values explain
this fitted proxy model's dependence; they cannot certify lead disease or a raw-time pathology mask.

## Frozen endpoints and checks

On every defined-label internal evaluation patient report per-background feature/group contributions,
base, unchanged logit/probability, absolute and positive top-group identity (positive requires >0),
top-three absolute-group overlap, within-patient Spearman absolute-rank correlation, fraction of groups
whose signed contribution changes sign, and per-patient/group L1 explanation shifts. Also report global
mean absolute grouped attribution and its rank order. Ties follow canonical lead order then global.

Require maximum absolute feature additivity error, grouped additivity error, and feature-to-group
conservation error <=1e-10. Reconstruct each logit from base + top-five positive groups + all remaining
groups. Verify that changing backgrounds leaves predictions identical and changes each linear feature
attribution by one patient-independent constant; its total cancels the base shift. Test the formula
against exact coalition enumeration on a small independent example. Require train-only background
membership and training-only transform statistics. Failure stops interpretation rather than triggering
feature, background or model tuning. No magnitude of background sensitivity establishes clinical truth.

Write aggregate group bars and a deterministic exemplar (lowest internal-evaluation patient ID) showing
signed grouped contributions under both backgrounds, logit and base. No score-based exemplar selection.
Only aggregate charts may be committed; patient/exemplar tables remain local.

## Execution and receipts

Commit this protocol before any new fit/score, implementation before execution, personal Git identity
only. Use existing venv, at most two numerical threads and five minutes CPU. Preserve failed output
directories and use fresh paths for reruns. Save committed source/input hashes, model/scaler parameters,
split/background identities, local predictions/attributions, numerical checks, aggregate result and
plots under `outputs/shap01_linear_background/`. Hash immutable artifacts after writing them; status is
mutable and must not be included in an earlier immutable-artifact hash. The executing agent writes the
results report from live outputs. Root owns queue/backlog updates; no follow-up experiment is run here.
