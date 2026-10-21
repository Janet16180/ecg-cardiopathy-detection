# CLUSTER-02: prospective stability selection across morphology views

18 October 2026. This exploratory development study follows CLUSTER-01's negative frozen gate. Its
Gaussian mixture was stable but was not the selected winner. This new protocol prospectively changes
selection to prioritize stability; it does not revise CLUSTER-01 or rescue its decision after the fact.
Commit this protocol before any new clustering score and commit implementation before execution.

## Question, cohort and integrity

Does physical ECG morphology admit partitions that are stable to fitting-patient subsampling and remain
similar across prespecified representations? Compare covariance models and a consensus partition while
preserving coexisting ECG annotations. Clusters are morphology organizations, never causes of disease.
All results remain exploratory: CLUSTER-01 already used every patient in this study.

Reuse exactly CLUSTER-01's 1,999 analyzed patients and 53 morphology features. Verify its original
result hash `74f0c82918f158a61368d308acdab5d5e98fe759852c966de76867e5c99bc8e7`, feature/measurement/identity
artifact hashes, metadata hash and copied reservation manifest. Check every patient against PR67 train
roles and folds 1–8; one ECG per patient. No raw waveform reads or new data/model downloads. Reproduce
CLUSTER-01's raw and residualized k-means assignments exactly up to label names and silhouettes within
1e-10 using its frozen helper and saved measurements. Do not rerun its whole-waveform extraction or any
NTT-01 label audit. The known mutable-status hash defect is excluded from this immutable-input check.

Within this cohort, assign patients to an internal anchor set if SHA256(`cluster02_anchor:20261018:` +
canonical integer patient ID), integer digest modulo 5, equals zero; others form the fitting pool.
The anchor patients are excluded from all new representation/model fits, but participate in the frozen
stability selection rule. They are a development selection set, not a final test or untouched population.
No age/sex subgroup or new clinical label definitions are introduced.

## Fixed representations and models

Use three views of the same measurements:

1. Raw physical morphology, using the CLUSTER-01 feature definition.
2. Fixed nuisance residualization using the CLUSTER-01 device/site indicators, heart rate, signal standard
   deviation, baseline fraction, beat consistency, constant leads and source quality flag. Ridge alpha=10.
3. Amplitude-normalized shape: divide the 48 R/S/ST/T amplitude features by each ECG's signal standard
   deviation (floor 1e-6 mV), retaining intervals, RR variation and axis features. No fitted normalization
   constant or diagnostic label enters this per-record transformation.

Fit all imputation, 1st/99th-percentile clipping, robust scaling, categorical encoding, ridge regression,
post-residual scaling and PCA only on the fitting patients for that fit. Unknown anchor device/site
categories map to an all-zero block. Retain 95% PCA variance subject to the fixed 16-component cap.
PCA is not whitened. The same transformed arrays feed every covariance model within each view/draw.
Residualization and amplitude normalization may remove real physiology; they are sensitivity views,
not proof that any removed signal was an artifact.

Models: k-means (`n_init=10`) and Gaussian mixtures with diagonal, tied and full covariance, each at
k=2/4/8. GMM uses `n_init=3`, `reg_covar=1e-3`, `max_iter=400`, `tol=1e-3`. Seed 20261018 for reference
fits; seed 20261019+j for refit j=0..7. Do not tune covariance regularization or dimensions on scores.
Reject nonconverged GMM configurations from selection and report them. No hard-assignment diagnosis model.

For each reference configuration, predict memberships of fitting and anchor patients. Then refit the
complete preprocessing and model on eight 80% subsamples without replacement of fitting patients,
using the same patient draws across all configurations and views. Predict all anchor patients without
refitting or adapting on them. Store each fit's preprocessing and model provenance through the frozen
source/config and sampled-patient hash; retain the anchor assignment matrices locally.

## Unsupervised selection and consensus

A configuration is eligible only if every fit converged; the reference fit has all k clusters on fitting
and anchor patients, with each cluster at least 5% of both populations; each refit has all k clusters and
at least 5% per cluster on anchors; reference anchor silhouette is >=0.05; median anchor ARI across eight
refits is >=0.80; and minimum ARI is >=0.60. Predicting every anchor into one cluster cannot qualify.
Silhouette uses the reference view's anchor coordinates and labels. Refit ARI compares predictions on
identical anchor patients with the reference fit's predictions. Eight refits are a robustness sample,
not a confidence interval.

Within each view, separately choose a k-means baseline and a mixture from the eligible candidates.
Sort by highest median stability ARI, then highest minimum ARI, then highest silhouette, then smaller k,
then covariance order diagonal/tied/full. No diagnostic/nuisance association metric enters this ranking.
If no candidate qualifies, mark that view/family ineligible and retain the same ranked descriptive
fallback using finite values; the fallback cannot pass a gate. Report all 36 configurations.

Construct consensus only if all three selected mixture views are eligible. On anchors, average the
three binary coassignment matrices and use average-linkage agglomerative clustering of one minus that
matrix. Consider consensus k=2/4/8. For each of the eight refits use the corresponding three view
assignments to form a new consensus; compare with the reference consensus by anchor ARI. Use raw-view
reference anchor coordinates for its silhouette and require the same anchor size/separation/stability
conditions. Select among eligible consensus k using the same stability-first rule, or report ineligible.
This consensus uses only selected unsupervised anchor assignments; it has no out-of-sample clinical
prediction rule. Do not fit a consensus when any component view is ineligible.

Primary robustness gate: all three mixture views must have eligible winners, every pairwise reference
anchor ARI among them must be >=0.50, and consensus must be eligible. A pass means recurring morphology
organization under these views only. Otherwise the conclusion is negative/inconclusive for persistence
across views. Separately report whether the residualized mixture improves minimum anchor stability by
>=0.10 over the residualized k-means baseline; both must be eligible to claim this secondary improvement.
Do not force larger k, switch winners using annotations, relax gates, or tune after scores.

## Post-selection interpretation

Only after unsupervised choices and the robustness decision are fixed, interpret original SCP codes for
these permitted patients. Keep NORM/MI/STTC/CD/HYP as overlapping flags, including ambiguous combinations.
For each selected baseline/mixture view and any eligible consensus, report anchor cluster sizes,
multilabel prevalences, fraction with at least two abnormal classes, and one-sided Fisher enrichment
against other anchors. One Benjamini-Hochberg family covers every displayed cluster-by-label enrichment
across all selected views and the consensus. Report effect sizes, not just p values. Ineligible fallbacks
may be described but cannot support a successful gate.

On anchors report AMI with annotation combinations, device, site, source quality flag, and rate/amplitude/
baseline/consistency quartiles. Also show descriptive full-cohort associations for selected mixture and
baseline predictions; those include fitting observations and are not an independent endpoint. Shared
whole-patient bootstrap helpers provide 200 conditional percentile intervals for the residualized
mixture-minus-k-means annotation AMI contrast on anchors. These intervals condition on selection/fits.
No source transfer claim follows from PTB acquisition-site metadata; no independent hospital is tested.

## Boundaries, artifacts and follow-up

Two numerical CPU threads, no GPU, no new libraries, no dependency/lock changes and no waveform rehash.
Stop admitting new fits after ten minutes elapsed CPU wall time and preserve partial output/status.
Do not resume or adapt a partial comparison to manufacture a winner. Save candidate tables, anchor
assignments/refit predictions, sample hashes, source/input hashes, timing, selected configurations,
aggregate JSON and standalone aggregate plots. Patient identities and individual features remain local.
Write an executed results report even for a negative gate, with deviations and practical limitations.
No pushes or PR changes. The coordinator owns queue/backlog updates and full union tests. Suggested
further experiments must enter that backlog before execution.
