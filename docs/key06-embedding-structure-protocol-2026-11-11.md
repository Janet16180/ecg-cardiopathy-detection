# KEY-06: structure of the xECG embedding space beyond the labelled conditions

Prospective, 11 November 2026. Frozen before any analysis of this experiment. Requested by the
user on 11 November 2026: study the vector space of the encoder behind the project's best
binary detector (xECG) and its clusters, which may indicate cardiac findings that we have no
labels for, or something else.

It follows Experiment 024 (clusters of ECG-CPC embeddings followed the device and the heart rate
more than the diagnosis), MLC-02 (the source is identifiable from xECG features with accuracy
0.852) and MLC-03 Part A (signal-quality flags, `docs/mlc03-signal-quality-results-2026-11-11.md`).

## Questions

1. How much of the embedding's variation is explained by what we already know: the 14
   conditions, the source, the heart rate and other measurements, signal defects, age and sex?
2. After those are accounted for, are there stable groups or directions left?
3. For each group: is it a cardiac finding outside the 14 conditions (named by the sources' own
   diagnostic statements or by measurements), or a recording artefact, a source habit, or
   unexplained?
4. Which statements outside the 14 conditions does the embedding already separate well enough
   to be worth labelling later?

## Data

- MLC-01 dataset v4, included records. Train (56,250) is the discovery set; validation (15,138)
  is the replication set. Test roles (PTB-XL fold 10, both SPH halves) are not read. No Challenge
  test group, EchoNext or MIMIC record is opened.
- Features: pooled xECG embeddings (1,024 values per record,
  `outputs/mlc02_condition_models_20261110/features/xecg.npy`), standardized on train. ECG-CPC
  embeddings (512), whose pretraining saw none of these sources, are used only for the
  cross-encoder check in step 5.
- Known factors per record: the 14 labels (positive, negative, undefined); source and family;
  age and sex; the 130 measurements (heart rate, intervals, axis, amplitudes); the 18 MLC-03
  signal flags; the raw diagnostic statements of each source (`raw_codes` in the manifest and
  `mapping.csv`), including those outside the 14 conditions.
- Integrity: P0's validation selection metric (0.9643) is reproduced from its stored
  predictions, and the feature rows are checked against the manifest keys, before any analysis.

## Steps

1. Variance accounting. For each factor group (source, 14 labels, heart rate, other
   measurements, signal flags, age and sex), the share of embedding variance it explains
   (cross-validated linear regression onto the embedding's principal components, fitted on
   train, read on validation), alone and after the others. In the other direction: how well
   each factor is predicted from the embedding within source.
2. Residual space. The linear effects of source, heart rate, age, sex and the reliable signal
   flags are removed (fitted on train). Every later step is run on the raw standardized space
   and on the residual space, and both are reported.
3. Clustering on train. Principal components keeping 90% of the variance; k-means with k in
   {10, 20, 40, 80} and HDBSCAN (minimum cluster size 100). Stability: 20 subsamples of 80% of
   train, per-cluster Jaccard overlap with the full solution. Validation records are assigned
   to the nearest train cluster. The same procedure is repeated on the label-negative subset:
   records with no positive among their defined labels.
4. Cluster description, on train and separately on validation: size; share per source and
   family; prevalence and enrichment of each of the 14 labels; enrichment of every raw statement
   outside the 14 conditions that has at least 30 records in train (odds ratio against the rest,
   within-source stratified, Benjamini-Hochberg at 5% over all cluster-statement pairs);
   measurement profile; signal-flag prevalence; age and sex. For each cluster, the 24 ECGs
   nearest its centre and 24 random members are plotted to review sheets under
   `outputs/key06_embedding_structure_20261111/review/`, and the analyst reads them and records
   what is measurable in them.
5. Cross-encoder check. For each candidate cluster of step 6, whether its train members are
   also closer to each other than to others in ECG-CPC space (cluster membership predicted from
   ECG-CPC embeddings, within source, AUROC on validation).
6. Outliers. The 1% of train records farthest from their 20 nearest neighbours: what they are
   (flags, sources, statements, plots of a fixed random 48).
7. Statements outside the 14 conditions. For every such statement with at least 100 positives
   in train and present in at least two source families: a logistic regression on the
   standardized embedding (C = 1), within-source AUROC on validation with a patient bootstrap
   interval (2,000), beside the statement's counts per source. This lists what the embedding
   already encodes without having been asked to.

## Reading the result

A cluster is called a candidate finding only if all hold:

- stable: median Jaccard at least 0.6 over the subsamples;
- not one source: at least two source families each hold at least 10% of its members, or at
  least 50 members;
- replicated: its strongest enrichment on train has the same sign on validation with an
  interval excluding no effect;
- not an artefact: no reliable signal flag is more than twice as frequent in it as outside;
- not already labelled: no one of the 14 conditions is positive in more than half its members.

A candidate is then named by its enriched statements or measurements, or listed as unexplained.
Clusters failing a rule are reported with the rule they fail; a cluster explained by source,
artefact or heart rate is a result, not a failure. Nothing here is a clinical finding: the
sources code statements differently, no second reading exists, and clinical review is pending.

## Limits known in advance

Clusters depend on the method and on k; the stability and replication rules are the guard, not
a guarantee. Raw statements are coded differently and incompletely between sources, so a
finding that only one source codes cannot be confirmed across sources. Removing linear effects
leaves non-linear ones. The label-negative subset contains wrongly negative labels.

## Resources and deliverables

CPU only, limited threads. New modules and scripts only, with tests on synthetic data.
Outputs in `outputs/key06_embedding_structure_20261111/`. Report
`docs/key06-embedding-structure-results-2026-11-11.md` written from the outputs, with the
cluster table, the candidate list, the list of statements the embedding separates, the review
sheets, departures and doubts. Queue and backlog updates.
