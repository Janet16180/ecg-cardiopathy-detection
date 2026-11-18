# KEY-05 results: condition keys on xECG embeddings for the 14 conditions

11 November 2026. This report executes the frozen [KEY-05 protocol](key05-condition-keys-protocol-2026-11-11.md)
(commit `ba89f83`) on dataset v5. It was written from the outputs by the agent that ran the experiment. Every
number below is printed from `outputs/key05_condition_keys_20261111/` by

```
PYTHONPATH=. .venv/bin/python -m scripts.reports.key05_tables --section <name>
```

and cited as [section `<name>`]. The script computes no metric; it selects, subtracts, takes medians and counts
stored values. The labels are ECG-annotation proxies, not diagnoses. No cardiologist has reviewed any of this
work; clinical validation is pending (validación pendiente). The review-sheet reading in section 9 is a
non-clinical reading.

A key is one vector per condition (K1 to K6), or a few (K7), built from train records without iterative
training; a record is scored by its similarity to the key. P0, the v5-trained independent logistic regressions on
the same pooled xECG embeddings (MLC-03 Part D), is the reference.

## Result of the test read [sections `test`, `test-conditions`]

- Selected on validation and committed before any test-role label was read (`83c9fe7`,
  [selection record](key05-selection-record-2026-11-11.md)): single key **K6**, the shrunk linear discriminant
  direction on source-centred embeddings; and **K7-m4**, four whitened keys per condition.
- Primary comparison, K6 minus P0, macro AUROC over the 12 conditions with at least 100 positives and 100
  negatives at the SPH evaluation half (12,698 records, 12,246 patients): **-0.0045 (-0.0060, -0.0032)**.
- Decision by the protocol's rule: the lower bound, -0.0060, is above -0.010, so **the single key is equivalent to
  the classifier for ranking** within the protocol's margin. The whole interval is below zero: P0 ranks slightly
  better, by less than the margin. Nothing is promoted; P0 stays the reference.
- Secondary, without a decision: K7-m4 minus P0 at SPH -0.0071 (-0.0085, -0.0058); at PTB-XL fold 10 (9
  conditions, 2,197 records) K6 minus P0 -0.0031 (-0.0059, -0.0006) and K7-m4 minus P0 -0.0100 (-0.0135,
  -0.0069).
- Per condition at SPH, by the same rule without a decision: K6 reads equivalent for 9 of 12 conditions,
  undecided for incomplete RBBB (-0.0191 (-0.0295, -0.0094)) and LVH (-0.0036 (-0.0120, +0.0009)), and clearly
  worse for ST depression (-0.0264 (-0.0354, -0.0180)). For sinus arrhythmia K6 is above P0 (+0.0074 (+0.0042,
  +0.0105)).

| role | model | conditions | macro AUROC (95% interval) |
|---|---|---|---|
| SPH evaluation half | K6 | 12 | 0.9692 (0.9668, 0.9714) |
| SPH evaluation half | K7-m4 | 12 | 0.9666 (0.9643, 0.9688) |
| SPH evaluation half | P0 | 12 | 0.9737 (0.9720, 0.9753) |
| SPH evaluation half | K6 minus P0 | 12 | -0.0045 (-0.0060, -0.0032) |
| SPH evaluation half | K7-m4 minus P0 | 12 | -0.0071 (-0.0085, -0.0058) |
| PTB-XL fold 10 | K6 | 9 | 0.9574 (0.9522, 0.9620) |
| PTB-XL fold 10 | K7-m4 | 9 | 0.9504 (0.9449, 0.9554) |
| PTB-XL fold 10 | P0 | 9 | 0.9605 (0.9558, 0.9647) |
| PTB-XL fold 10 | K6 minus P0 | 9 | -0.0031 (-0.0059, -0.0006) |
| PTB-XL fold 10 | K7-m4 minus P0 | 9 | -0.0100 (-0.0135, -0.0069) |

## Answers

1. **How close one key comes, and which construction matters.** On validation the best single key, K6, reaches a
   selection metric of 0.9643 against P0's 0.9698, a paired difference of -0.0054 (-0.0083, -0.0029); K4, the same
   discriminant without source centring, reaches 0.9634. Keys built from class means alone (K1, K2, K3, K5) stay
   at 0.897-0.924. The construction that matters is whitening by the within-class covariance: the step from K2 to
   K4 gains +0.0443 (+0.0397, +0.0489), and every other step moves the metric by less than 0.01 [sections
   `constructions`, `ladder`]. At the SPH evaluation half K6 is equivalent to P0 for ranking by the protocol's
   rule.
2. **Why keys fall short.**
   - Covariance: whitening raises the key's AUROC for 10 of 14 conditions with the interval above zero, most for sinus
     arrhythmia (+0.172), LVH (+0.113), ST depression (+0.092) and incomplete RBBB (+0.089) [section
     `ladder-conditions`].
   - Source: centring by source adds +0.0009 (+0.0005, +0.0014) within sources and loses -0.0097 (-0.0108,
     -0.0087) on pooled records. Keys built without a source family lose most on that family for ST depression,
     MI, incomplete RBBB and T-wave abnormality (section 7).
   - Several forms of one condition: four k-means keys per condition (K7-m4) lose -0.0026 (-0.0047, -0.0001)
     against K6 within sources; only complete RBBB gains (+0.0054 (-0.0015, +0.0201)) [section `ladder`].
   - What remains to P0 after K6 is 0.0054 on validation, largest for incomplete RBBB, first-degree AV block and
     ST depression (-0.012 to -0.013) [section `conditions-selected`].
   - Co-occurring conditions do not explain the gap: whitened keys respond to other conditions less than mean
     keys and less than P0 (section 5).
   - Few references: mean-difference keys are near their plateau (0.91) from 50-100 records per class; whitened
     keys need the full train set (section 6).
3. **References and stability.** A mean-difference key built from 50 positives and 50 negatives reaches a mean
   within-source AUROC of 0.900 against 0.913 with all records; a whitened key built from 500 per class reaches
   0.942 against 0.9635 [section `references`]. Mean keys are almost identical under patient resampling (pairwise
   cosine at least 0.980); whitened keys change direction (pairwise median 0.73-0.93) while their validation AUROC
   stays within 0.9611-0.9628 for K6 [sections `stability`, `stability-auroc`].
4. **Agreement with the defining measurement.** The K6 score follows each condition's measurement within source,
   in the direction the definition implies (for example against heart rate: Spearman 0.42 for sinus tachycardia
   and -0.66 for sinus bradycardia), but the whitened keys follow it less than the mean keys (median absolute Spearman 0.41 against
   0.56) and less than P0 (0.53) [sections `measurements`, `measurements-constructions`].

## 1. Integrity [section `integrity`]

Run before any key was built (`receipts/integrity.json`):

- The xECG array and `rows.csv` have the SHA-256 recorded in the MLC-02 Stage 1 receipt, and the v5 manifest has
  the SHA-256 of the v5 receipt.
- Every feature row matches its v5 manifest row by record key, role, source and first-10-s hash: 71,230 records,
  56,133 train and 15,097 validation.
- v5 label counts of train and validation equal the v5 build's `support_v4_v5.csv` in all 168 role, source and
  condition cells; that file's SHA-256 was checked against the v5 receipt.
- P0's stored v5 validation predictions give the stored selection metric exactly: 0.9697697420282113.
- P0's stored weights reproduce those predictions within 4.1e-15, and the standardizer fitted here on v5 train
  equals P0's (largest difference 0 in mean and in scale).
- Through this experiment's rank transform P0's selection metric is again 0.9697697420282113 [section
  `constructions`, `constructions.csv`].
- At the start of the test read, the label counts of PTB-XL fold 10 and the SPH evaluation half equalled the same
  file in all 28 cells [section `test`].

## 2. Constructions on validation [sections `constructions`, `conditions`, `average-precision`, `per-source`]

Two macro AUROCs are reported, both with 2,000 patient-cluster resamples (patients for PTB-XL, records for the
Challenge sources; no redraw was needed) and the paired difference from P0 on the same resamples:

- the selection metric: within-source macro AUROC (conditions with at least 20 positives and 20 negatives in the
  source), averaged over the six validation sources, as in MLC-02 and MLC-03;
- the pooled macro AUROC: all 15,097 validation records together, all 14 conditions.

| construction | selection metric | minus P0 | pooled macro | pooled minus P0 |
|---|---|---|---|---|
| K1 mean of positives, cosine | 0.9240 (0.9200, 0.9279) | -0.0458 (-0.0491, -0.0426) | 0.9304 (0.9277, 0.9331) | -0.0478 (-0.0502, -0.0455) |
| K2 positives minus negatives | 0.9191 (0.9152, 0.9231) | -0.0507 (-0.0541, -0.0474) | 0.9251 (0.9222, 0.9279) | -0.0532 (-0.0558, -0.0506) |
| K3 positives minus no-positive records | 0.8968 (0.8925, 0.9012) | -0.0729 (-0.0771, -0.0690) | 0.8968 (0.8934, 0.9002) | -0.0815 (-0.0847, -0.0783) |
| K4 whitened K2 | 0.9634 (0.9588, 0.9677) | -0.0064 (-0.0095, -0.0037) | 0.9734 (0.9715, 0.9751) | -0.0049 (-0.0061, -0.0037) |
| K5 K2 within sources, averaged | 0.9240 (0.9203, 0.9277) | -0.0458 (-0.0491, -0.0426) | 0.9158 (0.9127, 0.9189) | -0.0625 (-0.0652, -0.0596) |
| K6 K4 on source-centred features | 0.9643 (0.9601, 0.9684) | -0.0054 (-0.0083, -0.0029) | 0.9637 (0.9613, 0.9660) | -0.0145 (-0.0163, -0.0130) |
| K7-m2 | 0.9602 (0.9557, 0.9644) | -0.0096 (-0.0126, -0.0069) | 0.9690 (0.9671, 0.9708) | -0.0092 (-0.0105, -0.0081) |
| K7-m4 | 0.9617 (0.9581, 0.9652) | -0.0080 (-0.0101, -0.0060) | 0.9682 (0.9664, 0.9699) | -0.0101 (-0.0112, -0.0090) |
| K7-m8 | 0.9542 (0.9503, 0.9578) | -0.0155 (-0.0179, -0.0134) | 0.9619 (0.9601, 0.9636) | -0.0164 (-0.0176, -0.0152) |
| N1 20 nearest train records | 0.9249 (0.9196, 0.9299) | -0.0449 (-0.0498, -0.0400) | 0.9348 (0.9315, 0.9381) | -0.0435 (-0.0465, -0.0406) |
| P0 | 0.9698 (0.9664, 0.9728) | | 0.9783 (0.9769, 0.9795) | |

- Every construction is below P0 on both metrics, with the whole interval below zero.
- The two metrics order K4 and K6 differently: K6 is first on the selection metric (0.9643 against 0.9634), K4 on
  the pooled one (0.9734 against 0.9637). Among K7, m = 4 leads on the selection metric and m = 2 on the pooled
  one. The two readings of the protocol's "validation macro AUROC" therefore select different constructions:
  K6 and K7-m4 under the selection metric, K4 and K7-m2 under the pooled macro. Both are shown for every
  construction; the selection used the selection metric (section 10), and only K6 and K7-m4 were scored on a test
  role.
- The `keys` step took 138 s on 8 threads, about 105 s of it the 20-neighbour search of N1 (`receipts/keys.json`,
  `keys.log`). The Ledoit-Wolf shrinkage of the within-class covariance was 0.0007-0.0009
  for every condition (`keys_summary.json`), so K4 is close to the unshrunk linear discriminant.
- Mean average precision over the 14 conditions, pooled: K4 0.776, K6 0.756, K7-m4 0.717, K1 0.618, K2 0.584,
  N1 0.656, P0 0.792 [section `average-precision`].

Within-source AUROC per condition, the mean over the sources where the condition enters [section `conditions`]:

| condition | K1 | K2 | K3 | K4 | K5 | K6 | K7-m2 | K7-m4 | K7-m8 | N1 | P0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sinus tachycardia | 0.988 | 0.974 | 0.962 | 0.996 | 0.974 | 0.996 | 0.996 | 0.995 | 0.994 | 0.991 | 0.997 |
| sinus bradycardia | 0.966 | 0.975 | 0.986 | 0.996 | 0.982 | 0.996 | 0.996 | 0.996 | 0.994 | 0.982 | 0.997 |
| sinus arrhythmia | 0.786 | 0.786 | 0.784 | 0.959 | 0.805 | 0.961 | 0.957 | 0.947 | 0.927 | 0.822 | 0.963 |
| AF or flutter | 0.985 | 0.985 | 0.985 | 0.982 | 0.986 | 0.982 | 0.983 | 0.981 | 0.980 | 0.976 | 0.989 |
| PVC | 0.899 | 0.886 | 0.851 | 0.949 | 0.890 | 0.949 | 0.944 | 0.946 | 0.947 | 0.842 | 0.946 |
| first-degree AV block | 0.971 | 0.970 | 0.948 | 0.963 | 0.969 | 0.964 | 0.964 | 0.966 | 0.953 | 0.947 | 0.976 |
| complete RBBB | 0.991 | 0.990 | 0.983 | 0.987 | 0.991 | 0.988 | 0.988 | 0.994 | 0.993 | 0.989 | 0.996 |
| incomplete RBBB | 0.857 | 0.859 | 0.806 | 0.947 | 0.831 | 0.945 | 0.943 | 0.941 | 0.934 | 0.832 | 0.959 |
| LAFB | 0.970 | 0.971 | 0.966 | 0.973 | 0.970 | 0.975 | 0.972 | 0.973 | 0.963 | 0.967 | 0.979 |
| left axis deviation | 0.978 | 0.976 | 0.962 | 0.985 | 0.974 | 0.985 | 0.982 | 0.981 | 0.976 | 0.970 | 0.987 |
| LVH | 0.854 | 0.847 | 0.807 | 0.960 | 0.863 | 0.964 | 0.954 | 0.954 | 0.930 | 0.877 | 0.968 |
| ST depression | 0.854 | 0.845 | 0.797 | 0.937 | 0.863 | 0.937 | 0.925 | 0.937 | 0.922 | 0.906 | 0.949 |
| T-wave abnormality | 0.868 | 0.852 | 0.824 | 0.928 | 0.856 | 0.929 | 0.919 | 0.921 | 0.913 | 0.903 | 0.932 |
| any MI | 0.860 | 0.862 | 0.845 | 0.926 | 0.872 | 0.930 | 0.924 | 0.913 | 0.904 | 0.872 | 0.925 |

- For AF or flutter, first-degree AV block and complete RBBB the mean keys (K1, K2) are as good as or better than
  the whitened ones; for the other 11 conditions the whitened keys are better.
- K6 is at or above P0 for PVC (0.949 against 0.946) and any MI (0.930 against 0.925); the per-condition
  differences with intervals are in section 3.

Macro AUROC within each validation source (conditions entering that source) [section `per-source`]:

| source | conditions | K1 | K2 | K4 | K6 | K7-m4 | N1 | P0 |
|---|---|---|---|---|---|---|---|---|
| PTB-XL | 14 | 0.913 | 0.911 | 0.963 | 0.964 | 0.958 | 0.916 | 0.968 |
| Ningbo | 14 | 0.951 | 0.950 | 0.983 | 0.983 | 0.982 | 0.950 | 0.987 |
| Chapman | 9 | 0.957 | 0.947 | 0.985 | 0.985 | 0.985 | 0.964 | 0.985 |
| Georgia | 12 | 0.909 | 0.907 | 0.953 | 0.954 | 0.948 | 0.904 | 0.958 |
| CPSC 2018 | 5 | 0.901 | 0.891 | 0.959 | 0.960 | 0.956 | 0.925 | 0.967 |
| CPSC-Extra | 7 | 0.912 | 0.909 | 0.937 | 0.940 | 0.941 | 0.891 | 0.953 |

## 3. Ladder of gains [sections `ladder`, `ladder-conditions`, `conditions-selected`, `cosine-p0`]

The steps K1, K2, K4, K6, K7, P0, paired on the same resamples. Selection-metric gain of each step (K7 as
selected, m = 4):

| step | what changes | within-source gain | pooled gain |
|---|---|---|---|
| K1 to K2 | subtract the negative mean, dot product instead of cosine | -0.0049 (-0.0058, -0.0040) | -0.0053 (-0.0060, -0.0046) |
| K2 to K4 | whiten by the within-class covariance | +0.0443 (+0.0397, +0.0489) | +0.0483 (+0.0456, +0.0510) |
| K4 to K6 | centre each source first | +0.0009 (+0.0005, +0.0014) | -0.0097 (-0.0108, -0.0087) |
| K6 to K7-m4 | four keys per condition, maximum score | -0.0026 (-0.0047, -0.0001) | +0.0045 (+0.0031, +0.0060) |
| K7-m4 to P0 | the trained classifier | +0.0080 (+0.0060, +0.0101) | +0.0101 (+0.0090, +0.0112) |

With m = 2 the K6-to-K7 step is -0.0042 (-0.0054, -0.0028), with m = 8 -0.0101 (-0.0130, -0.0066); K6 to P0 is
therefore +0.0054 [section `ladder`].

Per condition, within-source gain of each step [section `ladder-conditions`]:

| condition | K1 to K2 | K2 to K4 | K4 to K6 | K6 to K7-m4 | K7-m4 to P0 |
|---|---|---|---|---|---|
| sinus tachycardia | -0.0136 (-0.0159, -0.0114) | +0.0218 (+0.0190, +0.0249) | -0.0000 (-0.0002, +0.0001) | -0.0008 (-0.0013, -0.0002) | +0.0022 (+0.0014, +0.0032) |
| sinus bradycardia | +0.0084 (+0.0064, +0.0106) | +0.0216 (+0.0169, +0.0272) | -0.0003 (-0.0013, +0.0002) | -0.0002 (-0.0011, +0.0012) | +0.0015 (+0.0006, +0.0028) |
| sinus arrhythmia | +0.0005 (-0.0022, +0.0033) | +0.1723 (+0.1470, +0.1972) | +0.0023 (+0.0012, +0.0035) | -0.0142 (-0.0192, -0.0097) | +0.0158 (+0.0098, +0.0223) |
| AF or flutter | +0.0000 (-0.0007, +0.0007) | -0.0030 (-0.0113, +0.0038) | +0.0002 (-0.0008, +0.0012) | -0.0012 (-0.0032, +0.0015) | +0.0082 (+0.0022, +0.0166) |
| PVC | -0.0132 (-0.0157, -0.0108) | +0.0626 (+0.0447, +0.0810) | +0.0000 (-0.0003, +0.0004) | -0.0028 (-0.0070, +0.0013) | +0.0000 (-0.0094, +0.0096) |
| first-degree AV block | -0.0011 (-0.0028, +0.0006) | -0.0074 (-0.0188, +0.0025) | +0.0016 (+0.0006, +0.0028) | +0.0016 (-0.0049, +0.0087) | +0.0106 (+0.0040, +0.0187) |
| complete RBBB | -0.0010 (-0.0026, +0.0014) | -0.0023 (-0.0193, +0.0072) | +0.0007 (-0.0002, +0.0023) | +0.0054 (-0.0015, +0.0201) | +0.0024 (+0.0010, +0.0046) |
| incomplete RBBB | +0.0013 (-0.0060, +0.0083) | +0.0885 (+0.0652, +0.1104) | -0.0019 (-0.0046, +0.0003) | -0.0044 (-0.0146, +0.0104) | +0.0177 (+0.0091, +0.0266) |
| LAFB | +0.0008 (-0.0020, +0.0037) | +0.0020 (-0.0103, +0.0108) | +0.0014 (-0.0017, +0.0059) | -0.0014 (-0.0059, +0.0049) | +0.0061 (+0.0014, +0.0123) |
| left axis deviation | -0.0016 (-0.0032, -0.0001) | +0.0089 (+0.0063, +0.0116) | -0.0005 (-0.0014, +0.0001) | -0.0033 (-0.0054, -0.0010) | +0.0054 (+0.0036, +0.0073) |
| LVH | -0.0062 (-0.0116, -0.0003) | +0.1127 (+0.0915, +0.1336) | +0.0034 (+0.0018, +0.0054) | -0.0091 (-0.0160, +0.0001) | +0.0138 (+0.0063, +0.0200) |
| ST depression | -0.0092 (-0.0121, -0.0062) | +0.0916 (+0.0788, +0.1042) | +0.0003 (-0.0007, +0.0014) | -0.0000 (-0.0051, +0.0061) | +0.0123 (+0.0058, +0.0184) |
| T-wave abnormality | -0.0154 (-0.0174, -0.0135) | +0.0759 (+0.0680, +0.0837) | +0.0009 (+0.0003, +0.0015) | -0.0081 (-0.0105, -0.0055) | +0.0109 (+0.0081, +0.0137) |
| any MI | +0.0021 (-0.0014, +0.0055) | +0.0641 (+0.0496, +0.0795) | +0.0046 (+0.0004, +0.0090) | -0.0171 (-0.0245, -0.0098) | +0.0115 (+0.0049, +0.0180) |

- Whitening is the largest gain for 10 of 14 conditions. For AF or flutter, first-degree AV block, complete RBBB
  and LAFB its interval includes zero; for complete RBBB the largest gain is K6 to K7-m4.
- The last step, K7-m4 to P0, is the largest gain for AF or flutter, first-degree AV block and LAFB.
- K6 minus P0 per condition on validation [section `conditions-selected`]: the interval lies below zero for 9 of
  14 conditions and includes zero for sinus bradycardia, sinus arrhythmia, PVC, LAFB and any MI, with the largest
  gaps for incomplete RBBB -0.0134 (-0.0277, -0.0027), ST depression -0.0123
  (-0.0196, -0.0051) and first-degree AV block -0.0122 (-0.0213, -0.0042).

Cosine of each key with P0's weight vector for the same condition, median over the 14 conditions [section
`cosine-p0`]: K1 0.328, K2 0.327, K3 0.281, K4 0.439, K5 0.319, K6 0.426, and for K7 (mean over centres) 0.347,
0.293, 0.223. Per condition the whitened key's cosine runs from 0.264 (complete RBBB, K4) to 0.686 (T-wave
abnormality, K4). The mean keys are closer to P0 than the whitened ones for AF or flutter (0.684 against 0.270)
and complete RBBB (0.763 against 0.264), the conditions where the mean keys also rank as well.

## 4. Source [sections `sources`, `sources-selected`, `pooled-within`]

A multinomial logistic regression on standardized train features separates the six train sources with accuracy
0.810 on train and 0.739 on validation (converged in 907 iterations). Its six coefficient vectors are the source
directions. For reference, 2,000 random unit vectors have an absolute cosine with them of median 0.021, 95th
percentile 0.060 and 99th percentile 0.079 (`sources.json`).

| construction | median over conditions of the largest absolute cosine with a source direction | largest |
|---|---|---|
| K1 | 0.013 | 0.039 |
| K2 | 0.012 | 0.037 |
| K3 | 0.012 | 0.031 |
| K4 | 0.115 | 0.301 |
| K5 | 0.006 | 0.010 |
| K6 | 0.052 | 0.119 |
| K7-m4 | 0.098 | 0.254 |

- The whitened key K4 is aligned with a source direction beyond the random 99th percentile for at least half of
  the conditions (median 0.115), most for any MI (0.301); centring by source (K6) halves the median alignment. The mean keys are
  weight vectors of another kind (mean differences, dominated by high-variance directions), so their small cosines
  say little.
- The share of a score's validation variance that lies between source means [section `sources`] has a median
  over conditions of 0.029 for K4, 0.026 for P0, 0.022 for K1 and 0.000 for K6. For any MI it is 0.181 for K4 and
  0.161 for P0.
- Pooled minus within-source AUROC [section `pooled-within`]: for any MI, +0.027 for K4, +0.035 for P0 and -0.075
  for K6, the largest difference of the table. The validation MI positives sit mostly in PTB-XL (520) and
  CPSC-Extra (272) [section `support`]; K4 and P0 keep between-source differences in their MI scores, K6 removes
  them.
- The largest absolute cosine of a K6 key with a source direction is 0.119, for PVC and the Ningbo direction
  [section `sources-selected`].

## 5. Co-occurring and related conditions [sections `pairs-related`, `pairs-summary`, `pairs-top`, `projected-own`]

For an ordered pair (A, B): AUROC of key A's score for B-positives against B-negatives among records negative for
A. 0.5 means key A ignores B. Reported for all 180 of 182 ordered pairs with at least 100 such positives, pooled and
as the mean within-source AUROC (sources with at least 20 of each).

How many pairs the keys respond to, within-source AUROC [section `pairs-summary`]:

| model | keys | above 0.70 | below 0.30 | median distance from 0.5 |
|---|---|---|---|---|
| K1 | raw | 51 | 10 | 0.154 |
| K1 | other 13 keys projected out | 0 | 0 | 0.032 |
| K2 | raw | 52 | 12 | 0.154 |
| K4 | raw | 6 | 0 | 0.034 |
| K4 | projected | 31 | 3 | 0.095 |
| K6 | raw | 6 | 0 | 0.039 |
| K6 | projected | 25 | 3 | 0.086 |
| K7-m4 | raw | 11 | 0 | 0.061 |
| P0 | raw | 38 | 2 | 0.125 |

- Mean keys respond to many other conditions. Projecting the other 13 keys out removes that response but costs the
  mean keys' own AUROC: the mean over conditions of K1 drops from 0.9162 to 0.8523 [section `projected-own`].
- Whitened keys already respond to few other conditions, fewer than P0. Projecting the other keys out of a whitened
  key increases its response to other conditions and lowers its own mean AUROC by about 0.010 (K6 0.9644 to
  0.9541), most for ST depression (0.937 to 0.884) and LAFB (0.975 to 0.930).

The related pairs named in the protocol, for K6 [section `pairs-related`]:

| key A | target B | B positives among A negatives | K6 pooled | K6 within | K6 projected, within | P0 within |
|---|---|---|---|---|---|---|
| complete RBBB | incomplete RBBB | 243 | 0.796 | 0.812 | 0.847 | 0.898 |
| incomplete RBBB | complete RBBB | 427 | 0.780 | 0.811 | 0.980 | 0.918 |
| sinus tachycardia | AF or flutter | 2310 | 0.409 | 0.439 | 0.840 | 0.703 |
| AF or flutter | sinus tachycardia | 1824 | 0.422 | 0.516 | 0.887 | 0.683 |
| LAFB | left axis deviation | 460 | 0.920 | 0.917 | 0.666 | 0.979 |
| left axis deviation | LAFB | fewer than 100 | | | | |
| LVH | ST depression | 193 | 0.682 | 0.701 | 0.513 | 0.772 |
| LVH | T-wave abnormality | 1807 | 0.583 | 0.588 | 0.537 | 0.649 |
| ST depression | LVH | 109 | 0.695 | 0.652 | 0.427 | 0.627 |
| T-wave abnormality | LVH | 212 | 0.611 | 0.605 | 0.558 | 0.648 |

- Complete and incomplete RBBB: each key scores the other condition's positives high (0.81), less than P0 does
  (0.90-0.92).
- Sinus tachycardia and AF or flutter: K6's tachycardia key scores AF-or-flutter records slightly below other
  negatives (0.439); P0's tachycardia model scores them higher (0.703).
- LAFB and left axis deviation: the LAFB key scores left-axis-deviation records high (0.917; P0 0.979), as the two
  definitions overlap. Fewer than 100 LAFB positives are negative for left axis deviation.
- LVH and ST or T changes respond to each other weakly to moderately (0.59-0.70).
- The MI conditions among themselves: v5 has one MI condition (any MI), so there is no MI pair.
- The pairs K6 responds to most are LAFB on left axis deviation (0.917), the two RBBB pairs (0.81), and any MI on
  LAFB (0.780) and on left axis deviation (0.742) [section `pairs-top`].

So co-occurrence is not where whitened keys lose: they separate related conditions more than P0 does, and removing
the other keys' directions hurts them.

## 6. Number of reference ECGs [sections `references`, `references-conditions`, `references-reach`]

Keys built from n positives and n negatives drawn from train, 50 draws per n; the mean over the 14 conditions of
the median within-source AUROC, with the mean 5th and 95th percentiles:

| n per class | K2 | K4, covariance of all train records | K4, covariance of the drawn records |
|---|---|---|---|
| 5 | 0.837 (0.742-0.886) | 0.599 (0.551-0.643) | 0.838 (0.742-0.890) |
| 10 | 0.863 (0.804-0.896) | 0.638 (0.592-0.679) | 0.861 (0.790-0.902) |
| 20 | 0.884 (0.845-0.910) | 0.686 (0.645-0.723) | 0.886 (0.837-0.916) |
| 50 | 0.900 (0.876-0.919) | 0.764 (0.734-0.791) | 0.901 (0.872-0.922) |
| 100 | 0.907 (0.890-0.921) | 0.823 (0.798-0.845) | 0.913 (0.892-0.928) |
| 200 | 0.910 (0.898-0.920) | 0.873 (0.854-0.889) | 0.925 (0.911-0.937) |
| 500 | 0.911 (0.905-0.918) | 0.921 (0.908-0.931) | 0.942 (0.934-0.949) |
| all | 0.913 | 0.961 | 0.9635 |

- The mean-difference key K2 is within 0.01 of its all-record value at 10 per class for 4 conditions, at 50-200
  for 9, and at 500 for incomplete RBBB [section `references-reach`].
- Whitened with the covariance of the drawn records, the key behaves like K2 up to 50 per class (0.838 against
  0.837 at 5, 0.901 against 0.900 at 50). It gains over K2 from about 100 per class, and at 500 per class it is
  still 0.021 below the key built on all train records; for 7 of 14 conditions it does not come within 0.01 of the
  full key at 500.
- Whitening with the nearly unshrunk covariance of all train records (shrinkage 0.0007, `receipts/references.json`;
  no labels used) is worse than no whitening below 500 per class: 0.599 at 5 per class.

## 7. Stability [sections `stability`, `stability-auroc`, `leave-family-out`]

Keys built on 200 patient-cluster bootstrap resamples of train (patients for PTB-XL, records otherwise). The
lowest value over the 14 conditions of each summary:

| construction | median pairwise cosine | 5th percentile pairwise | median cosine with the full key | 5th percentile |
|---|---|---|---|---|
| K1 | 0.9856 | 0.9775 | 0.9928 | 0.9891 |
| K2 | 0.9867 | 0.9793 | 0.9933 | 0.9901 |
| K3 | 0.9914 | 0.9867 | 0.9957 | 0.9936 |
| K4 | 0.7296 | 0.7000 | 0.8537 | 0.8362 |
| K5 | 0.9801 | 0.9625 | 0.9763 | 0.9666 |
| K6 | 0.7304 | 0.7008 | 0.8546 | 0.8367 |

- For K6 the median pairwise cosine runs from 0.730 (T-wave abnormality) to 0.934 (sinus bradycardia).
- The validation AUROC of the resampled keys barely moves: for K6 the mean over conditions has 5th percentile
  0.9611, median 0.9619 and 95th percentile 0.9628; the per-condition 5-95% spread has median 0.0061 and maximum
  0.0113 [section `stability-auroc`]. The whitened direction is not identified by the data, yet any of the
  resampled directions ranks about equally well.

Leave one source family out: the key built without the family, scored on that family's validation records (mean
within-source AUROC over the family's sources), against the key built with it [section `leave-family-out`].
Mean change over the conditions that enter the family's sources, and the largest loss:

| construction | PTB-XL | Chapman-Ningbo | Georgia | CPSC |
|---|---|---|---|---|
| K2 | -0.0121 (largest -0.0703) | +0.0016 (-0.0124) | -0.0049 (-0.0308) | -0.0182 (-0.0929) |
| K4 | -0.0240 (-0.0749) | -0.0189 (-0.1444) | -0.0078 (-0.0359) | -0.0244 (-0.1678) |
| K6 | -0.0251 (-0.0884) | -0.0190 (-0.1440) | -0.0085 (-0.0375) | -0.0257 (-0.1680) |

For K6 the largest losses are ST depression without CPSC (-0.1680 (-0.1951, -0.1413)) or without Chapman-Ningbo
(-0.1440 (-0.1771, -0.1132)), any MI without PTB-XL (-0.0884 (-0.1082, -0.0696)), ST depression without PTB-XL
(-0.0742), T-wave abnormality without Chapman-Ningbo (-0.0577) and incomplete RBBB without PTB-XL (-0.0529,
with 338 train positives left). Whitened keys lose more than K2 when a family is left out; what a condition's key
learns from one family does not carry fully to another, consistent with the source-coding differences of MLC-03
Part B.

## 8. Agreement with the defining measurement [sections `measurements`, `measurements-constructions`]

Spearman correlation of the K6 score and of P0's logit with the condition's measurement, within each validation
source with at least 30 measured records (median and range over the six sources). Measurements are MLC-03 Part
B's: consensus heart rate, QRS, PR and axis; method B's voltages, lowest ST 60 ms after the QRS, lowest T in I, II
and V3-V6, and longest Q in the Q-group leads. PR and RR irregularity are unreliable (Part B) and marked.

| condition | measure | records | K6 median rho (range) | P0 median rho (range) |
|---|---|---|---|---|
| sinus tachycardia | heart rate | 14,445 | 0.422 (0.20 to 0.53) | 0.609 (0.45 to 0.70) |
| sinus bradycardia | heart rate | 14,445 | -0.656 (-0.81 to -0.53) | -0.799 (-0.85 to -0.74) |
| sinus arrhythmia | RR range, unreliable | 14,968 | 0.313 (0.28 to 0.41) | 0.301 (0.13 to 0.51) |
| AF or flutter | RR irregularity, unreliable | 15,025 | 0.329 (0.16 to 0.45) | 0.340 (0.18 to 0.51) |
| first-degree AV block | PR, unreliable | 7,889 | 0.272 (0.20 to 0.30) | 0.749 (0.72 to 0.77) |
| complete RBBB | QRS | 14,425 | 0.259 (0.22 to 0.55) | 0.521 (0.41 to 0.72) |
| incomplete RBBB | QRS | 14,425 | 0.119 (0.07 to 0.32) | 0.158 (0.12 to 0.38) |
| LAFB | axis | 14,584 | -0.223 (-0.44 to -0.15) | -0.432 (-0.69 to -0.34) |
| left axis deviation | axis | 14,584 | -0.448 (-0.54 to -0.33) | -0.534 (-0.73 to -0.38) |
| LVH | Sokolow-Lyon | 15,025 | 0.516 (0.39 to 0.60) | 0.579 (0.41 to 0.65) |
| LVH | Cornell | 15,025 | 0.304 (0.24 to 0.42) | 0.387 (0.30 to 0.50) |
| ST depression | lowest ST | 15,025 | -0.489 (-0.61 to -0.31) | -0.650 (-0.70 to -0.48) |
| ST depression | lowest T | 15,025 | -0.385 (-0.52 to -0.24) | -0.573 (-0.66 to -0.34) |
| T-wave abnormality | lowest ST | 15,025 | -0.414 (-0.52 to -0.35) | -0.464 (-0.56 to -0.35) |
| T-wave abnormality | lowest T | 15,025 | -0.704 (-0.74 to -0.68) | -0.746 (-0.80 to -0.70) |
| any MI | longest Q | 15,025 | 0.176 (0.15 to 0.48) | 0.130 (0.06 to 0.42) |

- For K6 and P0 every correlation has the sign the definition implies, in every source.
- K6 follows the measurement less than P0 for 14 of the 16 rows; it follows Q duration a little more for MI and
  the unreliable RR range a little more for sinus arrhythmia. The largest difference is PR for first-degree AV block (0.27 against 0.75), on the unreliable measure.
- Median over the 13 reliable condition-measure rows of the absolute correlation [section
  `measurements-constructions`]: K1 0.556, K2 0.568, K3 0.512, K4 0.407, K6 0.414, K7-m4 0.437, P0 0.534. The
  mean keys follow the measurement best and rank worst; the whitened keys rank well and follow it least.
- The binned curves (score median per measurement decile and source) are in `measurement_curves.csv` for every
  construction. No measurement is used for PVC; the protocol names none.

## 9. Review sheets, non-clinical reading [sections `review`, `review-measured`]

For each condition, the 12 highest-scoring validation negatives and the 12 lowest-scoring positives of K6 are
plotted (`review/<condition>__top_negatives.png`, `__bottom_positives.png`: leads I, II, V1, V5 over the first
10 s with the Part B measurements; one standard 12-lead sheet per record under `review/<condition>/`). The analyst
read all 28 sheets by eye; the reading is `review/reading.csv`. This is a non-clinical reading: it records what is
measurable and visible, not a diagnosis, and no cardiologist has seen these sheets.

Records whose measurement meets the condition's measured bound, among those measured [section `review-measured`]:

| condition | bound | top negatives | bottom positives |
|---|---|---|---|
| sinus tachycardia | rate above 100 | 9 of 9 | 10 of 10 |
| sinus bradycardia | rate below 60 | 12 of 12 | 7 of 9 |
| sinus arrhythmia | RR range above 120 ms (unreliable) | 12 of 12 | 4 of 12 |
| AF or flutter | irregular RR (unreliable) | 11 of 12 | 6 of 12 |
| first-degree AV block | PR above 200 ms (unreliable) | 4 of 4 | 0 of 5 |
| complete RBBB | QRS at least 120 ms | 10 of 10 | 3 of 10 |
| incomplete RBBB | QRS 110-119 ms | 1 of 10 | 1 of 12 |
| LAFB | axis -45 to -90 | 11 of 12 | 3 of 12 |
| left axis deviation | axis -30 to -90 | 12 of 12 | 2 of 12 |
| LVH | Sokolow-Lyon at least 3.5 mV | 0 of 12 | 0 of 12 |
| ST depression | lowest ST at most -0.05 mV | 12 of 12 | 7 of 12 |
| T-wave abnormality | lowest T below 0.05 mV | 12 of 12 | 2 of 12 |
| any MI | longest Q at least 30 ms | 8 of 12 | 1 of 11 |

What the reading records:

- The highest-scoring negatives mostly meet the measured definition of the condition they are negative for:
  all 12 for left axis deviation (axis -45 to -78), ST depression and T-wave abnormality, 11 of 12 for LAFB, all
  10 measured for complete RBBB (QRS 120-166 ms). For PVC, most panels show a premature beat of different shape.
  These are measurable disagreements between label and record, not shown to be label errors.
- The lowest-scoring positives often do not meet their definition: left axis deviation 2 of 12, T-wave
  abnormality 2 of 12 (all Georgia), MI 1 of 11, LAFB 3 of 12, complete RBBB 3 of 10 (9 of 12 from CPSC 2018, four
  at 110-114 ms, just above rule C9's 110 ms cut). For PVC, most panels show no premature beat in the 10 s shown
  (6 CPSC 2018, 5 Georgia).
- Where the measurement does meet the definition, the low-scoring positives carry another finding: wide QRS,
  premature beats, irregular RR or noise for the rate conditions; four regular narrow-complex rhythms at 166-214
  per minute among AF-or-flutter positives.
- Sinus bradycardia: 9 of the 12 top negatives are PTB-XL records with consensus rate 41-55 per minute, still
  negative after rule C10.
- LVH: no top negative meets Sokolow-Lyon 3.5 mV (rule C12 removed those), and no bottom positive does either
  (0.43-3.02 mV).
- Data quality: one validation record, `cpsc_2018_extra:Q0400`, is flat in all 12 leads in the first 10 s (manifest
  `n_flat_leads` 12, the only such included train or validation record [section `data-quality`]) and keeps a
  positive MI label in v5. One
  MI positive (`cpsc_2018_extra:Q3289`) has rail-height deflections in I and V5 and a baseline-step flag.

## 10. Selection [section `selection`]

The selection record was written by the `select` step after every validation analysis and committed as `83c9fe7`
before the test step ran; the test step refuses to open a test role unless that file is committed unchanged.

- Single key: the highest validation selection metric among K1 to K6 is K6 (0.9643; K4 0.9634).
- K7: K7-m4 (0.9617; K7-m2 0.9602, K7-m8 0.9542).
- The protocol's "validation macro AUROC" can be read as either metric of section 2, and the two select different
  constructions: the pooled reading would have selected K4 and K7-m2. The selection metric was stated as the rule
  in a message to the orchestrator before any key was scored, and no reply had arrived when the record was
  committed. The orchestrator's ruling for the same reading, with the pooled value reported beside it and the
  seven readings of the Departures section accepted, reached this agent after the test read; it is recorded as
  Amendment 1 of the protocol (`4a9ae78`), whose first paragraph states that it was written after the selection
  record and the test read.

## 11. The test read [sections `test`, `test-conditions`]

Scored once, on v5 labels: the selected K6 and K7-m4 (keys fitted on train, as stored at the `keys` step) and the
stored v5-trained P0, at PTB-XL fold 10 and the SPH evaluation half. The SPH calibration half was not opened.
Conditions with at least 100 positives and 100 negatives in the role enter: 9 at fold 10 (AF or flutter, PVC,
incomplete RBBB, LAFB, left axis deviation, LVH, ST depression, T-wave abnormality, any MI) and 12 at SPH (all but
LAFB and left axis deviation). 2,000 patient-cluster resamples, no redraw needed; 1,903 patients at fold 10 and
12,246 at SPH. For K6 at SPH, which has no train mean, the records were centred on the unlabelled mean of the
scored evaluation half: this uses test-role features but no test-role labels, and a constant shift leaves the
within-role ranking unchanged.

The macro table and the decision are at the top of this report. Per condition (the protocol's reading applied
without a decision; "equivalent" means the lower bound is above -0.010):

| role | condition | positives | K6 | P0 | K6 minus P0 | reading |
|---|---|---|---|---|---|---|
| SPH | sinus tachycardia | 359 | 0.999 | 0.999 | -0.0002 (-0.0006, +0.0000) | equivalent |
| SPH | sinus bradycardia | 1298 | 0.993 | 0.995 | -0.0020 (-0.0043, -0.0003) | equivalent |
| SPH | sinus arrhythmia | 781 | 0.947 | 0.940 | +0.0074 (+0.0042, +0.0105) | equivalent |
| SPH | AF or flutter | 374 | 1.000 | 1.000 | -0.0000 (-0.0001, +0.0000) | equivalent |
| SPH | PVC | 511 | 0.990 | 0.994 | -0.0045 (-0.0095, -0.0011) | equivalent |
| SPH | first-degree AV block | 112 | 0.995 | 0.995 | -0.0002 (-0.0011, +0.0007) | equivalent |
| SPH | complete RBBB | 350 | 0.998 | 0.999 | -0.0010 (-0.0031, +0.0007) | equivalent |
| SPH | incomplete RBBB | 619 | 0.848 | 0.867 | -0.0191 (-0.0295, -0.0094) | undecided |
| SPH | LVH | 100 | 0.992 | 0.996 | -0.0036 (-0.0120, +0.0009) | undecided |
| SPH | ST depression | 497 | 0.913 | 0.940 | -0.0264 (-0.0354, -0.0180) | clearly worse |
| SPH | T-wave abnormality | 1633 | 0.971 | 0.972 | -0.0013 (-0.0031, +0.0003) | equivalent |
| SPH | any MI | 118 | 0.985 | 0.988 | -0.0033 (-0.0082, +0.0009) | equivalent |
| fold 10 | AF or flutter | 155 | 0.973 | 0.982 | -0.0093 (-0.0254, +0.0031) | undecided |
| fold 10 | PVC | 114 | 0.993 | 0.992 | +0.0009 (-0.0005, +0.0026) | equivalent |
| fold 10 | incomplete RBBB | 112 | 0.963 | 0.965 | -0.0013 (-0.0079, +0.0050) | equivalent |
| fold 10 | LAFB | 160 | 0.982 | 0.983 | -0.0016 (-0.0058, +0.0020) | equivalent |
| fold 10 | left axis deviation | 134 | 0.997 | 0.998 | -0.0008 (-0.0025, +0.0004) | equivalent |
| fold 10 | LVH | 213 | 0.953 | 0.956 | -0.0025 (-0.0110, +0.0050) | undecided |
| fold 10 | ST depression | 101 | 0.929 | 0.934 | -0.0051 (-0.0189, +0.0094) | undecided |
| fold 10 | T-wave abnormality | 255 | 0.918 | 0.923 | -0.0055 (-0.0120, +0.0012) | undecided |
| fold 10 | any MI | 531 | 0.908 | 0.911 | -0.0026 (-0.0095, +0.0047) | equivalent |

- At SPH, K7-m4 reads clearly worse than P0 for incomplete RBBB (-0.0322) and ST depression (-0.0254), and
  equivalent for 8 conditions [section `test-conditions`].
- ST depression is where the keys fall furthest behind at SPH, as on validation (section 3) and in the
  leave-family-out test, where the ST-depression key lost the most when a family was left out (section 7).
- The validation gap of K6 to P0 (-0.0054) and the SPH gap (-0.0045) are similar.

## Departures and doubts

- **Metric reading.** Selection used the within-source selection metric (section 10); the orchestrator was told
  before any score; its ruling for the same reading did not arrive before the run ended, and the reading is
  recorded as Amendment 1 of the protocol, written after the run. The pooled reading would have selected K4 and
  K7-m2. Neither K4 nor K7-m2 was scored on a test role.
- **Readings stated to the orchestrator before any score**, followed as written:
  - K3's "records with no positive label" are train records with no positive among the 14 conditions, undefined
    labels allowed.
  - K5 averages the train sources with at least 20 positives and 20 negatives of the condition (3 to 6 sources,
    `keys_summary.json`).
  - K6 centres every record on its train source's mean; SPH, which has no train mean, on the unlabelled mean of the
    scored half (test-role features, no labels).
  - K4, K6 and K7 use Ledoit-Wolf on the class-centred residuals of the defined train records, both classes
    pooled (tested against scikit-learn's estimator on repeated rows).
  - N1 uses the 20 nearest train records among those where the condition is defined.
  - In analysis 5, n = all is the key on all train records, one value without draws; draws are of records.
- **Ordering of one integrity check.** The protocol lists the v5 label counts per role among the integrity checks
  before any analysis. Counting test-role labels before the selection commit would have parsed them, which the
  orchestrator's rules forbid; train and validation were checked first and the two test roles at the start of the
  test read (28 cells, all equal).
- **Analyses where the protocol left a choice.**
  - Stability resamples cover K1 to K6. K7's centres have no fixed order between resamples, so cosines between
    them are not defined.
  - Leave-family-out scores the family's validation sources within source; for K6 the held-out family's records
    are centred on the full-train source means, a constant per source.
  - The pair analysis and the measurement agreement were computed for every construction (`pairs.csv`,
    `measurement_spearman.csv`); the report shows the selected K6 and P0. "The MI conditions among themselves"
    has no pair in v5.
  - For the ST and T conditions both ST level and T amplitude are reported. PVC has no measurement.
  - Analysis 5 reports the mean within-source AUROC; pooled values are in `references.csv`.
- **The review step** first stopped on a caption bug before writing any file; it was fixed (`eb6f3db`) and rerun.
  All other steps ran at `fad4ef5`; the receipts record the commit, no modified tracked file, and the SHA-256 of
  every code file and output.
- **The selection record** was annotated by hand with the paragraph on the metric reading after the `select` step
  wrote it and before it was committed; its JSON block is unchanged, and `receipts/select.json` holds the SHA-256
  of the version before the annotation.
- **Optimism.** Validation chose between constructions and also gives every descriptive number, so the validation
  numbers of K6 are optimistic; P0's C was also chosen on validation in MLC-03 Part D. The test read is the guard.
- **SPH** has been read by earlier experiments and is development data; its labels are cleaned less than the
  other sources' (MLC-03 Amendment 2).
- **Data quality noted, not acted on:** the all-flat validation record with a positive MI label, and PTB-XL sinus
  bradycardia negatives at 41-55 per minute that rule C10 did not reach (section 9).
- **Environment.** CPU only, 8 BLAS threads, `OPENBLAS_CORETYPE=Haswell`; P0's weights reproduced its stored
  predictions within 4.1e-15 under this kernel. The uncorrected-memory-error counter was 1 in every receipt, as in
  MLC-03.

## Limits

- The labels are annotation proxies from sources that code conditions differently; a key inherits that coding.
  Sections 7 and 9 show where.
- The whitened keys were built from 56,133 train records. Section 6 shows that with 100 or fewer records per class
  a key reaches a mean within-source AUROC of about 0.91 at best.
- Keys are compared with one classifier on one embedding. Token-level keys were out of scope.
- The review reading is one non-clinical reader's.

## How to rerun

From the worktree, with the `data`, `outputs` and `.venv` links to the main checkout, CPU only:

```
export PYTHONPATH=. OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_CORETYPE=Haswell
for step in integrity keys evaluate pairs sources references measurements stability review select; do
    .venv/bin/python -m scripts.experiments.run_key05_condition_keys $step
done
# commit docs/key05-selection-record-2026-11-11.md, then, once:
.venv/bin/python -m scripts.experiments.run_key05_condition_keys test
```

Wall times on 8 threads (`receipts/`): integrity 3 s, keys 138 s, evaluate 113 s, pairs 27 s, sources 34 s,
references 301 s, measurements 2 s, stability 3,163 s, review 57 s. The test step refuses to run a second time.
Tables: `PYTHONPATH=. .venv/bin/python -m scripts.reports.key05_tables [--section NAME]`. Tests:
`PYTHONPATH=. .venv/bin/python -m pytest tests/test_key05_keys.py` (15 tests, synthetic data; the one that reads
local data skips without it).

## Files

Code: `ecg_experiment/key05_keys.py` (constructions), `key05_data.py` (records, features, P0 and the test-role
guard), `key05_analysis.py` (analyses), `key05_review.py` (measurements and sheets),
`scripts/experiments/run_key05_condition_keys.py`, `scripts/reports/key05_tables.py`, `tests/test_key05_keys.py`.

Outputs in `outputs/key05_condition_keys_20261111/`: `integrity.json`; `keys.npz`, `keys_summary.json`,
`scores_validation.npz`; `constructions.csv`, `conditions.csv`, `per_source.csv`, `ladder.csv`, `cosine_p0.csv`,
`draws_validation.npz`, `resamples_validation.json`; `pairs.csv`, `projected_own_auroc.csv`; `source_cosines.csv`,
`source_variance_share.csv`, `source_directions.npz`, `sources.json`; `references.csv`; `stability_cosines.csv`,
`stability_auroc.csv`, `leave_family_out.csv`; `measurement_spearman.csv`, `measurement_curves.csv`,
`measurements.json`; `review/` (28 sheets, 336 record sheets, `records.csv`, `reading.csv`); `selection.json`;
`test.json`, `test_conditions.csv`; `receipts/` and the step logs.
