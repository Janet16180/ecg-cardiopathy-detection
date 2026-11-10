# KEY-03 results: ECGs behind the reference averages

Completed 6 November 2026 under the [reference-audit protocol](reference-ecg-audit-protocol-2026-11-06.md),
source `438adfbce29a6f5aa279fe1edb577e80b15bf23d`. [Aggregate notebook](../notebooks/ReferenceECGAudit.ipynb) and
[aggregate output](../reports/reference-ecg-audit-2026-11-06/aggregate.json). The twenty-panel
waveform atlas and executed individual-waveform notebook remain local in
`outputs/reference_ecg_audit_20261106/ReferenceECGAtlas.pdf` and `ReferenceWaveforms.ipynb`.
They are not Git artifacts. Every displayed individual ECG is an admitted training reference.

## Main findings

The averages are sensitive to source composition and overlapping annotations. Short-region
selection makes references more aligned with their new keys, while moving those keys away from
the whole-record directions. This geometry is consistent with the earlier poor short-region
transfer; it does not establish a causal explanation or validate new diagnostic rules.

| Pool | References | Exclusive group | Largest family | Share |
| --- | ---: | ---: | --- | ---: |
| MI | 2,041 | 812 | ptbxl | 79.7% |
| STTC | 12,356 | 8578 | chapman_ningbo | 63.7% |
| CD | 6,049 | 3084 | chapman_ningbo | 50.0% |
| HYP | 2,376 | 593 | georgia | 43.6% |
| normal | 7,393 | all | chapman_ningbo | 46.0% |

Only 812 of 2,041 MI references carry MI without another abnormal group: 60.2% co-occur with
another group. HYP is more mixed: 593 of 2,376 exclusive, so 75.0% co-occur. Among HYP references,
1,470 also carry STTC (61.9%). These are cohort annotation frequencies, not disease prevalence.
The reference mean is therefore not an isolated representation of one verified disease.

The averaging procedure standardizes embeddings and normalizes each reference vector to unit
length before averaging; every record has equal literal weight. It does not average raw ECG
waveforms. High raw voltage does not directly receive a larger arithmetic coefficient, although
waveform properties can still affect the embedding and the selected region.

## Source dependence of reference directions

MI references are 79.7% PTB-XL; 45 of the 50 nearest whole-key references are PTB-XL. Removing
that family's vector sum rotates the MI key by 46.2 degrees. STTC references are 63.7%
Chapman/Ningbo; 47 of the 50 nearest whole-key references come from that family. Removing it
rotates STTC by 60.7 degrees. Both effects combine large sample shares, waveform/acquisition
patterns and annotation differences. This is not a causal estimate of device bias.

CD has more distributed nearest references: PTB-XL 20, Ningbo 15, Chapman 5, Georgia 7 and
CPSC 3 among its nearest fifty. The largest family-removal rotation is still 33.2 degrees for
Chapman/Ningbo. HYP's nearest fifty comprise Georgia 24, PTB-XL 13, Ningbo 11 and CPSC-Extra 2;
removing Georgia rotates the whole key by 32.8 degrees.

The notebook provides every source-removal angle for every positive and class-specific normal
key, alongside source shares and nearest-reference counts. No source-balanced model was fit
or scored here. Nearest examples should not be described as clinician-approved typical cases.

## Short-region selection changes the geometry

| Positive key | Short rotation | 5-second rotation | 7.5-second rotation | Whole coherence | Short coherence |
| --- | ---: | ---: | ---: | ---: | ---: |
| MI | 45.72° | 19.39° | 13.64° | 0.228 | 0.419 |
| STTC | 44.67° | 20.87° | 16.00° | 0.155 | 0.266 |
| CD | 38.96° | 13.17° | 9.56° | 0.229 | 0.336 |
| HYP | 47.84° | 26.00° | 18.77° | 0.186 | 0.306 |

Coherence is the norm of the mean of unit reference vectors. A larger value means less vector
cancellation, not greater clinical specificity or predictive accuracy. All four positive short
banks become more coherent while their earlier external macro AUROC fell to 0.774. Five- and
7.5-second keys rotate less, consistent with their measured recovery in KEY-02. Candidate count,
selection extremes and contextual information also change with window length; these mechanisms
are not isolated by the present comparisons.

Normal keys matter too. Although all four use the same 7,393 reference records, their selected
regions depend on the group-specific negative margin. For example the short MI-normal key rotates
39.7 degrees from the common whole-record normal key; five seconds reduces this to 16.5 degrees.
A positive-minus-normal score can change through either key, so examining only positive references
would miss part of the change.

Short-region selection also treats positive and normal references differently. The half-second
window is selected for 57.5% of MI positives, but only 28.9% of MI normals. For STTC, 47.2% of
positives select half a second, while 85.1% of normals select two seconds. Candidate lengths have
different counts (39, 37 and 33); these frequencies are descriptive, not an equal-opportunity test.
The notebook shows start distributions and source-specific mean starts for both roles. A selected
span remains a contextual token region, not a clinical disease boundary.

## Identity, duplicate and quality evidence

There are 6,828 records with known patient identity, representing 6,359 patients: 469 additional
records from repeated known patients. The other 18,172 records lack patient identity. It is
incorrect to call the full pool 25,000 unique patients. Equal-known-patient weighting rotates
whole keys by approximately 0.5–2.1 degrees, much less than dominant-source removal, but cannot
correct repeated patients hidden in the unknown-identity sources.

No repeated canonical or full-signal hashes occur in the admitted pool. This does not exclude
near-duplicates or repeated people. Every record passed the existing admission policy, but 148
have a recorded amplitude-over-10-mV review flag and three have a strong-baseline-wander flag.
A pass and an empty review-flag field are not certification of a noise-free waveform.

All 6,828 PTB reference records have an individual human-validation marker. The 13,002
Chapman/Ningbo references carry source-process physician/secondary-validation evidence; 3,974
Georgia and 1,196 CPSC records carry human-reading/overread source evidence without a per-record
reader-count proof. Missing per-record validation must not be relabeled as software-generated.

## Twenty actual ECG panels

All twenty prespecified review roles resolved to distinct training ECGs. Raw release hashes and
canonical signal hashes passed before rendering. Each panel shows all twelve leads in physical
mV and seconds, with short/five-/7.5-second selection bars. Vertical limits are independent per lead; comparisons must use the numeric
mV ticks, not trace height. This is not a standard calibrated ECG-paper display. Selection included two nearest-family
examples, a low-similarity outlier and a large-selection-change example for each group and normals.
The full local manifest binds those anonymous panels to their original records.

All twenty were finite and had zero flat leads. Their observed maximum absolute amplitudes range
from 1.058 to 4.850 mV, and none has a stored review flag. These twenty geometry-selected examples
are not a random sample for estimating overall signal quality.

Post hoc visual observation: Reference 16, selected for the largest HYP short-region vector change,
shows a pronounced slow V4 level shift. The short selected span is 0–0.5 seconds; the five- and
7.5-second spans start at zero as well and include much more of the recording. Its stored review
flags are empty. This warrants a quality review, but does not prove that the shift caused the
embedding change or invalidate its annotation. Reference 20 likewise illustrates that a normal
annotation proxy can coexist with visible waveform variability. No new clinical diagnosis,
quality exclusion or label correction was made from these plots.

## Verification and limits

All 48 positive/normal key entries across six methods reconstructed within 1e-12 using saved
reference selections; the shared whole-record normal key is repeated across the four groups.
All 24 saved predecessor metric dictionaries reproduced exactly. Input receipt hashes were
verified. The audit took 87.292 seconds on two CPU threads. No new evaluation
prediction or classifier was generated, and no raw evaluation or closed-test waveform was read.

The aggregate notebook checks source denominators, identity counts, symmetric co-occurrence,
key reconstruction and region histograms. The waveform notebook is local and has not received
cardiologist adjudication. Clinical interpretations remain outside this audit.

One import-order lint finding was discovered after the audit source had been committed and
recorded in its scientific receipt. The pinned source remains unchanged; this style finding
has no numerical effect. The notebook builder and scientific reconstruction tests are checked
separately. Detailed delivery verification is recorded alongside the aggregate report.

## Follow-up priorities

The evidence favors a prospective source- and co-annotation-controlled reference-bank comparison
before assuming a single unweighted mean captures an isolated condition. Reference 16 is a concrete
quality-review candidate for the cardiologist. Do not automatically remove it or redefine HYP.
A stricter representative-reference bank needs a new protocol and clinical input for any label change.
The lead-preserving ECG-JEPA experiment remains a distinct follow-up; none of these geometry
checks evaluates CroPA or confirms a model's clinical reasoning.

Delivery CPU validation: 1,567 passed, two skipped, 78.98 seconds with CUDA disabled.

The executed aggregate notebook has 22 cells and passed 150 consistency checks. The separate
local waveform notebook executed without errors. Source, input artifact and notebook hashes
are bound in [delivery verification](../reports/reference-ecg-audit-2026-11-06/verification.json).
