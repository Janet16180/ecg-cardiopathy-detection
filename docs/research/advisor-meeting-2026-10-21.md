# From ECG alerts to supporting evidence

Advisor meeting brief, 21 October 2026. Branch: `research/ecg-reference-localization`.
[Draft PR #71](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/71).
This document summarizes executed development studies. The full reports retain their original
protocols, controls, failed advancement decisions and limitations.

## The progress to lead with

We now have a working, training-free reference-matching pipeline that reads ECG signals directly,
returns lead/time candidates and can reconstruct its classification margin from signed contextual
beat/token contributions. We also have evidence that reference selection affects explanations and
that a simpler aggregation rule improves their consistency in the fixed development study.

The strongest new control is NTT-10: with exactly the same unique reference patients, equal weighting
of each patient achieved mean beat-set overlap of 0.7708, versus 0.6559 for normalized bank averaging.
This is a conditional consistency result, not a clinical localization score. Four of sixty patients
materially favored bank averaging, so the mean improvement is not universal.

The objective remains: detect possible ECG alterations and identify the ECG lead/time evidence
supporting an alert. No Time to Train supplies a methodological idea; clustering and SHAP help test
the representations and explanations. None identifies the biological cause of heart disease.

## What the advisor's recommended paper contributes

[No time to train! Training-Free Reference-Based Instance Segmentation, v4](https://arxiv.org/abs/2507.02798v4)
addresses image segmentation using a small annotated reference set and frozen pretrained features.
Its method constructs a reference memory, aggregates representations and matches semantic features
to produce instance masks. Training-free means no task-specific weight fitting; its foundation models
were pretrained. Its reported image benchmarks are not ECG results.

The [official implementation](https://github.com/miquel-espinosa/no-time-to-train) uses SAM2 and DINOv2
and requires annotated examples of the target classes. The published recipe specifies large models;
our image experiment used compact models. We did not reproduce the complete original large/large
configuration or its benchmark results. For raw ECGs, we adapted reference matching and frozen-feature
pooling rather than claiming the original image-segmentation algorithm was run unchanged.

In simple terms: show the system examples of the pattern of interest, represent those examples
numerically, and compare a new ECG with them. The crucial research questions are whether similarity
distinguishes the relevant finding and whether the highlighted evidence is clinically meaningful.
The second question remains open because we lack independent specialist evidence labels.

## Image approach: what we actually tried

### NTT-03: can visual matching recognize a supplied reference region?

We rendered physical ECG signals as fixed black traces on white images and ran frozen DINOv2-S/14.
The experiment supplied the named finding, selected lead and alternative candidate windows. There
were six independent target patients, nine separate reference patients, and three annotation categories:
complete left bundle branch block, left anterior fascicular block and left ventricular hypertrophy.
These are existing source annotation names; we did not redefine clinical labels.

Visual reference matching selected the nominated silver answer for all six targets. With references
from the wrong window, it selected zero of six in the primary one-second view. The full ten-second
view also selected six of six, versus two of six with wrong-window references. This shows a functioning
reference-sensitive matching component. A generic contextual QRS-phase comparator also selected all
six answers, so the result did not establish abnormality-specific visual recognition.

We separately tested automatic SAM2-Tiny proposals on three full traces, one per category. Only one
of the three traces had any proposed region with temporal IoU of at least 0.5 with its nominated
interval. Zero of three trials produced a selected region overlapping the nominated interval; one
had no selectable semantic mask because its proposal contained no trace pixels. The best
proposal overlaps were 0.016, 0.014 and 0.879. This proposal stage failed its advancement gate.
These overlaps measure horizontal temporal extent, not pixel segmentation or clinical accuracy.

These endpoints must remain separate: choosing among supplied windows worked in this tiny pilot;
finding the relevant interval automatically did not pass. A different recurring beat may also be
valid evidence, so zero overlap with one nominated interval is not proof of clinical irrelevance.
The study ran in 327.07 seconds on CPU, with no training.

[Full NTT-03 report](../ntt-03-visual-reference-pilot-results.md).

### NTT-04: can visual features distinguish findings within the same wave type?

The next study removed the easy distinction between different wave types. Every ECG received an
automatically centered QRS region. We compared strict project normal, complete right bundle branch
block (CRBBB) and complete left bundle branch block (CLBBB), with twenty target patients per category.
All methods used the same sixty targets, twenty-seven supports, three fixed reference draws and
three supports per category per draw. Thus there were 180 repeated patient-draw predictions, not
180 independent patients.

The image pipeline rendered V1/V6 median beats into fixed images, encoded them with DINOv2-S/14 and
pooled the programmed QRS region. SAM was not used in NTT-04. Balanced accuracy was 82.78%.
Simple physical full-beat cosine matching achieved 92.78%, and physical QRS matching achieved 88.33%.
The primary visual difference from the strongest prespecified baseline was minus 10.00 percentage
points, with a paired patient-bootstrap interval of [-17.73, -2.34]. The visual improvement gate failed.

Visual CLBBB recall was 95.00%, compared with 88.33% for physical full-beat matching, a useful
descriptive success case. Overall performance still favored the physical baseline; this subgroup
observation does not justify a post-hoc fusion. Four selected records had some image clipping,
while the physical vectors were unclipped. This compares our fixed pipelines, not every possible
way of presenting an ECG image.

[Full NTT-04 report](../ntt-04-same-wave-finding-discrimination-results.md).

## Raw ECG approach: two distinct methods

### Simple physical-signal matching

The simple method uses actual voltage samples, without an image encoder or learned signal encoder.
After automatic beat detection, physical morphology is represented as vectors. Reference patients
are combined into finding prototypes; cosine similarity assigns a target to its closest prototype.
Full median-beat morphology scored 92.78% in NTT-04. It is a strong control and the best overall
method in the original matched three-finding comparison. It is not a new disease-diagnosis estimate.

The earlier NTT-02 native candidate-window study used 1,507 questions from 827 ECGs and 769
train-role patients. With three support patients, patient-macro exact released-option agreement
was 75.12% for masked physical morphology, 58.67% for duration and 87.07% for contextual QRS phase.
Morphology beat duration by 16.45 percentage points [13.31, 19.52], but its full advancement gate
also required beating every simple comparator, so the gate failed. NTT-02 is a supplied-candidate
task and cannot be pooled with NTT-04's
record-finding accuracy. Its predecessor annotation replay also accessed reserved-role aggregates;
that historical limitation is disclosed rather than treated as untouched evaluation.

[Full NTT-02 report](../ntt-02-native-reference-transfer-results.md).

### Learned native-signal matching with ECG-JEPA

NTT-05 first built the native lead/time adapter, reference-bank contracts, abstention and redundancy
handling. Its engineering checks did not yet execute learned-native recognition. NTT-06 then loaded
the real frozen ECG-JEPA checkpoint and ran it on the same eighty-seven permitted waveforms as
NTT-04: sixty targets and twenty-seven supports. No image entered the native model.

The model receives ten seconds of signal at 250 Hz from eight leads: I, II and V1-V6. It returns
400 contextual tokens, each with 768 numerical features: fifty 200 ms anchors per lead. We pool
tokens around automatically detected QRS regions in V1/V6, normalize regional vectors and form
reference prototypes. The encoder is frozen; no new weight training was performed.

| Fixed-pipeline method | Balanced accuracy | Correct repeated choices |
| --- | ---: | ---: |
| Physical full median beat | 92.78% | 167/180 |
| Native ECG-JEPA QRS | 91.67% | 165/180 |
| Physical QRS | 88.33% | 159/180 |
| Compact DINO image QRS | 82.78% | 149/180 |

All four rows concern the same sixty unique target people and three fixed support draws. Physical
and visual methods observe median beats; the native model observes the ten-second recording.
This is a pipeline comparison rather than an isolated encoder experiment. The image and native
scores were obtained in separate prospectively specified studies, with historical controls replayed.

Native minus strongest physical baseline was -1.11 percentage points, with a paired interval
[-5.98, +4.31]. Its required gain was at least five points with a positive lower interval bound;
the advancement gate failed. The interval does not establish equivalence or noninferiority.
The descriptive native-minus-image difference is +8.89 points; it is not a new prespecified
confirmatory superiority test.

The informative positive control was native QRS versus native shifted-window pooling: 91.67%
versus 83.33%, a difference of +8.33 points [3.67, 13.01]. Pooling location therefore carries useful
finding information in this study. Native QRS versus full-record V1/V6 pooling was +3.33 points
[-0.17, 7.35], so that comparison was inconclusive. These are exploratory secondary comparisons.

Native recall was 98.33% for normal, 76.67% for CRBBB and 100.00% for CLBBB. The native CLBBB
result is promising descriptively, but all methods used only twenty unique targets per category
and correlated coexisting annotations. It does not validate class-specific model selection.

The run produced 14,802 candidate assignments, representing 4,934 unique target-coordinate candidates
evaluated against three banks, including 674 unique automatic QRS intervals. Each candidate preserves
lead/time coordinates, similarities and margins. These are algorithmic proposals, not expert labels.
The measured development stage took 60.27 seconds on two CPU threads, with no GPU or new training.

[NTT-05 report](../ntt-05-native-signal-adapter-results.md) and
[full NTT-06 report](localization/ntt06-native-encoder/results.md).

## Why neither accuracy table proves correct highlighting

The finding labels describe entire records. They do not say which beat, lead or interval supports
the finding. Current pooling uses fixed automatic QRS windows and V1/V6; the model has not established
independent selection among P, QRS, ST and T evidence. Tokens have 200 ms anchors and retain global
recording context, so an anchored contribution does not imply the evidence originated only there.

The sample is explored PTB development data, not a new hospital or final clinical test. Source
annotations can co-occur: twelve of twenty CRBBB targets also had MI annotations, versus two of
twenty CLBBB targets. This prevents attributing discrimination solely to isolated bundle branch
block morphology. The released encoder checkpoint also lacks an independently auditable
patient-level pretraining manifest.

We should distinguish three tests: does the method propose useful regions; does it recognize the
finding; and does its evidence agree with acceptable expert evidence? We have engineering and
development answers to parts of the first two. We do not have the third answer.

## Labels and review readiness

NTT-01 audited a public ECG reasoning resource. The permitted subset contains 2,499 cases from
2,322 ECGs and 2,174 patients in folds 1-8. Its wave questions provide sparse, task-conditioned
silver evidence rather than exhaustive, independent pathology masks. Option-order and lead-choice
shortcuts matter: a first-option rule reached 64.31% on one lead-choice slice versus a 39.76%
uniform-choice comparator. These data can support qualified engineering tests, not manufacture
specialist ground truth.

NTT-07 prepared thirty blinded review cases: 180 PDF pages, 180 PNGs and thirty pending forms, with
zero completed clinical annotations. An original clipping failure is preserved; the wider-layout
repair uses exactly the same thirty patients. Semantic validation checks coordinates, lead identity,
signal binding, provenance and alternative acceptable evidence. This infrastructure is ready for a
future reviewer, but a cardiologist is currently unavailable. No private review packet is published.

[NTT-01 report](../ntt-01-reference-label-audit-results.md),
[NTT-07 report](localization/ntt07-blinded-review/results.md).

## What the later explanation studies discovered

NTT-08 reconstructs the model's signed winner-versus-runner score margin exactly from contextual
beat/token contributions for all 180 original evaluations. It finds diffuse contributions: the median
effective beat fraction is 0.9662, and the top quarter of beats carries only about 34.4% of absolute
contribution mass. A highlighted top-ranked set therefore omits much of the model's evidence.
Within comparable original reference-bank contrasts, median rank correlation is 0.4045 and median
top-quarter Jaccard is 0.3333. Representation-deletion experiments test model behavior, not physical
waveform causation or specialist relevance.

NTT-09 holds each target's full-pool finding contrast fixed and compares fifty three-shot banks with
ten five-bank ensembles. Mean patient top-positive-quarter overlap increases from 0.4418 to 0.6559,
a +0.2141 conditional gain; all sixty patients pass coverage. However, an ensemble uses fifteen
support slots per finding and six to nine unique people, versus three people in a single bank.
The earlier 0.3333 median is a different estimand and must not be used as this comparison's baseline.

NTT-10 holds each block's exact unique support patients fixed. Method A averages normalized bank
prototypes; B gives each unique support patient equal weight once; C weights patients by their
number of appearances across banks. Mean overlap is A 0.6559, B 0.7708 and C 0.6545. The primary
A-minus-B difference is -0.11498, meeting the frozen symmetric rule favoring B. Fifty patients favor
B, four favor A and six tie; forty-two have a material B advantage, four a material A advantage and
fourteen fall within the +/-0.05 resolution band.

All 1,800 new decompositions, 600 analytic identities, 8,100 within-method pairs and 1,800 matched
block comparisons reproduce independently. This isolates weighting choices within the fixed
development design. It does not prove a population effect, equal computational cost or correct
pathology localization. The common full-nine uniform reference contrast is itself a conditioning
choice. Method B shares that uniform-weighting convention; performance under other contrasts
remains untested.

[NTT-08](localization/ntt08-contextual-contributions/results.md),
[NTT-09](localization/ntt09-reference-contribution-stability/results.md),
[NTT-10](localization/ntt10-matched-union/results.md).

## Clustering both the native model and the reference model

We did cluster both: native ECG-JEPA vectors and the actual reference-affinity outputs from the
No Time to Train adaptation, alongside physical morphology controls. CLUSTER-03 uses 1,993 matched
development patients, split into 1,582 fitting patients and 411 anchors, with reference supports
excluded. Its native/reference partition agreement is only ARI 0.1026, and its worst bank agreement
is 0.3867; the persistence gate fails. Stable individual partitions do not imply a shared disease taxonomy.

CLUSTER-04 demonstrates conditional reference robustness: 381/411 anchors are stable for single
banks and 406/411 for five-bank means. The smaller original group improves from 25/43 to 41/43,
showing why an overall stability percentage can conceal a fragile minority group.

CLUSTER-05 controls dimensions, pooling and scale but still fails native/reference concordance
(ARI 0.1529). CLUSTER-06 then identifies a concrete numerical mechanism: scaling features while
holding a Gaussian mixture's covariance floor fixed changes the effective regularization. Transforming
that floor by the corresponding squared scale restores the original partitions in all 183 controls.
In the fixed regional tied-GMM condition, averaging raises p10 bank ARI from 0.5873 to 0.8416. A secondary
record-level condition worsens from 0.4861 to 0.3858, so the benefit is configuration-specific.

CLUSTER-07 fits projections to regional covariance. Regional variance retained increases from
10.79% to 14.41%, but native/reference agreement remains low (ARI 0.1532) and fails its primary gate.
Heart-rate association remains substantially stronger than association with annotation combinations.
Changing projection alone did not turn these groups into disease categories.

The useful discovery is methodological: reference choice and numerical scale can change clusters,
and native partitions associate strongly with heart rate. These controls help us avoid naming unstable or rate-associated groups as
cardiac diseases. The CLUSTER-01/02 physical studies likewise found morphology structure but failed
their prespecified robustness gates. Full reports supply their separate cohort/selection details.

[Clustering index](clustering/README.md),
[CLUSTER-03](clustering/cluster03-native-reference/results.md),
[CLUSTER-04](clustering/cluster04-bank-robustness/results.md),
[CLUSTER-05](clustering/cluster05-matched-controls/results.md),
[CLUSTER-06](clustering/cluster06-bank-scale/results.md),
[CLUSTER-07](clustering/cluster07-regional-projection/results.md).

## What happened with SHAP

SHAP-01 computes exact grouped explanations for a linear model and changes the reference population
while leaving predictions identical. The leading absolute feature group changes for 175 of 400
patients (43.75%). Thus the explanation's background population is a material modeling choice.
This study concerns grouped physical features, not a validated lead/time mask for ECG-JEPA.
Neither SHAP nor exact score decomposition supplies clinical causal labels.

[SHAP-01 report](../shap-01-linear-background-sensitivity-results.md).

## Verification, delivery and next decision

The NTT-10 integrated local CPU suite passes 1,730 tests with six skipped in 111.42 seconds.
Python code lint, formatting of the four new Python files and the offline package build pass.
At the user's request, further GitHub runner execution is paused; the next push skips CI. The
preceding remote run failed two PDF tests because its runner lacked `pdftotext`. A system-package
workflow repair is prepared for future use; it has not been verified on a GitHub runner.
The wheel's new module matches source bytes. Independent reconstruction checks numerical results,
hashes, reservations and reference/target separation; it does not replace clinical validation.
Source commits executed in isolated copies are retained in receipts; integration uses cherry-picks.

PR #71 remains draft and targets main. The user authorized pushing code and safe aggregate reports;
merging is not authorized. Raw signals, individual features/maps, identities, checkpoints and the
review packet remain local. No closed Challenge/EchoNext test or final frozen evaluation was used.
Historical reports retain their original no-push execution context.

Recommended next decision for the advisor: keep the native reference-matching approach as an
interpretable research tool, with the physical baseline retained. Preserve NTT-10's equal-person rule
as the leading consistency candidate, then prospectively test whether its advantage survives changed
reference pools and contrast definitions. Separately define what acceptable evidence would mean
without pretending that currently unavailable expert annotations exist. The public-silver typed
importer remains future work; no new scientific run is scheduled by this presentation.

An accurate thirty-second explanation is: “We tested both ECG images and raw signals. Images could
match supplied examples, but automatic region proposals struggled. The raw-signal adaptation now
runs and recognizes the three development categories well, although simple physical matching still
wins overall. We can explain its score exactly and have improved consistency by weighting reference
patients equally. We have not yet shown that these highlights are the regions a cardiologist would
choose.”
