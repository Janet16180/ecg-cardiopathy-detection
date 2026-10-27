# Scientific interpretation of matched background and held-out editor transfer

21 October 2026. The existing generator-trained B localizers passed the six prospectively defined
numerical-transfer checks on the fixed real-background/held-out-editor panel. Independent audit36
passed with zero mismatches. This is a useful narrow bridge beyond generated backgrounds: the
models respond to some added numerical changes on real ECG inputs, including a target waveform
construction absent from their training. It is not validation of real disease localization.

The [protocol](ssl-synthetic-localization-background-editor-transfer-v1.md),
[design review](ssl-synthetic-localization-background-editor-transfer-science-review-v1.md) and
[source review](ssl-synthetic-localization-background-editor-transfer-source-review-v1.md) were
committed before new outcomes. This review reads actual aggregate outputs, per-fit/mode summaries
and terminal receipts; it adds no scores, raw clinical reads or training. No arm is selected.

## What passed, and how large it is

For fixed B on real backgrounds with the biphasic editor, mean direct AP is
0.032520172625431776. The three paired components are:

| Prespecified contrast | Mean | Conditional 95% interval |
|---|---:|---|
| Edited-map AP minus same-target nuisance-map AP | 0.021488701 | [0.011647258, 0.034289499] |
| Direct AP minus absolute observed-input amplitude AP | 0.010528918 | [0.000416452, 0.023331458] |
| Direct AP minus fixed donor-mask AP | 0.019137361 | [0.006767303, 0.033732480] |

Each point and lower bound is positive, meeting all six frozen checks. Nuisance-site AP is
0.011031471982812758; amplitude AP is
0.02199125442500118. These controls matter: final map overlap
alone could reflect pre-existing background ranking, a simple amplitude cue or support geometry.
The response and comparator gains provide bounded evidence beyond those explanations.

The effect is modest. The amplitude-gain lower bound is only
0.0004164524580701302, and no minimum useful-effect margin was required. AP around
0.0325 is a ranking statistic on sparse numerical support, not 3.25% accuracy, a sensitivity,
a calibrated probability or evidence of precise clinical boundaries.

All six B real/biphasic fit-by-mode point rows are retained. Every listed point gap is positive,
but repeated-edit amplitude gains are small. These are points, not individually significant
training-population conclusions:

| Fixed fit | Mode | Direct AP | AP response | Amplitude gain | Donor gain |
|---|---|---:|---:|---:|---:|
| 1 | focal | 0.027548 | +0.024521 | +0.021830 | +0.018394 |
| 1 | repeated | 0.040003 | +0.021440 | +0.001738 | +0.021530 |
| 2 | focal | 0.023468 | +0.020557 | +0.017750 | +0.017291 |
| 2 | repeated | 0.043889 | +0.024755 | +0.005625 | +0.024113 |
| 3 | focal | 0.020089 | +0.017433 | +0.014371 | +0.013768 |
| 3 | repeated | 0.040124 | +0.020226 | +0.001860 | +0.019728 |

## Source and editor transfer remain incomplete

| Background source | Target editor | B AP | P AP | K AP | PK AP |
|---|---|---:|---:|---:|---:|
| generated | hann | 0.100072 | 0.202605 | 0.086129 | 0.141980 |
| generated | biphasic | 0.073252 | 0.108126 | 0.051369 | 0.071662 |
| real | hann | 0.043713 | 0.074506 | 0.048676 | 0.066598 |
| real | biphasic | 0.032520 | 0.042486 | 0.028624 | 0.036108 |

All four arms lose direct AP on the real cohort relative to the paired generated cohort, and on
the held-out biphasic editor relative to the original Hann editor. These are exploratory, fixed-
cohort comparisons. Shared geometry and nominal kernel L2 do not equalize every kernel peak,
mask occupancy, morphology or acquisition characteristic, and sources are not matched patients.
They do not identify a causal physiological generator defect or justify choosing a winning arm.

For B, the smaller absolute editor-related drop on real data produces a positive interaction,
0.015626534816212914. Real AP still starts and ends below generated
AP. A favorable interaction sign must not be presented as better real-data performance.

The three fitted states stay fixed in the shared 2,000 whole-slot bootstrap. These backgrounds
are spent development evidence; fresh64 originally meant held out from localizer fitting, not
untouched after earlier evaluation. Donor assignments and target banks stay fixed. Intervals are
conditional on this cohort, checkpoint set and geometry/control assignment, not a new-patient,
training-population or universal permutation-null result.

## Control responses and the preceding timing result

B sham maximum exceeds the uncalibrated 0.5 cutoff for approximately
92.2% of real case-model outputs versus
58.9% on generated backgrounds.
Sham means no added target editor, not a healthy recording; maxima are not clinical alarm rates.
High background responses therefore remain a practical research concern despite the new transfer
conjunction passing. Lower control scores alone would not establish correct localization either.

The earlier negative exact-N concentration result and this positive intervention-response result
answer different questions and can coexist. The model need not rank an exact released wave marker
highly to change its ranking when a numerical perturbation is added. Neither result establishes
that existing high scores represent disease or that all wave-related attraction is absent.
The old paired/suppression conjunction remains FAIL with fourteen of fifteen gates passing.

The frozen first-slot profile plot displays lead II, an untargeted control channel for both
first-slot modes. It must not be described as a selected-target localization illustration.
Any additional target-lead gallery must be separately bounded, use metadata-selected fixed cases
and preserve the original figure and output catalog; no favorable-map selection is justified.

## Verification and actual operations

The independent reconstruction covers 1536 model-background rows,
6144 positive rows, 192 profile rows and all
2000 by 352 bootstrap values. The declared new-arithmetic
comparison tolerance is 1e-12, with no clipping or gate change; maximum saved-draw difference is
1.3877787807814457e-16. Inputs, masks and frozen identity/closure checks
retain their exact contracts. All 96 fixed CPU model-record replay
checks pass; largest CUDA difference is 8.344650268554688e-07, below the declared 1e-5 absolute tolerance.
No optimizer, generator or new raw-record read occurred.

Main35 used 211.44608792399998 CPU / 262.52704470278695 wall /
51.14985655294731 GPU-lock seconds. Audit36 added 99.84471085999999 CPU /
103.04366322467104 wall seconds and zero GPU. All resource gates pass. Actual cumulative36 costs
are 1071.22394269 CPU and
524.1993697541766 GPU-lock seconds. Historical absent-artifact
costs remain unknown; setup/development is not represented as scientific execution.

| Actual receipt | SHA-256 |
|---|---|
| Main35 summary | `c3ea28105ef99659b0f8854fd32430169ecb48f456b02b8b709fec306da93ba7` |
| Main35 terminal ledger | `8cf80790df2ef3ce3a5c0d2cf539b1e6311697a01a4394e672c7f6b1d819e304` |
| Independent36 summary | `e4f60657ab5c19007bdf91a53b4d27b58408d51dd732ebbc7b7ac3aa9c03afd6` |
| Latest terminal ledger | `d64d6ee087dd5547ae02f02a3adc58e141be0d53b9f1d2474a4f2f01aaef1b12` |

## Next decision: adapt the localizer before the generator

Pulse2Pulse is a usable research starting point: its outputs supported numerical edit learning
that now shows some controlled transfer to real backgrounds. This experiment does not compare
against a model trained on real backgrounds, prove that synthetic training is preferable, or show
that generator fine-tuning would improve localization.

The more direct next hypothesis is a small, prospectively matched localizer-training comparison:
generated-only versus mixed real/generated backgrounds, with the same architecture, loss,
initializations, update budget and controls. Freeze real training/evaluation patient separation
before fitting; retain all current LUDB evaluation material as spent development. Use permitted
training sources and a new declared cohort if needed, with unchanged labels and closed tests.
Keep held-out target constructions out of fitting, report all fixed fits/modes and retain control
responses rather than selecting a favorable checkpoint. Actual-path training profiles and shared
GPU/resource gates must precede any run. This is a recommendation, not training authorization or
a prediction that mixing will necessarily help.

The localizer has 35,660 parameters versus 10,590,997 in the released generator; parameter count
alone is not a measured training-speed forecast. Localizer adaptation directly tests the observed
transfer/control gap against known numerical truth. Fine-tuning Pulse2Pulse instead changes the
background distribution without creating regional disease labels. It would need a separately
justified training objective/recipe (including a discriminator if following its GAN route),
patient-disjoint data, normalization/provenance, resource profiles and renewed generator diversity/
quality evaluation. The current descriptor gap does not establish that this is the bottleneck.

No generated or real ECG is labeled healthy here. Physiological realism, disease conditioning,
true disease masks, general clinical localization and SSL benefit remain unproven. All data,
weights and individual maps stay local. Clinical definitions, closed tests and draft PR67 remain
unchanged; the user still forbids pushing or triggering GitHub runners.
