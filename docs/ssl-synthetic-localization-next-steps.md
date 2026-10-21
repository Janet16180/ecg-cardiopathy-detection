# ECG localization plan after the first experiments

18 October 2026. Continue the real-data self-supervision hypothesis as a controlled research question.
The completed experiments justify improving the intervention benchmark before investing in a GAN.
The clinical objective remains highlighting expert-identified abnormalities; the currently executable
objective is locating a precisely introduced voltage change.

## What the experiments established

The original simulator pilot used matched random-initialized (R), self-supervised (S) and frozen-encoder
models. Its primary S−R average-precision gain was 0.00264, with a 95% interval of
[−0.00355, 0.00833]. It failed the frozen advancement rule. See the
[pilot results](ssl-synthetic-localization-pilot-results.md).

A separate frozen-model study qualified all 192 real ECG backgrounds and 768 variants without a
failure or patient replacement. On 128 evaluation patients, average precision was 0.23209 for R and
0.28940 for S: difference 0.05731 [0.03700, 0.07753]. The improvement was concentrated in repeated
changes; the focal difference was −0.00575 [−0.03358, 0.01955]. These are introduced-change labels,
not native disease annotations. See the [qualification results](ssl-synthetic-localization-real-background-qualification-results.md)
and [transfer results](ssl-synthetic-localization-real-transfer-results.md).

The matched unedited controls materially change the interpretation. At the locations where an edit
would later be introduced, S already had average precision 0.22399 versus R's 0.14473. Adding the edit
increased AP by 0.06541 for S and 0.08736 for R. Both models respond to the intervention, but the
higher edited AP alone does not establish that SSL learned a stronger response. Also, the frozen
record-level thresholds flagged only 34/256 introduced-change positives for S and 16/256 for R.
Ranking performance therefore does not establish a useful screening detector.

## Next controlled experiment

Prepare a separate protocol for learning from real backgrounds with matched controls. Reserve the
existing 500 disjoint TRAIN patients for this purpose; their waveforms remain unread. Generate both
positive modes and both controls for every background, preserving patient groups and nuisance matching.
This tests whether matching reduces the current benchmark's reward for highlighting a plausible
location before any change is present.

Compare R and S with the same architecture, initial head, examples, optimization schedule and label
budget. Use the existing SSL initialization without new pretraining. Freeze the primary paired response
measure and decision rule before training, and report edited AP, baseline-site AP, both positive modes,
control flags and fixed-area coverage together. Any comparison with the old toy-trained models should
be described as a comparison of training recipes; it cannot isolate background realism alone.

Select calibration/evaluation patients new to this follow-up from the remaining original VALIDATION
patients before payload access. They remain previously used PTB-XL development data, rather than an
untouched clinical confirmation set. Exclude all 192 patients used in the completed transfer, every training/pretraining
patient and all protected partitions. Existing transfer results are now development evidence.
Require complete generator qualification and a real profile of the new training path before fitting.
The existing storage and runtime ceilings still apply. This is a proposal, not a launched experiment.

## Expert annotation work

Recover and technically audit the intended expert-annotation development artifacts before comparing
clinical localization methods, including the parallel No Time to Train implementation. Verify record
matching, lead order, timing, resampling and unknown/unannotated regions. Retain the uncertain masks;
model agreement cannot repair them into clinical truth.

The local historical studies distinguish wave-boundary annotations, beat-type annotations, ST/T
change episodes and abnormal-region masks. Each supports a different task. Those previously evaluated
sources are development evidence, and their historical reports do not establish that their arrays are
currently available. The current [setup report](ssl-synthetic-localization-setup-results.md) records
missing ECGAD prerequisites. Until compatible expert truth is available, describe the thesis result as
engineering localization of controlled changes. Keep student-screening labels and operating points
under their existing clinical restrictions.

## When learned generation becomes useful

Consider a conditional GAN only after a controlled comparison identifies a generator limitation that
it can test. Require it to outperform the simple editor at the same patient and label budget on a
fixed endpoint, with tracked changes, matched controls and unchanged-region checks. Visual realism
and a label supplied to the generator do not establish that the generated disease or its supporting
region is clinically correct.

Keep the No Time to Train stream independent. Compare the methods only after a common annotation
contract and permitted development split are frozen. All current changes, patient artifacts and model
outputs remain local; no pushes or PR updates are scheduled.
