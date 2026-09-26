# Investigation: does ECG fine-tuning leave its readout behind?

## The question in plain language

A pretrained ECG encoder turns a waveform into features; a small prediction head
turns those features into a score. If we update both parts together, the encoder
may change faster than the head learns to use its new features. The waveform
information could still be present, even when the final prediction gets worse.
This is a testable **readout mismatch** hypothesis, not a claim that fine-tuning
is generally harmful.

The immediate experiment compares two otherwise matched training runs. In
**Moving**, the encoder can update. In **Frozen**, its learning rate is zero,
although the encoder still participates in backward passes and global gradient
clipping. Both receive the same records, initial weights, head optimizer, and
240 updates. Afterward, a separate fixed-procedure linear head is fit on each
encoder's training features. If Moving's jointly trained head is worse but its
refitted head still works, that supports a mismatch at this fixed endpoint.
It does not isolate why the mismatch arose: encoder drift, optimizer state,
clipping, and head adaptation interact.

## Evidence so far

The [fixed-feature head study](experiment-016-head-mechanism-v8-results.md)
showed that a head can recover good discrimination when the encoder does not
move. The standardized-feature explanation did not pass its own screen. The
[matched encoder-update study](experiment-016-encoder-motion-v9-results.md)
found, on one training-order seed, that Frozen's joint-head AUROC exceeded
Moving's by 0.00607 (paired-patient 95% interval 0.00170 to 0.01090). Moving's
refitted head scored 0.96410, versus 0.95642 for its jointly trained head.
This is evidence for a readout gap under this exact training recipe, not proof
that updating the encoder destroys ECG information.

The first [second-seed attempt](experiment-016-encoder-motion-replication-v10-results.md)
failed before any seed-47 GPU update and supplies no result. The later
[versioned v14 replication](experiment-016-encoder-motion-replication-v14-results.md)
completed the same Moving/Frozen question under seed 47. Frozen minus Moving
joint-head AUROC was +0.008252 (paired-patient 95% interval +0.002745 to
+0.014180), while Moving's refitted head scored 0.962963 versus 0.953001
for its joint head. Both seeds show the same direction on the repeatedly used
development patients. The two seeds do not establish independent patient
generalization or uniquely identify the mechanism.

## Why this may be worth a paper

The useful contribution would be a careful **failure diagnosis for ECG model
adaptation**: measuring the final trained head and the information available to
a freshly fit head under a tightly controlled encoder-update intervention.
That can guide practitioners choosing between a frozen probe and end-to-end
fine-tuning when labels and GPU time are limited. The claim must stay narrow:
these experiments use one pretrained xECG checkpoint, one cleaned public-data
task, one short warmup epoch, and a binary diagnostic-annotation proxy. They do
not measure actual patient health or referral outcomes.

The broad ideas are not new. [Classifier decoupling](https://arxiv.org/abs/1910.09217)
and [fine-tuning distortion](https://arxiv.org/abs/2202.10054) precede this
work. ECG-specific studies already compare frozen and adapted representations,
including [Self-DANA adaptation](https://www.nature.com/articles/s41598-026-71351-2)
and [ECGFounder post-training](https://arxiv.org/abs/2509.12991), which includes
preview linear probing. A [2026 ECG scaling preprint](https://pubmed.ncbi.nlm.nih.gov/42528512/)
also directly studies unlabeled pretraining volume with a different model and
tasks. Therefore neither “refitting a head helps” nor “more ECGs are not always
better” is a defensible novelty claim by itself. The stronger possible paper
question is **when continued ECG representation learning or supervised encoder
motion changes useful information versus when the existing readout simply
fails to follow it**, measured with matched frozen controls and a second
encoder/task. The compact CPC 25k follow-up is a control for one data-size
confounder, not the main paper result.

The CPC and xECG observations remain separate until tested together. The CPC
scaling readout fits a fresh fixed head for every encoder, so its negative
115k result cannot itself be called a stale-head failure. The xECG
Moving/Frozen studies do measure a gap between a joint head and a refitted
head, but do not prove feature information was preserved for other tasks.
The completed [25k CPC follow-up](experiment-019-cpc-25k-readout-results.md)
has a directionally higher development AUROC than the 115k continuation at
both label budgets, but both paired intervals include zero and the unchanged
encoder remains the highest point estimate. It is a useful sampling control,
not evidence of an optimal data size or an independent paper contribution.

## What would change the evidence level

1. Replicate the Moving/Frozen and refit/joint contrasts with another pretrained
   encoder checkpoint or architecture and a distinct ECG dataset/task, holding
   splits and patient identities clean. Two training-order seeds on one encoder
   and one repeatedly inspected development set are already complete.
2. Predeclare a mechanism-focused intervention, such as a frozen-head warmup
   before encoder unfreezing, against the matched Moving/Frozen controls.
   Measure both representation/refit and joint-head performance so any rescue
   can be attributed more narrowly. Profile and run only one intervention at
   a time; do not choose it from development outcomes after seeing them.
3. Once the hypothesis and analysis are frozen, evaluate on untouched patients
   and report discrimination, calibration, and clinically interpretable error
   patterns. Keep the diagnostic-annotation endpoint explicit.
4. Isolate head learning-rate, clipping and encoder drift in separate controls
   instead of calling the total Moving/Frozen difference a single mechanism.

Until these checks, the result is an interesting development-set investigation,
not an established general rule or a clinical claim.
