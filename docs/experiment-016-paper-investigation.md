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

The [predeclared second-seed protocol](experiment-016-encoder-motion-replication-v10.md)
was designed to check whether the pattern survives a new record order. Its
bridge failed before any seed-47 GPU update, and the frozen cost gate then
prevented a corrected rerun. The attempted replication supplies **no second
performance result**. See its [stopped-run result note](experiment-016-encoder-motion-replication-v10-results.md)
for the implementation failure and measured budget.

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
and [ECGFounder post-training](https://arxiv.org/abs/2509.12991). A publishable
angle would need to show that the controlled readout-gap pattern is robust and
useful **in ECGs**, while positioning it against these results. “Refitting a
head helps” alone is not a novelty claim.

## What would change the evidence level

1. Repeat the frozen Moving/Frozen comparison with independent training orders
   under a new, affordable, fully profiled protocol. Report failures as well as
   passes; do not tune the rule on development patients after seeing outcomes.
2. Repeat with another pretrained encoder checkpoint or architecture and a
   distinct ECG dataset/task, holding splits and patient identities clean.
3. Once the hypothesis and analysis are frozen, evaluate on untouched patients
   and report discrimination, calibration, and clinically interpretable error
   patterns. Keep the diagnostic-annotation endpoint explicit.
4. Isolate candidate mechanisms in separate controlled experiments (for
   example, head learning-rate/clipping interactions) instead of calling the
   total Moving/Frozen difference an encoder-drift mechanism.

Until these checks, the result is an interesting development-set investigation,
not an established general rule or a clinical claim.
