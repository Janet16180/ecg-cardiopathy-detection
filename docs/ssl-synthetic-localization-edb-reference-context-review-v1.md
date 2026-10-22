# Reference context required by the proposed EDB bridge

Prospective scientific clarification,22October2026. No waveform,annotation payload,
new score or training result was read. This does not amend the frozen annotation
qualification or authorize a model experiment. It limits the later interpretation
before a convenient input recipe becomes an unsupported disease-localization claim.

The [official EDB annotation specification](https://physionet.org/content/edb/1.0.0/annotations.shtml)
defines transient ST deviations relative to a reference waveform from the recording's
initial30seconds. The exact reference waveform was not retained and the original
templates are unavailable. Fixed ST displacement may coexist with a separately
annotated transient deviation. Consequently,the annotated episode target is not
simply the instantaneous absolute ST level visible in one short window.

A single10second window cannot generally distinguish a persistent baseline ST
displacement from a transient change relative to this record's earlier reference.
A window-only learner could still predict statistical correlates of episode
membership,but such performance would not establish that it measured the documented
within-record deviation. Do not infer that improving a GAN would resolve missing
reference information. This is a target/input compatibility issue,not evidence
against all synthetic localization or proof of an impossible learning task.

A possible later context-aware protocol could supply the fixed first30seconds of
the observed channel to every evaluated window,with a frozen,label-free processing
rule. It must not choose a normal-looking interval,highest-quality beat or reference
waveform using expert episode/quality annotations,held-out labels,or the target
window's eventual score. A deployed reference policy must be available from the
actual provided signal and declared metadata without an oracle label. Poor initial
quality or atypical reference morphology remains a limitation/failure to retain;
another segment cannot quietly replace it after labels are known.

This fixed initial context would not reproduce the lost expert template exactly.
It also changes the input requirement to longitudinal ambulatory context. An EDB
episode application would therefore be a narrower research bridge,not evidence
that a student's isolated10second twelve-lead ECG can be interpreted with that
unavailable history. Both this domain gap and the two-channel restriction must
remain visible even if the later real-episode endpoint improves.

Pulse2Pulse currently supplies independently generated ten-second backgrounds.
They do not establish a coherent thirty-second same-patient reference trajectory.
Synthetic supervision using an exact unedited counterpart as reference could
create a difference-oracle shortcut absent from real recordings,where beats/rates
and noise differ between reference and target. A future protocol must explicitly
control reference matching,ordinary temporal variation and nuisance changes; it
cannot silently construct a long clinical history by concatenating unrelated
generated ECGs or compare exact-counterpart synthetic references to arbitrary
real references as if the information were equivalent.

Qualification42 can correctly report interval/quality/grid availability without
solving this modeling question. The later source/architecture/training protocol
must state whether its objective is episode presence,within-record change or
short-record morphology,what context is available,and which time/lead labels the
evidence actually validates. Existing public twelve-lead point-mask resources may
be more compatible with the final short-ECG goal,but their provenance,coverage,
patient linkage and spent status require qualification first. No existing binary
clinical definition,closed test or public-label meaning changes here.
