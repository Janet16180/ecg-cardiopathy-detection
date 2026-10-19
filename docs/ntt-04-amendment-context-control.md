# NTT-04 amendment: full-context physical comparator

18 October 2026, before any new diagnosis counts, waveform extraction or model score. Scientific preflight
noted that QRS-pooled DINO patch tokens remain globally contextual: the encoder receives the entire
[-0.30,+0.65) median-beat image. They can contain P/T information even when pooling only QRS ink.
The original image representation remains fixed; do not describe it as QRS-exclusive evidence.

Add one conservative predefined baseline to the primary best-baseline family: raw V1/V6 full median-beat
samples, unit-normalized and matched to the same three-patient class prototypes by cosine, in addition
to the original raw QRS and scalar baselines. The bootstrap still recomputes the maximum across every
predefined baseline in each draw, and the >=5 percentage-point / positive-lower-bound gate is unchanged.
This gives the physical representation access to the visual arm's temporal context, without fitting
features, weights or a target transform. No representation or threshold was chosen using a score.

For exact render receipts, each 259-pixel panel centers at `(panel + 0.5) * 259` pixels; +/-3 mV maps to
center minus/plus 110 pixels, leaving 19.5-pixel margins. Both programmed ROI widths are 300 ms. The
fixed pre-QRS baseline uses sample bounds rounded from [-0.28,-0.20) seconds, exactly samples 10:50
at 500 Hz. These details apply identically to every class and patient.
