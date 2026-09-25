# Clean cohort cached probe fusion screen v1

25 September 2026. Repeat the Experiment 014 development screen using the new
clean-cohort JEPA and ordinary local-plus-context CPC linear probes. Each
encoder retains historical pretraining and preprocessing. The full budget has
15,359 clean labels; the fixed 1,518-label budget is unchanged. This is a
PTB-XL diagnostic annotation proxy study, separate from the 76,598-record
expanded union and Challenge data.

For each budget, compute each refitted probe's raw logits on its clean labeled
training rows and fixed 1,306-record development split. Standardize each
model's logits using only that budget's labeled training mean and population
standard deviation. Score `alpha * JEPA + (1-alpha) * CPC` for JEPA weights
0, .25, .5, .75, and 1. Assign development patients to five stratified group
folds with seed 14042. Each fold's 95%-sensitivity inclusive threshold comes
from the other four folds. Select highest pooled specificity, then AUROC, then
distance to .5, then lower alpha. Repeat 300 patient-cluster bootstrap draws
with the frozen fold assignments and seed 14042.

The Experiment 014 development gate is unchanged: selected alpha interior,
specificity at least .02 above the better endpoint, AUROC no more than .002
below the better endpoint, adjacent interior weight also above the endpoint,
at least half of bootstrap selections interior, and at least 75% of bootstrap
draws favoring the selected weight over the better endpoint. Save all arms and
gate checks. No calibration or test records are scored. A positive gate would
require a separately frozen follow-up.
