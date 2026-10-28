# Local threshold-impact graphs

28 October 2026. The local app now links its exact saved confusion matrix to three views: TP/FN
counts among proxy-positive reference ECGs, FP/TN counts among proxy-negative reference ECGs, and
sensitivity against referral workload. Dots represent saved aggregate cutoffs; counts are not
interpolated. The selected point uses a ring and the frozen calibration point uses a square.

All seven JavaScript tests passed on the host, including exact marker values, stable calibration
markers, class denominators, undefined sensitivity, and translated accessible descriptions. Targeted
Ruff, JavaScript syntax and whitespace checks passed.

A real Firefox check against the completed from-raw aggregate API passed at both PTB and SPH for
the fixed baseline and the secondary general-age blend. It checked slider endpoints and middle
saved points, current and frozen marker values, model/site changes, English/Spanish rendering, and
mobile layout without page overflow. Existing fixed 25,000-record cohort comparisons, the primary
770-record cards, baseline default, exact confusion matrices and status-only localization also passed.

The screenshots use one saved exploratory point distinct from the frozen cutoff to show the two
markers separately. That point is a presentation example, not a new operating-point selection:

- [Desktop graphs](figures/pr75-threshold-impact/desktop.png).
- [Mobile graphs](figures/pr75-threshold-impact/mobile.png).

The app retains the strict `score > cutoff` rule and the separate 5% calibration-normal budget.
Scores remain uncalibrated and the data remain reused development evidence. No model was fitted,
no new sample was scored, and no dataset, threshold or completed scientific artifact was changed.
The preview and browser checks ran locally; nothing was deployed or uploaded.
