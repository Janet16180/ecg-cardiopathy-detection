# Real ECG localization notebook

The notebook explains the completed research with real patient recordings, saved screening
decisions, and expert-annotated comparisons. It uses no synthetic ECGs and runs no training or new
performance evaluation. Its Markdown sections explain the method, findings, and conclusions.

## Run the notebook directly

Open [notebook 14](../notebooks/14-jr-real-ecg-localization.ipynb) in Jupyter, select the project's
`.venv` Python kernel, and choose **Restart Kernel and Run All Cells**. Start Jupyter in the project
root or its `notebooks` folder. You do not need to run `build_real_ecg_notebook` first.

The first loading cell finds the checkout and imports shared helpers. They read original local
PTB-XL, INCART, and QTDB recordings and verify cached result hashes before drawing the examples.
The project environment supplies the needed Python packages. Existing data and completed outputs
must be available; no neural model weights, GPU, internet, training, or threshold refitting is needed.
Missing local files raise their actual file errors instead of directing you to a builder.

The tracked notebook has executable cells and explanations, with no stored patient waveform payload
or figure outputs. Running it loads the data into memory and displays the figures. Clear outputs
before committing the notebook so raw recordings and patient figures remain outside Git.

## What the reader sees

Nine original 12-lead PTB-XL development ECGs illustrate normal annotations (47 and 91), a premature
ventricular beat (219), left and right bundle branch blocks (2135 and 1123), nonspecific T-wave changes
(127), combined infarction-pattern annotations (184), a probable old inferior infarction annotation
(1577), and an LVH annotation (30). These are label-based examples, not a new accuracy estimate.
Every code is retained, so a mixed label is not presented as a single confirmed diagnosis.

The full recordings and two-second close-ups show the original voltages and existing candidate
marks. The saved combined screening rule flags five examples and does not flag four. The T-change
and LVH examples are explicitly shown without referral or highlights. Voltage alone remains an
undefined clinical referral criterion; its unflagged example does not establish a clinical error.

Two INCART excerpts show a successful abnormal-beat selection and a failure of the newer method.
Two QTDB excerpts show improved QRS boundaries and a deliberately selected boundary failure.
Their legends distinguish algorithm candidates from expert annotation timing. The QRS truth is
joint timing across two channels, not a disease-location annotation for an individual lead.

## Findings and conclusions

The completed [INCART evaluation](experiment-056-incart-beat-localization-results.md) increased
patient-averaged ventricular ectopic beat selection from 74.1% to 89.6% in mixed segments across
30 eligible patients. The candidate still flagged 19.1% of expert-normal beats in the 31-patient
false-alarm comparison. Finding an unusual beat helps inspection but does not prove that its
highlighted segment is the pathological wave.

The completed [QTDB evaluation](experiment-059-hybrid-wave-boundaries-results.md) increased QRS
overlap from 68.2% to 79.4%, and the share of endpoints within 30 milliseconds of the expert mark
from 65.5% to 91.5%, across 72 records and 2,582 complete QRS intervals. This supports more precise
physiological timing. It does not validate disease localization or anatomical diagnosis. Patient
independence is unknown in QTDB, and T-wave improvement remains unproven.

The notebook keeps screening, abnormal-beat selection, and wave boundaries separate. It explains
validated components and their limits without promoting them into a fully validated clinical
localization system or changing the classifier or label definitions.

## Interactive localization explanation

The final cell creates `notebooks/localization-explained.executed.html`. Open it in a browser to
choose a beat and lead, compare real beat/reference shapes, slide a scoring window, click the
lead-by-beat map, and inspect expert QRS boundaries. All page resources are embedded and work
offline. No separate builder command is needed. Details and executed checks are in
[the visual explanation guide](localization-visual-explanation.md).

## Direct-run verification

On 2026-10-02, notebook 14 was executed directly with the project environment from its `notebooks`
folder, without invoking the builder. All 18 code cells completed, showing 13 real examples and
24 inline figures, with zero errors. Its last cell created the interactive HTML. The tracked copy was then saved with cleared outputs and
execution counts. Shared loading helpers verified the saved cache and selected input hashes.
