# Visual explanation of ECG localization

Open `notebooks/localization-explained.executed.html` in a browser. It includes its real ECG data,
plots, and controls and works offline. Running all cells in notebook 14 creates the page directly;
no command-line builder is needed. The previous executed notebook and its HTML export were removed.

## How the candidate chooses coordinates

The unsupervised residual method detects R peaks, subtracts each beat's local voltage baseline,
and allows an alignment correction of up to 10 ms. It builds a median reference from the other
complete beats, excluding the beat under inspection. It measures squared voltage differences
after dividing by a typical QRS amplitude range for each lead.

The method averages those sample differences in 140 ms windows advancing by 10 ms. Each window
belongs to the nearest detected R peak according to its center; it can include samples assigned
to neighboring beats. A beat receives the maximum candidate score across its windows and leads.
The output is an unusual beat, an electrical lead, and a short time interval for inspection.

The interactive page shows the real trace, the beat/reference overlay, the actual residual field,
a movable scoring window, and a clickable lead-by-beat map. Changing a control explores existing
scores and does not change the saved algorithm choice. Its selected examples are INCART I01 cores
6, 13, and 0: a clear morphology example, a ventricular beat with a post-QRS mark and drift, and
a normal-labeled beat selected in error. Examples explain behavior and are not a new benchmark.

## QRS timing is a separate task

The boundary method follows a smoothed absolute derivative of the filtered waveform. Its local
threshold and connected activity define a QRS interval in each channel. The joint interval uses
the earliest channel onset and latest offset. The page shows the restricted search neighborhood,
local maximum, activity threshold, candidate borders, and real expert timing for QTDB sel100 and
sel49. The latter is a deliberately selected failure.

The expert annotations establish beat identity in INCART and joint physiological QRS timing in
QTDB. They do not establish an exact pathological lead, pathological-wave interval, or anatomical
injury. Neither candidate was integrated into a newly validated clinical screening system.
The classifier and label definitions remain unchanged.

## Executed verification

The helper imports the existing frozen methods rather than copying the scoring implementation.
Both saved per-beat methods reproduced their scores, selected leads, and selected starts for all
three INCART examples; the two QTDB joint boundaries reproduced exactly. Source, recording,
annotation, and saved-output hashes were verified. No synthetic waveforms, training, new accuracy
evaluation, or protected final-test data were used.

The exact repository notebook completed Run All from `notebooks/`: 18 code cells, 24 figures,
zero errors, and the interactive HTML written by its last cell. Its tracked outputs remain empty.

Browser checks exercised every example, lead/beat selectors, the slider, map clicks, expert-label
toggle, boundary comparisons, and a 390-pixel mobile layout. There were no browser errors or
external network requests. All 25,560 lead-window scores agreed with the displayed field averages
within 0.000000149; 68 windows crossing beat boundaries were included. Drawing arrays are rounded
to six decimal places, while saved numeric scores and chosen coordinates remain unchanged.

The full repository suite passed 1,373 tests and Ruff passed. Real waveform payloads and generated
HTML remain local; Git contains the executable notebook, shared helpers, and data-free HTML template.
