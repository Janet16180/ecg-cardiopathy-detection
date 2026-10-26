# NTT-03 results: reference matching works on supplied windows; proposals remain a separate problem

18 October 2026. A compact DINOv2/SAM2 engineering pilot completed on six permitted target patients in
327.07 seconds on CPU. With the finding, lead and candidate windows supplied, visual reference matching
selected the nominated silver answer for all six targets. Wrong-window references selected none in the
primary one-second view. On the three full-trace proposal trials, SAM2 offered a sufficiently overlapping
candidate for one nominated interval, and reference-based mask selection overlapped none of the three
nominated intervals. Another highlighted beat could still be acceptable diagnostic evidence; this study
does not establish its clinical correctness or incorrectness.

The useful distinction is between matching regions and generating them. The matching component runs
and responds to the reference region. That does not yet demonstrate abnormality-specific discrimination:
the contextual QRS-phase comparator also selected all six supplied answers. The automatic proposal stage
needs its own ECG-specific evaluation. The original large/large paper configuration remains unexecuted.

## Prospective design and input integrity

The [protocol](ntt-03-visual-reference-pilot.md) was committed as `8198380`. The
[selection/primary-view amendment](ntt-03-amendment-selection.md) and implementation were frozen as
`9d3ee35` before real-ECG inference. Scientific preflight identified and corrected ambiguous lead parsing,
diagnosis/class grouping, the primary view and per-image deadline checks before scoring.

Saved NTT-02 result, question, prediction and reservation-manifest hashes were verified. Its 1,507-question
inventory reproduced exactly; no broad annotation replay was repeated. Every selected source belonged
to PR67's training role. Selected waveform bytes matched their NTT-02 hashes. All reference patients were
outside inner fold zero; every target was in fold zero. There was no cross-class support/target overlap.

Of the input questions, 167 had an explicit resolved lead and 91 had one unambiguous selected-lead answer;
756 had ambiguous selected-lead information and 493 had no resolved lead. No missing lead was guessed.
Grouping and patient-count rules yielded three eligible diagnoses, not the four-class maximum: complete
LBBB, LAFB and LVH. There were nine independent reference patients and six independent target patients,
with three supports and two targets per class. These are source annotation names; this experiment made
no clinical label change or inference about high voltage alone.

Physical 100 Hz mV signals were median-centered and rendered on a fixed white canvas with a two-pixel
black trace. DINO features came from the public frozen small/14 checkpoint. Trace-ink occupancy inside
each interval was pooled softly to preserve sub-patch strokes, then averaged across reference instances.
Full views covered ten seconds; candidate-centered crops covered one second. The primary view was the
one-second crop, fixed before scores. No selected target answer determined a crop: every supplied option
received the same rendering rule. The selected lead and named finding were experimental inputs.

## Supplied-window matching

| Method | Exact nominated answers / six targets |
| --- | ---: |
| DINO reference matching, primary one-second crop | 6 / 6 |
| Wrong-window reference control, one-second crop | 0 / 6 |
| DINO reference matching, secondary ten-second view | 6 / 6 |
| Wrong-window reference control, ten-second view | 2 / 6 |
| Duration matching using this pilot's three references | 4 / 6 |
| NTT-02 QRS-phase contextual comparator | 6 / 6 |

These are six-case engineering observations, not a clinical accuracy estimate. No confidence interval
or population-level superiority claim is made. The contextual phase comparator uses twelve leads and
its separate fixed NTT-02 reference draw; it is not an input-matched comparison with the one-lead visual
method. It nevertheless shows that every nominated answer in this small subset was already selectable
with generic wave timing. The wrong-reference control distinguishes supplied wave regions, not necessarily
different pathological morphologies of the same wave.

The fixed first target of each class and its cached similarity scores are available locally in
`outputs/ntt03_visual_reference/reference_matching_examples.png`. Coordinates, candidate choices and
reference manifests remain local. The figure shows supplied alternatives and nominated silver evidence;
it does not add expert annotations.

## Automatic proposals on full traces

SAM2-Tiny used the fixed 32x32 prompt grid, CPU float32, quality threshold 0.4, stability threshold 0.95
and box NMS 0.5. It received the full ten-second selected-lead image without a target-region prompt.
At most 100 masks were retained; actual counts were much smaller.

| Supplied finding | Masks | Best proposed temporal IoU with nominated interval | Reference-selected temporal IoU |
| --- | ---: | ---: | ---: |
| Complete LBBB | 1 | 0.016 | 0.000 |
| LAFB | 3 | 0.014 | 0.000 |
| LVH | 7 | 0.879 | 0.000 |

These IoUs compare the mask's horizontal extent with one nominated time interval. They are neither
trace-ink segmentation accuracy nor expert abnormality-mask accuracy. The LBBB proposal was background
and contained no trace pixels for feature pooling, so no semantic mask was selected. For LAFB, the
chosen region was at 8.491–8.627 seconds; for LVH, at 4.217–4.313 seconds. Their zero overlap with the
nominated interval does not prove that they carry no relevant finding. Repeated beats may provide
equivalent evidence that the selected-answer file does not enumerate.

The frozen advancement gate failed: the primary crop view did not beat the contextual phase comparator,
and only one of three proposal trials reached oracle IoU 0.5, below the required half. No rendering,
threshold, support, model size or candidate rule was tuned afterward. This is a negative engineering
admission result with a functioning matching component, not evidence that the full paper can never help.

## Resources, artifacts and checks

The run used two CPU threads, no training and no GPU. Official source revisions were DINOv2
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` and SAM2
`2b90b9f5ceec907a1c18123530e92e794ad901a4`. Actual executed Python source files and both checkpoint files
were hashed in the local receipt. The isolated helper overlay used iopath 0.1.10 and portalocker 2.10.1;
the pinned project environment, `pyproject.toml` and `uv.lock` were unchanged. Optional CUDA cleanup was
disabled; this is one more reason to keep the compact pilot distinct from the original recipe.

Original result SHA-256: `aa9b76d825a9aeff067d20521dc46d4013a637b8601c04573978fa3c3cdbce5c`.
The portable [aggregate summary](../reports/ntt03_visual_reference_summary.json) omits patient identifiers,
per-record waveforms and predictions. Raw result, selected references, exact render transforms, progress
receipts and three fixed proposal overlays remain in `outputs/ntt03_visual_reference/`. All plotted traces
were already used in this frozen run. No figure or raw annotation was uploaded.

Repository-wide Ruff passed. The integrated CPU suite passed 1,266 tests with six skipped in 78.72 seconds.
Coordinate, thin-mask, lead-link and deadline tests passed before the scored run. Scientific preflight
checked source eligibility and interpretation without model scoring. Future interpretation must retain
the selected-interval/equivalent-evidence distinction.

## Follow-up

Preserve DINO reference matching as a candidate component. Prospectively compare ECG-native beat/wave
proposals with SAM proposals and distinguish three endpoints: proposal coverage, recognition of the
named finding within the same wave type, and agreement with acceptable specialist evidence. Evaluate
equivalent recurring beats as sets of possible evidence rather than verified negatives. The current
selected-answer benchmark and spent ECGAD holdout cannot supply a new independent clinical claim.
Clustering and SHAP studies address morphology structure and model dependence separately.
