# Frozen rendering layout

The approved protocol is committed as `2b517af`. Before any patient selection, the implementation
fixes a single 570 mm by 590 mm vector PDF page per case. Twelve full ten-second leads occupy six
rows and two columns. Every signal panel is exactly 250 mm by 80 mm: 25 mm/s horizontally and
10 mm/mV vertically, spanning 0–10 seconds and -4–4 mV. Any sample outside that fixed range aborts
without replacement. The reviewer PNG uses 120 dpi; the PDF uses physical page coordinates.
The PNG's displayed size depends on the viewing software. Print PDF at actual size for calibration.
A 1 mV /200 ms calibration pulse occupies the separate margin, never the signal panel. The grid
uses 1 mm fine squares and 5 mm major squares. Every trace retains its original 500 Hz samples;
no filtering, baseline removal, normalization, clipping or model annotation is applied.

The identical sinus-rhythm prompt is a request for specialist review. It is distinct from the
source NORM annotation and does not mean clinical normality. The four prompts are a draft checklist,
not a diagnostic rubric or exhaustive abnormality taxonomy. The reviewer instructions state that
the rubric awaits clinician approval. The packet is prepared locally, without delivery or contact.
