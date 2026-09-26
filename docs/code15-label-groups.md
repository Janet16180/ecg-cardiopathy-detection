# CODE-15 source-label ID groups

`scripts/reports/count_code15_label_groups.py` reads only the 34 MB official
`exams.csv`, verifies its Zenodo MD5 against the completed acquisition receipt,
and writes a compact local-only table of exam IDs, patient IDs, archive-part
names and released positive labels. It does not extract archives, copy ECG
waveforms, modify raw data, or run the CPC model. The output is
`outputs/code15_label_groups/source_positive_ids.csv.gz`; aggregate counts and
source hashes are in `report.json` beside it.

Across **345,779** metadata exams from **233,770** patient IDs, **37,775**
exams from **28,483** patients have at least one of six released diagnostic
flags. Counts by flag are 1dAVb **5,716**, RBBB **9,672**, LBBB **6,026**,
SB **5,605**, ST **7,584**, and AF **7,033**. These counts overlap: **3,671**
exams have multiple flags. The separate automatic `normal_ecg` flag is set on
**134,657** exams. These are **source annotations**, not CPC findings or a
validated count of all cardiac conditions. No signal-quality filter was applied
to this metadata-only report. The flags cover six named conditions only, so
absence of all six does not mean a patient is healthy.

CODE-15's 400 Hz waveforms still need a verified conversion into the CPC
model's 250 Hz, 10-second mV input contract before the model can score them.
Once that exists, patient-independent evaluation can compare CPC's broad binary
score with these released source labels. That comparison would still need an
endpoint definition and careful handling of unlabeled conditions.
