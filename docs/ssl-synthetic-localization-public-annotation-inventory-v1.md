# Public annotation source inventory and proposed qualification

Status: executed directory-only inventory plus public documentation/source-code review on 22 October 2026. This is setup evidence, not a new clinical result, annotation qualification or model experiment. Base source: `6953178`. No frozen source was edited.

## What is actually local

The permitted shared raw-data root contains nine source directories: PTB-XL, Challenge 2020, Challenge 2021, SPH, CODE-15%, MIMIC-IV-ECG, EchoNext, LUDB and the ECG reasoning benchmark. EDB, INCART, LTST and QTDB source folders were absent under both that root and `baseline/data/raw/`, including the checked alternative source names. No signal or annotation payload was opened.

The local Challenge training source folders are CPSC 2018, CPSC Extra, Georgia, Chapman-Shaoxing and Ningbo. No INCART source folder is present there. The inventory stops at source/chunk directory metadata; it does not inspect closed-group contents or establish that every dataset archive has been exhaustively searched.

The five PTB-XL aliases (database CSV, statement CSV, checksum table, records100 and records500) exist and have existing targets beside their five tracked `.dvc` pointers. The reasoning JSONL alias also has an existing target. The baseline LUDB alias `data/raw/ludb/1.0.1/data` points to our local `outputs/ssl_synthetic_localization/ludb_acquisition_v4/data`; the separate shared raw LUDB version directory remains empty. SPH's source directory exists, but this inventory makes no completeness or record-validity claim about it. No aliases or pointers were replaced or staged.

Local aggregate evidence: `outputs/ssl_synthetic_localization/public_localization_source_inventory_v1/summary.json`, SHA-256 `c658c5e443cb4dc4d0347dd5fa4bda60b94579994a620a988bdc17c1f83b66bc`. The saved directory-count receipt reports 0.379 seconds wall, 0.0037 seconds CPU and 15.6 MiB peak RSS; these numbers cover that final enumeration/serialization, not the preceding interactive source review. All annotation-content reads, waveform decodes, downloads, model forwards and GPU operations were zero. This development/setup cost has not been inserted as a fabricated scored-experiment stage.

## What the public documentation establishes

The [European ST-T Database](https://physionet.org/content/edb/1.0.0/) provides two observed channels at 250 Hz and expert-reviewed episodes. Its [annotation specification](https://physionet.org/content/edb/1.0.0/annotations.shtml) defines channel-specific ST/T start, extremum and end fields, separate axis-shift annotations and quality states. These can potentially support episode-time by observed-channel validation. They are not a twelve-lead or dense within-beat pathology mask. The primary metadata page and specification were accessible in this review; no individual EDB record was fetched.

[Long-Term ST Database](https://physionet.org/content/ltstdb/1.0.0/) documentation describes long, two- or three-channel records and several explicit episode definitions. It is a possible later source, not necessary for an initial bounded qualification. Official INCART and QTDB metadata requests timed out twice in the browser tool; this inventory claims no fresh verification of their release semantics. Their absence locally is separately supported by the directory checks.

## Reader reuse and important limits

- `ecg_experiment/ssl_synthetic_ludb_qualification.py::read_signal` sanitizes structural header lines and validates finite calibrated samples. It explicitly requires twelve channels, 500 Hz, 5,000 samples and interleaved format16. Do not call it unchanged for a long two-channel EDB record.
- `ssl_synthetic_ludb_audit.py::waveform` independently checks the same fixed geometry and format16. Its artificial tests and decoding principles are useful; its fixed geometry is not a general reader.
- `ssl_synthetic_ludb_raw_events.py::decode` preserves numeric timed events and metadata while retaining only AUX lengths. Its existing decoder may supply independent timing/field checks. It cannot recover episode semantics from discarded AUX text; a new, narrowly specified independent AUX parser is needed.
- Reuse `files.sha256_file`, `paths.to_stored/from_stored`, `intervals.patient_groups/patient_resample` and the existing exclusive-output, source-snapshot and terminal-accounting conventions. Import helpers from the package, never from scripts. No neural model is needed for annotation qualification.

## Proposed next implementation, not an executed protocol

Following scientific review, start with EDB annotation-only qualification. Root must first commit the exact source URLs/version/license, allowlisted files and byte caps, selection/roles, semantics, parser decision rule and resources. New package files should be source-specific, for example `ssl_synthetic_edb_acquisition.py` and `ssl_synthetic_edb_qualification.py`, with thin entry points under `scripts/data/` and `scripts/experiments/`. No changes to existing frozen modules are required.

1. A metadata admission binds official `RECORDS`, `ANNOTATORS`, `SHA256SUMS.txt`, license and annotation specification. Fetch only allowed release files under an exclusive new baseline output directory, never directly into the other checkout. No recursive download, implicit retries, redirects to unadmitted hosts or whole-release ZIP is needed. A failed/partial acquisition keeps its receipt, byte identities and actual cost.
2. A separately frozen annotation-only acquisition can allow all declared `.atr` files and `.hea` files, with no `.dat` requests. Suggested initial caps: 8 MiB received, 16 MiB retained, 30 CPU seconds, 600 active wall seconds, 512 MiB RSS and one CPU thread. Measure before expanding these prospective caps. Header bytes can be retained/hash-checked, but interpretation reads only the first structural line and its declared signal lines; never clinical comments.
3. A source qualification compares WFDB and independent binary timing/numeric fields and exact allowlisted AUX episode tokens. Preserve original event order, multiplicity and unknown tokens; never repair, silently discard, merge or infer boundaries. Diagnose unbalanced start/end, channel/polarity disagreement, nonfinite values, out-of-bounds/decreasing events and malformed quality transitions. Preserve lowercase axis-shift and quality labels separately. No clinical normality follows from the absence of an episode. Suggested qualification caps: 60 CPU seconds, 180 wall seconds, 512 MiB RSS and 32 MiB outputs, zero GPU/forwards/scores.
4. Output local event/episode tables, explicit observed-channel identities, subject-group declarations from authoritative metadata, annotation coverage/exclusions and independent verification. Write a report from those outputs. Only a subsequent protocol can admit waveform files, windows, preprocessing, controls and a model comparison.

The existing localizers require twelve channels. Two recorded channels must not be silently padded, duplicated or renamed to satisfy that interface. The later model design must explicitly resolve this representation mismatch, preserve the existing synthetic-generation objective and distinguish annotated episode timing from a diagnosis or per-sample disease truth. The negative/mixed previous results and their exact predecessor closures remain intact.
