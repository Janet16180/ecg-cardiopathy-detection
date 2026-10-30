# Reconstructing the 25,000-record cohort

Status: prerequisite audit; no cleaning, model fitting or scoring has run.

The user requested rebuilding missing cleaning/cohort artifacts and then trying the 25k cohort,
without age-based selection or analysis. This document scopes the reconstruction; it is not a
model-comparison protocol or permission to open the final test for evaluation.

## What is available

The original cleaning, deduplication, split and cohort builders are tracked in Git. The v3 builder
was committed as `b45bec5`, and the Ningbo builder as `15ea617`. Missing generated artifacts do not
establish that code was left uncommitted. DVC pointer files alone do not supply their payloads.

The executed inventory found all 11 inspected source files tracked and all 14 listed generated
prerequisites absent. Raw source directories and official waveform checksum lists are present;
this does not yet verify every raw file. The official Challenge diagnosis-mapping CSVs also need
restoration and verification against the pinned mapping hashes before any label reconstruction.
The aggregate inventory is in `reports/cohort25-reconstruction-2026-10-28/input-audit.json`.

Run the non-reading inventory with:

```bash
python3 scripts/validation/check_cohort25_rebuild_inputs.py --root /path/to/data-checkout
```

## Required reconstruction order

1. Restore or reproduce the original PTB training identities and held-out references using the
   existing PTB processing contract. Preserve the distinction between historical folds 1–8 and
   the current SCREEN01 training folds 1–7; fold 8 must not enter the new supervised fit.
2. Reproduce canonical source windows, file verification, waveform hashes and quality decisions
   for PTB and the curated Challenge sources. Reuse the frozen conversion and quality functions.
3. Reconstruct the original cross-source exact-duplicate groups and Challenge split. Original
   seed is 20260929. Splits depend on source and primary-label strata, plus duplicate groups;
   counts and a random seed alone are insufficient.
4. Require the reconstructed Challenge `rows.csv` SHA-256 to equal
   `5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8` before admitting training
   records. A mismatch is a failed reconstruction, not permission to assign a new test split.
5. Apply original training quality, duplicate and held-out exclusions. Reuse v2/v3 source ordering
   and interleaving functions for the curated block, then take its first 25,000 records.
6. Check documented source counts: PTB 17,405; Ningbo 3,982; Chapman 1,232; Georgia 1,227;
   CPSC 791; CPSC-Extra 363. Counts alone do not prove record-level equivalence. Where original
   manifests cannot be recovered for identity comparison, explicitly report the remaining limit.
7. Publish to a new reconstruction namespace, with input/code hashes and a reconstruction report.
   Preserve existing artifacts and frozen source files. Do not call changed backend pointers or
   metadata byte-identical to the historical cohort. v3 and v4 original 25k selections are identical.

The old all-tier builder eagerly loads MIMIC and CODE-15 prerequisites even for 25k. Its curated
helper also calls the MIMIC candidate pool before filtering. A new curated-only adapter can avoid
those dependencies, but must preserve all original curated selection and deduplication semantics.
No MIMIC, CODE-15 or EchoNext rows belong in the target 25k cohort.

## Explicit data-access boundary

`AGENTS.md` says: "Closed data: the Challenge test groups ... Never read them before a user-approved
final test." Without the saved header/hash/split tables, rebuilding the original split requires
reading raw Challenge headers (including labels) and waveforms across all records, including records
whose original assignment was test. This is mechanical reconstruction, but it still crosses that
literal access rule. Obtain an explicit exception limited to reconstruction before these reads.

The requested exception would permit checksum verification, canonical hashing, duplicate resolution,
fixed quality checks and original split reconstruction only. It would not permit model predictions,
performance evaluation, model selection, changing label definitions, or opening EchoNext test data.
No patient-level data leave the machine. If granted, record the exception in the reconstruction receipt.

## Subsequent model comparison

After reconstruction passes, freeze a separate comparison protocol before fitting or scoring:
released frozen xECG with the same logistic head, PTB-only versus admissible labeled members of the
25k cohort. Reproduce SCREEN01's PTB-only reference, retain the same SPH calibration/evaluation rows
and 5% reference-normal budget, and report sensitivity, false-positive rate and AUROC. Training uses
fewer than 25k records because undefined labels and current held-out rows remain excluded. No age
subgroup analysis. This compares training composition and possibly size, not a causal isolation of
hospital diversity, and does not perform new self-supervised pretraining.
