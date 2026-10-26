# Importing completed evidence later

The packet forms use `ecg-pending-review-1.0` and intentionally cannot satisfy the finding-specific
evidence contract. Pending is not clinical uncertainty. Do not populate reviewer provenance until a
review actually occurs, and obtain clinician rubric approval before clinical review or evaluation.

The reusable validator is `ecg_experiment.localization.evidence_validation.validate_evidence`.
It accepts an annotation and separately established `SignalBinding`, optional `ReviewerBinding` and
optional `BeatBinding`. It remains the general technical validator pinned by the packet receipt.
For future completed independent expert imports, use the separate
[strict import adapter](../../../../ecg_experiment/localization/evidence_import_v2.py):

```bash
uv run --no-sync python -m scripts.experiments.localization.import_evidence_v2 \
  local-annotation.json --signal-binding trusted-signal-binding.json \
  --reviewer-binding trusted-reviewer-attestation.json
```

Keep imported annotations and bindings local under `outputs/`. A binding copied from the annotation
is not independent verification. Establish the signal binding from the actual canonical signal bytes.
The private packet key declares the byte convention: samples by canonical lead, little-endian float64,
C order. Establish reviewer identity, expertise and the review conditions through a trusted local
process. Passing an invented attestation to the CLI does not establish that a clinician reviewed a case.
The policy reference identifies that external process; this code does not authenticate a clinician.

The validator preserves alternative sets and jointly considered regions, checks dates even when
optional JSON Schema format libraries are absent, and validates original-coordinate bounds without
converting them. Ordinal coordinates require a frozen original-sample beat list, source revision and
signal-bound digest computed with `beat_digest`. Measurement units use an explicit vocabulary; code
callers can prospectively supply another declared vocabulary without implicit unit conversion.

Independent physical rules or synthetic examples may be independent of model outputs while retaining
nonexpert provenance. Reviewing them does not promote them to independent expert evidence. The strict
policy requires completed expert provenance, trusted reviewer/signal/policy binding, no visible model
output and direct support in every proposed alternative set. Context-only sets produce warnings under
the general policy and rejection under the strict policy. That is an evaluation admission rule, not a
medical judgment that any other set is sufficient. Structural and semantic validity never certify
clinical truth, evidence sufficiency, population independence or exhaustive abnormality annotation.

The current reviewer binding does not authenticate reviewer identity or clinical expertise, or verify
adjudication lineage back to the original annotations. The forward strict import adapter rejects any
adjudication kind or adjudicated status until a lineage-aware policy exists. It requires finite
standard JSON values, rejecting unsupported NumPy scalar objects as well as nonfinite numbers, then
reuses the frozen validator. A future strict evaluation must independently verify lineage and
reviewer authorization. Preserve original valid timestamp strings, including
`-00:00`: it identifies a known UTC instant while leaving the local offset unknown
([RFC 3339 section 4.3](https://www.rfc-editor.org/rfc/rfc3339#section-4.3)). The updated interpretation
also permits `Z` with unknown local-offset information
([RFC 9557 section 2](https://www.rfc-editor.org/rfc/rfc9557#section-2)). This does not permit malformed
offset components such as `+00:99`. No completed human annotation was imported in NTT-07.
