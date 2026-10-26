# Strict import admission for completed independent expert annotations

Use the forward import command for future completed independent-expert evidence:

```bash
uv run --no-sync python -m scripts.experiments.localization.import_evidence_v2 \
  local-annotation.json --signal-binding trusted-signal-binding.json \
  --reviewer-binding trusted-reviewer-attestation.json --beat-binding trusted-beats.json
```

Omit the beat binding only when no ordinal coordinates occur. Keep all inputs local. Pending packet
forms remain a separate contract and do not qualify as completed annotations. The new reusable
`evidence_import_v2` module requires finite standard JSON values, rejecting unsupported NumPy scalars
and NaN/infinity through its Python API as well as its JSON-loading command. It delegates coordinate, schema, timestamp, lead, measurement and provenance
checks to the frozen validator with strict blinded-expert admission enabled. It preserves alternative
sets and original coordinate objects. It does not rewrite imported files.

The forward policy rejects either `expert_adjudication` provenance or `adjudicated` review status
until a separate lineage-aware admission policy can
verify the original reviewed annotations. A reviewer expertise flag does not establish that lineage.
General technical validation of an adjudication remains distinct from its admission to a clinical
evaluation. The initial review packet has no actual expert annotations or adjudications.

Signal and reviewer attestations must be established separately through a trusted local process.
Software neither authenticates the reviewer nor establishes expertise, independence, clinical truth,
or sufficient evidence. Every technical result retains those limitations explicitly. The prospective
clinical rubric and metric still require review before evaluation.

Valid timestamp strings remain unchanged. In particular, `-00:00` represents a known UTC instant with
unknown local offset; it is not an unknown instant. See [RFC3339 section4.3](https://www.rfc-editor.org/rfc/rfc3339#section-4.3)
and the [RFC9557 section2 update](https://www.rfc-editor.org/rfc/rfc9557#section-2). The frozen validator's
acceptance of that valid representation is preserved. Malformed date/time components are rejected.

The forward guard is engineering work, without new patient reads, model scores or clinical labels.
Tests use invented annotations, bindings, reviewer identities and beat anchors only. The initial packet
receipt still binds the original validator; no frozen source or failed rendering artifact was edited.
