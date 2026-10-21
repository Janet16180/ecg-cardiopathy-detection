# NTT-01 amendment: quarantine reasoning content before analysis

18 October 2026, before any aggregate reasoning-annotation result. The scientific review requires the
same quarantine for the public JSONL as for local metadata: use only source/ECG/case identifiers to join
each case to its fold, then skip folds 9/10 before accessing diagnoses, questions, answers or reasoning.
All grounding, diagnostic and shortcut denominators refer exclusively to folds 1–8. Only identifier
counts and fold counts may include the quarantined cases. Unknown identifiers are counted and skipped.

Two examples were inspected for file format, not scored; both were verified to be fold 8. No fold-9/10
reasoning example was inspected. The entire public file is downloaded and hashed as bytes; quarantined
case content is not interpreted. No waveform or model output is read. This amendment narrows the
[original protocol](ntt-01-reference-label-audit.md); it changes no model or decision threshold.
