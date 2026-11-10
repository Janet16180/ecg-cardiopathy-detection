# KEY-02: longer sections and temporal agreement

Prospective 6 November 2026. User requested larger sections and an approach informed by ECG-JEPA.
Freeze before any new scores. Successor of KEY-01, using its same verified 25,000 training references,
10,434 SPH evaluation ECGs, four overlapping annotation groups and training-only scaler. No labels,
clinical thresholds, patient roles or source permissions change. SPH remains reused development.

## Paper and scope

ECG-JEPA's CroPA connects patches along a lead and across leads at the same time; its masking is
synchronized across leads. This is an interpretation-inspired inductive bias, not validation of a
model's clinical reasoning. Source: https://arxiv.org/html/2410.08559v2, sections 3.1–3.3.
The cached xECG vectors have already combined leads. They cannot implement CroPA or evaluate
cross-lead consistency. This CPU experiment tests longer temporal context and a temporal-agreement
control only. Actual lead-preserving JEPA extraction is a distinct follow-up; no JEPA training or
SAM execution is claimed. Tokens remain contextual, not independent encodings of cropped ECGs.

## Arms

Reuse frozen `condition_keys.refine`, `region_vectors`, `region_scores` and `whole_bank`.
For each length independently, generate every stride-one span of 10, 20 or 30 tokens: respectively
2.5, 5 and 7.5 seconds. Refit one positive and one normal key per group using exactly KEY-01's
single initial-margin selection pass. Query scores are the maximum refined-key margin over that
length's windows. These are three separate fixed arms; never select the best after scoring.
Retain the full-record and short-window KEY-01 predictions as frozen comparators.

Secondary controls at five seconds: (a) maximum window margin using unchanged whole-record keys;
(b) median window margin using unchanged whole-record keys. The median tests whether recurring
rather than isolated evidence helps; it is an experimental heuristic, not a JEPA method or clinical rule.
No threshold fitting, GPU inference, optimization, new cluster fit or novel-condition claim.

## Primary and decision

Primary contrast: five-second selected keys minus KEY-01 short selected keys, macro one-versus-rest
AUROC, all four groups on every SPH evaluation record. Paired whole-patient bootstrap, 2,000 draws,
seed 20261103, same shared helper as KEY-01. Report 95% percentile interval and skipped draws.
Advance longer sections only if gain >=0.01 and lower bound >0, and five-second macro AUROC is
at least whole-record macro AUROC minus 0.01. The last guard is descriptive, not noninferiority.
Also report the five-second versus whole-record paired interval descriptively. Secondary arms:
point macro/per-group AUROC and AP, and condition-versus-normal metrics; no winner selection.

## Integrity and resources

Before new scores, verify all predecessor direct input hashes; verify KEY-01 artifact hash and
recompute every saved arm's all-row and normal-only metric dictionaries exactly. Recompute the
saved KEY-01 primary interval exactly. Verify new whole-record scores agree with saved ones within
1e-12, report maximum error, and require its metric dictionary exactly. Preserve old scientific files.
Save new sources/protocol hashes and require them committed before running. Profile the actual first
128 training rows at all three lengths before full construction; conservative forecast = batch seconds
x (25000+10434)/128 x 2, gate 1800 s. Two CPU threads; no full run above a 3600-second stage-boundary
ceiling. Refuse output overwrite and preserve failure receipts. Save keys, selections and predictions
locally; publish aggregate results only. Output `outputs/long_section_keys_20261106/`.

Write an executed supplementary notebook `notebooks/LongSectionKeys.ipynb`, results report,
both queue updates and ranked follow-ups. Preserve the prior notebook and cluster findings.
