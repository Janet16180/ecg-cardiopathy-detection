# Scientific interpretation of mixed-background training

Output-derived review,22October2026. Main stage38 and independent recovery40 pass
technical verification with zero numerical mismatches. The prospective scientific
conjunction fails:9of14checks pass. Successful verification confirms the negative
engineering result; it does not turn that result into a scientific pass.

Reviewed actual local mixed_background_v1 main/recovery summaries and ledgers, all
source/family endpoints, control endpoints, fixed-fit/mode points and aggregate figure.
No new forward pass, training, waveform decode or outcome recomputation was run.

| Local evidence | SHA256 |
| --- | --- |
| summary.json | c5c8c83de8e07225b4763a61a5f3e24727076c7daa53a120b5c6e62049b29dfa |
| prospective_ledger.json | dcc6ee167ae3f8267a8f2e723285253d2230d580df8e5df81359c86cd3301c04 |
| independent_audit_v1/summary.json,failed39 | f5436ad157a203dfa310ef60191ec0f6130147fa53e8016830c95e52625df434 |
| independent_audit_v1/prospective_ledger.json | 309ee61bd827362b26f247c60082f9f9ff375efaa656c8f6136c21248b6bd86a |
| independent_audit_v2/summary.json,successful40 | 4f2f5e4943c454f7aa45572cfcafc512781702a53fd25a1dc23a011ccbba483a |
| independent_audit_v2/prospective_ledger.json | 0f7def28c1f5c5135e6096be38ca0d57826d5ab4d1d3e90acc7953a573a90feb |

The coordinator's [executed results report](ssl-synthetic-localization-mixed-background-v1-results.md)
contains the complete training and artifact record. This review interprets those
saved results; independent arithmetic and functional replays are distinct from
independent physical observation of the original GPU run.

## Result and tradeoff

Replacing half the generated backgrounds with the fixed40real PTB training
patients did not improve the primary unfamiliar-change localization endpoint on
the spent real LUDB development panel. Generated-only AP was0.03252017 and mixed
AP0.02984406. The paired mixed-minus-generated contrast was−0.00267612, with the
conditional95%interval[−0.00686795,0.00130617]. It failed both the planned0.01gain
and positive-lower-bound checks. The interval includes zero, so this is not proof
of a universally harmful intervention or exact equality; it does not support the
planned minimum benefit in this fixed comparison.

| Evaluation source/editor | Generated-only AP | Mixed AP | Paired difference,95%interval |
| --- | ---: | ---: | --- |
| Generated/Hann | 0.10007214 | 0.07971880 | −0.02035334[−0.03171313,−0.01013427] |
| Generated/biphasic | 0.07325239 | 0.05316966 | −0.02008273[−0.03313032,−0.00882911] |
| Real/Hann | 0.04371339 | 0.04645254 | 0.00273916[−0.00421082,0.00906888] |
| Real/biphasic,primary | 0.03252017 | 0.02984406 | −0.00267612[−0.00686795,0.00130617] |

The unfamiliar real focal contrast was−0.00007632, interval[−0.00769152,0.00691952].
The repeated contrast was−0.00527591, interval[−0.00975551,−0.00131435]. Both mode
preservation checks passed their fixed−0.01lower-bound criterion; this is compatible
with the repeated point estimate decreasing. Both generated-family preservation
checks failed. The small favorable real/Hann point cannot replace the primary.

Mixed real/biphasic response AP remained positive at0.01820882, interval
[0.00946009,0.02955297], as did donor gain0.01530970[0.00293815,0.02922265]. However,
its amplitude-baseline gain0.00785280 had interval[−0.00156959,0.01931225], failing
that guard. Added edits still affect the predictions, but the result does not
establish reliable advantage over the declared amplitude baseline on this task.

The background-score finding is substantial and distinct from localization:

| Real control,mean top600 probability | Generated-only | Mixed | Paired difference,95%interval |
| --- | ---: | ---: | --- |
| Sham | 0.68275956 | 0.52717540 | −0.15558416[−0.17472349,−0.13464845] |
| Nuisance | 0.68402706 | 0.52739380 | −0.15663326[−0.17576173,−0.13601413] |

Both prespecified upper control-score guards passed. These controls contain no
ADDED target editor, not necessarily no genuine abnormality. The reductions are
uncalibrated score changes, not clinical false-positive reductions. AP assesses
within-record ranking and need not improve when probability magnitudes decline.
The observations are consistent with lower background scores without better
unfamiliar-edit ranking; they do not isolate the mechanism causing that tradeoff.

## Integrity and limits

The actual operation counters report2400scientific updates across two arms and
three matched fits, plus10disposable profile updates,36frozen-B profile batches,
36mixed subset batches and288full mixed batches. Main reports exact replay of all
three generated-only400-step loss/component/model/Adam trajectories, complete
input/source reverification and all resource gates passing. The actual stage38
cost was96.13543CPU/128.33847wall/40.25288GPU-lock seconds, peak RSS1,719,930,880bytes,
with820,392,950retained bytes before its terminal ledger. These are actual main
ledger measurements, not training costs inferred from faster kernels. The audit
also independently verifies the main resource limits, identities and accounting.

Independent attempt39 stopped on its caller's input-rank contract before its first
CPU convolution: a single recording lacked the batch dimension expected by the
frozen functional helper. Its failed receipt and35.69943CPU/38.49025wall seconds
remain preserved. This checker failure does not reverse the main scientific
verdict, and its unsaved partial computations are not claimed as successful
verification. The separately committed recovery changes only that batch adapter
and audit catalog/accounting paths; main training and all endpoints remain frozen.

Recovery40 independently reconstructs the fixed40training records and complete
training banks, all six model/Adam states, original400-step baseline closures,
profile/recovery evidence and all768evaluation rows/3,072positive cases. It compares
51,144new numerical scalars and all2000×348bootstrap entries with zero mismatches;
maximum draw difference is6.94e−17, below the fixed1e−12tolerance. Its24functional
CPU replays have maximum difference3.87430e−7, below the fixed1e−5tolerance. No new
training, GPU forward, raw LUDB read or clinical-label interpretation occurs.

The independent attempt uses65.75398CPU/71.58270wall seconds, zeroGPU and
991,080,448byte peak RSS; all audit resource gates pass. Its actual stage40ledger
binds failed39 directly and retains both that cost and the main cost once. Current
machine cumulative totals are1274.365159998CPU/564.4522484582849GPU-lock seconds.
Unavailable historical costs remain unknown; developer/setup costs are separate.

The interval scope remains the fixed training cohort, three fixed initializations,
spent64-slot source panels and fixed donor assignments. It does not include
uncertainty over new training cohorts or a patient population. Distinct real
training/evaluation roles do not establish biological disjointness across data
sources or GAN pretraining membership. Missing historical v4/partition payloads
remain missing. Numerical final-difference supports remain numerical target truth,
not pathology masks. Earlier14of15FAIL and the negative real-N result are unchanged.

## Next decision

Do not promote this50:50replacement scheme or fine-tune the generator on the basis
of this result. The experiment changes a finite set of background identities and
source composition; it does not identify Pulse2Pulse as the cause of weak transfer.
More plausible-looking signals alone would not establish disease control or true
disease localization, and a generator training run would lack a validated clinical
localization endpoint for judging improvement.

Prioritize a bounded public-annotation feasibility bridge: identify whether released
expert annotations can define a real abnormality-event timing task, retain the
actual time/lead scope and annotation uncertainty, and freeze permitted patient
roles and an endpoint before any model score or new training. Existing LUDB N
markers and wave delineations alone cannot become abnormality labels. Preserve a
concrete annotation-availability blocker if suitable truth cannot be established.

If that bridge is feasible, the useful next causal comparison is whether synthetic
numerical-change supervision helps learning on that real annotated task versus a
matched real-only learning budget. It needs its own prospective controls, resource
profile and protocol. This recommendation authorizes no additional training,
generator adaptation, closed-test read, disease-label change, push or CI operation.
