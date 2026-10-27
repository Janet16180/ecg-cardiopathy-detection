# Next real-annotation comparison: scientific recommendation

23October2026. Read-only design memo,not an executable experiment protocol or a
clinical result. No annotation payload,waveform,mask array or new model prediction
was read for this memo. The acquisition43 summary alone was checked:180newrequests,
1,662,449received bytes and zero annotation/waveform/model operations; summary SHA
c8e59e4227bb87e88fcc42bae307719bef6848a03b0cb92697d5bab93c54cdcd.

## Current operative contract

Reviewed current coordinator8b5119a and the separately frozen
[execution v2](ssl-synthetic-localization-edb-qualification-execution-v2.md),
[acquisition v3](ssl-synthetic-localization-edb-acquisition-recovery-v3.md),original
annotation/grid protocols and both earlier source/reference reviews. Current rules
supersede the older draft assumptions without editing their historical bytes:

- The released roster does not contain e0120. Its omission from the published
  range yields79actual released-record linkage components. All90released records
  remain; the old78expectation is corrected history,not a continuing contradiction.
- An interpreted-code custom definition is allowed only when BOTH symbol and
  description exactly match the pinned installed WFDB standard. There is no
  semantic synonym exception. Preserve the exact declarations and all raw events.
- Qualification/audit reported RSS caps are2GiB; original CPU120/wall300/output16MiB
  limits remain. Failed acquisition stages and their actual costs stay retained.
- Extreme intervals need at least one valid containing base; retain all matching
  parents and diagnostics. Unsafe/unperformed installed-decoder comparisons are
  explicit record issues,UNKNOWN128 grids and overall engineering failure.

My earlier message proposing an any-custom-definition veto was stale relative to
the committed overlay and is withdrawn. No frozen source or current rule change
is requested. Qualification and independent verification must finish before its
availability is described as verified or used to choose a later model protocol.

## Choice of endpoint

ECGAD is the closer candidate for the final isolated10second,twelve-lead goal.
Its authors provide expert point-mask benchmark links and code with that geometry.
However,the [source feasibility review](ssl-synthetic-localization-ecgad-source-feasibility-v1.md)
documents unresolved actual asset shape/lead order,mask meaning/coverage,patient
mapping,package provenance and licensing scope. The old holdout is spent. A new
official acquisition can be justified as recovery/qualification of a development
resource; it does not create an untouched clinical test or verify missing old data.

EDB can qualify an observed-channel transient-ST episode task,but the
[reference-context review](ssl-synthetic-localization-edb-reference-context-review-v1.md)
remains decisive. One10second segment cannot generally distinguish fixed ST level
from deviation relative to this recording's earlier reference. A window-only result
would be a statistical association with episode labels,not proof of the source-
defined within-record change. Two ambulatory channels also do not validate twelve
standard leads. Do not zero-fill the old network or concatenate independent GAN
records into an invented longitudinal patient reference.

If a later EDB model is pursued,its input must explicitly include a label-free
reference policy,for example the fixed first30seconds available to every window.
Never use annotations to select the clean/normal reference. The lost expert
template,bad initial quality and changed deployment input remain limitations.
That new architecture/input/training study is larger and less directly compatible
with the current short-record goal than a qualified ECGAD diagnostic.

## Smallest useful generator-to-real comparison

After a separately prospective ECGAD acquisition and mask/source qualification,
start with a SIX-state frozen comparison rather than another training run:
the three supervised and three matched permuted states from
generated_fit_replication_v1. Metadata-only file checks confirm all six final_state.pt
files exist locally,each144,776bytes,alongside original main/ledger/audit receipts.
Their actual hashes,tensor identities and exact parent closure still require the
new executable admission. Existing paired-study B states reproduce the supervised
states; do not count duplicate copies as additional independent fits.

The direct question is: does correctly aligned numerical supervision on released
Pulse2Pulse backgrounds improve agreement with expert real ECG time/lead regions,
relative to the same generated backgrounds,initializations,schedules and permuted
targets? This preserves the pretrained-synthetic research question and its matched
control. It does not isolate the generator's unique advantage over real-only
training or establish that synthetic clinical labels exist.

Evaluate all released benchmark cases under fixed pre-score eligibility. Require
the acquired record count/geometry to match the declared source contract; do not
choose a favorable subset or fit. Candidate comparison coordinates are the source
evaluator's central100:4900sample grid on all12leads,with the full masks/edges
retained and coverage stated. No crop may be selected after scores. Cases without
both mask classes have undefined within-record AP: retain every such case and its
predictions as declared control/coverage evidence,not a silently dropped patient.

Use one fixed real-input recipe for both arms,preferably the already reviewed
native-I/II/V1--V6 robust transformation followed by canonical lead construction,
only if the asset's actual lead identities/coordinates support it. Verify the input
and mask alignment first; no guessed lead order,mV calibration,new filter chain
or inverse preprocessing. Any representation change relative to the annotation
display must be explicit. If that mapping is unresolved,do not admit model scores.

Primary should be equal-record within-ECG AP difference,averaging the three FIXED
matched fits before aggregation. With actual patient linkage,average repeat records
within patient before the patient-macro estimand and use2000whole-patient draws.
Do not pool all cells or records into one AUROC as the primary: whole-record score
differences can inflate that statistic without correct local ranking. A consistent
engineering rule could require primary gain>=0.01 and lower95%bound>0,with absolute
AP and target prevalence always visible. This is a proposed rule for a future
protocol,not one frozen or tested by this memo.

Prespecify comparisons with absolute-input and the fixed derivative-activity proxy,
plus time-shifted and lead-assignment controls. These identify simple amplitude/
beat/assignment explanations; phase shifts are not universal clinical nulls. A
reasonable engineering conjunction would require positive lower-bound advantages
over both amplitude and derivative controls in addition to the primary contrast.
Show every fixed fit,all annotation eligibility denominators,and time-versus-lead
performance separately. Shared masks across leads cannot establish independent
lead selectivity merely because the array has twelve columns.

The originally reported globally pooled AUROC can be a secondary source-comparison
descriptor,not a replacement for a failed primary. No diagnostic subtype,age,
clinical operating point or new disease definition is needed. No label or benchmark
waveform enters optimization in this smallest comparison. New mixed/paired-objective
arms can wait for a later bounded question; including every existing model would
multiply results without strengthening this first matched-supervision test.

## Hard prerequisites and bounded resources

Before any model output,freeze an acquisition/header-only asset protocol,then a
separate mask/linkage qualification and literal admission. Check finite numeric
arrays with pickle disabled,mask domains,exact time/lead alignment,all-zero/all-one
coverage,duplicates,expert provenance and actual source identifiers. Search only
released maps or independently verifiable deterministic source matching. The
published preprocessing script's fold10 split does not prove the benchmark's
membership. Approximate waveform similarity is not established patient identity.

Patient linkage is required for the patient-level version and any benchmark-label
training. If unavailable,a new explicitly finite-record diagnostic could report
point differences and complete case distributions,but must not silently substitute
row-bootstrap intervals or claim patient separation. Current real-training40 and
other local source roles must be checked where linkage exists. Generator cohort
membership and prior project-wide benchmark exposure remain separate unknowns.

A prospective six-state inference study can be small: at most400records,300batch-
eight forward calls for full outputs,and approximately576MB full float32 maps.
Budget1GiB new outputs,2GiB RSS/CUDA,600CPU/900wall/180GPU-lock seconds,subject to an
actual six-state load/forward/persist/hash/score profile and measured forecast.
No CPU fallback or new training. Reproduce a fixed original synthetic case per
state as integrity closure and save each comparison before its tolerance gate.
An independent CPU audit could have300CPU/600wall/2GiB RSS/64MiB outputs and24fixed
model-record replays,plus complete masks/metrics/draw reconstruction. These ceilings
are recommendations only; asset sizes and the actual profile must be prospectively
admitted. Keep every signal,weight,map and record identifier local.

If this diagnostic demonstrates useful alignment,it would justify the stronger
next learning comparison: matched real-only annotation training versus the same
real annotation budget with a frozen synthetic-supervision initialization. That
requires verified patient roles and a new training/profile protocol. Neither a
positive diagnostic nor a visually plausible generator alone completes clinically
validated suspicious-time/lead localization. Preserve all old failures,current
clinical definitions,closed tests and the24hour mission's no-push restrictions.
