# NTT-09: reference-averaging stability of fixed-contrast contextual contributions

## Prospective question and primary rule

Does averaging reference prototypes improve consistency of contextual beat highlighting within the
existing finite support pool, when every bank explains the same patient-specific ordered contrast?
This is conditional model-behavior research. Stability does not establish pathological localization,
clinical sufficiency, disease etiology or agreement with an expert. No clinical endpoint is claimed.
Commit this protocol before any new score, then commit source and meaningful synthetic tests before
execution. Scientific review and a passed independent NTT-08 audit are execution prerequisites.

For each of the unchanged 60 targets, compute top-positive-quarter beat-set Jaccard for all 1,225
pairs among 50 single banks and all 45 pairs among ten five-bank ensembles. First average defined
pairs separately within that patient and bank family. Define the patient's difference as ensemble
mean minus single-bank mean. The primary finite-design benefit requires the unweighted mean of all
60 patient differences to be >=0.10, all 60 differences defined, and at least 95% defined pairs within
EACH patient and family: >=1,164/1,225 singles and >=43/45 ensembles. Never pool pair counts across
patients to satisfy coverage. Report the signed effect even when negative. If any patient difference
is undefined, the all-patient primary mean is null and the primary gate fails; no patient is dropped.
The threshold describes conditional improvement, not adequate clinical localization.

The positive-quarter set contains strictly positive contributions within the first ceil(B/4) beats
ordered by signed contribution descending, breaking ties by original beat index. It may have fewer
than ceil(B/4) members. Both-empty sets give null Jaccard; one-empty gives zero. Preserve complete
pair denominators, zero/negative contributions, rank ties and undefined counts. No bank or target
is selected by correctness, margin, label, apparent stability or contribution sign. Pairs share
supports and people; no independent-pair confidence interval or confirmatory p-value is inferred.

## Inputs, roles and exact replay

Use only NTT-08/NTT-06's 87 existing cached records: 60 targets and 27 disjoint reference patients.
Keep original automatic QRS regions, lead/time anchors and dense 400-by-768 float32 tokens. Retain
the existing patient/role and whole-patient PTB fold-9/10 gates before feature/token payloads. No new
patient, waveform, encoder inference, model training, GPU, download, age/sex analysis, clinical label,
expert annotation, closed Challenge/EchoNext test payload, upload or external contact.

Pin NTT-08 result SHA256 `0d137e4f8618a99f2d6994179b9ff5b1b3de0d82e309fd6ab4d915d5a3001058`
and CLUSTER-04 result SHA256 `c672be6b2fc9fe6d8c14d974179f04345381b068349df611d1656723e0427baa`.
Verify completed statuses and every immutable source/input/artifact binding. Reuse frozen
`replay_cached_inputs`, occupancy, mass/rank/summary and predecessor-evaluation helpers; never edit
their pinned files. Before new comparisons, reconstruct all 180 NTT-08 original decompositions,
original choices, signed contributions, interventions, bank pairs and complete patient-first summary.
Require all saved arrays and numeric summaries within 1e-12 (sum identities within 1e-10), with exact
identities, integer counts and choices. Preserve the preceding 87-descriptor and native-score replay.
Abort on any source, input, role, identity or replay disagreement, retaining a typed failure receipt.

Reuse the exact ordered 50 three-shot-per-finding CLUSTER-04 banks and ten consecutive disjoint
five-bank blocks. Reconstruct the manifest with the original support-draw helper and require exact
agreement with the bound file before new projections. Support pools contain nine distinct patients
per existing finding; no target enters any prototype.

## Fixed contrast and exact contributions

Build one full-pool prototype per finding by unit-normalizing the mean of all nine unit patient-QRS
support vectors, giving each support equal weight. For each target rank its three full-pool cosines
descending, breaking exact ties by lower finding index. Freeze that winner w and runner-up r in a
local manifest before any single-bank/ensemble comparison. This is an algorithmic reference contrast,
without target annotation selection. Every subsequent bank retains the SAME ordered w/r even when
its margin is negative or its own highest-scoring finding changes. Do not rerank the contrast.

Each single-bank prototype is the unit-normalized mean of its three unit support vectors. Each
five-bank ensemble prototype is unit(mean of its five single-bank unit prototypes), separately per
finding. This gives each bank equal weight and may repeat support patients; it differs from averaging
unnormalized cosine scores or contribution vectors. No alternative ensemble rule will be selected.
Each ensemble uses 15 support slots per finding, compared with three for a single bank, with repeats
within the nine-patient pool. Freeze identity-only distinct-support counts for every bank/ensemble
and report their distributions. The common full-nine-pool contrast is shared by both comparisons.
The comparison changes support exposure and computation; it cannot establish an averaging benefit
at equal support or computational cost.

Use a new thin fixed-contrast projection helper over the unchanged NTT-08 occupancy-weighted region
means v_b, unit beat vectors u_b=v_b/||v_b|| and D=||sum_b u_b||. For each bank's frozen contrast
delta=p_w-p_r, c_b=delta dot u_b/D and c_bt=a_bt*(delta dot h_t)/(||v_b||*D). Sum beat-token
contributions over beats with the original unique token identity. Require beat and token sums to
reconstruct the fixed-contrast cosine margin within 1e-10 for every bank/ensemble/target. Undefined
region or aggregate normalization aborts with a typed error rather than silently dropping a target.
Do not call NTT-08's dynamic-contrast decomposition to choose the NTT-09 contrast.

## Secondary descriptions, privacy and runtime

Describe within-patient signed-rank Spearman agreement, absolute-contribution rank agreement and
top-absolute-quarter set Jaccard across all pairs; constant ranks remain null. Describe patient-first
beat/active-unique-token positive/negative/absolute mass, cancellation, effective fraction and top
absolute shares, preserving zero-mass nulls. Report each family's fixed-margin range and negative/
zero margin counts, positive-set sizes, signed/absolute rank boundary ties, pair coverage and every
patient's contribution variability locally. Aggregate variability summaries use equal-patient means
of per-beat/per-active-token population standard deviations across banks, for signed and absolute
contributions separately. These are sensitivity summaries, not pathology masks or probabilities.
No deletion experiment, new target classification accuracy or clinical recommendation is added.

Use two CPU threads and a measured 1,200-second ceiling including integrity replay, 3,600 new
bank/ensemble-target decompositions, all pair summaries and artifact writing. Check time between
patients. Preserve typed failure/progress and never silently retry, discard or replace a bank.
Save versions, commit, all source/input/artifact hashes through repository-relative stored paths,
exact local identities/splits/manifests, coordinates, contributions, margins, pairs and summaries
under `outputs/ntt09_reference_contribution_stability/`. Only explicit aggregate allowlists and
standalone aggregate figures enter `reports/localization/ntt09-reference-contribution-stability/`;
no real traces, individual identities, hashes, coordinates or maps leave the local outputs.
The executing agent writes results from live outputs, including failed coverage or negative effects,
and arranges independent reconstruction. The coordinator owns queue/backlog/index updates. Existing
clinical review remains pending. No result rescues NTT-06's failed primary or substitutes stability
for expert annotation. All work stays local; no push or merge.
