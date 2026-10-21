# Scientific review of the frozen-localizer landmark diagnostic

21 October 2026. I reviewed the concrete protocol/configuration committed as
b829ad1 and prospective profile-decomposition amendment 71d740c. The proposed
study is scientifically coherent as a diagnostic of landmark and activity
attraction in frozen time-by-lead heatmaps. It does not test anatomical lesions,
clinical discrimination, real disease support or a newly trained model. There
is no scientific-success or clinical PASS threshold. This is a prospective
design review, not an experimental result.

I recovered root a0d8cf6, read AGENTS.md, CONTRIBUTING.md, the current handoff,
queues and earlier synthetic/real reports, and inspected local output metadata.
All twelve final states are present under the paired-suppression study: three
fixed fits by B/P/K/PK. Each file is 144776 bytes, and every file-byte SHA256
matches its original main output manifest. This verifies asset availability,
not model performance. I loaded no waveform, annotation, heatmap values or
score array and ran no inference. The model source is the observed-only
three-Conv1d network with two ReLU stages; no alternative architecture is
substituted. File identities and tensor-state identities must remain separate.

The fixed B primary is a useful reference without choosing a winner. It
averages released-N event ranks within each native lead, eight leads within a
record, and the three fixed B fits within each record, then 160 records equally.
Six within-span circular map rolls preserve each map's distribution. They test
relative alignment to the retained timestamps and can preserve periodic
alignment; they are not causal controls or a universal null. Constant maps
correctly produce 0.5 at every landmark and zero gain, without exclusions.
Shared 2000 whole-record bootstrap draws, seed 2026106501, keep the paired arm
contrasts aligned. Fixed fits are not independent patients or a representative
sample of training runs. Uncertainty remains conditional on these models,
public development records and released annotation provenance.

The primary point and interval should be reported together. Concentration near
released N landmarks supports landmark attraction under the prescribed phase
controls. It does not show that maps mainly ignore introduced changes, nor
that these activations are clinical false positives. A low gain cannot show
clinical specificity: a constant high map also has zero gain. Preserve the
original 14/15 scientific FAIL and exactly reproduce its saved all-twelve
full-grid AP/aggregate closure, rather than replacing it with the new ranking
view. Establishing a causal beat-versus-edit mechanism requires a later matched
intervention experiment with known numerical support.

The input path matches the trained generator representation: physical native
I,II,V1..V6 cast float32, fixed per-record robust standardization, then canonical
twelve-lead derivation. This does not validate physical voltage calibration or
clinical deployment preprocessing. Scoring only the eight native outputs is
necessary: independently standardizing I/II changes their reconstructed limb
combinations relative to the originally annotated limb signals. Saving the
other four maps preserves outputs without asserting annotation equivalence.
All 160 measurement IDs remain; the 40 parser IDs remain nonscored. LUDB is spent
public development evidence. No held-out clinical cohort, normal label or
disease mask is created by its inclusion or released N code.

The energy secondary uses the exact standardized native model input, full
20-sample derivative calculation before cropping, and the same unlabeled
annotation span. This matches the representation the network sees. Spearman,
strict-q90 map/energy Jaccard and their fixed rolls describe association, not
causal attribution. Undefined correlations retain explicit nulls/reasons and
defined-only denominators; any such means condition on the stated subset and
must not be compared as though all arms had identical denominators. Empty
Jaccard unions are retained with zero and flags. Fixed raw probability
mean/q95/maximum and constant counts are important safeguards because ranks,
correlations and top-decile masks are insensitive to output magnitude. These
uncalibrated probabilities supply no referral or disease threshold. Every
fit/arm/lead result remains visible, including PK-minus-B as secondary only.

The technical closure strategy is appropriate: exact prior synthetic/timing/
coherence statistics and bootstrap draws, fixed synthetic background 2000's
four-variant replay through all twelve CUDA states at absolute 1e-6/rtol 0,
and independent functional CPU replay for real IDs 1 and 199 at absolute 1e-5/
rtol 0. Record every difference and reject an exceeded tolerance; do not choose
new examples, devices, omissions or tolerances after inspecting outcomes.
Independent saved-map arithmetic and physical-input reconstruction are needed
in addition to the selected forward checks.

The resource amendment correctly separates fixed CUDA initialization/checkpoint
loading/replay/figure work from recurring batch-eight input/inference/map-write/
hash work. Count fixed work once in a conservative forecast, rather than
extrapolating one figure twenty times. Include actual setup elapsed time and
all closure/summary/persistence work. The amendment retains every hard cap:
900 CPU/wall seconds, 240 GPU-lock seconds, 2 GiB RSS, 768 MiB outputs and 2 GiB CUDA
allocation, with declared free-memory admission; independent audit remains
900 CPU/1200 wall, 2 GiB RSS/128 MiB outputs and zero GPU. The actual small CUDA
operation, shared nonblocking lock and complete-path profile must precede new
landmark/activity summaries. A failed gate or occupied GPU is a concrete stop;
it does not justify a classification replacement or unprofiled training.

Scientific design GO is conditional on complete committed main/audit source,
artificial tests and identity admission implementing these exact definitions,
all twelve states, all 160 records and the profile amendment. Actual technical
completion still requires successful replay/resource gates and independent
numerical agreement. No outcome or score is certified by this review. Any later
editor-family/background-source study needs a new prospective protocol and
resource gates before execution; this diagnostic authorizes no optimizer step.

Reviewed main protocol SHA256:
`eadd41535559e9ea579d44e7431f4169f2624018fd12875d4c589f21e74e0d70`;
main configuration SHA256
`71803d46ffc5e3e4e582c7172d3bbc07791a48c599cbd3f43d01618b00da22a2`;
profile amendment SHA256
`bf784c594ae2a830e4c9f8fba65a0157f108b2ce3e8f762684436d5828a3af43`;
profile configuration SHA256
`21923246b26f67341b1701cd1d9aa7a151e452130aa0f6633479047e2269ff2e`.
