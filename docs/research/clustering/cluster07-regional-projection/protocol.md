# CLUSTER-07: equal-patient regional covariance projection

## Question and primary decision

CLUSTER-05's patient-vector PCA3 captured only 10.79% of equal-patient regional variance and its
matched regional concordance gate failed. This successor asks whether choosing three directions from
regional covariance changes concordance with the frozen NTT regional view. It cannot reverse the
negative CLUSTER-03 or CLUSTER-05 primary decisions. Increased regional variance retention is expected
by construction and is not scientific or clinical success by itself.

The primary comparison is `regional_covariance9_rms` versus `ntt_region9_rms`. Both selected candidates
must satisfy the unchanged CLUSTER-03/05 eligibility rule and validation-anchor ARI must be >=0.50.
Either ineligibility or ARI below 0.50 is a negative primary result. The original patient-PCA regional
view, all pairwise comparisons, variance diagnostics and annotation/nuisance associations are
descriptive and cannot rescue this decision. No configuration or projection is optimized for passing.
This evaluates concordance on explored development patients, not disease discovery, clinical
diagnosis, specialist localization, physiological causes or referral performance.

Scientific reviewer and coordinator approval precede protocol commit. Commit this protocol before any
new score, then commit source and tests before execution. The executing agent writes the results.

## Data, integrity and frozen replay

Use only verified existing caches and the unchanged 1,993 patients, 1,582 fitting patients, 411 anchors
and eight saved 80% fitting-patient subsamples. Whole patients touching PTB folds 9/10, all 27 support
patients and PR67 non-training roles remain excluded. Run the complete existing identity and role gate
before reading feature payloads. Verify all receipt-bound input/source/artifact hashes and CLUSTER-05
result SHA256 `cbe2d5af704e6a1ee86ba535ea03e2087bd667b33573ec676952d4e65275f885`.
Preserve CLUSTER-04 result SHA256
`c672be6b2fc9fe6d8c14d974179f04345381b068349df611d1656723e0427baa` as contextual evidence;
no CLUSTER-04-derived model is selected here.

Before new scores, reconstruct CLUSTER-05 transforms for all nine fitting samples, all 1,440 grid fits,
all candidate diagnostics and selections, the complete unsupervised decision, and 23 original support
control fits. Require transform error <=1e-12, identical labels, candidate numeric error <=1e-12,
identical selection/decision and matching original sample hashes. Also reproduce the five selected
CLUSTER-03 fits and three original bank fits using the frozen helpers. Abort on disagreement.
After the new unsupervised decision is frozen, reproduce the predecessor annotation summaries too;
annotation parsing cannot occur during the preceding selection/replay phase.

## Equal-patient fitting and matched views

For each fitting sample independently, with n fitting patients and m_i regions per patient, use
weights w_ij=1/(n*m_i), mean mu=sum_ij w_ij*r_ij and covariance
C=sum_ij w_ij*(r_ij-mu)*(r_ij-mu)^T. Each patient has equal total weight. Do not use ordinary
region-count-weighted PCA. Accumulate one covariance matrix through weighted centered rows; do not
allocate a matrix per patient. Use float64 symmetric eigendecomposition, descending eigenvalue order,
three unwhitened directions, and orient each by its first largest-absolute loading positive before
nonlinear pooling. Exact/near eigenvalue ties do not justify changing dimensions or favorable signs;
report the third/fourth eigenvalue gap, relative gap, and deterministic-library basis limitation.

Project every unit region vector using the fitting regional mean and three directions. Concatenate
per-patient coordinate means, maxima and population standard deviations to obtain nine axes. Refit
covariance, mean and basis independently in each frozen subsample; anchors never enter fitting.
Compare three RMS-scaled nine-dimensional views: `regional_covariance9_rms`, CLUSTER-05's
`native_region9_rms` (patient PCA3 directions), and original `ntt_region9_rms`. Refit each view's
patient-coordinate centering and single scalar sqrt(mean(centered_fitting_coordinates**2)) using only
the current fitting sample.
No per-axis standardization, whitening, label residualization, added component or support-bank change.

Report weighted regional variance retained by the new and patient-PCA3 bases and patient-QRS variance
retained by both bases for the reference and all eight refits. Variance fractions are centered on the
appropriate fitting population mean. Also report eigenvalues, spectral gap and projection runtime.
Regional PCA still differs from NTT's source-label-conditioned directions and nonlinear three-bank
averaging. Direction signs affect maxima. These differences limit attribution even if concordance passes.

## Grid, budget and interpretation

Run all 16 unchanged candidates per view: k=2/3/4/8 and k-means/diagonal/tied/full GMM, with frozen
CLUSTER-03 seeds, n_init, regularization and iteration limits. Use its eight refits, convergence checks,
minimum 5% fitting/anchor occupancy, silhouette >=0.05, median stability >=0.80 and minimum >=0.60.
Rank by median stability, minimum stability, silhouette, smaller k, then frozen algorithm order.
Select the same-ranked ineligible fallback only for description. Preserve every candidate and failure.
There are 48 new configurations and 432 new grid fits, plus 1,440 predecessor grid fits and 31
predecessor reference/support fits. No conditional grid truncation or new support-label search.

Use two CPU threads and a measured 1,200-second wall budget covering verification, replay, projection,
clustering, interpretation and output writing, checked between operations. Save partial candidate
progress and typed failure on budget exhaustion. No GPU, waveform reads, encoder inference, protected
test data, downloads, dependency changes, external API, push or merge. No age/sex subgroup or clinical
label-definition change.

Save and hash complete label-free selection and decision before reading existing target annotations.
Only afterward describe the unchanged overlapping NORM/MI/STTC/CD/HYP combination AMI and existing
nuisance/quality associations, including heart rate and amplitude; no significance/enrichment claim.
Keep all patient identities, vectors, bases, assignments, splits and individual feature/input bindings
local under `outputs/cluster07_regional_projection/`. Commit only aggregate-safe reports under
`reports/clustering/cluster07-regional-projection/` and this study's results report. Record source,
protocol, predecessor, input/artifact hashes, runtime, selected conditions and failures. The coordinator
owns both queue files and backlog updates. No follow-up study is automatically authorized by this protocol.
