"""Build and sequentially execute the aggregate KEY-01 explanation notebook."""

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).absolute().parents[2]


def main() -> None:
    """Write and execute the notebook with the current project interpreter."""
    cells = []
    evidence = json.loads((ROOT / "reports/condition-keys-2026-11-03/aggregate.json").read_text())

    def markdown(text: str) -> None:
        """Append a Markdown cell."""
        cells.append(nbformat.v4.new_markdown_cell(text.strip()))

    def code(text: str) -> None:
        """Append a Python cell."""
        cells.append(nbformat.v4.new_code_cell(text.strip()))

    markdown(r"""
# ECG condition keys: whole records, selected regions and clusters

KEY-01 · 3 November 2026 · Executed analysis of reused external development data

This notebook asks whether averaging **selected ECG-region embeddings** gives better reference
keys than averaging **whole-record embeddings**. It explains the algorithm, compares prediction
rankings and describes clusters. Four existing annotation groups are examined: myocardial
infarction annotations (MI), ST/T changes (STTC), conduction findings (CD), and hypertrophy
annotations (HYP). An ECG can belong to several groups. These annotations are proxies; the
experiment does not establish a person's disease, screening eligibility or clinical diagnosis.

The prospective protocol was committed before scoring. Its primary endpoint, controls and gates
remain fixed. All values and charts below come from the executed KEY-01 artifacts. No example
patient identifiers, raw waveforms or per-record feature vectors are displayed.
""")
    code("""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

ROOT = Path.cwd()
if not (ROOT / 'ecg_experiment').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from ecg_experiment.files import sha256_file

report_path = ROOT / 'reports/condition-keys-2026-11-03/aggregate.json'
result = json.loads(report_path.read_text())
assert result['status'] == 'completed'
artifact_path = ROOT / 'outputs/condition_keys_20261103_v2/artifacts.npz'
assert sha256_file(artifact_path) == result['artifact_sha256'][str(artifact_path.relative_to(ROOT))]
private = np.load(artifact_path, allow_pickle=False)
groups = result['groups']
arms = list(result['metrics'])
names = {
    'whole': 'Whole-record keys',
    'window_whole_keys': 'Region search, whole-record keys',
    'window_selected_keys': 'Selected-region keys',
    'fixed_selected_keys': 'Fixed 1-second region keys',
    'shuffled_keys': 'Permuted condition keys',
}
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
print('Study:', result['study'], '| source commit:', result['git_commit'][:12])
print('Scope:', result['scope'], '| CPU seconds:', round(result['wall_seconds'], 2))
""")
    markdown(r"""
## 1. What the professor's idea means

A key is a reference direction in an embedding space. The encoder turns a recording or region into
a vector. Averaging labeled reference vectors produces a key for that group. A new recording is
ranked by how much its vector resembles that group's key compared with a normal-reference key.

The [2025 No time to train! paper](https://arxiv.org/html/2507.02798v1) combines image-region
proposals with pretrained features and reference prototypes. KEY-01 adapts the reference-matching
idea to native ECG embeddings. **SAM2 and DINOv2 are not run here.** Regions are contiguous
windows of existing xECG tokens, not image masks or physiological P/QRS/T delineations.

```text
Training ECGs -> frozen xECG vectors -> training-only standardization
             -> whole-record keys -> select reference regions once -> region keys

New ECG -> candidate token windows -> similarity to frozen keys -> one score per group
```

Training-free matching still needs labeled references and a pretrained encoder. It cannot create
an averaged key for a condition with no reference examples. Adding a key later is possible without
retraining the encoder, but recognition of an unseen disease category is not tested in this notebook.
""")
    markdown(r"""
## 2. Data and separation of roles

Keys use the unchanged all-age 25,000-record training cohort. Human-read sources supply the
labels. SPH supplies evaluation ECGs and never contributes reference vectors, scaler estimates,
cluster centers or selection rules. All four groups are already represented in the training pool.

The SPH evaluation role is patient-separated from its existing calibration role. Calibration
records are not used here because this is a ranking experiment. SPH has been examined in prior
experiments and is **development data**. No closed Challenge or EchoNext test data are opened.
Source-family Challenge splits are record-based, so their training patient identities are incomplete.
No age subgroup is analysed. The encoder's original pretraining does not establish class-unseen learning.
""")
    code("""
counts = result['counts']
display(pd.DataFrame([
    {'role': 'Reference training', 'records': counts['training'],
     'normal annotations': counts['training_normals'],
     'multiple abnormal groups': counts['training_multiple_groups']},
    {'role': 'SPH evaluation', 'records': counts['query'],
     'normal annotations': counts['query_normals'],
     'multiple abnormal groups': counts['query_multiple_groups']},
]).set_index('role'))
print('SPH evaluation patients:', counts['query_patients'])
support_counts = pd.DataFrame({'positive reference ECGs': counts['training_positive']})
support_counts['SPH positive ECGs'] = [
    result['metrics']['whole']['groups'][g]['positives'] for g in support_counts.index
]
display(support_counts)
source_counts = pd.Series(counts['training_sources']).sort_values()
source_counts.plot.barh(figsize=(8, 3), color='#427a9f', xlabel='Reference ECGs', title='Training sources')
plt.tight_layout()
plt.show()
""")
    markdown(r"""
## 3. Integrity before new scores

Every directly used input is checked against predecessor receipt hashes. The saved HEADS-01
SPH prediction metrics are recomputed exactly, without refitting its models. Separately, averaging
the 40 float16 cached tokens must reproduce the saved float32 pooled vector to relative L2 error
at most 0.001. That second test qualifies pooling semantics at cache precision; it does not imply
bit-identical embeddings. A mismatch stops the experiment rather than silently dropping records.
The first attempt stopped before scoring because it could not parse decimal membership strings.
The committed v2 execution correction validates those strings as exact numeric zero/one values;
the scientific comparison and cohort identities are unchanged and the failed attempt is preserved.

Earlier local visual/native reference-matching studies did not establish improvement over their
strongest controls. KEY-01 does not change those outcomes. Experiments 024/024b motivated the
prototype comparator; their unavailable original outputs were not claimed as reproduced.
""")
    code("""
print('Exact predecessor replay:', result['integrity']['predecessor'])
display(pd.DataFrame(result['integrity']['pooling']).T)
print('Verified directly used inputs:', len(result['input_sha256']))
display(pd.DataFrame([result['profile']]))
""")
    markdown(r"""
## 4. The keys and similarity score

Fit a StandardScaler on full-record **training** embeddings only. Apply that same scaler to
whole vectors and window means, then normalize each vector to unit length. Each reference ECG
has equal weight within a key. The positive key for group $c$ and its score are:

$$k_c^+ = \operatorname{unit}\left(\frac{1}{N_c}\sum_{i:y_{ic}=1}z_i\right),
\qquad s_c(z)=z^\top k_c^+ - z^\top k_c^0.$$

$k_c^0$ is the normal-reference key. Other abnormal groups are not treated as normal support.
Multiple-label ECGs contribute to every applicable positive key. Similarity margins are uncalibrated
scores, not disease probabilities. A whole-record key is one mean direction; variation within a
group and coexisting annotations can make that direction an incomplete description.

For selected-region keys, positive references choose their largest initial margin and normal
references choose their smallest margin. These selected vectors are averaged into new keys
**once**. Query labels never enter region selection. Selecting regions from record-level labels
is weak supervision, not a regional annotation of where a finding occurs.
""")
    code("""
# Verify the geometric meaning of the saved keys using only local private artifacts.
key_norms = []
for arm in ['whole', 'window_selected_keys', 'fixed_selected_keys']:
    for kind in ['positive', 'normal']:
        values = private[f'{arm}_{kind}']
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-12)
        key_norms.append({'arm': names[arm], 'key': kind,
                          'number': len(values), 'dimensions': values.shape[1]})
display(pd.DataFrame(key_norms))
""")
    markdown(r"""
## 5. Candidate regions and controls

xECG uses 25 samples per patch at 100 Hz: 40 contextual tokens across a ten-second ECG.
Variable proposals include every stride-one span of 2, 4 or 8 tokens: 109 windows of nominal
length 0.5, 1 or 2 seconds. The fixed-window control uses ten nonoverlapping one-second windows.
Ties choose the first candidate in length/start order. No window length is tuned on SPH.

The tokens contain context from the encoder. Pooling tokens belonging to a span is not the same
as encoding that cropped waveform alone. These indices do not establish a clinical boundary.

The five arms isolate the whole-record baseline, region search using unchanged keys, reference
region selection, a simpler fixed-window selection control, and a cyclic permutation of group keys.
Each query's group score is the maximum margin across its candidate windows. No single winning
condition is forced on a recording: the four scores can all be high or low.
""")
    code("""
from ecg_experiment.condition_keys import spans

display(pd.DataFrame([
    {'proposal': 'Variable', 'candidates per ECG': len(spans()), 'lengths in seconds': '0.5, 1, 2'},
    {'proposal': 'Fixed', 'candidates per ECG': len(spans(True)), 'lengths in seconds': '1'},
]))
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for ax, group in zip(axes, groups):
    hist = result['selections']['window_selected_keys']['support_positive'][group]['lengths']
    ax.bar([float(k) * 0.25 for k in hist], list(hist.values()), width=0.2, color='#427a9f')
    ax.set(title=group, xlabel='Selected nominal seconds')
axes[0].set_ylabel('Positive reference ECGs')
fig.suptitle('Which reference lengths did the single selection pass choose?')
plt.tight_layout()
plt.show()
""")
    markdown(r"""
## 6. Primary prediction results

For each group, every other SPH evaluation ECG counts as a negative, including recordings with
other abnormal groups. This evaluates whether the key distinguishes the requested group rather
than only separating abnormal ECGs from normal ones. Missing or imperfect annotations limit that
interpretation. AUROC measures ranking, with 0.5 indicating chance ranking; average precision (AP)
also depends on the group's prevalence. Neither metric sets a clinical referral threshold.

The primary endpoint is the unweighted mean of the four AUROCs. The frozen contrast is selected
region keys minus whole-record keys. Its interval uses 2,000 paired whole-patient bootstrap draws,
seed 20261103. The same patients are sampled for all groups and both arms. It holds the support
cohort, learned keys and pretrained encoder fixed.
""")
    code("""
summary = pd.DataFrame([
    {'arm': names[arm], 'macro AUROC': result['metrics'][arm]['macro_auroc'],
     'macro AP': result['metrics'][arm]['macro_average_precision']}
    for arm in arms
]).set_index('arm')
display(summary.style.format('{:.4f}'))
primary = result['primary']
display(pd.DataFrame([primary]).style.format(precision=6))
print('Frozen decision gates:', result['decision'])
fig, ax = plt.subplots(figsize=(8, 3))
summary['macro AUROC'].iloc[::-1].plot.barh(ax=ax, color='#427a9f')
ax.set(xlim=(0, 1), xlabel='Macro one-versus-rest AUROC', ylabel='')
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()
""")
    code("""
per_group = pd.DataFrame([
    {'group': group, 'arm': names[arm], **result['metrics'][arm]['groups'][group]}
    for group in groups for arm in arms
])
display(per_group.pivot(index='group', columns='arm', values='auroc').style.format('{:.4f}'))
display(per_group.pivot(index='group', columns='arm', values='average_precision').style.format('{:.4f}'))

fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
for c, (ax, group) in enumerate(zip(axes, groups)):
    y = private['query_flags'][:, c]
    for arm in ['whole', 'window_selected_keys', 'fixed_selected_keys']:
        fpr, tpr, _ = roc_curve(y, private[f'score_{arm}'][:, c])
        ax.plot(fpr, tpr, label=names[arm])
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8)
    ax.set(title=group, xlabel='False positive rate', ylabel='True positive rate')
axes[-1].legend(loc='lower right', fontsize=7)
fig.suptitle('Ranking each annotation group against all other evaluated ECGs')
plt.tight_layout()
plt.show()
""")
    code("""
fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
for c, (ax, group) in enumerate(zip(axes, groups)):
    y = private['query_flags'][:, c]
    for arm in ['whole', 'window_selected_keys', 'fixed_selected_keys']:
        precision, recall, _ = precision_recall_curve(y, private[f'score_{arm}'][:, c])
        ax.plot(recall, precision, label=names[arm])
    ax.axhline(y.mean(), linestyle='--', color='gray', linewidth=0.8)
    ax.set(title=f'{group} (prevalence {y.mean():.1%})', xlabel='Recall', ylabel='Precision', ylim=(0, 1.03))
axes[-1].legend(fontsize=7)
fig.suptitle('Precision–recall curves; dashed line is the annotation prevalence')
plt.tight_layout()
plt.show()
""")
    markdown(r"""
## 7. Why the negative comparison matters

Condition-versus-normal discrimination is an easier, different task: other abnormalities are
removed from the negatives. The next table shows both definitions without substituting the
easier one for the primary endpoint. If performance is higher against normals, part of the
representation may reflect general abnormality rather than a specific annotation group.
""")
    code("""
negative_comparison = pd.DataFrame([
    {'group': group, 'comparison': task, 'AUROC': values[arm]['groups'][group]['auroc'],
     'AP': values[arm]['groups'][group]['average_precision']}
    for group in groups
    for task, values in [('All other ECGs', result['metrics']),
                         ('Normal ECGs only', result['condition_vs_normal'])]
    for arm in ['window_selected_keys']
])
comparison_table = negative_comparison.pivot(index='group', columns='comparison', values=['AUROC', 'AP'])
display(comparison_table.style.format('{:.4f}'))
""")
    markdown(r"""
## 8. Do the keys point in similar directions?

Cosine similarity between positive keys describes embedding geometry. High similarity can reflect
shared morphology, coexisting annotations, general abnormality or source effects. It is not a
measure of clinical equivalence or the probability that two diagnoses occur together.
""")
    code("""
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, arm in zip(axes, ['whole', 'window_selected_keys', 'fixed_selected_keys']):
    values = np.asarray(result['positive_key_cosines'][arm])
    im = ax.imshow(values, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set(xticks=range(4), yticks=range(4), xticklabels=groups, yticklabels=groups, title=names[arm])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', fontsize=9)
fig.colorbar(im, ax=list(axes), shrink=0.7, label='Cosine similarity')
plt.show()
""")
    markdown(r"""
## 9. Descriptive clusters: fitting without condition labels

PCA is fitted on standardized, unit whole-record **training** embeddings (16 components).
K-means then partitions those training coordinates into five clusters, with ten initializations
and seed 20261103. SPH is projected and assigned to the frozen centers. Annotation memberships
do not enter PCA or k-means; they are used afterward to describe each cluster.

Five is a prespecified number of clusters, not a finding that five types of disease exist.
Cluster numbers have no clinical names. A cluster can contain normals, several conditions and
unannotated variation. We compare adjusted mutual information (AMI) with source family and the
exact combination of the four annotations. AMI is descriptive, adjusts for chance and depends on
the factors being compared; it does not prove removal of hospital effects or clinical validity.
""")
    code("""
cluster = result['clusters']
variance = np.asarray(cluster['pca_explained_variance_ratio'])
print(f'First two PCA components explain {variance[:2].sum():.1%} of training variance.')
print(f'All sixteen explain {variance.sum():.1%}.')
display(pd.Series(cluster['training_ami'], name='Training cluster AMI').to_frame())
fig, ax = plt.subplots(figsize=(7, 2.5))
ax.plot(np.arange(1, 17), np.cumsum(variance), marker='o', markersize=3)
ax.set(xlabel='PCA components', ylabel='Cumulative variance explained', ylim=(0, 1))
plt.tight_layout()
plt.show()
""")
    code("""
train_clusters = pd.DataFrame(cluster['training']).set_index('cluster')
query_clusters = pd.DataFrame(cluster['sph']).set_index('cluster')
display(train_clusters.rename_axis('Training cluster'))
display(query_clusters.rename_axis('SPH cluster'))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, table, title in zip(axes, [train_clusters, query_clusters],
                           ['Training composition', 'SPH composition']):
    prevalence = table[['normal', *groups]].div(table['records'], axis=0).fillna(0)
    im = ax.imshow(prevalence, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    ax.set(xticks=range(5), yticks=range(5), xticklabels=['Normal', *groups],
           yticklabels=range(5), xlabel='Annotation', ylabel='Cluster', title=title)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{prevalence.iloc[i, j]:.0%}', ha='center', va='center')
fig.colorbar(im, ax=list(axes), shrink=0.8, label='Fraction of cluster carrying annotation')
plt.show()
print('Group memberships overlap, so the annotation fractions need not add to 100%.')
""")
    code("""
sources = pd.DataFrame(cluster['training_source_counts']).set_index('cluster')
display(sources)
sources.div(sources.sum(axis=1), axis=0).plot.bar(stacked=True, figsize=(9, 3.5), colormap='tab10')
plt.ylabel('Fraction of training cluster')
plt.title('Source-family composition: a check for hospital structure')
plt.legend(title='Source family', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
""")
    markdown(r"""
## 10. A two-dimensional view of SPH embeddings

The following charts are aggregated density grids, not individual patient scatter plots. All use
the same PCA axes fitted on training data. Each panel includes records carrying its annotation;
multiple-label records appear in more than one panel. For visual comparison, each panel's density
is normalized by its own record count, with a shared color scale. The grid bounds come from the
0.5th–99.5th training percentiles, and out-of-grid counts are reported. Two-dimensional overlap
does not establish overlap in all sixteen clustering dimensions, and apparent separation does
not establish useful classification accuracy.
""")
    code("""
# Show the frozen cluster assignments on the same training-fitted PCA plane.
edges = [np.asarray(e) for e in cluster['grid_edges']]
coordinates = private['query_pca']
assigned = private['query_cluster']
cluster_grids = []
for k in range(5):
    keep = assigned == k
    grid, _, _ = np.histogram2d(coordinates[keep, 0], coordinates[keep, 1], bins=edges)
    cluster_grids.append(grid.T / keep.sum())
vmax = max(g.max() for g in cluster_grids)
fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True)
for k, (ax, grid) in enumerate(zip(axes, cluster_grids)):
    im = ax.pcolormesh(*edges, grid, vmin=0, vmax=vmax, cmap='magma', shading='auto')
    ax.set(title=f'Cluster {k} (n={(assigned == k).sum():,})', xlabel='Training PC1')
axes[0].set_ylabel('Training PC2')
fig.colorbar(im, ax=list(axes), shrink=0.75, label='Fraction of cluster records per bin')
plt.show()
""")
    code("""
panel_names = ['normal', *groups]
grids = cluster['sph_pca_grids']
xedges, yedges = [np.asarray(e) for e in cluster['grid_edges']]
densities = {g: np.asarray(grids[g]['counts']).T / grids[g]['records'] for g in panel_names}
vmax = max(a.max() for a in densities.values())
fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True)
for ax, group in zip(axes, panel_names):
    im = ax.pcolormesh(xedges, yedges, densities[group], cmap='magma', vmin=0, vmax=vmax, shading='auto')
    ax.set(title=f'{group} (n={grids[group]["records"]:,})', xlabel='Training PC1')
axes[0].set_ylabel('Training PC2')
fig.colorbar(im, ax=list(axes), shrink=0.75, label='Fraction of panel records per bin')
plt.show()
display(pd.DataFrame({g: {k: v for k, v in grids[g].items() if k != 'counts'} for g in panel_names}).T)
""")
    markdown(r"""
## 11. Arithmetic checks on the saved result

These checks reload the local private scores and recompute every per-group AUROC and AP,
including the condition-versus-normal task. They also check key geometry, region histograms
and cluster accounting. They do not refit keys or choose a new analysis after seeing results.
The saved patient-bootstrap interval and decision remain the prospectively specified comparison.
""")
    code("""
checks = 0
flags = private['query_flags']
for arm in arms:
    scores = private[f'score_{arm}']
    for c, group in enumerate(groups):
        for normal_only, container in [(False, result['metrics']), (True, result['condition_vs_normal'])]:
            keep = (flags[:, c] == 1) | ~flags.any(axis=1) if normal_only else np.ones(len(flags), bool)
            expected = container[arm]['groups'][group]
            assert roc_auc_score(flags[keep, c], scores[keep, c]) == expected['auroc']
            assert average_precision_score(flags[keep, c], scores[keep, c]) == expected['average_precision']
            checks += 2
for arm in ['whole', 'window_selected_keys', 'fixed_selected_keys']:
    key = private[f'{arm}_positive']
    np.testing.assert_array_equal(key @ key.T, np.asarray(result['positive_key_cosines'][arm]))
    checks += 1
assert sum(row['records'] for row in cluster['training']) == counts['training']
assert sum(row['records'] for row in cluster['sph']) == counts['query']
checks += 2
for arm in ['window_selected_keys', 'fixed_selected_keys']:
    for group in groups:
        selection = result['selections'][arm]['query_all'][group]
        assert sum(selection['starts']) == sum(selection['lengths'].values()) == counts['query']
        checks += 1
assert len(np.unique(private['query_patient'])) == counts['query_patients']
checks += 1
for prefix, rows in [('training', cluster['training']), ('query', cluster['sph'])]:
    local_flags = private['train_flags'] if prefix == 'training' else flags
    local_cluster = private[f'{prefix}_cluster']
    for row in rows:
        mask = local_cluster == row['cluster']
        assert int(mask.sum()) == row['records']
        assert int((~local_flags[mask].any(axis=1)).sum()) == row['normal']
        checks += 2
        for c, group in enumerate(groups):
            assert int(local_flags[mask, c].sum()) == row[group]
            checks += 1
print(f'{checks} saved-output arithmetic checks passed.')
verification = {
    'study': 'KEY-01', 'status': 'passed', 'checks': checks,
    'artifact_sha256': sha256_file(artifact_path), 'report_sha256': sha256_file(report_path),
    'refitted': False, 'raw_waveforms_read': False,
}
verification_path = ROOT / 'reports/condition-keys-2026-11-03/notebook-verification.json'
verification_path.write_text(json.dumps(verification, indent=2) + '\\n')
""")
    markdown(r"""
## 12. What this experiment establishes

The executed result below describes the fixed contrast, not a search for the best-looking arm.
Advancement requires a macro-AUROC gain of at least 0.01, a 95% lower bound above zero, and
a candidate point estimate at least as high as the fixed-window control. Cluster composition
and easier condition-versus-normal comparisons do not replace a failed primary gate.

This experiment measures recognition of four existing annotation groups on reused hospital
development ECGs. It does not test absent-reference diseases, physiological segmentation,
clinical diagnosis, university-student referral performance or an untouched final test. The interval
does not quantify uncertainty from resampling reference patients or retraining the encoder.
Project recommendations and follow-up priorities are recorded in the separate results report.
""")
    primary = evidence["primary"]
    verdict = "passed" if evidence["decision"]["advance"] else "did not pass"
    markdown(f"""
The selected-region versus whole-record macro-AUROC difference was
**{primary['difference']:+.4f}**, with a paired 95% interval
**[{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}]**. The selected-region candidate
{verdict} all three frozen advancement gates. This is the prespecified result even when a
secondary control or an individual annotation group has a more favorable point estimate.
""")
    group_changes = []
    for group in evidence["groups"]:
        candidate = evidence["metrics"]["window_selected_keys"]["groups"][group]
        whole = evidence["metrics"]["whole"]["groups"][group]
        group_changes.append(
            f"- {group}: selected-region AUROC {candidate['auroc']:.4f}, "
            f"whole-record {whole['auroc']:.4f}; difference {candidate['auroc'] - whole['auroc']:+.4f}."
        )
    markdown("Per-group changes are descriptive and have no separate confirmatory intervals:\n\n"
             + "\n".join(group_changes))
    cluster_text = []
    for group in evidence["groups"]:
        rows = evidence["clusters"]["sph"]
        best = max(rows, key=lambda row, g=group: row[g] / max(row["records"], 1))
        prevalence = best[group] / best["records"]
        base = evidence["metrics"]["whole"]["groups"][group]["positives"] / evidence["counts"]["query"]
        cluster_text.append(
            f"- {group} is most concentrated in cluster {best['cluster']}: "
            f"{prevalence:.1%} of its {best['records']:,} ECGs carry the annotation, "
            f"against {base:.1%} across all SPH evaluation ECGs."
        )
    markdown("The cluster composition tables give these descriptive concentrations:\n\n"
             + "\n".join(cluster_text)
             + "\n\nThese are annotation enrichments, not cluster diagnoses. The same cluster may be "
               "most enriched for several groups, and a group's ECGs may span multiple clusters.")
    code("""
print(f'Selected-region macro AUROC: {summary.loc[names["window_selected_keys"], "macro AUROC"]:.6f}')
print(f'Whole-record macro AUROC: {summary.loc[names["whole"], "macro AUROC"]:.6f}')
print(f'Difference: {primary["difference"]:+.6f}; '
      f'95% interval [{primary["ci_low"]:+.6f}, {primary["ci_high"]:+.6f}]')
print('All frozen advancement gates passed:', result['decision']['advance'])
print('SAM executed:', result['sam_executed'])
print('Previously unsupported diseases evaluated:', result['unseen_disease_evaluated'])
""")
    markdown(r"""
## Reproduction and AI assistance

The full scientific run is `uv run --no-sync python -m scripts.experiments.run_condition_keys_v2`.
It requires the local, hash-bound caches and refuses to overwrite its output directory. The
notebook builder is `uv run --no-sync python -m scripts.reports.build_condition_keys_notebook`.
The notebook replays the completed outputs; it does not repeat encoder extraction or training.
See `docs/condition-keys-protocol-2026-11-03.md` and
`docs/condition-keys-results-2026-11-03.md` for the frozen design and results report.

OpenAI Codex assisted in developing the protocol, Python implementation, analysis and explanatory
text. Numerical results were computed locally from the project's existing artifacts. No external
model inference was used on ECG data in KEY-01. Private per-record arrays remain local; the
notebook displays aggregate tables, curves and density grids. The project authors remain
responsible for reviewing the interpretation and its clinical scope.
""")
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
    destination = ROOT / "notebooks/ConditionKeys.ipynb"
    client = NotebookClient(notebook, timeout=300, resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(notebook, destination)
    print(f"Executed {len(cells)} cells: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
