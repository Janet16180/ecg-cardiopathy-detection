"""Execute aggregate and private-waveform notebooks for the reference audit."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).absolute().parents[2]


def execute(cells: list, path: Path) -> None:
    """Execute cells sequentially and save the notebook at its authorized local destination."""
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    NotebookClient(notebook, timeout=300, resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(notebook, path)


def main() -> None:
    """Create an aggregate notebook and a separate local waveform-review notebook."""
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md('''# Which ECGs create our condition keys?

KEY-03 · Training-reference audit · 6 November 2026

This notebook examines the ECGs actually averaged in KEY-01 and KEY-02. It covers reference
composition, annotation overlap, repeated patients, source effects, embedding geometry and region
selection. The companion local waveform notebook and PDF contain twenty prespecified review
panels. Individual waveform panels and record mappings remain in ignored local outputs.

A key averages **standardized, unit-normalized embedding vectors**, not raw ECG voltages.
Each reference record has equal weight. A nearest reference is an example selected by the model's
geometry, not a clinician-approved typical case. These groups are overlapping annotation proxies.
'''), code('''import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT = Path.cwd()
if not (ROOT / 'ecg_experiment').exists():
    ROOT = ROOT.parent
r = json.loads((ROOT / 'reports/reference-ecg-audit-2026-11-06/aggregate.json').read_text())
assert r['status'] == 'completed'
groups = ['MI', 'STTC', 'CD', 'HYP']
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
print('Study:', r['study'], '| Source:', r['git_commit'])
print('Saved predecessor metric dictionaries reproduced:', r['metric_dictionaries_reproduced'])
print('New evaluation scores:', r['new_evaluation_scores'])
'''), md('''## 1. Population and what is known about its labels

All 25,000 reference records are from permitted training roles. SPH contributes no references.
Source-level physician-reading processes and individual human-validation fields are different
kinds of evidence. A missing per-record validation field does not establish that a label came
from software. The counts below preserve that distinction.

Patient identity is known only where the source supplies it. Unknown identities are not converted
into a claimed number of unique people. Exact signal-hash duplicates are a narrow duplication
check: absence of equal hashes cannot exclude near-duplicates or repeated patients.
'''), code('''overall = r['overall']
display(pd.Series({k: v for k, v in overall.items() if not isinstance(v, dict)}, name='count').to_frame())
display(pd.Series(overall['label_evidence'], name='records').to_frame())
validation = pd.Series(overall['individual_human_validation'], name='records')
display(validation.rename(index={'': 'unavailable'}).to_frame())
review = pd.Series(overall['review_flags'], name='records')
display(review.rename(index={'': 'no recorded review flag'}).to_frame())
'''), md('''## 2. Which sources contribute to each average?

The four positive pools overlap. Normal references are the shared negative pool under the frozen
binary definition; they are not newly certified clinically normal. The number of records from each
family is its literal contribution to the arithmetic average before normalization. The direction
of its contribution also depends on its vectors, examined later.
'''), code('''
composition = pd.DataFrame({g: v['families'] for g, v in r['composition'].items()}).fillna(0).astype(int).T
composition['total'] = composition.sum(axis=1)
display(composition)
shares = composition.drop(columns='total').div(composition['total'], axis=0)
shares.plot.bar(stacked=True, figsize=(10, 4), colormap='tab10')
plt.ylabel('Fraction of reference ECGs')
plt.title('Source-family composition of each reference pool')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
identity = pd.DataFrame({g: {k: v[k] for k in ['records', 'known_patients',
    'known_identity_records', 'unknown_identity_records', 'known_patient_extra_record_count']}
    for g, v in r['composition'].items()}).T
display(identity)
'''), md('''## 3. Averaging mixed annotations

A vector can contribute to several condition keys. The diagonal below is each positive reference
count; off-diagonal entries count shared reference ECGs. Row-normalized values answer: among
references for this row's group, what fraction also carries the column's group? This is annotation
co-occurrence in the support cohort, not population disease prevalence or causation.
'''), code('''co = np.asarray(r['cooccurrence'])
assert np.array_equal(co, co.T)
display(pd.DataFrame(co, index=groups, columns=groups))
display(pd.DataFrame({'positive references': np.diag(co),
    'exclusive group references': [r['exclusive_counts'][g] for g in groups]}, index=groups))
conditional = co / np.diag(co)[:, None]
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(conditional, vmin=0, vmax=1, cmap='Blues')
ax.set(xticks=range(4), yticks=range(4), xticklabels=groups, yticklabels=groups)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{conditional[i,j]:.0%}', ha='center', va='center',
                color='white' if conditional[i,j] > .6 else 'black')
fig.colorbar(im, ax=ax, label='Fraction of row-group references')
plt.tight_layout()
plt.show()
'''), md('''## 4. Reconstruction and compactness of the averages

Every positive and class-specific normal key is reconstructed directly from the original vectors
and saved window selections. Maximum coordinate error must be at most 1e-12, allowing only
summation-order differences. Selected references are not reselected in this audit.

Resultant length is the norm of the mean of unit vectors: values near one indicate stronger
alignment, while small values indicate cancellation. It describes support geometry; it is not
classification accuracy. Selecting regions to maximize training margins can create compact
reference directions without improving external ranking.
'''), code('''print('Key reconstruction errors:', {a: v['key_max_error'] for a, v in r['geometry'].items()})
geometry = pd.DataFrame([{'arm': a, 'key': g, **v}
    for a, details in r['geometry'].items() for g, v in details['groups'].items()])
coherence = geometry.pivot(index='key', columns='arm', values='resultant_length')
display(coherence.style.format('{:.4f}'))
coherence.T.plot(figsize=(11, 4), marker='o', markersize=3)
plt.ylabel('Norm of mean unit vector')
plt.title('Reference compactness across whole-record and selected-region keys')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()
'''), md('''## 5. How far do region keys rotate?

The angle below compares each key to its corresponding whole-record key. Normal-region keys
are class-specific after selection even though they use the same normal records. Consequently,
positive-key changes alone cannot explain a positive-minus-normal similarity score.
'''), code('''rotation = geometry.pivot(index='key', columns='arm', values='whole_key_rotation_degrees')
display(rotation.style.format('{:.2f}'))
rotation.T.plot(figsize=(11, 4), marker='o', markersize=3)
plt.ylabel('Angle from whole-record key (degrees)')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()
'''), md('''## 6. Source removal and known-patient weighting

Source-removed rotation is a geometric sensitivity check: subtract that family's vector sum and
measure the remaining key's angle from the original. It is not a retrained classifier or a hospital
transfer score. Large rotation can reflect both sample share and different recording/label patterns.

The equal-patient diagnostic gives equal weight to each known patient's mean representation;
unknown-patient records retain record weights. It cannot correct repeated patients that are not
identified by the data. No alternative weighting is promoted or evaluated on SPH.
'''), code('''removed = pd.DataFrame([{'arm': a, 'key': g, **v['source_removed_rotation_degrees']}
    for a, details in r['geometry'].items() for g, v in details['groups'].items()]).set_index(['arm', 'key'])
display(removed.loc[['whole', 'short', '5s', '7.5s']].style.format('{:.2f}'))
patient_rotation = geometry.pivot(index='key', columns='arm', values='equal_patient_rotation_degrees')
display(patient_rotation.style.format('{:.3f}'))
exclusive = geometry.pivot(index='key', columns='arm', values='exclusive_vs_cooccurring_angle_degrees')
display(exclusive.dropna(how='all').style.format('{:.2f}', na_rep='not applicable'))
'''), md('''## 7. Which ECGs are closest to a key?

The fifty highest reference-to-key cosines have the largest projections onto that key direction.
They still had the same averaging weight as other reference ECGs. Their source distribution can
be much more concentrated than the full support pool. This is why showing a handful of nearest
ECGs alone is insufficient to justify the average as representative.
'''), code('''nearest = pd.DataFrame([{'arm': a, 'key': g, **v['top50_sources']}
    for a, details in r['geometry'].items() for g, v in details['groups'].items()]).fillna(0)
display(nearest.loc[nearest['arm'].isin(['whole', 'short', '5s', '7.5s'])].set_index(['arm', 'key']))
quantiles = pd.DataFrame([{'arm': a, 'key': g,
    **dict(zip(['min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'], v['cosine_quantiles']))}
    for a, details in r['geometry'].items() for g, v in details['groups'].items()])
display(quantiles.set_index(['arm', 'key']).round(3))
'''), md('''## 8. Where do the selected reference sections occur?

These are nominal token starts and lengths, with four tokens per second. Each token is contextual:
a selected span is not an independently encoded crop and does not locate disease. Positive
references maximize the initial margin; normal references minimize it. The two selection rules
can therefore emphasize different time positions even when there is no clinical regional truth.
'''), code('''fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
for row, kind in enumerate(['positive', 'normal']):
    for col, group in enumerate(groups):
        ax = axes[row, col]
        for arm in ['short', '5s', '7.5s']:
            v = r['geometry'][arm]['groups'][kind + '_' + group]
            ax.plot(np.arange(40)/4, np.asarray(v['start_counts'])/v['records'], label=arm)
        ax.set(title=kind + ' ' + group, xlabel='Nominal start seconds', ylabel='Fraction of references')
axes[0, -1].legend()
plt.tight_layout()
plt.show()
source_starts = pd.DataFrame([{'arm': a, 'key': g, **v.get('source_mean_start_seconds', {})}
    for a, details in r['geometry'].items() if a != 'whole'
    for g, v in details['groups'].items()]).set_index(['arm', 'key'])
display(source_starts.loc[['short', '5s', '7.5s']].round(3))
'''), md('''## 9. Inspecting actual training waveforms

Four roles per group and the normal pool were fixed before selection: nearest to the whole key
within each of the two largest source families; least similar to the whole key; largest vector
change after short-section selection. Repeated ECGs are loaded once, but all twenty review roles
are retained. The normal change example uses the MI-specific normal selection.

Each lead uses its own vertical axis limits; compare the numeric mV ticks rather than apparent
trace heights across leads. These are scientific plots, not a calibrated ECG-paper layout.

Each waveform is checked against raw-file and canonical-signal hashes by the existing training
loader. The private atlas plots all twelve leads in mV with nominal short, five- and 7.5-second
selection bars. Labels are existing annotations, not a new reading. Finite/flat-lead observations
are descriptive quality checks and cannot certify clinical validity.

Local artifacts:
- `outputs/reference_ecg_audit_20261106/ReferenceECGAtlas.pdf`
- `outputs/reference_ecg_audit_20261106/ReferenceWaveforms.ipynb`

Individual waveform images and original record mappings are deliberately kept out of the aggregate
Git notebook. These can be reviewed locally with the cardiologist without changing the frozen labels.
'''), code('''display(pd.Series(r['waveform_review'], name='executed waveform review').to_frame())
checks = 0
for group, values in r['composition'].items():
    assert sum(values['sources'].values()) == values['records']
    assert sum(values['families'].values()) == values['records']
    assert values['known_identity_records'] + values['unknown_identity_records'] == values['records']
    checks += 3
for arm, details in r['geometry'].items():
    assert details['key_max_error'] <= 1e-12
    checks += 1
    for key, values in details['groups'].items():
        assert sum(values['top50_sources'].values()) == 50
        checks += 1
        if 'start_counts' in values:
            assert sum(values['start_counts']) == values['records']
            assert sum(values['length_counts'].values()) == values['records']
            checks += 2
assert np.array_equal(co, co.T)
checks += 1
print('Aggregate consistency checks passed:', checks)
print('Clinical adjudication performed:', r['clinical_review_performed'])
'''), md('''## Observed patterns in these references

The MI support pool is 79.7% PTB-XL, and 45 of its 50 nearest whole-key references are PTB-XL.
The STTC pool is 63.7% Chapman/Ningbo, with 47 of its 50 nearest references from that family.
Removing these families rotates the respective whole keys by about 46 and 61 degrees.
These geometry shifts combine sample-share, acquisition and annotation differences; they do not
isolate a hospital artifact or prove that source balancing would improve prediction.

Only 812 of 2,041 MI references and 593 of 2,376 HYP references carry their group exclusively.
Thus about 60% of MI and 75% of HYP references also contribute other abnormal annotations.
The keys summarize mixtures of findings, not isolated examples of one disease.

Short selection increases positive-reference coherence but rotates their keys by roughly 39–48
degrees. Five- and 7.5-second keys rotate less. Together with KEY-01/02's measured performance,
this shows that a more tightly aligned training reference bank can transfer worse. The cause is
not established by these descriptive comparisons.

Among identified PTB patients, record repetition accounts for 469 extra records across the full
cohort. Equal-known-patient weighting rotates whole keys by about 0.5–2.1 degrees. That limited
check is much smaller than dominant-source removal, but 18,172 records lack patient identity.
It cannot rule out repeated-patient effects in those sources.

All twenty sampled waveforms were finite and had no flat lead. That does not establish freedom
from noise or drift. In the local atlas, Reference 16 shows a pronounced slow V4 level shift and
its short selected window covers the initial half-second. Its existing review-flag field is empty.
This visual observation is post hoc, warrants quality review, and does not prove why its key moved.
It is not a new diagnosis or an automatically applied exclusion.
'''), md('''## Interpretation and reproduction

This is a descriptive audit of fixed reference averages. Geometry shifts and source concentration
can motivate follow-up hypotheses but do not establish why a particular prediction was wrong.
No reference is removed, relabeled or reweighted in the deployed classifier. Recommendations and
observed findings are recorded in the separate results report.

Run the frozen audit with `python -m scripts.experiments.run_reference_ecg_audit` in a fresh output
directory with original local caches and permitted raw training data. Rebuild the notebooks with
`python -m scripts.reports.build_reference_audit_notebooks`. The prior KEY-01/02 conclusions remain.

OpenAI Codex assisted with implementation, analysis and prose. Computations and waveform rendering
were local; no external ECG inference API was used. The waveform panels have not been clinically
adjudicated, and project authors are responsible for reviewing their interpretation.
''')]
    execute(cells, ROOT / 'notebooks/ReferenceECGAudit.ipynb')
    private = [md('''# Local reference waveform review

Twenty prespecified training-only review panels. These are anonymous examples of reference
selection, not clinical diagnoses or validated localization. Existing labels may coexist.
Each lead has independent vertical limits: read its numeric mV ticks rather than comparing
trace heights. This is a scientific display, not standard calibrated ECG paper.
Normal-pool selection-change examples use the MI-specific normal key. Original IDs remain in the
local manifest. Bars show contextual token indices. Review the aggregate notebook and results
report alongside these panels. Do not upload this notebook or atlas as aggregate code artifacts.
'''), code('''import json
from pathlib import Path
from IPython.display import Image, display, Markdown
ROOT = Path.cwd()
if not (ROOT / 'ecg_experiment').exists():
    ROOT = ROOT.parent
folder = ROOT / 'outputs/reference_ecg_audit_20261106'
manifest = json.loads((folder / 'waveform_manifest.json').read_text())
for row in manifest:
    display(Markdown(f"### {row['alias']} — {row['group']}: {row['role']}"))
    display(Markdown(f"Recorded review flags: {row['review_flags'] or 'none'}. "
                     f"Maximum absolute amplitude: {row['max_abs_mv']:.3f} mV. "
                     f"Flat leads: {row['flat_leads']}."))
    display(Image(filename=str(folder / row['file'])))
''')]
    execute(private, ROOT / 'outputs/reference_ecg_audit_20261106/ReferenceWaveforms.ipynb')
    print(f'Executed aggregate notebook ({len(cells)} cells) and local waveform notebook')


if __name__ == '__main__':
    main()
