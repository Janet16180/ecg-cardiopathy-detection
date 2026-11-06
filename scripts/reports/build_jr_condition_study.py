"""Consolidate all condition-key results and execute the JR classifier comparison notebook."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from ecg_experiment.files import sha256_file

ROOT = Path(__file__).absolute().parents[2]
DESTINATION = ROOT / "notebooks/11-jr-ecg-condition-keys-and-classification.ipynb"


def previous_cells(filename: str, anchor: str, title: str) -> list:
    """
    Copy historical notebook sources into an explicitly labeled consolidated section.

    Parameters
    ----------
    filename : str
        Preserved aggregate-only notebook name.
    anchor, title : str
        Consolidated navigation anchor and section title.

    Returns
    -------
    list
        Fresh cells without historical outputs or writes to old verification receipts.
    """
    original = nbformat.read(ROOT / "notebooks" / filename, as_version=4)
    cells = []
    for position, previous in enumerate(original.cells):
        cell = copy.deepcopy(previous)
        if cell.cell_type == "markdown":
            lines = cell.source.splitlines()
            lines = ["#" + line if line.startswith("#") else line for line in lines]
            if position == 0:
                lines[0] = f'<a id="{anchor}"></a>\n## {title}'
            cell.source = "\n".join(lines)
        elif cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.source = "\n".join(
                line for line in cell.source.splitlines() if not line.strip().startswith("verification_path")
            )
        cells.append(cell)
    return cells


def classification_cells() -> list:
    """
    Explain and analyze the registered multiclass comparison and reference resampling.

    Returns
    -------
    list
        Notebook cells drawing only on completed local results.
    """
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    return [
        md("""<a id="classifiers"></a>
## IV. Matched multiclass and independent-label classification — KEY-04

The previous studies show how reference averages behave. This comparison asks whether a
learned classifier makes better use of the **same normalized xECG features and same training
labels**. No encoder, new data source or clinical definition is added.

An ECG can have several findings. To provide a genuine multiclass method without throwing away
co-occurrence, multinomial logistic regression predicts sixteen existing annotation combinations.
The state code is `MI + 2*STTC + 4*CD + 8*HYP`. Zero is the normal annotation proxy. These are
sixteen combinations of four groups, not sixteen diseases. Each combination is mutually exclusive
as a state, while its four component annotations can overlap.

For comparison with keys, sum each predicted state's probability into the groups it contains.
For example, probability assigned to MI+STTC contributes to both MI and STTC marginal scores.
A second family uses four independent binary logistic regressions. The common primary endpoint
is the four-group macro one-versus-rest AUROC, not exact-state accuracy.
"""),
        code("""from sklearn.metrics import roc_curve, precision_recall_curve

K4 = json.loads((ROOT / 'reports/condition-classifiers-2026-11-06/aggregate.json').read_text())
NC = json.loads((ROOT / 'reports/condition-classifiers-2026-11-06/negative-control.json').read_text())
AUDIT4 = json.loads((ROOT / 'reports/condition-classifiers-2026-11-06/verification.json').read_text())
KP = np.load(ROOT / 'outputs/condition_classifiers_20261106/predictions.npz', allow_pickle=False)
negative_path = ROOT / 'outputs/condition_classifiers_negative_control_20261106/artifacts.npz'
NP = np.load(negative_path, allow_pickle=False)
assert K4['status'] == NC['status'] == 'completed'
assert AUDIT4['status'] == 'passed'
assert np.array_equal(KP['flags'], NP['flags'])
print('KEY-04 source:', K4['git_commit'])
print('Independent arithmetic verification:', AUDIT4['status'])
display(pd.DataFrame(K4['qualification']['folds'])[['fold', 'records', 'groups']])
"""),
        md("""### Training-only model selection

Both families use the same three folds, grouped by known patient identity. Unknown-patient records
retain distinct record groups; this cannot establish patient independence in those sources.
Each fold refits its own StandardScaler on fitting rows, then normalizes vectors. Neither SPH
features nor labels fit a scaler or select regularization.

C is selected separately from 0.1, 1 and 10 using mean fold macro AUROC; ties choose smaller C.
All fits use L2 logistic regression, lbfgs, tolerance 1e-8 and at most 5,000 iterations. Nonconverged
fits stop the study. The table includes every registered candidate, not just the selected one.
If the best C is at the grid boundary, this is a bounded comparison, not a claim of global optimum.
"""),
        code("""cv_table = pd.DataFrame(K4['cross_validation']['selection_table'])
display(cv_table.pivot(index='C', columns='family', values='mean_fold_macro_auroc').style.format('{:.6f}'))
print('Training-selected C:', K4['cross_validation']['chosen_C'])
fold_table = pd.DataFrame([{'family': row['family'], 'C': row['C'], 'fold': row['fold'],
                           'macro AUROC': row['metrics']['macro_auroc'],
                           'fit seconds': row['fit_seconds']}
                          for row in K4['cross_validation']['fold_metrics']])
display(fold_table)
cv_table.pivot(index='C', columns='family', values='mean_fold_macro_auroc').plot(
    marker='o', figsize=(7, 3), logx=True)
plt.ylabel('Mean training-fold macro AUROC')
plt.title('Regularization chosen without SPH scores')
plt.tight_layout()
plt.show()
"""),
        md("""### A control for negative-reference composition

The original positive-minus-normal keys use only normals as negative references. The classifiers
also learn from other abnormal combinations. A secondary control was registered before the new
SPH predictions: retain the same positive keys, but construct each negative key from **all training
ECGs lacking that group**. A co-occurring positive never enters its own group's negative pool.

This control isolates negative-key composition. It does not equalize classifier capacity or turn
prototype scores into calibrated probabilities. Its interval is exploratory, and it cannot replace
the original primary comparison after observing results.
"""),
        code("""comparison = pd.DataFrame({name: {'macro AUROC': values['macro_auroc'],
                                  'macro AP': values['macro_average_precision']}
                           for name, values in K4['metrics'].items()}).T
comparison.loc['all_other_negative_keys'] = [NC['metrics']['macro_auroc'],
                                            NC['metrics']['macro_average_precision']]
display(comparison.style.format('{:.6f}'))
display(pd.DataFrame({'primary joint minus whole': K4['primary'],
                     'exploratory independent minus whole': K4['exploratory_independent_contrast'],
                     'exploratory negative control minus whole': NC['versus_whole']}).T)
print('Registered ranking decision:', K4['decision'])
print('Clinical pipeline promoted:', K4['decision']['clinical_pipeline_promoted'])
"""),
        md("""### Per-group ranking and precision–recall

All evaluated records contribute to each one-versus-rest task, including other abnormalities.
Average precision depends on prevalence; comparing all methods on the identical evaluation rows
keeps that prevalence fixed. Rare MI/HYP groups should not be judged by AUROC alone.
Probabilities come from the source-trained models and have not been clinically calibrated at SPH.
"""),
        code("""model_scores = {name: KP['score_' + name] for name in K4['metrics']}
model_scores['all_other_negative_keys'] = NP['scores']
per_group = pd.DataFrame([{'model': name, 'group': g, **values['groups'][g]}
                          for name, values in {**K4['metrics'],
                                               'all_other_negative_keys': NC['metrics']}.items()
                          for g in ['MI', 'STTC', 'CD', 'HYP']])
display(per_group.pivot(index='group', columns='model', values='auroc').style.format('{:.4f}'))
display(per_group.pivot(index='group', columns='model', values='average_precision').style.format('{:.4f}'))
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for c, g in enumerate(['MI', 'STTC', 'CD', 'HYP']):
    y = KP['flags'][:, c]
    for name, values in model_scores.items():
        fpr, tpr, _ = roc_curve(y, values[:, c])
        precision, recall, _ = precision_recall_curve(y, values[:, c])
        axes[0, c].plot(fpr, tpr, label=name)
        axes[1, c].plot(recall, precision, label=name)
    axes[0, c].plot([0, 1], [0, 1], '--', color='gray', linewidth=.6)
    axes[1, c].axhline(y.mean(), linestyle='--', color='gray', linewidth=.6)
    axes[0, c].set(title=g, xlabel='False positive rate', ylabel='True positive rate')
    axes[1, c].set(xlabel='Recall', ylabel='Precision', ylim=(0, 1.02))
axes[0, -1].legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.show()
"""),
        md("""### Genuine multiclass performance and rare combinations

The joint model chooses one of sixteen annotation states by probability argmax. This is a
classification endpoint, not a calibrated referral rule. Keys do not define joint-state probabilities,
so no artificial multiclass accuracy or log loss is assigned to them.

Exact-state accuracy can be dominated by common combinations. Supported-state macro recall and
the full confusion matrix expose that. Recall is undefined where a state has no evaluation examples;
precision is undefined if no record was predicted into that state. These entries remain missing.
The dummy always chooses the most common training state, never the most common SPH state.
"""),
        code("""multi = K4['multiclass']
state_counts = pd.DataFrame({'code': range(16), 'annotations': multi['state_names'],
                            'training ECGs': K4['qualification']['training_state_counts'],
                            'SPH ECGs': multi['support'], 'predicted ECGs': multi['predicted_count'],
                            'precision': multi['precision'], 'recall': multi['recall'], 'F1': multi['f1']})
display(state_counts)
display(pd.Series({k: multi[k] for k in ['exact_state_accuracy', 'macro_recall_supported_states',
    'top_two_accuracy', 'log_loss', 'training_majority_state', 'training_majority_accuracy']},
    name='multiclass result').to_frame())
matrix = np.asarray(multi['confusion'])
display(pd.DataFrame(matrix, index=pd.Index(range(16), name='true state'),
                     columns=pd.Index(range(16), name='predicted state')))
row_totals = matrix.sum(axis=1)
normalized = np.divide(matrix, row_totals[:, None], out=np.full(matrix.shape, np.nan),
                       where=row_totals[:, None] != 0)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
ax.set(xticks=range(16), yticks=range(16), xlabel='Predicted annotation-state code',
       ylabel='True annotation-state code', title='Row-normalized multiclass confusion')
for i in range(16):
    for j in range(16):
        if matrix[i, j]:
            ax.text(j, i, str(matrix[i,j]), ha='center', va='center', fontsize=7,
                    color='white' if normalized[i,j] > .55 else 'black')
fig.colorbar(im, ax=ax, label='Fraction of true state')
plt.tight_layout()
plt.show()
"""),
        md("""### Error patterns and co-occurring findings

The next table ranks aggregate exact-state errors. It describes annotation-combination mistakes;
it is not a list of missed diseases or clinical referral decisions. The following subset comparison
uses exclusive or co-occurring positives against the same normal negative pool. Its denominators
are different from the primary all-other-negative task, so its numbers cannot replace that result.
"""),
        code("""errors = [{'true annotations': multi['state_names'][i],
           'predicted annotations': multi['state_names'][j], 'ECGs': int(matrix[i,j])}
          for i in range(16) for j in range(16) if i != j and matrix[i,j] > 0]
display(pd.DataFrame(errors).sort_values('ECGs', ascending=False).head(15))
subsets = pd.DataFrame([{'group': group, 'positive subset': subset, 'model': model,
                        'positive ECGs': values['positives'], 'normal ECGs': values['normals'], **point}
                       for group, parts in K4['error_subsets'].items()
                       for subset, values in parts.items()
                       for model, point in values['models'].items()])
display(subsets)
"""),
        md("""### Does the advantage survive different reference samples?

Ten source-family-stratified reference bootstraps were registered in advance. Known patients are
resampled as whole groups; unknown identities remain record groups. Each draw refits its scaler,
prototypes and both classifiers from the same sampled rows, using C selected before SPH scoring.
Every draw is retained. The ranges below are **not confidence intervals** and the draws are not
ten independent hospitals. They assess reference-sampling sensitivity conditional on the fixed
encoder, selected C and the same SPH evaluation set.
"""),
        code("""draw_rows = []
for draw in K4['reference_sensitivity']:
    for model, contrast in draw['differences'].items():
        draw_rows.append({'seed': draw['seed'], 'model': model,
                          'training rows': draw['training_rows'],
                          'unique training groups': draw['unique_training_groups'],
                          'AUROC difference': contrast['macro_auroc'],
                          'AP difference': contrast['macro_average_precision']})
draws = pd.DataFrame(draw_rows)
display(draws)
display(draws.groupby('model')[['AUROC difference', 'AP difference']].agg(['min', 'median', 'max']))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for ax, metric in zip(axes, ['AUROC difference', 'AP difference']):
    for model, part in draws.groupby('model'):
        ax.plot(range(1, len(part)+1), part[metric], marker='o', label=model)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set(xlabel='Registered reference draw', ylabel=metric + ' versus same-draw keys')
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
"""),
        md("""<a id="interpretation"></a>
## V. What this deeper comparison does and does not establish

- All models use the same permitted source pool, but they use its labels differently. The extra
  negative-reference control isolates one such difference; it does not equalize capacity.
- Training-only model selection keeps SPH out of hyperparameter fitting. SPH has still been read
  repeatedly across this research sequence, so these results remain external development evidence.
- Paired patient intervals quantify evaluation sampling conditional on fitted models. The ten
  reference resamples assess another sensitivity, but do not include reselecting C, new encoders,
  unknown repeated patients or independently acquired hospitals.
- High four-group AUROC can coexist with poor exact recognition of rare combinations. Review
  per-group AP, per-state support and confusion before interpreting a single summary.
- A source-removed key rotation combines sample share, acquisition and label differences. It is
  not a causal estimate of hospital/device bias. Likewise, the single cyclic-key control from
  KEY-01 is a diagnostic example, not a permutation significance test.
- The twenty waveform examples were selected for geometry review, not sampled to estimate
  population signal quality. Finite samples and no flat leads do not certify absence of artifacts.
- Our region analyses pool contextual xECG tokens. They do not implement SAM segmentation,
  lead-preserving JEPA/CroPA, physiological localization or a cardiologist's reasoning process.
- The four memberships are annotation proxies; this work has not established diagnosis,
  unseen-condition recognition, student screening performance or a final clinical operating point.

The software and arithmetic audit below checks reproducibility. Passing it does not validate the
clinical target or remove these scientific limitations. Project recommendations remain in the
separate results report.
"""),
        code("""display(pd.Series(AUDIT4, name='saved-output audit').to_frame())
assert AUDIT4['new_fits'] == 0
assert AUDIT4['primary_interval_exact']
assert AUDIT4['known_patient_fold_separation_verified']
assert AUDIT4['bootstrap_group_multiplicities_exact']
print('Registered ranking improvement:', K4['decision']['ranking_improvement'])
print('No clinical pipeline promotion:', not K4['decision']['clinical_pipeline_promoted'])
"""),
        md("""### Reproduction and AI assistance

Scientific execution: `python -m scripts.experiments.run_key_classifiers`, followed by the
registered `run_key_negative_control` and `scripts.reports.audit_key_classifiers`. Original
local cache identities and a fresh output directory are required; scientific runs refuse overwrite.
Build this consolidated notebook with `python -m scripts.reports.build_jr_condition_study`.

The earlier notebooks and their scientific receipts remain intact. This notebook re-executes their
aggregate analysis cells and adds KEY-04; it does not repeat encoder extraction or display private
individual-waveform images. The separate local atlas remains available for review.

OpenAI Codex assisted with the prospective designs, code, plots and explanatory text. Numerical
results were executed locally. The project authors remain responsible for reviewing the methods,
interpretations and clinical scope. No external model inference was used on ECG data here.
"""),
    ]


def main() -> None:
    """Execute one JR-format notebook containing all four studies and the matched comparison."""
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md("""# JR — ECG condition keys, reference audit and classification

**Consolidated executed research notebook · 6 November 2026**

This notebook brings together the complete condition-key experiments, longer-section comparison,
cluster analysis, reference-ECG audit and a matched multiclass classification study. Each section
identifies its prospective design, executed results and limits. The classification comparison adds
training-only grouped model selection, rare-combination error analysis, a negative-reference control
and ten registered reference-sampling refits.

Contents:
1. [All performance results at a glance](#overview)
2. [Whole-record and selected-region keys, with clusters — KEY-01](#key01)
3. [Longer sections and temporal agreement — KEY-02](#key02)
4. [Which ECGs form the averages? — KEY-03](#key03)
5. [Matched multiclass comparison and reference stability — KEY-04](#classifiers)
6. [Interpretation, evidence limits and verification](#interpretation)

All four studies concern existing ECG annotation groups on reused hospital development data.
They do not establish clinical diagnosis or an untouched final test. Individual waveform panels
remain in the local `outputs/reference_ecg_audit_20261106/ReferenceECGAtlas.pdf` and its
`ReferenceWaveforms.ipynb` companion.

<a id="overview"></a>
## All performance results at a glance
"""),
        code("""import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / 'ecg_experiment').exists():
    ROOT = ROOT.parent
K1 = json.loads((ROOT / 'reports/condition-keys-2026-11-03/aggregate.json').read_text())
K2 = json.loads((ROOT / 'reports/long-section-keys-2026-11-06/aggregate.json').read_text())
K4 = json.loads((ROOT / 'reports/condition-classifiers-2026-11-06/aggregate.json').read_text())
NC = json.loads((ROOT / 'reports/condition-classifiers-2026-11-06/negative-control.json').read_text())
all_rows = []
for study, metrics in [('KEY-01', K1['metrics']), ('KEY-02', K2['metrics']), ('KEY-04', K4['metrics'])]:
    for name, value in metrics.items():
        if study == 'KEY-02' and name in ['whole', 'short_selected']:
            continue
        if study == 'KEY-04' and name == 'whole':
            continue
        all_rows.append({'study': study, 'method': name, 'macro AUROC': value['macro_auroc'],
                         'macro AP': value['macro_average_precision']})
all_rows.append({'study': 'KEY-04 control', 'method': 'all_other_negative_keys',
                 'macro AUROC': NC['metrics']['macro_auroc'],
                 'macro AP': NC['metrics']['macro_average_precision']})
all_results = pd.DataFrame(all_rows)
display(all_results.set_index(['study', 'method']).style.format('{:.6f}'))
print('All rows use the same 10,434 SPH evaluation ECGs and four annotation groups.')
print('The shuffled-key row is a diagnostic control, not a candidate classifier.')
print('Primary KEY-04 paired contrast:', K4['primary'])
"""),
    ]
    for filename, anchor, title in (
        ("ConditionKeys.ipynb", "key01", "I. Whole-record keys, selected regions and clusters — KEY-01"),
        ("LongSectionKeys.ipynb", "key02", "II. Longer sections and temporal agreement — KEY-02"),
        ("ReferenceECGAudit.ipynb", "key03", "III. The ECG references behind the averages — KEY-03"),
    ):
        cells.extend(previous_cells(filename, anchor, title))
    cells.extend(classification_cells())
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NotebookClient(notebook, timeout=600, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, DESTINATION)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    receipt = {
        "notebook": str(DESTINATION.relative_to(ROOT)),
        "cells": len(cells),
        "executed_code_cells": len(code_cells),
        "status": "executed_without_cell_errors",
        "notebook_sha256": sha256_file(DESTINATION),
        "builder_sha256": sha256_file(Path(__file__)),
        "historical_notebook_sha256": {
            name: sha256_file(ROOT / "notebooks" / name)
            for name in ("ConditionKeys.ipynb", "LongSectionKeys.ipynb", "ReferenceECGAudit.ipynb")
        },
    }
    path = ROOT / "reports/condition-classifiers-2026-11-06/jr-notebook-verification.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt)


if __name__ == "__main__":
    main()
