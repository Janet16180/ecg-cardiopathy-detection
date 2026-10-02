"""Build a portable, real-recording explanation of existing localization results."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat

from ecg_experiment.paths import to_stored

SETUP = """import base64
import gzip
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

plt.rcParams.update({"font.size": 10, "figure.dpi": 110})
"""

PLOTTING = '''def plot_record(case, marks=None, limits=None, title=None):
    """Show original recorded voltages and separately stored highlights."""
    signal = np.asarray(case["signal"], dtype=float)
    time = np.asarray(case["time"], dtype=float)
    count = len(case["leads"])
    columns = 2 if count > 2 else 1
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, max(3.5, rows * 1.45)),
                             squeeze=False, sharex=True)
    shown = case.get("marks", []) if marks is None else marks
    for index, axis in enumerate(axes.flat):
        if index >= count:
            axis.set_visible(False)
            continue
        axis.plot(time, signal[index], color="#222222", linewidth=0.65)
        axis.set_ylabel(case["leads"][index] + " / " + case.get("units", "mV"))
        axis.grid(alpha=0.2)
        for mark in shown:
            if mark.get("lead") in (None, -1, index):
                axis.axvspan(mark["start"], mark["end"], color=mark["color"],
                             alpha=0.23, label=mark["label"])
        if limits is not None:
            axis.set_xlim(*limits)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), fontsize=7, loc="upper right")
        axis.set_xlabel("Time in the original recording (seconds)")
    fig.suptitle(title or case["title"], fontsize=13)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_zoom(case):
    """Magnify recorded samples without adding or changing localization marks."""
    marks = case.get("marks", [])
    marked = [m['lead'] for m in marks if m.get('lead') is not None]
    leads = list(dict.fromkeys(marked))[:2]
    if not leads:
        leads = [case['leads'].index(name) for name in ['II', 'V5'] if name in case['leads']]
    center = (marks[0]['start'] + marks[0]['end']) / 2 if marks else case['time'][0] + 2
    start = max(case['time'][0], min(center - 1, case['time'][-1] - 2))
    end = start + 2
    signal = np.asarray(case['signal'])
    time = np.asarray(case['time'])
    selected = (time >= start) & (time <= end)
    fig, axes = plt.subplots(len(leads), 1, figsize=(14, 2.2 * len(leads)), squeeze=False)
    for lead, axis in zip(leads, axes.flat):
        axis.plot(time[selected], signal[lead, selected], color='#222222', linewidth=0.9)
        for mark in marks:
            if mark.get('lead') in (None, lead):
                axis.axvspan(mark['start'], mark['end'], color=mark['color'], alpha=0.23,
                             label=mark['label'])
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
        axis.set_xlim(start, end)
        axis.set_ylabel(case['leads'][lead] + ' / ' + case.get('units', 'mV'))
        axis.set_xlabel('Time in the original recording (seconds)')
        axis.grid(alpha=0.2)
    fig.suptitle('Closer view of the same real samples — ' + case['title'])
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def describe_record(case):
    """Display the saved labels, decisions, and interpretation limits."""
    notes = case.get("annotation_notes", "")
    if isinstance(notes, list):
        notes = "\\n\\n".join(notes)
    lines = [case.get("plain_language", "")]
    if case.get("codes"):
        lines.append("Recorded annotation codes: `" + str(case["codes"]) + "`.")
    if case.get("annotations"):
        labels = ["| Code | Recorded ECG annotation |", "|---|---|"]
        labels.extend("| " + row["code"] + " | " + row["description"] + " |"
                      for row in case["annotations"])
        lines.append("\\n".join(labels))
    if case.get("classifier"):
        result = case["classifier"]
        referred = result.get("referred", result.get("combined_v4_referred"))
        if isinstance(referred, bool):
            decision = "follow-up flagged" if referred else "follow-up not flagged"
            lines.append("Reconstructed saved screening decision: **" + decision + "**.")
        lines.append("<details><summary>Saved score and rule details</summary>\\n\\n```json\\n"
                     + json.dumps(result, indent=2) + "\\n```\\n</details>")
        if not case.get("marks"):
            lines.append("No region is colored in this saved display: its referral gate is off.")
        else:
            lines.append("Highlighted leads: " + ", ".join(
                sorted({case['leads'][m['lead']] for m in case['marks']})) + ".")
    if case.get("method_results"):
        for result in case["method_results"]:
            role = result.get("role", result.get("method", "Method")).capitalize()
            if "selected_beat_type" in result:
                beat_type = result["selected_beat_type"]
                names = {"V": "ventricular ectopic beat", "N": "beat annotated as normal"}
                lines.append(role + " selected: **" + names.get(beat_type, beat_type) + "**.")
            if result.get("iou") is not None:
                lines.append(role + " QRS overlap with the expert interval: **"
                             + f"{100 * result['iou']:.1f}%**.")
        lines.append("<details><summary>Saved comparison details</summary>\\n\\n```json\\n"
                     + json.dumps(case["method_results"], indent=2) + "\\n```\\n</details>")
    details = [notes, case.get("selection_reason", case.get("selection", "")),
               "Source: " + str(case["source"]) + "."]
    lines.append("<details><summary>Annotation and source notes</summary>\\n\\n"
                 + "\\n\\n".join(str(line) for line in details if line) + "\\n\\n</details>")
    display(Markdown("\\n\\n".join(str(line) for line in lines if line)))
'''


def _markdown(text: str) -> Any:
    """Create a Markdown cell."""
    return nbformat.v4.new_markdown_cell(text)


def _code(text: str) -> Any:
    """Create an executable cell."""
    return nbformat.v4.new_code_cell(text)


def _metrics(root: Path) -> dict[str, Any]:
    """Read previously completed experiments without running an evaluation."""
    paths = {
        "beat": root / "outputs/experiment056_incart_localization_v1/result.json",
        "boundary": root / "outputs/experiment059_qtdb_hybrid_v2/result.json",
    }
    receipts = {key: json.loads(path.read_text()) for key, path in paths.items()}
    return {
        "beat": receipts["beat"]["benchmark"],
        "boundary": receipts["boundary"]["primary_non_edb"],
        "receipts": {
            key: {"path": to_stored(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in paths.items()
        },
    }


def _findings(metrics: dict[str, Any]) -> str:
    """Describe executed aggregate results with their correct evaluation units."""
    beat = metrics["beat"]["methods"]
    paired = metrics["beat"]["paired"]["mixed_core_excess"]
    wave = metrics["boundary"]["waves"]["QRS"]
    rows = [
        (
            "Top-scoring beat is ventricular ectopic",
            beat["U_B_fixed"]["mixed_core_hit"],
            beat["aligned_residual"]["mixed_core_hit"],
        ),
        (
            "Normal beat incorrectly flagged",
            beat["U_B_fixed"]["normal_fpr"],
            beat["aligned_residual"]["normal_fpr"],
        ),
        ("QRS interval overlap", wave["fixed"]["iou"], wave["hybrid"]["iou"]),
        ("QRS endpoint within 30 milliseconds", wave["fixed"]["within_30ms"], wave["hybrid"]["within_30ms"]),
    ]
    table = "\n".join(f"| {name} | {100 * old:.1f}% | {100 * new:.1f}% |" for name, old, new in rows)
    return f"""### What the completed experiments found

The pictures above explain individual examples. The following numbers come from the full,
previously completed real-data evaluations, not from this small illustrated selection.

| Measurement | Earlier method | Candidate method |
|---|---:|---:|
{table}

The beat-selection result averages patients' success in mixed 10-second INCART segments:
30 eligible patients. The false-alarm result for beats annotated as normal uses 31 eligible patients at the
experiments' saved thresholds; lower is better. QRS overlap means intersection divided by union,
averaged across 72 QTDB records. Endpoint accuracy counts the start and end separately; it is
not the percentage of fully correct beats.

The estimated gain in selecting the ventricular ectopic beat was {100 * paired["value"]:.1f}
percentage points, with a patient-bootstrap 95% interval of
{100 * paired["ci_low"]:.1f} to {100 * paired["ci_high"]:.1f} points.
For QRS overlap, the record-bootstrap gain was
{100 * metrics["boundary"]["primary_gain"]["estimate"]:.1f} points, with a 95% interval of
{100 * metrics["boundary"]["primary_gain"]["ci_low"]:.1f} to
{100 * metrics["boundary"]["primary_gain"]["ci_high"]:.1f} points.
QTDB does not establish independent patient identities for these 72 records.

### Findings and conclusions

1. Comparing each beat with other beats from the same recording helped identify ventricular
   ectopic beats. It still flagged about {100 * beat["aligned_residual"]["normal_fpr"]:.0f}%
   of beats annotated as normal in that evaluation, so a colored region is a reason to inspect the ECG.
2. Following the waveform instead of drawing a fixed-width box improved QRS timing against
   expert annotations. The experiment evaluated 2,582 complete QRS intervals in the 72-record
   primary comparison. This supports better physiological boundaries.
3. A disease code describes the recording. An unusual beat score ranks differences.
   A QRS boundary identifies timing. These three kinds of evidence answer different questions.
   None of them alone identifies the anatomical cause of a heart problem.
4. The selected conduction, ST/T, infarction-pattern, and high-voltage examples show how different
   labeled ECGs look. Their colored marks are existing screening explanations; they are not
   cardiologist-confirmed disease locations. Missing marks and misleading marks remain possible.
5. The new boundary candidate preserves the existing P-wave boundaries. T-wave localization
   has not shown a reliable general improvement. The first-stage classifier was not retrained
   for this notebook, and these candidate localization methods are separate from its saved decisions.

All waveforms here are original patient recordings. No synthetic ECGs, artificial lesions,
augmentation, training, or new accuracy evaluation were used to make this notebook.
"""


def make_notebook(payload: dict[str, Any], include_data: bool = True) -> Any:
    """Create explanatory cells with optional portable waveform storage."""
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    cells = [
        _markdown("""# Understanding ECG localization with real recordings

This notebook explains the existing research in plain language. It includes ECGs labeled normal
and ECGs with different recorded abnormalities, then shows two improvements tested against expert
annotations. A normal ECG does not prove that a person has no heart disease.

The portable local version contains its selected real waveforms, saved model outputs, and source
notes. **Restart the kernel and run all cells:** it needs Python, NumPy, Matplotlib, and IPython,
but no repository checkout, model weights, GPU, internet connection, or original data folders.
The HTML export can be read without running Python.

This is a replay and explanation of completed research, not a diagnostic tool. The examples were
chosen for illustration from already-read development data. They do not form an independent test.
"""),
        _markdown("""## 1. Reading the picture

An ECG records voltage over time. Each lead is a different electrical view of the same heartbeat,
like cameras looking at one event from different angles. These views are not pictures of twelve
separate parts of the heart.

The small P wave reflects electrical activation of the upper chambers. The sharper QRS complex
reflects activation of the lower chambers. The ST segment and T wave describe part of their
electrical recovery. The horizontal axis is time; the vertical axis is recorded voltage.

A wider spike, a premature beat, or a changed recovery shape may deserve attention. Appearance
depends on the lead, and a pattern needs clinical context. Here we retain the database's original
labels instead of inventing diagnoses from the pictures.

Each plot adapts its vertical axis to make the recorded shape readable. Compare the voltage
numbers, rather than the drawn height, across leads and examples. The closer views magnify
the original samples; they do not modify the ECG or add new algorithm predictions.
"""),
        _code(SETUP),
    ]
    if include_data:
        cells.append(
            _code(
                f'packed_real_recordings = "{encoded}"\n'
                "decoded = gzip.decompress(base64.b64decode(packed_real_recordings))\n"
                f'if hashlib.sha256(decoded).hexdigest() != "{digest}":\n'
                '    raise ValueError("The embedded real-recording bundle changed.")\n'
                "bundle = json.loads(decoded)\n"
                "print(f\"Loaded {len(bundle['cases'])} real recording examples; no external files needed.\")"
            )
        )
    else:
        cells.append(
            _code(
                'raise RuntimeError("This Git copy omits patient waveforms. Generate the portable local "\n'
                '                   "version with: python -m scripts.reports.build_real_ecg_notebook")'
            )
        )
    cells.extend(
        [
            _code(PLOTTING),
            _markdown("""## 2. From screening to a highlight

The screening model asks: **does this ECG need follow-up under the saved referral rule?** Its
ranking score is not a calibrated probability that the person has heart disease. The project
sets its operating point with a referral budget fitted to normal ECGs. These saved development
examples used PTB-XL normal records, not recordings from the university's students.

The explanatory map asks: **which recorded region attracted attention or differs from a reference?**
The existing map uses learned model attention for most records and a beat comparison when its
PVC finding score crosses the saved switch threshold. Saved token contributions show which
regions contributed to that attention head's score;
a difference score shows unusual shape. Neither supplies an expert disease-location annotation.

The following 12-lead examples replay saved results. Colored areas identify the saved method and
lead. Some maps are displayed only when the saved referral rule is positive, so an empty map
can also mean that its referral gate was off. It does not rule out the record's annotated
abnormality. A referral and a localized mark can disagree.
Red marks use the saved normal-beat comparison; blue marks use saved attention contributions.
Amber marks show a forced top candidate whose score is below the usual display cutoff.
"""),
        ]
    )
    for index, case in enumerate(payload["cases"]):
        if case.get("kind") != "ptb":
            continue
        cells.extend(
            [
                _markdown(f"### {case['title']}"),
                _code(
                    f"case = bundle['cases'][{index}]\ndescribe_record(case)\n"
                    "plot_record(case)\nplot_zoom(case)"
                ),
            ]
        )
    cells.append(
        _markdown("""## 3. Finding an unusual beat in a real recording

The candidate finds heartbeats from the waveform, lines them up, and compares each one with
the typical shape of the other beats in the same recording. Each beat is left out of its own
reference. It looks for the largest local difference, and associates that region with its
nearest detected beat. This can identify a premature ventricular beat even when the difference
is in its recovery wave rather than exactly at its sharp spike.

The expert beat labels are used afterward to check the answer. They are not given to the
beat-scoring algorithm. Green bands in these examples orient us around expert-labeled ventricular
beats; they are **not expert QRS onset/offset boundaries**. The earlier and candidate marks show
their selected regions. One example succeeds and one illustrates a remaining failure.
""")
    )
    for index, case in enumerate(payload["cases"]):
        if case.get("kind") != "incart":
            continue
        cells.extend(
            [
                _markdown(f"### {case['title']}"),
                _code(f"case = bundle['cases'][{index}]\ndescribe_record(case)\nplot_record(case)"),
            ]
        )
    cells.append(
        _markdown("""## 4. Drawing a more accurate QRS box

The earlier method places a fixed interval around each detected R peak. That is easy to compute,
but the same box can be too short for one beat and too long for another.

The candidate follows local changes in the signal to find the QRS start and end, and combines
the timing information from the available leads. Green is the expert's joint QRS interval;
orange is the earlier fixed box; blue is the candidate box. The expert timing is shared across
the two leads, so these pictures cannot prove that one lead contains the disease.

The first example uses the first complete annotated QRS in a prespecified record. The second
deliberately shows a bad interval from the record with the largest average QRS regression.
Individual pictures explain behavior; the aggregate evaluation below measures improvement.
""")
    )
    for index, case in enumerate(payload["cases"]):
        if case.get("kind") != "qtdb":
            continue
        cells.extend(
            [
                _markdown(f"### {case['title']}"),
                _code(
                    f"case = bundle['cases'][{index}]\ndescribe_record(case)\n"
                    "truth = [m for m in case['marks'] if m.get('method') == 'expert']\n"
                    "for method in ['fixed', 'hybrid']:\n"
                    "    selected = truth + [m for m in case['marks'] if m.get('method') == method]\n"
                    "    plot_record(case, selected, title=case['title'] + ' — ' + method)"
                ),
            ]
        )
    cells.extend(
        [
            _markdown(_findings(payload["metrics"])),
            _markdown("""## Sources and reproducibility

Real examples come from [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/),
[INCART](https://physionet.org/content/incartdb/1.0.0/), and
[QT Database](https://physionet.org/content/qtdb/1.0.0/). The source versions, selected record IDs,
sampling rates, annotation notes, and saved-result receipt hashes are stored in this notebook.
The project reports for Experiments 056 and 059 describe the full protocols and comparison rules.

The embedded bundle stores original waveform samples and saved explanation results. Rerunning
this notebook redraws and explains those results; it does not run the neural networks or repeat
the full benchmarks. No protected final-test cohort is accessed.
"""),
            _code(
                "print(json.dumps(bundle['metrics']['receipts'], indent=2))\n"
                "print('Example IDs:', ', '.join(str(c['id']) for c in bundle['cases']))"
            ),
        ]
    )
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    notebook.metadata.real_ecg_bundle_sha256 = digest
    notebook.metadata.contains_patient_waveforms = include_data
    if include_data:
        notebook.cells[3].metadata.jupyter = {"source_hidden": True}
    else:
        notebook.cells.insert(
            0,
            _markdown("""**Repository blueprint:** this Git copy contains explanations
and plotting code, but no patient waveform payload or figure outputs. Generate the self-contained
local notebook and offline HTML with:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.reports.build_real_ecg_notebook
```

Use the resulting `notebooks/14-jr-real-ecg-localization.executed.ipynb`
for the portable, runnable version. Raw recordings remain local according to the project's data policy.
"""),
        )
    return notebook


def build_notebooks(root: Path, destination: Path, template: Path | None = None) -> Path:
    """Build a local standalone notebook and optionally a data-free Git template."""
    from ecg_experiment.real_notebook_cases import load_real_evidence_cases
    from ecg_experiment.real_notebook_ptb import load_real_ptb_cases

    cases = load_real_ptb_cases(root) + load_real_evidence_cases(root)
    payload = {"cases": cases, "metrics": _metrics(root)}
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "14-jr-real-ecg-localization.executed.ipynb"
    nbformat.write(make_notebook(payload), path)
    if template is not None:
        template.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(make_notebook(payload, include_data=False), template)
    return path
