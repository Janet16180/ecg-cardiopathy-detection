"""Print the tables of the MLC-02 results report from the Stage 2 and Stage 3 outputs.

Reads only files that the MLC-02 steps wrote: the selection record, the operating points, the
test-prediction receipt, ``evaluation/evaluation.json`` (written by ``run_mlc02_stage3 evaluate``), the
leave-one-family-out and source-identification outputs. It computes no metric: every number is read from
those files, and the only arithmetic is a difference of two stored values, a count or a minimum and
maximum. It also checks the SHA-256 of every file the receipts pin. Run from the repository root:
``PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc02_results_tables [--section <name>]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from ecg_experiment.files import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("outputs/mlc02_condition_models_20261110")
STAGE3 = OUTPUT / "stage3"
CANDIDATES = ("P0", "L-J", "L-C", "G-M", "H-X", "H-J", "H-C", "H-All", "S-R", "F", "E")
SETS = {"test_same_source": "PTB-XL fold 10", "test_external": "SPH evaluation half"}
NAMES = {
    "sinus_tachycardia": "sinus tachycardia",
    "sinus_bradycardia": "sinus bradycardia",
    "sinus_arrhythmia": "sinus arrhythmia",
    "af_or_flutter": "AF or flutter",
    "premature_ventricular_complex": "PVC",
    "first_degree_av_block": "first-degree AV block",
    "complete_rbbb": "complete RBBB",
    "incomplete_rbbb": "incomplete RBBB",
    "lafb": "LAFB",
    "left_axis_deviation": "left axis deviation",
    "lvh": "LVH",
    "st_depression": "ST depression",
    "t_wave_abnormality": "T-wave abnormality",
    "mi_any": "any MI",
}
LOFO_SOURCES = (
    ("ptbxl", "ptbxl", "PTB-XL"),
    ("chapman_ningbo", "ningbo", "Ningbo"),
    ("chapman_ningbo", "chapman_shaoxing", "Chapman"),
    ("georgia", "georgia", "Georgia"),
    ("cpsc", "cpsc_2018", "CPSC 2018"),
    ("cpsc", "cpsc_2018_extra", "CPSC-Extra"),
)


def load(path: Path) -> Any:
    """
    Read a JSON file under the repository root.

    Parameters
    ----------
    path : Path
        Repository-relative path.

    Returns
    -------
    Any
        Parsed content.
    """
    return json.loads((ROOT / path).read_text())


def number(value: float | None, digits: int = 4) -> str:
    """
    Format a stored value, with NaN or missing shown as ``NaN``.

    Parameters
    ----------
    value : float or None
        Value from a JSON file.
    digits : int
        Decimals.

    Returns
    -------
    str
        Formatted value.
    """
    if value is None or value != value:
        return "NaN"
    return f"{value:.{digits}f}"


def interval(value: float | None, low: float | None, high: float | None, digits: int = 4) -> str:
    """
    Format a point estimate with its 95% interval.

    Parameters
    ----------
    value, low, high : float or None
        Point estimate and interval ends.
    digits : int
        Decimals.

    Returns
    -------
    str
        ``value (low, high)``.
    """
    return f"{number(value, digits)} ({number(low, digits)}, {number(high, digits)})"


def table(header: list[str], rows: list[list[str]], right: int = 1) -> list[str]:
    """
    Build a Markdown table.

    Parameters
    ----------
    header : list[str]
        Column names.
    rows : list[list[str]]
        Cells.
    right : int
        Columns from this index on are right-aligned.

    Returns
    -------
    list[str]
        Lines.
    """
    align = ["---" if i < right else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def rows_by_condition(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Index an ``interval_report`` table by condition.

    Parameters
    ----------
    report : dict[str, Any]
        Table with a ``conditions`` list.

    Returns
    -------
    dict[str, dict[str, Any]]
        Condition to row.
    """
    return {row["condition"]: row for row in report["conditions"]}


def provenance(evaluation: dict[str, Any]) -> list[str]:
    """
    Check the SHA-256 of every file the receipts pin and list the commits and commands.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    receipt = load(STAGE3 / "test_predictions_receipt.json")
    select = load(STAGE3 / "receipts/select.json")
    fix = load(STAGE3 / "receipts/fix.json")
    pinned = evaluation["provenance"]
    checks = [
        ("selection record", STAGE3 / "selection_record.json", receipt["selection_record_sha256"]),
        ("operating points", STAGE3 / "operating_points.json", receipt["operating_points_sha256"]),
        (
            "test-prediction receipt",
            STAGE3 / "test_predictions_receipt.json",
            pinned["test_predictions_receipt_sha256"],
        ),
        (
            "selection record document",
            Path(receipt["record_doc"]["doc"]),
            receipt["record_doc"]["doc_sha256"],
        ),
        ("manifest", Path(pinned["manifest"]), pinned["manifest_sha256"]),
    ]
    rows = [
        [name, f"`{path}`", f"`{expected[:12]}`", str(sha256_file(ROOT / path) == expected)]
        for name, path, expected in checks
    ]
    files = [entry for by_role in receipt["files"].values() for entry in by_role.values()]
    equal = sum(sha256_file(ROOT / entry["path"]) == entry["sha256"] for entry in files)
    rows.append(
        [
            "test prediction files",
            "`stage3/predictions/<id>/<role>__seeds-<...>.npz`",
            "receipt",
            f"{equal} of {len(files)}",
        ]
    )
    lines = table(["file", "path", "pinned SHA-256", "equal"], rows, right=3)
    lines += [
        "",
        "| step | commit | modified tracked files | command | started (local time) | wall seconds |",
        "|---|---|---:|---|---|---:|",
    ]
    for step, record in (("select", select), ("fix", fix), ("test-predict", receipt), ("evaluate", pinned)):
        command = " ".join(record["command"][1:])
        started = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(record["started_unix"]))
        lines.append(
            f"| {step} | `{record['git']['commit'][:7]}` | {len(record['git']['modified_tracked'])} | "
            f"`{command}` | {started} | {record['wall_seconds']:.0f} |"
        )
    variables = pinned["environment"]["variables"]
    lines += [
        "",
        f"evaluate threads: OPENBLAS_NUM_THREADS={variables['OPENBLAS_NUM_THREADS']}; "
        f"packages {pinned['environment']['packages']}",
    ]
    for role, info in evaluation["sets"].items():
        lines.append(
            f"- {SETS[role]}: {info['records']} records, {info['groups']} patients, "
            f"{info['draws']} resamples, {info['redraws']} redraws, seed {info['seed']}, "
            f"{len(info['entering'])} entering conditions, "
            f"resample fingerprint `{info['fingerprint'][:12]}`"
        )
    return lines


def csv_agreement(evaluation: dict[str, Any]) -> list[str]:
    """
    Check that the CSV summaries written by ``evaluate`` equal ``evaluation.json`` to the CSV text precision.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.

    Returns
    -------
    list[str]
        Rows checked and rows that differ, per CSV file.
    """

    def same(stored: float | None, text: str) -> bool:
        a = math.nan if stored is None else float(stored)
        b = float(text) if text else math.nan
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-14 * max(1.0, abs(a))

    def read(name: str) -> list[dict[str, str]]:
        with open(ROOT / STAGE3 / "evaluation" / name, newline="") as handle:
            return list(csv.DictReader(handle))

    lines = []
    rows = read("macro.csv")
    differ = sum(
        not all(
            same(evaluation["candidates"][r["set"]][r["candidate"]][r["probabilities"]][key], r[column])
            for key, column in (
                ("macro_auroc", "macro_auroc"),
                ("macro_auroc_ci_low", "ci_low"),
                ("macro_auroc_ci_high", "ci_high"),
            )
        )
        for r in rows
    )
    lines.append(f"- macro.csv: {len(rows)} rows, {differ} differ")
    rows = read("comparisons.csv")
    differ = 0
    for r in rows:
        comparison = evaluation["comparisons"][r["set"]][r["candidate"]]
        if r["condition"] == "macro":
            stored = comparison["macro"]
        else:
            stored = next(x for x in comparison["conditions"] if x["condition"] == r["condition"])
        differ += not all(same(stored[key], r[key]) for key in ("difference", "ci_low", "ci_high"))
    lines.append(f"- comparisons.csv: {len(rows)} rows, {differ} differ")
    rows = read("conditions.csv")
    differ = 0
    for r in rows:
        stored = rows_by_condition(evaluation["candidates"][r["set"]][r["candidate"]][r["probabilities"]])
        entry = stored[r["condition"]]
        keys = [k for k in ("auroc", "brier", "sensitivity", "calibration_slope") if k in entry]
        differ += not all(same(entry[key], r[key]) for key in keys)
    lines.append(f"- conditions.csv: {len(rows)} rows, {differ} differ")
    return lines


def confirmatory(evaluation: dict[str, Any]) -> list[str]:
    """
    Show the confirmatory entry as stored.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    entry = evaluation["confirmatory"]
    macro = entry["macro"]
    return [
        f"set {entry['set']}; selected {entry['selected']}; reference {entry['reference']}; "
        f"selected is reference: {entry['selected_is_reference']}",
        f"macro AUROC difference {entry['selected']} minus {entry['reference']}: "
        f"{interval(macro['difference'], macro['ci_low'], macro['ci_high'], 5)}; better: {entry['better']}",
        f"interval includes zero: {macro['ci_low'] <= 0 <= macro['ci_high']}",
        f"Holm family {entry['holm_family']}; resample fingerprint `{entry['fingerprint'][:12]}`",
    ]


def macro_table(evaluation: dict[str, Any], record: dict[str, Any]) -> list[str]:
    """
    Macro AUROC of every candidate on both test sets beside the validation selection metric.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.
    record : dict[str, Any]
        ``selection_record.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    validation = {row["id"]: row["selection_metric"] for row in record["candidates"]}
    rows = []
    for name in CANDIDATES:
        cells = [name, number(validation[name])]
        for role in SETS:
            raw = evaluation["candidates"][role][name]["raw"]
            cells.append(interval(raw["macro_auroc"], raw["macro_auroc_ci_low"], raw["macro_auroc_ci_high"]))
        rows.append(cells)
    lines = table(["candidate", "validation selection metric", *SETS.values()], rows)
    for role, label in SETS.items():
        order = sorted(CANDIDATES, key=lambda n: -evaluation["candidates"][role][n]["raw"]["macro_auroc"])
        lines.append(f"- rank at {label}: {', '.join(order)}")
    order = sorted(CANDIDATES, key=lambda n: -validation[n])
    lines.append(f"- rank on validation: {', '.join(order)}")
    shift = [
        abs(
            evaluation["candidates"][role][n]["platt"]["macro_auroc"]
            - evaluation["candidates"][role][n]["raw"]["macro_auroc"]
        )
        for role in SETS
        for n in CANDIDATES
    ]
    lines.append(f"- largest |Platt minus raw| macro AUROC over candidates and sets: {max(shift):.2e}")
    return lines


def macro_differences(evaluation: dict[str, Any]) -> list[str]:
    """
    Paired macro AUROC differences of every candidate against P0 on both test sets.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    rows = []
    for name in CANDIDATES[1:]:
        cells = [name]
        for role in SETS:
            macro = evaluation["comparisons"][role][name]["macro"]
            cells += [interval(macro["difference"], macro["ci_low"], macro["ci_high"]), str(macro["better"])]
        rows.append(cells)
    header = ["candidate minus P0"]
    for label in SETS.values():
        header += [label, "lower end above 0"]
    return table(header, rows)


def condition_differences(evaluation: dict[str, Any], role: str, name: str = "E") -> list[str]:
    """
    Per-condition AUROC differences of one candidate against P0 with Holm-adjusted p-values.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.
    role : str
        Test set.
    name : str
        Candidate.

    Returns
    -------
    list[str]
        Lines.
    """
    comparison = evaluation["comparisons"][role][name]
    rows = [
        [
            NAMES[row["condition"]],
            interval(row["difference"], row["ci_low"], row["ci_high"]),
            number(row["p_value"], 3),
            number(row["p_holm"], 3),
            "yes" if row["p_holm"] < 0.05 else "",
        ]
        for row in comparison["conditions"]
    ]
    lines = table(["condition", f"{name} minus P0 (95% interval)", "p", "Holm p", "Holm p < 0.05"], rows)
    lines.append(f"- Holm family: {comparison['holm_family']} conditions")
    return lines


def flags(row: dict[str, Any]) -> str:
    """
    Join the imprecise and usability flags of one condition row.

    Parameters
    ----------
    row : dict[str, Any]
        Row of an ``interval_report`` table.

    Returns
    -------
    str
        Comma-separated flags.
    """
    marks = []
    if row["imprecise"]:
        marks.append("imprecise")
    if not row["usable"]:
        marks.append("not usable")
    return ", ".join(marks)


def condition_table(evaluation: dict[str, Any], role: str, first: str = "E", second: str = "P0") -> list[str]:
    """
    Per-condition results of two candidates on one test set at the validation thresholds.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.
    role : str
        Test set.
    first, second : str
        Candidates.

    Returns
    -------
    list[str]
        Lines.
    """
    a = rows_by_condition(evaluation["candidates"][role][first]["raw"])
    b = rows_by_condition(evaluation["candidates"][role][second]["raw"])
    rows = []
    for condition in NAMES:
        x, y = a[condition], b[condition]
        marks = flags(x) if flags(x) == flags(y) else f"{first}: {flags(x)}; {second}: {flags(y)}"
        rows.append(
            [
                NAMES[condition],
                f"{x['positives']} ({x['prevalence']:.3f})",
                interval(x["auroc"], x["auroc_ci_low"], x["auroc_ci_high"], 3),
                interval(y["auroc"], y["auroc_ci_low"], y["auroc_ci_high"], 3),
                f"{x['average_precision']:.3f} / {y['average_precision']:.3f}",
                f"{x['sensitivity']:.3f} / {y['sensitivity']:.3f}",
                f"{x['specificity']:.3f} / {y['specificity']:.3f}",
                marks,
            ]
        )
    return table(
        [
            "condition",
            "positives (prevalence)",
            f"AUROC {first}",
            f"AUROC {second}",
            f"AP {first} / {second}",
            f"sensitivity {first} / {second}",
            f"specificity {first} / {second}",
            "flags",
        ],
        rows,
    )


def usability(evaluation: dict[str, Any]) -> list[str]:
    """
    Conditions not called usable, per candidate and test set, with the reason.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    lines = []
    for role, label in SETS.items():
        for name in CANDIDATES:
            reasons = []
            for row in evaluation["candidates"][role][name]["raw"]["conditions"]:
                if row["usable"]:
                    continue
                why = []
                if row["positives"] < 100:
                    why.append(f"{row['positives']} positives")
                if row["auroc_ci_low"] < 0.80:
                    why.append(f"lower end {row['auroc_ci_low']:.3f}")
                reasons.append(f"{NAMES[row['condition']]} ({'; '.join(why)})")
            lines.append(f"- {label}, {name}: {len(reasons)} not usable: {', '.join(reasons) or 'none'}")
    return lines


def calibration(evaluation: dict[str, Any], role: str, name: str) -> list[str]:
    """
    Damped-Newton calibration slope and intercept (raw and after validation Platt), library values and Brier.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.
    role : str
        Test set.
    name : str
        Candidate.

    Returns
    -------
    list[str]
        Lines.
    """
    tables = evaluation["candidates"][role][name]
    raw, platt = rows_by_condition(tables["raw"]), rows_by_condition(tables["platt"])
    raw_damped = rows_by_condition(tables["raw_calibration_damped"])
    platt_damped = rows_by_condition(tables["platt_calibration_damped"])
    rows = []
    for condition in NAMES:
        r, p, rd, pd_ = raw[condition], platt[condition], raw_damped[condition], platt_damped[condition]
        rows.append(
            [
                NAMES[condition],
                interval(
                    rd["calibration_slope"],
                    rd["calibration_slope_ci_low"],
                    rd["calibration_slope_ci_high"],
                    2,
                ),
                interval(
                    rd["calibration_intercept"],
                    rd["calibration_intercept_ci_low"],
                    rd["calibration_intercept_ci_high"],
                    2,
                ),
                number(r["calibration_in_the_large"], 2),
                number(r["calibration_slope"], 2),
                interval(
                    pd_["calibration_slope"],
                    pd_["calibration_slope_ci_low"],
                    pd_["calibration_slope_ci_high"],
                    2,
                ),
                number(p["calibration_in_the_large"], 2),
                number(r["brier"], 4),
                number(p["brier"], 4),
            ]
        )
    return table(
        [
            "condition",
            "raw slope (damped)",
            "raw intercept (damped)",
            "raw CITL",
            "raw slope (library)",
            "Platt slope (damped)",
            "Platt CITL",
            "Brier raw",
            "Brier Platt",
        ],
        rows,
    )


def calibration_summary(evaluation: dict[str, Any]) -> list[str]:
    """
    Library failures and damped-fit NaN resamples per candidate, and Brier raw against Platt.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    rows = []
    for role, label in SETS.items():
        for name in CANDIDATES:
            failures = evaluation["library_calibration_failures"][role][name]
            tables = evaluation["candidates"][role][name]
            damped_nan = sum(
                (row["calibration_slope_nan_draws"] or 0)
                for row in tables["raw_calibration_damped"]["conditions"]
            )
            damped_point_nan = sum(
                row["calibration_slope"] is None or row["calibration_slope"] != row["calibration_slope"]
                for row in tables["raw_calibration_damped"]["conditions"]
            )
            raw, platt = rows_by_condition(tables["raw"]), rows_by_condition(tables["platt"])
            lower = sum(platt[c]["brier"] < raw[c]["brier"] for c in NAMES)
            slopes = [row["calibration_slope"] for row in tables["raw_calibration_damped"]["conditions"]]
            platt_slopes = [
                row["calibration_slope"] for row in tables["platt_calibration_damped"]["conditions"]
            ]
            rows.append(
                [
                    label,
                    name,
                    str(len(failures["raw"]["point_slope_nan"])),
                    str(failures["raw"]["slope_nan_draws"]),
                    str(damped_point_nan),
                    str(damped_nan),
                    f"{min(slopes):.2f}-{max(slopes):.2f}",
                    f"{min(platt_slopes):.2f}-{max(platt_slopes):.2f}",
                    f"{lower} of 14",
                ]
            )
    return table(
        [
            "set",
            "candidate",
            "library point slopes NaN (of 14)",
            "library NaN resamples (raw)",
            "damped point slopes NaN",
            "damped NaN resamples (raw)",
            "damped raw slope range",
            "damped Platt slope range",
            "Brier lower after Platt",
        ],
        rows,
        right=2,
    )


def secondary(evaluation: dict[str, Any], name: str) -> list[str]:
    """
    SPH calibration-half refit applied to the SPH evaluation half, against the validation fit.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.
    name : str
        Candidate.

    Returns
    -------
    list[str]
        Lines.
    """
    entry = evaluation["secondary_sph_calibration"]["candidates"][name]
    primary = rows_by_condition(evaluation["candidates"]["test_external"][name]["raw"])
    local = rows_by_condition(entry["at_local_thresholds"])
    difference = entry["local_minus_validation"]
    damped_local = rows_by_condition(entry["after_local_platt_calibration_damped"])
    damped_validation = rows_by_condition(
        evaluation["candidates"]["test_external"][name]["platt_calibration_damped"]
    )
    rows = []
    for condition in NAMES:
        cells = [NAMES[condition]]
        for metric in ("sensitivity", "specificity"):
            ci = difference["interval"][metric][condition]
            cells += [
                number(primary[condition][metric], 3),
                number(local[condition][metric], 3),
                interval(difference["observed"][metric][condition], ci["ci_low"], ci["ci_high"], 3),
            ]
        slope = difference["calibration_damped"]["calibration_slope"][condition]
        cells += [
            number(damped_validation[condition]["calibration_slope"], 2),
            number(damped_local[condition]["calibration_slope"], 2),
            interval(slope["difference"], slope["ci_low"], slope["ci_high"], 2),
        ]
        rows.append(cells)
    return table(
        [
            "condition",
            "sens. validation thr.",
            "sens. local thr.",
            "difference",
            "spec. validation thr.",
            "spec. local thr.",
            "difference",
            "Platt slope, validation fit",
            "Platt slope, local fit",
            "difference",
        ],
        rows,
    )


def secondary_summary(evaluation: dict[str, Any]) -> list[str]:
    """
    Range of specificity over conditions at validation and at local thresholds, every candidate.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    rows = []
    for name in CANDIDATES:
        entry = evaluation["secondary_sph_calibration"]["candidates"][name]
        validation = [
            row["specificity"] for row in evaluation["candidates"]["test_external"][name]["raw"]["conditions"]
        ]
        local = [row["specificity"] for row in entry["at_local_thresholds"]["conditions"]]
        sensitivity = [
            row["sensitivity"] for row in evaluation["candidates"]["test_external"][name]["raw"]["conditions"]
        ]
        local_sensitivity = [row["sensitivity"] for row in entry["at_local_thresholds"]["conditions"]]
        slopes = [
            row["calibration_slope"]
            for row in evaluation["candidates"]["test_external"][name]["platt_calibration_damped"][
                "conditions"
            ]
        ]
        local_slopes = [
            row["calibration_slope"] for row in entry["after_local_platt_calibration_damped"]["conditions"]
        ]
        rows.append(
            [
                name,
                f"{min(validation):.3f}-{max(validation):.3f}",
                f"{min(local):.3f}-{max(local):.3f}",
                f"{min(sensitivity):.3f}-{max(sensitivity):.3f}",
                f"{min(local_sensitivity):.3f}-{max(local_sensitivity):.3f}",
                f"{min(slopes):.2f}-{max(slopes):.2f}",
                f"{min(local_slopes):.2f}-{max(local_slopes):.2f}",
            ]
        )
    lines = table(
        [
            "candidate",
            "specificity, validation thresholds",
            "specificity, local thresholds",
            "sensitivity, validation thresholds",
            "sensitivity, local thresholds",
            "Platt slope, validation fit",
            "Platt slope, local fit",
        ],
        rows,
    )
    lines.append(
        f"- fit records (SPH calibration half): {evaluation['secondary_sph_calibration']['fit_records']}"
    )
    return lines


def lofo() -> list[str]:
    """
    Leave-one-family-out macro AUROC per held-out source, from the Stage 2 outputs.

    Returns
    -------
    list[str]
        Lines.
    """
    macro = {
        name: load(OUTPUT / f"lofo/{name}/lofo.json")["macro_by_held_out_source"]
        for name in ("P0", "L-C", "H-All")
    }
    scratch = load(OUTPUT / "lofo/S-R/lofo.json")["families"]
    entering = {}
    for name in ("P0", "L-C", "H-All"):
        with open(ROOT / OUTPUT / f"lofo/{name}/lofo.csv", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (name, row["held_out"], row["source"])
                entering[key] = entering.get(key, 0) + (row["enters_macro"] == "True")
    rows = []
    for family, source, label in LOFO_SOURCES:
        held = scratch[family]["held_out"][source]
        counts = {entering[(name, family, source)] for name in ("P0", "L-C", "H-All")}
        rows.append(
            [
                label,
                str(held["records"]),
                f"{len(held['entering'])} / {', '.join(str(c) for c in sorted(counts))}",
                *(number(macro[name][f"{family}/{source}"]) for name in ("P0", "L-C", "H-All")),
                number(held["macro_auroc"]),
            ]
        )
    means = [
        sum(macro[name][f"{f}/{s}"] for f, s, _ in LOFO_SOURCES) / len(LOFO_SOURCES)
        for name in ("P0", "L-C", "H-All")
    ]
    means.append(
        sum(scratch[f]["held_out"][s]["macro_auroc"] for f, s, _ in LOFO_SOURCES) / len(LOFO_SOURCES)
    )
    rows.append(["unweighted mean", "", "", *(number(value) for value in means)])
    return table(
        [
            "held-out source",
            "records",
            "entering (S-R / others)",
            "P0",
            "L-C",
            "H-All",
            "S-R (seed 20261110)",
        ],
        rows,
    )


def source_identification() -> list[str]:
    """
    Source-family identification from each feature set, from the Stage 2 output.

    Returns
    -------
    list[str]
        Lines.
    """
    result = load(OUTPUT / "source_identification/source_identification.json")
    rows = [
        [
            name,
            str(value["features"]),
            number(value["accuracy"]),
            number(value["balanced_accuracy"]),
            number(value["macro_ovr_auroc"]),
            number(value["majority_accuracy"]),
            number(value["chance_balanced_accuracy"], 2),
        ]
        for name, value in result.items()
    ]
    return table(
        [
            "features",
            "width",
            "accuracy",
            "balanced accuracy",
            "macro one-vs-rest AUROC",
            "majority class",
            "1/K",
        ],
        rows,
    )


def subgroup_macros(evaluation: dict[str, Any]) -> list[str]:
    """
    Macro AUROC within the sex, flat-lead and one-against-several subgroups (point estimates).

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    rows = []
    for role, label in SETS.items():
        for name in CANDIDATES:
            audit = evaluation["audits"][role][name]
            cells = [label, name]
            for kind, group in (
                ("sex", "female"),
                ("sex", "male"),
                ("cooccurrence", "one_condition"),
                ("cooccurrence", "several_conditions"),
            ):
                entry = audit[kind][group]
                cells.append(f"{entry['macro_auroc']:.4f} ({len(entry['entering'])})")
            cells.append(
                ", ".join(f"{group} {entry['records']}" for group, entry in audit["flat_leads"].items())
            )
            rows.append(cells)
    lines = table(
        ["set", "candidate", "female", "male", "one condition", "several conditions", "flat-lead groups"],
        rows,
        right=2,
    )
    audit = evaluation["audits"]
    for role, label in SETS.items():
        sizes = {g: e["records"] for g, e in audit[role]["E"]["sex"].items()}
        lines.append(f"- {label}: sex groups {sizes}")
        for group, entry in audit[role]["E"]["cooccurrence"].items():
            out = [
                f"{NAMES[r['condition']]} {r['positives']}"
                for r in entry["conditions"]
                if not r["enters_macro"]
            ]
            lines.append(f"- {label}, {group}: not entering (positives): {', '.join(out) or 'none'}")
        for group, entry in audit[role]["E"]["sex"].items():
            out = [
                f"{NAMES[r['condition']]} {r['positives']}"
                for r in entry["conditions"]
                if not r["enters_macro"]
            ]
            lines.append(f"- {label}, {group}: not entering (positives): {', '.join(out) or 'none'}")
    return lines


def likelihood(evaluation: dict[str, Any]) -> list[str]:
    """
    PTB-XL fold 10 with likelihood below 50 undefined, beside the main labels.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    entry = evaluation["likelihood_50"]
    rows = []
    for name in CANDIDATES:
        main = evaluation["candidates"]["test_same_source"][name]["raw"]
        lik = entry["candidates"][name]
        mi = rows_by_condition(lik)["mi_any"]
        lvh = rows_by_condition(lik)["lvh"]
        main_rows = rows_by_condition(main)
        rows.append(
            [
                name,
                number(main["macro_auroc"]),
                interval(lik["macro_auroc"], lik["macro_auroc_ci_low"], lik["macro_auroc_ci_high"]),
                f"{number(main_rows['mi_any']['auroc'], 3)} / {number(mi['auroc'], 3)}",
                f"{number(main_rows['lvh']['auroc'], 3)} / {number(lvh['auroc'], 3)}",
            ]
        )
    lines = table(
        [
            "candidate",
            "main labels",
            "likelihood >= 50 (95% interval)",
            "any MI main / lik",
            "LVH main / lik",
        ],
        rows,
    )
    lik_rows = rows_by_condition(entry["candidates"]["E"])
    lines.append(
        f"- entries made undefined: {entry['entries_made_undefined']}; "
        f"MI positives {lik_rows['mi_any']['positives']}, "
        f"LVH positives {lik_rows['lvh']['positives']}; entering {len(entry['resampling']['entering'])}; "
        f"redraws {entry['resampling']['redraws']}"
    )
    return lines


def multiclass(evaluation: dict[str, Any]) -> list[str]:
    """
    Show derived top-1, top-2 and exact-set accuracy under both readings of Addendum 1.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    rows = []
    for role, label in SETS.items():
        for name in CANDIDATES:
            audit = evaluation["audits"][role][name]
            view, platt = audit["multiclass"], audit["top_k_platt"]
            cells = [label, name]
            for k in ("top_1", "top_2"):
                for reading in ("undefined_allowed", "all_defined"):
                    cells.append(f"{view[k][reading]['accuracy']:.3f} / {platt[k][reading]['accuracy']:.3f}")
            exact = view["exact_set"]
            cells += [number(exact["accuracy"], 3), number(exact["accuracy_all_defined"], 3)]
            rows.append(cells)
    lines = table(
        [
            "set",
            "candidate",
            "top-1, undefined allowed",
            "top-1, all defined",
            "top-2, undefined allowed",
            "top-2, all defined",
            "exact set",
            "exact set, all defined",
        ],
        rows,
        right=2,
    )
    for role, label in SETS.items():
        view = evaluation["audits"][role]["E"]["multiclass"]
        lines.append(
            f"- {label}: records undefined allowed {view['top_1']['undefined_allowed']['records']}, "
            f"all defined {view['top_1']['all_defined']['records']}; "
            f"exact set {view['exact_set']['records']} records, "
            f"all defined {view['exact_set']['records_all_defined']}"
        )
    return lines


def multiclass_by_condition(evaluation: dict[str, Any], role: str) -> list[str]:
    """
    Top-1 accuracy of E and P0 per true condition, undefined entries allowed.

    Parameters
    ----------
    evaluation : dict[str, Any]
        ``evaluation.json``.
    role : str
        Test set.

    Returns
    -------
    list[str]
        Lines.
    """
    views = {
        name: evaluation["audits"][role][name]["multiclass"]["top_1"]["undefined_allowed"]
        for name in ("P0", "E")
    }
    rows = [
        [
            NAMES[c],
            str(views["E"]["records_by_condition"][c]),
            number(views["P0"]["accuracy_by_condition"][c], 3),
            number(views["E"]["accuracy_by_condition"][c], 3),
        ]
        for c in NAMES
    ]
    return table(["condition", "single-condition records", "top-1 P0", "top-1 E"], rows)


def selection(record: dict[str, Any]) -> list[str]:
    """
    E's members and the chosen logistic C values at the grid edge.

    Parameters
    ----------
    record : dict[str, Any]
        ``selection_record.json``.

    Returns
    -------
    list[str]
        Lines.
    """
    lines = [f"selected: {record['selected']}"]
    lines.append(f"members of E: {', '.join(record['ensemble']['members'])}")
    for row in record["candidates"]:
        chosen = row["settings"].get("chosen_c")
        if chosen:
            edge = sum(value == 100.0 for value in chosen.values())
            lines.append(f"- {row['id']}: C = 100 for {edge} of {len(chosen)} conditions")
    operating = load(STAGE3 / "operating_points.json")["candidates"]
    slopes = [fit["slope"] for name in CANDIDATES for fit in operating[name]["platt"].values()]
    points = [point for name in CANDIDATES for point in operating[name]["validation_at_threshold"].values()]
    specificity = [point["specificity"] for point in points]
    lines.append(
        f"- validation Platt slopes: {sum(slope > 0 for slope in slopes)} of {len(slopes)} positive, "
        f"range {min(slopes):.3f}-{max(slopes):.3f}"
    )
    lines.append(
        f"- validation specificity at the fitted thresholds: {min(specificity):.4f}-{max(specificity):.4f}"
    )
    return lines


SECTIONS = {
    "provenance": lambda e, r: provenance(e),
    "csv-agreement": lambda e, r: csv_agreement(e),
    "confirmatory": lambda e, r: confirmatory(e),
    "macro": macro_table,
    "macro-differences": lambda e, r: macro_differences(e),
    "condition-differences-sph": lambda e, r: condition_differences(e, "test_external"),
    "condition-differences-ptbxl": lambda e, r: condition_differences(e, "test_same_source"),
    "conditions-sph": lambda e, r: condition_table(e, "test_external"),
    "conditions-ptbxl": lambda e, r: condition_table(e, "test_same_source"),
    "usability": lambda e, r: usability(e),
    "calibration-E-sph": lambda e, r: calibration(e, "test_external", "E"),
    "calibration-P0-sph": lambda e, r: calibration(e, "test_external", "P0"),
    "calibration-E-ptbxl": lambda e, r: calibration(e, "test_same_source", "E"),
    "calibration-P0-ptbxl": lambda e, r: calibration(e, "test_same_source", "P0"),
    "calibration-summary": lambda e, r: calibration_summary(e),
    "secondary-E": lambda e, r: secondary(e, "E"),
    "secondary-P0": lambda e, r: secondary(e, "P0"),
    "secondary-summary": lambda e, r: secondary_summary(e),
    "lofo": lambda e, r: lofo(),
    "source-identification": lambda e, r: source_identification(),
    "subgroups": lambda e, r: subgroup_macros(e),
    "likelihood": lambda e, r: likelihood(e),
    "multiclass": lambda e, r: multiclass(e),
    "multiclass-by-condition-sph": lambda e, r: multiclass_by_condition(e, "test_external"),
    "multiclass-by-condition-ptbxl": lambda e, r: multiclass_by_condition(e, "test_same_source"),
    "selection": lambda e, r: selection(r),
    "not-implemented": lambda e, r: [f"- {key}: {value}" for key, value in e["not_implemented"].items()],
}


def main() -> None:
    """Print every section, or the one named by ``--section``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=sorted(SECTIONS), default=None)
    args = parser.parse_args()
    evaluation = load(STAGE3 / "evaluation/evaluation.json")
    record = load(STAGE3 / "selection_record.json")
    for name, build in SECTIONS.items():
        if args.section in (None, name):
            print(f"## {name}\n")
            print("\n".join(build(evaluation, record)))
            print()


if __name__ == "__main__":
    main()
