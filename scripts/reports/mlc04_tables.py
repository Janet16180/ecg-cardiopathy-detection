"""
Print the tables of the MLC-04 report from its outputs.

The script computes no metric: it reads ``outputs/mlc04_final_test_20261111/`` (the evaluation of each role,
the dry-run checks, the rule flows, the manifests' counts, the thresholds and the receipts), selects, counts
and formats. Every table names its section:

``PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc04_tables [--section NAME]``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

import pandas as pd

from ecg_experiment.files import read_json, sha256_file
from ecg_experiment.mlc02_data import CHALLENGE_SOURCES, CONDITIONS
from ecg_experiment.mlc04_data import OUTPUT
from ecg_experiment.paths import from_stored

ROLES = ("challenge_test", "validation")
ROLE_NAMES = {"challenge_test": "test groups", "validation": "validation (calibration groups)"}
SOURCE_NAMES = {
    "ningbo": "Ningbo",
    "chapman_shaoxing": "Chapman",
    "georgia": "Georgia",
    "cpsc_2018": "CPSC 2018",
    "cpsc_2018_extra": "CPSC-Extra",
}
MODELS = ("P0-v5", "P0-v4", "H-All-v5", "H-All-v4", "K6")
COMPARISONS = {
    "cleaning_v5_labels": "P0-v5 minus P0-v4, v5-rule labels",
    "cleaning_v4_labels": "P0-v5 minus P0-v4, v4 labels",
    "larger_model": "H-All-v5 minus P0-v5",
    "key": "K6 minus P0-v5",
}


def table(rows: list[dict[str, Any]]) -> str:
    """
    Format rows as a Markdown table.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Rows with the same keys.

    Returns
    -------
    str
        The table, or a note when there is no row.
    """
    if not rows:
        return "(no rows)"
    names = list(rows[0])
    lines = ["| " + " | ".join(names) + " |", "|" + "|".join("---" for _ in names) + "|"]
    lines += ["| " + " | ".join(str(row[name]) for name in names) + " |" for row in rows]
    return "\n".join(lines)


def number(value: Any, digits: int = 4) -> str:
    """
    Format a number, or an empty cell for a missing one.

    Parameters
    ----------
    value : Any
        Number or None.
    digits : int
        Decimals.

    Returns
    -------
    str
        Formatted text.
    """
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return f"{value:.{digits}f}"


def with_ci(value: Any, ci: Any, digits: int = 4) -> str:
    """
    Format a value with its interval.

    Parameters
    ----------
    value : Any
        Point value.
    ci : Any
        Two bounds, or None.

    digits : int
        Decimals.

    Returns
    -------
    str
        ``value (low, high)``.
    """
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if ci is None or ci[0] is None or (isinstance(ci[0], float) and ci[0] != ci[0]):
        return number(value, digits)
    return f"{number(value, digits)} ({number(ci[0], digits)}, {number(ci[1], digits)})"


def evaluation(role: str) -> dict[str, Any] | None:
    """
    Read a role's evaluation, if it exists.

    Parameters
    ----------
    role : str
        Role.

    Returns
    -------
    dict[str, Any] | None
        ``evaluation.json`` content.
    """
    path = OUTPUT / role / "score" / "evaluation.json"
    return read_json(path) if path.exists() else None


def conditions(role: str) -> pd.DataFrame | None:
    """
    Read a role's per-source, per-condition table, if it exists.

    Parameters
    ----------
    role : str
        Role.

    Returns
    -------
    pd.DataFrame | None
        ``conditions.csv``.
    """
    path = OUTPUT / role / "score" / "conditions.csv"
    return pd.read_csv(path) if path.exists() else None


def primary() -> str:
    """Primary result: P0-v5's metric on the test groups beside its validation values."""
    rows = []
    for role in ROLES:
        result = evaluation(role)
        if result is None:
            continue
        found = result["primary"]
        rule20 = result["models_validation_rule"]["P0-v5"]
        rows.append(
            {
                "set": ROLE_NAMES[role],
                "records": result["records"]["v5"],
                "metric, entry 100 (95% interval)": with_ci(found["value"], found["ci"]),
                "metric, entry 20 (95% interval)": with_ci(rule20["value"], rule20["ci"]),
                "sources": len(result["models_primary_rule"]["P0-v5"]["sources"]),
            }
        )
    test = evaluation("challenge_test")
    text = table(rows)
    if test is not None:
        found = test["primary"]
        text += (
            f"\n\nFrozen validation value {found['validation_value']!r} (six validation sources, entry 20); "
            f"bound {found['bound']!r}; reading: **{found['reading']}**."
        )
    return text


def secondary() -> str:
    """Secondary comparisons with their Holm-adjusted p-values and readings."""
    result = evaluation("challenge_test")
    if result is None:
        return "(no test evaluation)"
    holm = result["readings"]["holm"]
    family = {
        "cleaning_v5_labels": holm["cleaning"],
        "larger_model": holm["larger_model"],
        "key": holm["key"],
    }
    rows = [
        {
            "comparison": COMPARISONS[name],
            "difference (95% interval)": with_ci(found["difference"], found["ci"]),
            "null": found["null"],
            "p": number(found["p_value"], 4),
            "Holm p": number(family.get(name), 4),
        }
        for name, found in result["comparisons"].items()
    ]
    readings = result["readings"]
    return (
        table(rows) + f"\n\nCleaning: {readings['cleaning']['reading']} (v5 labels: "
        f"{readings['cleaning']['v5_labels']}; v4-label interval not below zero: "
        f"{readings['cleaning']['v4_labels_not_below_zero']}). Larger model: {readings['larger_model']}. "
        f"Key: {readings['key']}."
    )


def models() -> str:
    """Every model's metric on both roles, entry rules 100 and 20, and on v4 labels."""
    rows = []
    for name in MODELS:
        row: dict[str, Any] = {"model": name}
        for role in ROLES:
            result = evaluation(role)
            if result is None:
                continue
            short = "test" if role == "challenge_test" else "validation"
            found = result["models_primary_rule"][name]
            row[f"{short}, entry 100"] = with_ci(found["value"], found["ci"])
            row[f"{short}, entry 20"] = number(result["models_validation_rule"][name]["value"])
            v4 = result["models_v4_labels"].get(name)
            row[f"{short}, v4 labels"] = with_ci(v4["value"], v4["ci"]) if v4 else ""
        rows.append(row)
    return table(rows)


def sources() -> str:
    """Within-source macro AUROC (entry 100) of every model on both roles."""
    rows = []
    for role in ROLES:
        result = evaluation(role)
        if result is None:
            continue
        for source in CHALLENGE_SOURCES:
            found = result["models_primary_rule"]["P0-v5"]["macro_by_source"].get(source)
            if found is None:
                continue
            row = {
                "set": "test" if role == "challenge_test" else "validation",
                "source": SOURCE_NAMES[source],
                "conditions": len(found["entering"]),
            }
            for name in MODELS:
                part = result["models_primary_rule"][name]["macro_by_source"][source]
                row[name] = with_ci(part["value"], part["ci"]) if name == "P0-v5" else number(part["value"])
            rows.append(row)
    return table(rows)


def entering() -> str:
    """Conditions entering each source's macro under the primary entry rule (test groups)."""
    result = evaluation("challenge_test") or evaluation("validation")
    if result is None:
        return "(no evaluation)"
    found = result["primary"]["entering_by_source"]
    return table(
        [
            {"source": SOURCE_NAMES[s], "conditions": len(v), "entering": ", ".join(v)}
            for s, v in found.items()
        ]
    )


def condition_cells(frame: pd.DataFrame, source: str, condition: str) -> dict[str, Any]:
    """
    Select one source and condition of a conditions table.

    Parameters
    ----------
    frame : pd.DataFrame
        ``conditions.csv`` of a role.
    source, condition : str
        Cell.

    Returns
    -------
    dict[str, Any]
        The row as a dictionary, None values for intervals that were not resampled.
    """
    row = frame[(frame["source"] == source) & (frame["condition"] == condition)].iloc[0].to_dict()
    return {
        name: (None if isinstance(value, float) and value != value else value) for name, value in row.items()
    }


def metric_ci(row: dict[str, Any], name: str, digits: int = 3) -> str:
    """
    Format a metric of a conditions row with its interval.

    Parameters
    ----------
    row : dict[str, Any]
        Output of ``condition_cells``.
    name : str
        Metric.
    digits : int
        Decimals.

    Returns
    -------
    str
        Formatted cell.
    """
    return with_ci(row.get(name), [row.get(f"{name}_ci_low"), row.get(f"{name}_ci_high")], digits)


def condition_table(role: str) -> str:
    """
    P0-v5 per source and condition on one role: support, AUROC, AP, sensitivity and specificity.

    Parameters
    ----------
    role : str
        Role.

    Returns
    -------
    str
        Markdown table.
    """
    frame = conditions(role)
    if frame is None:
        return "(no evaluation)"
    rows = []
    for source in CHALLENGE_SOURCES:
        for condition in CONDITIONS:
            row = condition_cells(frame, source, condition)
            if not row["positives"] or not row["negatives"]:
                continue
            rows.append(
                {
                    "source": SOURCE_NAMES[source],
                    "condition": condition,
                    "pos / neg": f"{row['positives']} / {row['negatives']}",
                    "AUROC": metric_ci(row, "auroc"),
                    "AP": metric_ci(row, "average_precision"),
                    "prevalence": number(row["prevalence"], 3),
                    "sensitivity": metric_ci(row, "sensitivity"),
                    "specificity": metric_ci(row, "specificity"),
                    "entry": "100"
                    if row["enters_primary_rule"]
                    else ("20" if row["enters_validation_rule"] else ""),
                }
            )
    return table(rows)


def conditions_test() -> str:
    """P0-v5 per source and condition on the test groups."""
    return condition_table("challenge_test")


def conditions_validation() -> str:
    """P0-v5 per source and condition on the validation groups."""
    return condition_table("validation")


def condition_models() -> str:
    """AUROC of every model per source and condition on the test groups (cells with both classes)."""
    frame = conditions("challenge_test")
    if frame is None:
        return "(no test evaluation)"
    rows = []
    for _, row in frame[(frame["positives"] > 0) & (frame["negatives"] > 0)].iterrows():
        rows.append(
            {
                "source": SOURCE_NAMES[row["source"]],
                "condition": row["condition"],
                "positives": int(row["positives"]),
                **{name: number(row[f"auroc_{name}"], 3) for name in MODELS},
            }
        )
    return table(rows)


def thresholds() -> str:
    """List the frozen thresholds of P0-v5 and their pooled validation rates."""
    found = read_json(OUTPUT / "freeze" / "thresholds.json")
    rows = [
        {
            "condition": name,
            "threshold": number(found["thresholds"][name], 6),
            "validation sensitivity": number(found["validation_at_threshold"][name]["sensitivity"], 3),
            "validation specificity": number(found["validation_at_threshold"][name]["specificity"], 4),
            "positives / negatives": f"{found['validation_at_threshold'][name]['positives']} / "
            f"{found['validation_at_threshold'][name]['negatives']}",
        }
        for name in CONDITIONS
    ]
    return table(rows) + f"\n\nRecords: {found['records']}. Rule: {found['rule']}."


def calibration() -> str:
    """Brier score and reliability gaps of P0-v5 per source and condition, test beside validation."""
    test, validation = conditions("challenge_test"), conditions("validation")
    if test is None or validation is None:
        return "(evaluations missing)"
    rows = []
    for source in CHALLENGE_SOURCES:
        for condition in CONDITIONS:
            first, second = (
                condition_cells(test, source, condition),
                condition_cells(validation, source, condition),
            )
            if not first["positives"] or not first["negatives"]:
                continue
            rows.append(
                {
                    "source": SOURCE_NAMES[source],
                    "condition": condition,
                    "Brier test": metric_ci(first, "brier", 4),
                    "Brier validation": number(second["brier"], 4),
                    "mean gap test": number(first["calibration_gap_mean"], 3),
                    "largest gap test": number(first["calibration_gap_max"], 3),
                    "mean gap validation": number(second["calibration_gap_mean"], 3),
                }
            )
    return table(rows)


def flow(role: str) -> pd.DataFrame | None:
    """
    Read a role's rule flow, if it exists.

    Parameters
    ----------
    role : str
        Role.

    Returns
    -------
    pd.DataFrame | None
        ``rule_flow.csv``.
    """
    path = OUTPUT / role / "rule_flow.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame.assign(condition=frame["condition"].fillna(""))


def rule_counts(frame: pd.DataFrame, rule: str) -> str:
    """
    Count the records (C1-C4) or labels (C5-C12) a rule selects and changes in one role.

    Parameters
    ----------
    frame : pd.DataFrame
        A role's rule flow.
    rule : str
        Rule name.

    Returns
    -------
    str
        ``selected / changed`` for records, ``+p -n / +p -n`` for labels.
    """
    part = frame[frame["rule"] == rule]
    if part["condition"].eq("").all():
        return f"{int(part['selected'].sum())} / {int(part['excluded'].sum())}"
    numbers = [
        int(part[name].sum())
        for name in ("selected_positives", "selected_negatives", "undefined_positives", "undefined_negatives")
    ]
    return f"{numbers[0]}+ {numbers[1]}- / {numbers[2]}+ {numbers[3]}-"


def rules() -> str:
    """Count each rule C1-C12 on the test groups beside validation: selected / changed."""
    flows = {role: flow(role) for role in ROLES}
    rows = []
    for rule in [f"C{number_}" for number_ in range(1, 13)]:
        row = {"rule": rule}
        for role, frame in flows.items():
            if frame is not None:
                row["test" if role == "challenge_test" else "validation"] = rule_counts(frame, rule)
        rows.append(row)
    return table(rows)


def manifest_counts(role: str) -> list[dict[str, Any]]:
    """
    Count records per source along one role's flow: enumerated, excluded, duplicates, v4 and v5.

    Parameters
    ----------
    role : str
        Role.

    Returns
    -------
    list[dict[str, Any]]
        One row per source.
    """
    v4 = pd.read_csv(
        OUTPUT / role / "manifest_v4.csv",
        usecols=["source", "exclusion_reason", "duplicate_status", "included"],
        keep_default_na=False,
        dtype=str,
    )
    path = OUTPUT / role / "manifest_v5.csv"
    v5 = (
        pd.read_csv(
            path, usecols=["source", "included", "exclusion_reason"], keep_default_na=False, dtype=str
        )
        if path.exists()
        else None
    )
    rows = []
    for source in CHALLENGE_SOURCES:
        part = v4[v4["source"] == source]
        row = {
            "set": "test" if role == "challenge_test" else "validation",
            "source": SOURCE_NAMES[source],
            "records": len(part),
            "short": int((part["exclusion_reason"] == "short").sum()),
            "other exclusion": int(
                part["exclusion_reason"].isin(["unreadable", "wrong_leads", "wrong_rate"]).sum()
            ),
            "dropped copy": int((part["duplicate_status"] == "dropped_copy").sum()),
            "dropped conflict": int((part["duplicate_status"] == "dropped_conflict").sum()),
            "included v4": int((part["included"] == "True").sum()),
        }
        if v5 is not None:
            other = v5[v5["source"] == source]
            row["excluded C1-C4"] = int(other["exclusion_reason"].isin(["C1", "C2", "C3", "C4"]).sum())
            row["included v5"] = int((other["included"] == "True").sum())
        rows.append(row)
    return rows


def records() -> str:
    """Count records per source from the split table to dataset v5, both roles."""
    rows = []
    for role in ROLES:
        if (OUTPUT / role / "manifest_v4.csv").exists():
            rows += manifest_counts(role)
    return table(rows)


def support() -> str:
    """Positives / negatives / undefined per source and condition on the test groups, v5 beside v4."""
    paths = [OUTPUT / "challenge_test" / "score" / f"support_{name}.csv" for name in ("v5", "v4")]
    if not all(path.exists() for path in paths):
        return "(no test evaluation)"
    v5, v4 = (pd.read_csv(path) for path in paths)
    merged = v5.merge(v4, on=["source", "condition"], suffixes=("_v5", "_v4"))
    rows = [
        {
            "source": SOURCE_NAMES[row["source"]],
            "condition": row["condition"],
            "v5": f"{row['positives_v5']} / {row['negatives_v5']} / {row['undefined_v5']}",
            "v4": f"{row['positives_v4']} / {row['negatives_v4']} / {row['undefined_v4']}",
        }
        for _, row in merged.iterrows()
    ]
    return table(rows)


def check() -> str:
    """List the dry-run reproductions on the validation groups."""
    found = read_json(OUTPUT / "validation" / "check.json")
    rows = []
    for name, value in found.items():
        if name in (
            "failed",
            "models",
            "features",
            "outputs_stored_features",
            "outputs_reextracted_features",
        ):
            continue
        detail = {
            key: value[key]
            for key in ("records", "cells", "differing", "mismatches", "rows", "equal")
            if key in value
        }
        rows.append({"check": name, "passed": value.get("passed"), "detail": json.dumps(detail)})
    for name, value in found["features"].items():
        rows.append(
            {
                "check": f"features {name}",
                "passed": value["passed"],
                "detail": f"max abs {value['max_abs']:.3g}, max rel {value['max_rel']:.3g}, bit-identical "
                f"{value['bit_identical_records']} of {value['records']}",
            }
        )
    for group, label in (
        ("outputs_stored_features", "stored"),
        ("outputs_reextracted_features", "re-extracted"),
    ):
        for name, value in found[group].items():
            rows.append(
                {
                    "check": f"outputs {name}, {label} features",
                    "passed": value["passed"],
                    "detail": f"records {value['records']}, max abs {value['max_abs']:.3g}"
                    + (f", max rel {value['max_rel']:.3g}" if name == "K6" else ""),
                }
            )
    metric = found["metric"]
    return (
        table(rows) + f"\n\nMetric from stored features {metric['from_stored_features']!r} (difference "
        f"{metric['difference_stored_features']!r}); with re-extracted Challenge rows "
        f"{metric['with_reextracted_challenge_rows']!r} (difference {metric['difference_reextracted']!r}). "
        f"Failed: {found['failed']}."
    )


def integrity() -> str:
    """Receipts: commit, modified tracked files and whether each output still has its recorded SHA-256."""
    rows = []
    for path in sorted((OUTPUT / "receipts").glob("*.json")):
        found = read_json(path)
        outputs = found.get("outputs_sha256", {})
        equal = sum(
            sha256_file(from_stored(name)) == digest
            for name, digest in outputs.items()
            if from_stored(name).exists()
        )
        rows.append(
            {
                "receipt": path.stem,
                "commit": found["git"]["commit"][:7],
                "modified tracked": len(found["git"]["modified_tracked"]),
                "wall s": number(found["wall_seconds"], 0),
                "outputs equal": f"{equal} of {len(outputs)}",
            }
        )
    marker = OUTPUT / "challenge_test" / "score" / "marker.json"
    text = table(rows)
    if marker.exists():
        found = read_json(marker)
        text += (
            f"\n\nMarker: {len(found['attempts'])} attempt(s), "
            f"completed {found['completed_unix'] is not None}, "
            f"freeze commit {found['freeze']['freeze_commit'][:7]}, code commit "
            f"{found['freeze']['code_commit'][:7]}."
        )
    return text


SECTIONS: dict[str, Callable[[], str]] = {
    "primary": primary,
    "secondary": secondary,
    "models": models,
    "sources": sources,
    "entering": entering,
    "conditions-test": conditions_test,
    "conditions-validation": conditions_validation,
    "condition-models": condition_models,
    "thresholds": thresholds,
    "calibration": calibration,
    "rules": rules,
    "records": records,
    "support": support,
    "check": check,
    "integrity": integrity,
}


def main(argv: list[str] | None = None) -> None:
    """
    Print one section or all of them.

    Parameters
    ----------
    argv : list[str] | None
        Arguments; None reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--section", choices=sorted(SECTIONS))
    args = parser.parse_args(argv)
    names = [args.section] if args.section else list(SECTIONS)
    for name in names:
        print(f"## {name}\n\n{SECTIONS[name]()}\n")


if __name__ == "__main__":
    main()
