"""
Print the tables of the MLC-03 dataset v5 and overall reports from their outputs.

Sections: ``rules`` (per rule and role, selected and changed), ``gate`` (checks against Parts A and B),
``flow`` (records per source and role), ``undefined`` (labels made undefined per condition and role),
``test-roles`` (rule counts per source at the test roles), ``support`` (v4 against v5 per condition, role and
source), ``eligibility``, ``integrity``, ``effect`` (Part D selection metric), ``effect-conditions`` and
``effect-sources``.

``PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_v5_tables --section <name>``
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ecg_experiment import mlc03_dataset_v5 as mv
from ecg_experiment.mlc02_data import CONDITIONS

EFFECT = mv.QUALITY / "effect"
ROLE_NAMES = {
    "train": "train",
    "validation": "validation",
    "test_same_source": "fold 10",
    "sph_calibration": "SPH calibration",
    "test_external": "SPH test",
}
SOURCE_NAMES = {
    "ptbxl": "PTB-XL",
    "ningbo": "Ningbo",
    "chapman_shaoxing": "Chapman",
    "georgia": "Georgia",
    "cpsc_2018": "CPSC 2018",
    "cpsc_2018_extra": "CPSC-Extra",
    "sph": "SPH",
}


def markdown(frame: pd.DataFrame) -> str:
    """
    Render a table as GitHub Markdown without an index.

    Parameters
    ----------
    frame : pd.DataFrame
        Table.

    Returns
    -------
    str
        Markdown text.
    """
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(format_cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def format_cell(value: object) -> str:
    """
    Format one cell: floats to four decimals, the rest as text.

    Parameters
    ----------
    value : object
        Cell value.

    Returns
    -------
    str
        Text.
    """
    if isinstance(value, float):
        return "" if pd.isna(value) else f"{value:.4f}"
    return str(value)


def rules() -> pd.DataFrame:
    """
    Per rule and role: records or labels the rule selects before overlaps, and those it changes.

    Returns
    -------
    pd.DataFrame
        One row per rule, ``selected / changed`` per role.
    """
    summary = pd.read_csv(mv.V5 / "rule_summary.csv")
    summary["selected"] = summary["selected"].fillna(
        summary["selected_positives"] + summary["selected_negatives"]
    )
    summary["changed"] = summary["excluded"].fillna(
        summary["undefined_positives"] + summary["undefined_negatives"]
    )
    table = summary.pivot(index="rule", columns="role", values=["selected", "changed"]).fillna(0).astype(int)
    rows = []
    for rule in mv.RULE_ORDER:
        row = {"rule": rule, "action": "exclude records" if rule in mv.RECORD_RULES else "labels undefined"}
        for role, name in ROLE_NAMES.items():
            row[name] = f"{table.loc[rule, ('selected', role)]} / {table.loc[rule, ('changed', role)]}"
        rows.append(row)
    return pd.DataFrame(rows)


def label_rules_by_kind() -> pd.DataFrame:
    """
    Per label rule and role: positives and negatives selected, and made undefined.

    Returns
    -------
    pd.DataFrame
        One row per label rule.
    """
    summary = pd.read_csv(mv.V5 / "rule_summary.csv")
    summary = summary[summary["rule"].isin(mv.LABEL_RULES)]
    rows = []
    for rule in mv.LABEL_RULES:
        row = {"rule": rule}
        for role, name in ROLE_NAMES.items():
            part = summary[(summary["rule"] == rule) & (summary["role"] == role)].iloc[0]
            row[name] = (
                f"{int(part['selected_positives'])}+ {int(part['selected_negatives'])}- / "
                f"{int(part['undefined_positives'])}+ {int(part['undefined_negatives'])}-"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def gate() -> pd.DataFrame:
    """
    Summary of the integrity gate against Parts A and B.

    Returns
    -------
    pd.DataFrame
        One row per check.
    """
    integrity = json.loads((mv.V5 / "integrity.json").read_text())["gate"]
    part_a = pd.read_csv(mv.V5 / "part_a_count_check.csv")
    part_b = pd.read_csv(mv.V5 / "part_b_count_check.csv")
    rows = [
        {
            "check": "record reader equals Part B's",
            "result": "equal"
            if not integrity["records_equal_part_b_reader"]
            else str(integrity["records_equal_part_b_reader"]),
        },
        {
            "check": "method A equals Part B's",
            "result": "equal"
            if not integrity["method_a_equal_part_b"]
            else str(integrity["method_a_equal_part_b"]),
        },
        {
            "check": "analysis table equals Part B's analysis.parquet",
            "result": "equal"
            if not integrity["analysis_equal_part_b"]
            else str(integrity["analysis_equal_part_b"]),
        },
        {
            "check": f"method B remeasured on {integrity['method_b_sample']['records']} records",
            "result": "equal" if not integrity["method_b_sample"]["mismatches"] else "differs",
        },
        {
            "check": f"window summaries of {integrity['windows_equal_part_b']['records']} long records",
            "result": "equal" if not integrity["windows_equal_part_b"]["mismatches"] else "differs",
        },
        {
            "check": f"Part A counts, {len(part_a)} rule-role-source cells",
            "result": f"{int(part_a['equal'].sum())} equal",
        },
        {
            "check": f"Part B counts, {len(part_b)} rule-condition-source-role rows",
            "result": f"{int(part_b['equal'].sum())} equal",
        },
    ]
    return pd.DataFrame(rows)


def part_b_totals() -> pd.DataFrame:
    """
    Train plus validation totals of every adopted Part B rule part, as found and as Part B reported.

    Returns
    -------
    pd.DataFrame
        One row per Part B rule and condition.
    """
    check = pd.read_csv(mv.V5 / "part_b_count_check.csv")
    columns = [
        "positives_to_undefined_found",
        "positives_to_undefined_expected",
        "negatives_to_undefined_found",
        "negatives_to_undefined_expected",
    ]
    return check.groupby(["rule", "condition"], sort=False)[columns].sum().reset_index()


def flow() -> pd.DataFrame:
    """
    Count records per source and role in v4, excluded by each rule, and in v5.

    Returns
    -------
    pd.DataFrame
        The exclusion flow with readable names.
    """
    table = pd.read_csv(mv.V5 / "exclusion_flow_v5.csv")
    table = table[table["included_v4"] > 0]
    return table.assign(source=table["source"].map(SOURCE_NAMES), role=table["role"].map(ROLE_NAMES))


def undefined() -> pd.DataFrame:
    """
    Labels made undefined per condition and role, all label rules together, after overlaps.

    Returns
    -------
    pd.DataFrame
        ``positives / negatives`` per role.
    """
    rule_flow = pd.read_csv(mv.V5 / "rule_flow.csv")
    rule_flow = rule_flow[rule_flow["condition"].notna() & (rule_flow["condition"] != "")]
    sums = rule_flow.groupby(["condition", "role"])[["undefined_positives", "undefined_negatives"]].sum()
    rows = []
    for condition in CONDITIONS:
        row = {"condition": condition}
        for role, name in ROLE_NAMES.items():
            if (condition, role) in sums.index:
                values = sums.loc[(condition, role)]
                row[name] = f"{int(values['undefined_positives'])} / {int(values['undefined_negatives'])}"
            else:
                row[name] = "0 / 0"
        rows.append(row)
    return pd.DataFrame(rows)


def totals() -> pd.DataFrame:
    """
    Per role: records in v4 and v5, and labels made undefined (positives, negatives) by the label rules.

    Returns
    -------
    pd.DataFrame
        One row per role.
    """
    rule_flow = pd.read_csv(mv.V5 / "rule_flow.csv")
    labels = (
        rule_flow[rule_flow["rule"].isin(mv.LABEL_RULES)]
        .groupby("role")[["undefined_positives", "undefined_negatives"]]
        .sum()
    )
    records = (
        pd.read_csv(mv.V5 / "exclusion_flow_v5.csv").groupby("role")[["included_v4", "included_v5"]].sum()
    )
    rows = []
    for role, name in ROLE_NAMES.items():
        rows.append(
            {
                "role": name,
                "records v4": int(records.loc[role, "included_v4"]),
                "records v5": int(records.loc[role, "included_v5"]),
                "excluded": int(records.loc[role, "included_v4"] - records.loc[role, "included_v5"]),
                "positives made undefined": int(labels.loc[role, "undefined_positives"]),
                "negatives made undefined": int(labels.loc[role, "undefined_negatives"]),
            }
        )
    return pd.DataFrame(rows)


def test_roles() -> pd.DataFrame:
    """
    Rule counts at the test roles per source: records excluded and labels made undefined.

    Returns
    -------
    pd.DataFrame
        One row per rule and condition with a count above zero at a test role.
    """
    rule_flow = pd.read_csv(mv.V5 / "rule_flow.csv")
    rule_flow = rule_flow[rule_flow["role"].isin(mv.TEST_ROLES)].fillna({"condition": ""})
    rows = []
    for (rule, condition), part in rule_flow.groupby(["rule", "condition"], sort=False):
        row = {"rule": rule, "condition": condition}
        for role in mv.TEST_ROLES:
            values = part[part["role"] == role]
            if rule in mv.RECORD_RULES:
                row[ROLE_NAMES[role]] = f"{int(values['selected'].sum())} / {int(values['excluded'].sum())}"
            else:
                sums = (
                    values[
                        [
                            "selected_positives",
                            "selected_negatives",
                            "undefined_positives",
                            "undefined_negatives",
                        ]
                    ]
                    .sum()
                    .astype(int)
                    .tolist()
                )
                row[ROLE_NAMES[role]] = f"{sums[0]}+ {sums[1]}- / {sums[2]}+ {sums[3]}-"
        if any(
            not value.startswith(("0 / 0", "0+ 0- / 0+ 0-"))
            for key, value in row.items()
            if key in ROLE_NAMES.values()
        ):
            rows.append(row)
    return pd.DataFrame(rows)


def support(role: str | None = None) -> pd.DataFrame:
    """
    Positives, negatives and undefined labels per condition and source in v4 and v5.

    Parameters
    ----------
    role : str | None
        One role, or None for every role.

    Returns
    -------
    pd.DataFrame
        Source-level rows with ``v4 -> v5`` per state.
    """
    table = pd.read_csv(mv.V5 / "support_v4_v5.csv")
    table = table[(table["level"] == "source") & ((table["role"] == role) if role else True)]
    out = pd.DataFrame(
        {
            "condition": table["label"].str[2:],
            "role": table["role"].map(ROLE_NAMES),
            "source": table["unit"].map(SOURCE_NAMES),
            "positives": table["positive_v4"].astype(str) + " -> " + table["positive_v5"].astype(str),
            "negatives": table["negative_v4"].astype(str) + " -> " + table["negative_v5"].astype(str),
            "undefined": table["undefined_v4"].astype(str) + " -> " + table["undefined_v5"].astype(str),
            "positive change %": (
                100 * table["positive_change"] / table["positive_v4"].where(table["positive_v4"] > 0)
            ),
            "negative change %": (
                100 * table["negative_change"] / table["negative_v4"].where(table["negative_v4"] > 0)
            ),
        }
    )
    return out


def largest_changes(minimum: float = 5.0) -> pd.DataFrame:
    """
    Source-level cells that lose at least ``minimum`` percent of their positives or negatives.

    Parameters
    ----------
    minimum : float
        Percentage threshold.

    Returns
    -------
    pd.DataFrame
        The rows of ``support`` that meet it.
    """
    table = support()
    keep = (table["positive change %"] <= -minimum) | (table["negative change %"] <= -minimum)
    return table[keep]


def eligibility() -> pd.DataFrame:
    """
    Evaluation-role cells that lose macro eligibility or drop below 100 positives.

    Returns
    -------
    pd.DataFrame
        The rows of ``eligibility_changes.csv``.
    """
    table = pd.read_csv(mv.V5 / "eligibility_changes.csv")
    columns = [
        "label",
        "role",
        "level",
        "unit",
        "positive_v4",
        "positive_v5",
        "negative_v4",
        "negative_v5",
        "loses_eligibility",
        "drops_below_100",
    ]
    return table[columns]


def integrity() -> pd.DataFrame:
    """
    Cell-by-cell comparison of v5 with v4.

    Returns
    -------
    pd.DataFrame
        One row per check.
    """
    data = json.loads((mv.V5 / "integrity.json").read_text())
    rows = [
        {"check": key, "value": data[key]}
        for key in (
            "records",
            "columns",
            "cells_compared",
            "cells_changed",
            "only_defined_to_undefined",
            "changes_only_in_rule_columns",
            "identity_unchanged",
            "included_only_removed",
            "support_v4_reproduced",
        )
    ]
    rows += [
        {"check": f"changed cells in {name}", "value": count}
        for name, count in data["changed_by_column"].items()
    ]
    return pd.DataFrame(rows)


def effect() -> pd.DataFrame:
    """
    Part D selection metric per model, training set and scored set.

    Returns
    -------
    pd.DataFrame
        The rows of ``selection_metric.csv``.
    """
    return pd.read_csv(EFFECT / "selection_metric.csv")


def effect_conditions() -> pd.DataFrame:
    """
    Part D per-condition metric (within-source AUROC averaged over entering sources), v5 minus v4 trained.

    Returns
    -------
    pd.DataFrame
        The rows of ``difference_by_condition.csv``.
    """
    return pd.read_csv(EFFECT / "difference_by_condition.csv").drop(columns=["sources"])


def effect_sources() -> pd.DataFrame:
    """
    Part D within-source AUROC per condition, v5 minus v4 trained, with intervals.

    Returns
    -------
    pd.DataFrame
        The rows of ``difference_by_source.csv``.
    """
    table = pd.read_csv(EFFECT / "difference_by_source.csv")
    return table.assign(source=table["source"].map(SOURCE_NAMES))


SECTIONS = {
    "rules": rules,
    "label-rules": label_rules_by_kind,
    "gate": gate,
    "part-b-totals": part_b_totals,
    "flow": flow,
    "undefined": undefined,
    "totals": totals,
    "test-roles": test_roles,
    "support": support,
    "largest-changes": largest_changes,
    "eligibility": eligibility,
    "integrity": integrity,
    "effect": effect,
    "effect-conditions": effect_conditions,
    "effect-sources": effect_sources,
}


def main() -> None:
    """Print one section, or every section."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--section", choices=sorted(SECTIONS))
    args = parser.parse_args()
    for name in [args.section] if args.section else SECTIONS:
        print(f"## {name}\n\n{markdown(SECTIONS[name]())}\n")


if __name__ == "__main__":
    main()
