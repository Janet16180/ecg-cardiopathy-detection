"""
Print the tables of the MLC-03 label-quality report from its outputs.

Every number is read from ``outputs/mlc03_data_quality_20261111/labels/`` (and the review judgements); the
script only selects, sums, divides and formats. Run from the worktree:
``PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_label_quality_tables [--section NAME]``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ecg_experiment import mlc03_label_quality as lq

OUT = lq.OUTPUT
SHORT = {
    "ptbxl": "PTB-XL",
    "ningbo": "Ningbo",
    "chapman_shaoxing": "Chapman",
    "georgia": "Georgia",
    "cpsc_2018": "CPSC 2018",
    "cpsc_2018_extra": "CPSC-Extra",
}


def markdown(frame: pd.DataFrame) -> str:
    """
    Format a table as GitHub Markdown.

    Parameters
    ----------
    frame : pd.DataFrame
        Table with formatted cells.

    Returns
    -------
    str
        Markdown text.
    """
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def percent(value: float, digits: int = 1) -> str:
    """
    Format a share as a percentage.

    Parameters
    ----------
    value : float
        Share between 0 and 1.
    digits : int
        Decimals.

    Returns
    -------
    str
        Percentage text, or an empty string for NaN.
    """
    return "" if pd.isna(value) else f"{100 * value:.{digits}f}%"


def validation() -> str:
    """
    Measurement error against PTB-XL+ (held-out records), LUDB and synthetic records.

    Returns
    -------
    str
        Markdown tables.
    """
    text = []
    for name, subset, label in (
        (
            "ptbxl_plus_summary.csv",
            "validation outside calibration",
            "PTB-XL+ reference (12SL and Uni-G where they agree), PTB-XL validation records outside the "
            "calibration sample:",
        ),
        ("ludb_summary.csv", None, "LUDB manual delineation (200 records):"),
        ("synthetic_summary.csv", None, "Synthetic records (all noise levels):"),
    ):
        table = pd.read_csv(OUT / "measurement_validation" / name)
        if subset is not None:
            table = table[table["subset"] == subset]
        if "noise_mv" in table:
            table = table[table["noise_mv"].astype(str) == "all"]
        shown = pd.DataFrame(
            {
                "quantity": table["quantity"],
                "method": table["method"],
                "tolerance": table["tolerance"],
                "pairs": table["pairs"].astype(int),
                "failed": (1 - table["measured"]).map(percent),
                "bias (median difference)": table["bias"].map(lambda v: f"{v:.3g}"),
                "median absolute": table["median_abs"].map(lambda v: f"{v:.3g}"),
                "90th percentile absolute": table["p90_abs"].map(lambda v: f"{v:.3g}"),
                "within tolerance": table["within"].map(percent),
            }
        )
        text += [label, markdown(shown)]
    detection = pd.read_csv(OUT / "measurement_validation/synthetic_detection.csv")
    detection = detection.assign(
        flagged_with=detection["flagged_with"].map(percent),
        flagged_without=detection["flagged_without"].map(percent),
    )
    findings = pd.read_csv(OUT / "measurement_validation/ptbxl_plus_findings.csv")
    pivot = findings.pivot_table(
        index=["finding", "reference"], columns="method_b", values="records"
    ).reset_index()
    text += [
        "Synthetic categorical findings:",
        markdown(detection),
        "PTB-XL+ findings where 12SL and Uni-G agree:",
        markdown(pivot),
    ]
    return "\n\n".join(text)


def agreement() -> str:
    """
    Measurement coverage and agreement of methods A and B per source.

    Returns
    -------
    str
        Markdown table.
    """
    table = pd.read_csv(OUT / "measurement_validation/source_agreement.csv")
    rows = []
    for source, part in table.groupby("source", sort=False):
        row = {"source": SHORT[source], "records": int(part["records"].iloc[0])}
        status = part.iloc[0]
        row["A failed (flat lead / no beats)"] = (
            f"{int(np.nan_to_num(status.get('a_status_derivacion_plana', 0)))} / "
            f"{int(np.nan_to_num(status.get('a_status_sin_latidos', 0)))}"
        )
        failed_b = sum(
            int(np.nan_to_num(status.get(f"b_status_{k}", 0))) for k in ("no_dominant", "few_beats", "flat")
        )
        row["B failed"] = failed_b
        for quantity in ("hr", "qrs", "pr", "axis"):
            r = part[part["quantity"] == quantity].iloc[0]
            row[f"{quantity}: agree / consensus"] = (
                f"{percent(r['agree_given_both'])} / {percent(r['consensus'])}"
            )
        row["P wave present (B)"] = percent(part[part["quantity"] == "p_present"].iloc[0]["consensus"])
        rows.append(row)
    return markdown(pd.DataFrame(rows))


def count_share(row: pd.Series, strict: str, tolerant: str) -> str:
    """
    Format two counts with their shares of the measured records.

    Parameters
    ----------
    row : pd.Series
        Row of ``item1_label_vs_measurement.csv``.
    strict, tolerant : str
        Column names.

    Returns
    -------
    str
        ``count (share) / count (share)``.
    """
    measured = max(row["measured"], 1)
    return (
        f"{int(row[strict])} ({percent(row[strict] / measured)}) / "
        f"{int(row[tolerant])} ({percent(row[tolerant] / measured)})"
    )


def item1() -> str:
    """
    Labels contradicted by their measured definition, per condition and source.

    Returns
    -------
    str
        Markdown table: positives outside and negatives inside the definition, strictly and beyond tolerance.
    """
    table = pd.read_csv(OUT / "item1_label_vs_measurement.csv")
    rows = []
    for (condition, source), part in table.groupby(["condition", "source"], sort=False):
        states = {state: part[part["state"] == state] for state in ("positive", "negative")}
        if any(s.empty for s in states.values()):
            continue
        p, n = states["positive"].iloc[0], states["negative"].iloc[0]
        median = p["p5_p25_p50_p75_p95"].strip("[]").split(",")[2].strip()
        rows.append(
            {
                "condition": condition,
                "source": SHORT[source],
                "positives measured": int(p["measured"]),
                "positive median": median,
                "positives outside (strict / beyond tolerance)": count_share(p, "outside", "outside_tol"),
                "negatives measured": int(n["measured"]),
                "negatives inside (strict / beyond tolerance)": count_share(n, "inside", "inside_tol"),
            }
        )
    return markdown(pd.DataFrame(rows))


def records() -> str:
    """
    Open records by source and role, records longer than 10 s, and method B's status.

    Returns
    -------
    str
        Markdown tables read from ``analysis.parquet``.
    """
    table = pd.read_parquet(OUT / "analysis.parquet", columns=["source", "role", "duration_s", "status_b"])
    counts = pd.crosstab(table["source"], table["role"]).reindex(list(SHORT))
    counts["records"] = counts.sum(axis=1)
    counts["longer than 10 s"] = (table["duration_s"] > 10.0 + 1e-6).groupby(table["source"]).sum()
    counts.loc["total"] = counts.sum()
    counts = counts.reset_index().assign(source=lambda f: f["source"].map({**SHORT, "total": "total"}))
    status = table["status_b"].value_counts().rename_axis("method B status").reset_index(name="records")
    return "\n\n".join([markdown(counts), markdown(status)])


def definitions() -> str:
    """
    Measured definition of each condition used by item 1.

    Returns
    -------
    str
        Markdown table of ``item1_definitions.csv``.
    """
    table = pd.read_csv(OUT / "item1_definitions.csv")
    return markdown(table[["condition", "measure", "population", "bound", "tolerance", "reference"]])


def whole() -> str:
    """
    Count the findings of the records longer than 10 s by 10 s window.

    Returns
    -------
    str
        Markdown table of ``item1_whole_record_summary.csv``.
    """
    table = pd.read_csv(OUT / "item1_whole_record_summary.csv")
    return markdown(table.assign(source=table["source"].map(SHORT)))


def item2() -> str:
    """
    Inconsistent source pairs per analysis, implied rate cut-offs and sources not coding a distinction.

    Returns
    -------
    str
        Markdown tables.
    """
    pairs = pd.read_csv(OUT / "item2_source_pairs.csv")
    summary = []
    for name, part in pairs.groupby("name", sort=False):
        bad = part[part["inconsistent"]]
        sources = sorted({SHORT[s] for s in pd.concat([bad["higher"], bad["lower"]])})
        summary.append(
            {
                "analysis": name,
                "bin comparisons": len(part),
                "inconsistent": int(len(bad)),
                "bins with an inconsistent pair": int(
                    bad[["bin_low", "bin_high"]].drop_duplicates().shape[0]
                ),
                "sources involved": ", ".join(sources),
            }
        )
    cutoffs = pd.read_csv(OUT / "item2_implied_cutoffs.csv").dropna(subset=["cutoff_bpm"])
    cutoffs = cutoffs.assign(
        source=cutoffs["source"].map(SHORT),
        cutoff=[
            f"{c:.1f} ({lo:.1f}, {hi:.1f})"
            for c, lo, hi in zip(cutoffs["cutoff_bpm"], cutoffs["ci_low"], cutoffs["ci_high"], strict=True)
        ],
    )
    coding = pd.read_csv(OUT / "item2_separate_coding.csv")
    coding = coding[coding["not_coded_separately"]]
    coding = coding.assign(source=coding["source"].map(SHORT))
    return "\n\n".join(
        [
            markdown(pd.DataFrame(summary)),
            markdown(cutoffs[["condition", "source", "records", "positives", "cutoff"]]),
            markdown(coding[["condition", "source", "sibling", "sibling_codes", "positives", "negatives"]]),
        ]
    )


def prevalence(name: str) -> str:
    """
    Tabulate the matched prevalence of one analysis, one row per bin and one column per source.

    Parameters
    ----------
    name : str
        Analysis name in ``item2_matched_prevalence.csv``.

    Returns
    -------
    str
        Markdown table of ``prevalence (positives/defined)``; bins with fewer than ``lq.MIN_BIN`` records are
        bracketed.
    """
    table = pd.read_csv(OUT / "item2_matched_prevalence.csv")
    table = table[table["name"] == name]
    table = table.assign(
        cell=[
            (f"{p:.2f} ({k}/{n})" if n >= lq.MIN_BIN else f"[{k}/{n}]") if n else ""
            for p, k, n in zip(table["prevalence"], table["positives"], table["defined"], strict=True)
        ],
        bin=[f"{lo:g} to {hi:g}" for lo, hi in zip(table["bin_low"], table["bin_high"], strict=True)],
    )
    wide = table.pivot_table(
        index="bin", columns="source", values="cell", aggfunc="first", sort=False
    ).reset_index()
    return markdown(wide.rename(columns=SHORT).fillna(""))


def item3() -> str:
    """
    Label pairs that cannot both hold, with counts per source (train + validation).

    Returns
    -------
    str
        Markdown table of nonzero pairs.
    """
    table = pd.read_csv(OUT / "item3_contradictions.csv")
    table["records"] = table["train"] + table["validation"]
    wide = table.pivot_table(
        index=["first", "second", "reason"], columns="source", values="records", aggfunc="sum", sort=False
    ).reset_index()
    wide["total"] = wide[[c for c in wide.columns if c in SHORT]].sum(axis=1)
    return markdown(wide.rename(columns=SHORT))


def item4() -> str:
    """
    Confident-learning estimates per condition and source, and the review tally.

    Returns
    -------
    str
        Markdown tables.
    """
    table = pd.read_csv(OUT / "item4_confident_learning.csv")
    table = table.assign(
        source=table["source"].map(lambda s: SHORT.get(s, s)),
        estimate=[
            f"{fp:.0f} / {fn:.0f} ({percent(share)})"
            for fp, fn, share in zip(
                table["estimated_false_positives"],
                table["estimated_false_negatives"],
                table["estimated_wrong_share"],
                strict=True,
            )
        ],
    )
    wide = table.pivot_table(
        index="condition", columns="source", values="estimate", aggfunc="first", sort=False
    ).reset_index()
    return markdown(wide.fillna(""))


def review() -> str:
    """
    Judgements of the sheets read by eye.

    By condition, by source, by rhythm statement and against the Part A signal flags.

    Returns
    -------
    str
        Markdown tables.
    """
    tally = pd.read_csv(OUT / "item4_review_tally.csv")
    tally["sheets"] = tally[["label error", "model error", "cannot tell"]].sum(axis=1)
    tally["label error share"] = (tally["label error"] / tally["sheets"]).map(percent)
    by_source = pd.read_csv(OUT / "item4_review_by_source.csv")
    by_source["source"] = by_source["source"].map(SHORT)
    rhythm = pd.read_csv(OUT / "item4_review_rhythm_statement.csv")
    signal = pd.read_csv(OUT / "item4_review_signal_flags.csv")
    columns = [
        "any_reliable_flag",
        "baseline_wander",
        "baseline_step",
        "isolated_spike",
        "flat_lead",
        "contact_loss",
    ]
    signal = signal.assign(**{column: signal[column].map(percent) for column in columns})
    counted = ["label error", "model error", "cannot tell", "sheets"]
    totals = tally.groupby("kind")[counted].sum().reset_index()
    return "\n\n".join(
        [
            markdown(tally),
            markdown(totals),
            markdown(by_source),
            markdown(rhythm),
            markdown(signal[["group", "records", *columns]]),
        ]
    )


def item5() -> str:
    """
    Undefined entries by rule, with the share meeting the measured definition.

    Returns
    -------
    str
        Markdown table.
    """
    table = pd.read_csv(OUT / "item5_undefined.csv")
    table = table.assign(
        records=table["train"] + table["validation"],
        source=table["source"].map(SHORT),
        meeting=table["share_meeting_definition"].map(percent),
        positives=table["share_positives"].map(percent),
        negatives=table["share_negatives"].map(percent),
    )
    return markdown(
        table[
            ["condition", "source", "reason", "records", "in_population", "meeting", "positives", "negatives"]
        ]
    )


def rules_summary() -> str:
    """
    Candidate cleaning rules summed over sources and roles.

    Returns
    -------
    str
        Markdown table: per rule and condition, the sources it touches, the positives and negatives it
        would set undefined, and their share of the defined positives and negatives of those sources.
    """
    table = pd.read_csv(OUT / "rules_candidate_counts.csv")
    table = table.assign(
        touched=(table["positives_to_undefined"] + table["negatives_to_undefined"]) > 0,
        short=table["source"].map(SHORT),
    )
    rows = []
    for (rule, condition), part in table.groupby(["rule", "condition"], sort=False):
        touched = part[part["touched"]]
        sources = touched.groupby("short", sort=False).size().index
        within = part[part["short"].isin(sources)]
        positives = touched["positives_to_undefined"].sum()
        negatives = touched["negatives_to_undefined"].sum()
        rows.append(
            {
                "rule": rule,
                "condition": condition,
                "sources": ", ".join(sources),
                "positives to undefined": f"{positives} ({percent(positives / within['positives'].sum())})"
                if within["positives"].sum()
                else str(positives),
                "negatives to undefined": f"{negatives} ({percent(negatives / within['negatives'].sum())})",
            }
        )
    return markdown(pd.DataFrame(rows))


def rules() -> str:
    """
    Candidate cleaning rules with the entries they would set undefined, by source and role.

    Returns
    -------
    str
        Markdown table.
    """
    table = pd.read_csv(OUT / "rules_candidate_counts.csv")
    table["entries"] = [
        f"{p} / {n}"
        for p, n in zip(table["positives_to_undefined"], table["negatives_to_undefined"], strict=True)
    ]
    wide = table.pivot_table(
        index=["rule", "condition", "source"], columns="role", values="entries", aggfunc="first", sort=False
    ).reset_index()
    return markdown(wide.assign(source=wide["source"].map(SHORT)).fillna(""))


def integrity() -> str:
    """
    P0 integrity checks from the ``oof`` receipt.

    Returns
    -------
    str
        JSON text of the checks.
    """
    receipt = json.loads((OUT / "receipts/oof.json").read_text())
    return json.dumps(receipt["integrity"], indent=1)


SECTIONS = {
    "records": records,
    "validation": validation,
    "agreement": agreement,
    "definitions": definitions,
    "item1": item1,
    "whole": whole,
    "item2": item2,
    "item3": item3,
    "item4": item4,
    "review": review,
    "item5": item5,
    "rules_summary": rules_summary,
    "rules": rules,
    "integrity": integrity,
}


def main() -> None:
    """Print one section or every section."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--section", choices=[*SECTIONS, "prevalence"])
    parser.add_argument("--name", help="analysis name for --section prevalence")
    arguments = parser.parse_args()
    if arguments.section == "prevalence":
        print(prevalence(arguments.name))
        return
    for name, function in SECTIONS.items():
        if arguments.section in (None, name):
            print(f"## {name}\n\n{function()}\n")


if __name__ == "__main__":
    main()
