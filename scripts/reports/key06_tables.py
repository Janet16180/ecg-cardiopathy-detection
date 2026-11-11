"""
Print the tables of the KEY-06 results report from the step outputs; nothing is recomputed.

Example
-------
PYTHONPATH=. .venv/bin/python -m scripts.reports.key06_tables --section variance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import key06_embedding_structure as key06

RULES = ("rule_stable", "rule_source", "rule_replicated", "rule_artefact", "rule_labelled")


def markdown(frame: pd.DataFrame, digits: int = 3) -> str:
    """
    Format a table as Markdown.

    Parameters
    ----------
    frame : pd.DataFrame
        Table.
    digits : int
        Decimal places of floats.

    Returns
    -------
    str
        Markdown text.
    """

    def cell(value: object) -> str:
        if isinstance(value, float | np.floating):
            return "" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False)
    )
    return "\n".join(lines)


def read_json(path: Path) -> dict:
    """
    Read a JSON file.

    Parameters
    ----------
    path : Path
        File.

    Returns
    -------
    dict
        Content.
    """
    return json.loads(path.read_text())


def section_integrity(output: Path) -> str:
    """
    Show the integrity checks.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.

    Returns
    -------
    str
        Markdown.
    """
    content = read_json(output / "integrity" / "integrity.json")
    rows = pd.DataFrame([{"check": name, "value": value} for name, value in content["checks"].items()])
    records = pd.DataFrame(content["records"]).T.reset_index(names="source")
    extra = (
        f"\n\nStatements outside the 14 conditions: {content['statements_outside']}; with at least 30 train "
        f"records: {content['statements_at_least_30_train']}."
    )
    return markdown(rows, 16) + "\n\n" + markdown(records, 0) + extra


def section_variance(output: Path) -> str:
    """
    Show the share of embedding variance per factor group.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.

    Returns
    -------
    str
        Markdown.
    """
    table = pd.read_csv(output / "variance" / "variance_accounting.csv")
    columns = [
        "group",
        "columns",
        "r2_alone_validation",
        "r2_unique_validation",
        "r2_alone_validation_kept",
        "r2_unique_validation_kept",
        "r2_alone_train",
    ]
    return markdown(table[columns]) + f"\n\nKept components (90%): {int(table['kept_components'].iloc[0])}."


def section_components(output: Path, count: int = 10) -> str:
    """
    Show the validation R^2 of each factor group on the leading principal components.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.
    count : int
        Components shown.

    Returns
    -------
    str
        Markdown.
    """
    return markdown(pd.read_csv(output / "variance" / "components.csv").head(count))


def section_reverse(output: Path) -> str:
    """
    Show how well each factor is predicted from the embedding, within source.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.

    Returns
    -------
    str
        Markdown.
    """
    source = read_json(output / "variance" / "source_prediction.json")
    binary = pd.read_csv(output / "variance" / "factor_prediction_binary.csv")
    columns = [
        "kind",
        "name",
        "train_positives",
        "validation_positives",
        "auroc",
        *[f"auroc_{name}" for name in key06.VALIDATION_SOURCES if f"auroc_{name}" in binary],
    ]
    continuous = pd.read_csv(output / "variance" / "factor_prediction_continuous.csv")
    head = continuous[continuous["name"].isin(["hr_bpm", "age"])]
    others = continuous[~continuous["name"].isin(["hr_bpm", "age"])]
    summary = pd.DataFrame(
        [
            {
                "measurements": len(others),
                **{
                    f"within_source_r2_q{q}": float(np.quantile(others["within_source_r2"], q / 100))
                    for q in (0, 10, 50, 90, 100)
                },
                "above_0.5": int((others["within_source_r2"] > 0.5).sum()),
            }
        ]
    )
    ranked = others.sort_values("within_source_r2")
    text = (
        f"Source from the embedding (multinomial logistic, train to validation): accuracy "
        f"{source['accuracy']:.4f}, balanced accuracy {source['balanced_accuracy']:.4f}, majority share "
        f"{source['majority_share']:.4f}."
    )
    return "\n\n".join(
        [
            text,
            markdown(binary[columns]),
            markdown(head),
            markdown(summary),
            "Lowest:\n\n" + markdown(ranked.head(8)),
            "Highest:\n\n" + markdown(ranked.tail(8)),
        ]
    )


def section_residual(output: Path) -> str:
    """
    Summarize the residual space.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.

    Returns
    -------
    str
        Markdown.
    """
    summary = read_json(output / "residual" / "residual_summary.json")
    norms = pd.DataFrame(
        {"raw": summary["source_mean_norm_raw"], "residual": summary["source_mean_norm_residual"]}
    ).reset_index(names="source")
    probe = summary["source_prediction_residual"]
    return (
        f"Variance removed on validation: {summary['variance_removed_validation']:.4f}; left: "
        f"{summary['residual_variance_share_validation']:.4f}. Source from the residual space: accuracy "
        f"{probe['accuracy']:.4f}, balanced {probe['balanced_accuracy']:.4f}, majority "
        f"{probe['majority_share']:.4f}.\n\nNorm of each source's validation mean:\n\n" + markdown(norms)
    )


def section_solutions(output: Path) -> str:
    """
    Count clusters, noise and the clusters passing each rule, per solution.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.

    Returns
    -------
    str
        Markdown.
    """
    clusters = pd.read_csv(output / "describe" / "clusters.csv", keep_default_na=False, na_values=[""])
    summary = read_json(output / "clusters" / "summary.json")
    rows = []
    for solution, part in clusters.groupby("solution", sort=False):
        subset, space, method = solution.split("/")
        info = summary[f"{subset}__{space}/{method}"]
        stable_source = part["rule_stable"] & part["rule_source"]
        rows.append(
            {
                "solution": solution,
                "clusters": len(part),
                "noise_train": info["noise_train"],
                **{rule.removeprefix("rule_"): int(part[rule].sum()) for rule in RULES},
                "stable_and_source": int(stable_source.sum()),
                "stable_source_replicated": int((stable_source & part["rule_replicated"]).sum()),
                "artefact_min5": int(part["rule_artefact_min5"].sum()),
                "candidates": int(part["candidate"].sum()),
            }
        )
    frame = pd.DataFrame(rows)
    total = frame.drop(columns=["solution", "noise_train"]).sum().to_dict()
    frame = pd.concat([frame, pd.DataFrame([{"solution": "total", "noise_train": np.nan, **total}])])
    return markdown(frame)


def cluster_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Select the report columns of cluster rows.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows of ``clusters.csv``.

    Returns
    -------
    pd.DataFrame
        Compact columns.
    """
    families = [
        column for column in frame if column.startswith("family_") and column != "families_qualifying"
    ]
    shares = frame[families].fillna(0).div(frame["size"], axis=0)
    family_text = shares.apply(
        lambda row: ", ".join(
            f"{name.removeprefix('family_')} {value:.2f}"
            for name, value in row.sort_values(ascending=False).items()
            if value >= 0.05
        ),
        axis=1,
    )
    return pd.DataFrame(
        {
            "solution": frame["solution"],
            "c": frame["cluster"],
            "n train": frame["size"],
            "n val": frame["size_validation"],
            "jaccard": frame["median_jaccard"],
            "families": family_text,
            "top label": frame["top_label"] + " " + frame["top_label_share"].map("{:.2f}".format),
            "strongest (train/val log OR)": frame["strongest_feature"].fillna("")
            + " "
            + frame["strongest_log_odds_train"].map("{:+.2f}".format)
            + "/"
            + frame["strongest_log_odds_validation"].map("{:+.2f}".format),
            "worst flag": frame["worst_flag"],
            "HR": frame["median_hr_bpm"].map("{:.0f}".format),
            "QRS": frame["median_qrs_dur_ms"].map("{:.0f}".format),
            "failed": frame["failed_rules"].fillna(""),
        }
    )


def section_clusters(output: Path, which: str) -> str:
    """
    Show the cluster table: ``stable`` (stable and not one source), ``candidates`` or ``all``.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.
    which : str
        Rows shown.

    Returns
    -------
    str
        Markdown.
    """
    clusters = pd.read_csv(output / "describe" / "clusters.csv")
    if which == "stable":
        clusters = clusters[clusters["rule_stable"] & clusters["rule_source"]]
    elif which == "candidates":
        clusters = clusters[clusters["candidate"]]
    return markdown(cluster_columns(clusters), 2)


def section_statements_of(output: Path, which: str) -> str:
    """
    Show the enriched statements and measurements of the clusters of ``section_clusters``.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.
    which : str
        ``stable`` or ``candidates``.

    Returns
    -------
    str
        Markdown.
    """
    clusters = pd.read_csv(output / "describe" / "clusters.csv", keep_default_na=False, na_values=[""])
    keep = (
        clusters["candidate"] if which == "candidates" else clusters["rule_stable"] & clusters["rule_source"]
    )
    part = clusters[keep]
    return markdown(
        pd.DataFrame(
            {
                "solution": part["solution"],
                "c": part["cluster"],
                "top statements (OR train/val [members with it/members])": part["top_statements"].fillna(""),
                "top measurements (within-source SMD)": part["top_measurements"].fillna(""),
            }
        )
    )


def section_cross_encoder(output: Path) -> str:
    """
    Show the ECG-CPC and xECG membership AUROC of the checked clusters.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.

    Returns
    -------
    str
        Markdown.
    """
    table = pd.read_csv(output / "cross_encoder" / "cross_encoder.csv")
    if table.empty:
        return "No cluster was checked."
    wide = table.pivot_table(
        index=["solution", "cluster", "candidate"], columns="encoder", values=["auroc", "ci_low", "ci_high"]
    ).reset_index()
    wide.columns = [" ".join(str(part) for part in column if part) for column in wide.columns]
    counts = table[table["encoder"] == "cpc"][
        ["solution", "cluster", "train_positives", "validation_positives"]
    ]
    return markdown(wide.merge(counts, on=["solution", "cluster"]))


def section_outliers(output: Path) -> str:
    """
    Describe the outliers of each space: sources, flags, labels and enriched statements.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.

    Returns
    -------
    str
        Markdown.
    """
    parts = []
    for space in key06.SPACES:
        directory = output / "outliers" / space
        profile = pd.read_csv(directory / "profile.csv").set_index("cluster")
        flags = pd.read_csv(directory / "flags.csv")
        enrichment = pd.read_csv(directory / "enrichment.csv")
        sources = profile[[c for c in profile if c.startswith("source_")]].T.rename(
            columns={0: "outliers", -1: "others"}
        )
        top = enrichment[(enrichment["q_value"] < key06.BH_LEVEL)].sort_values(
            "log_odds_ratio", ascending=False
        )
        parts.extend(
            [
                f"### {space}",
                markdown(sources.reset_index(names="source"), 0),
                markdown(flags[flags["inside"] > 0], 2),
                markdown(
                    profile[
                        ["size", "any_flag_share", "median_hr_bpm", "median_qrs_dur_ms", "age_median"]
                    ].reset_index(),
                    2,
                ),
                markdown(top[["kind", "feature", "a", "members", "odds_ratio", "q_value"]].head(15), 3),
            ]
        )
    return "\n\n".join(parts)


def section_statements(output: Path, rule: bool) -> str:
    """
    List the statements outside the 14 conditions by within-source AUROC.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.
    rule : bool
        True for the protocol's list (100 train positives, two families); False for statements with 100
        train positives in one family only.

    Returns
    -------
    str
        Markdown.
    """
    table = pd.read_csv(output / "statements" / "statements.csv")
    statements = table[table["kind"] == "statement"]
    statements = statements[statements["probe_rule"] == rule]
    wide = statements.pivot_table(index="feature", columns="space", values=["auroc", "ci_low", "ci_high"])
    wide.columns = [f"{value} {space}" for value, space in wide.columns]
    info = statements.drop_duplicates("feature").set_index("feature")
    counts = (
        info[[f"train_{s}" for s in key06.VALIDATION_SOURCES]].astype(int).astype(str).agg("/".join, axis=1)
    )
    wide = wide.assign(
        description=info["description"],
        mapping=info["mapping"].fillna(""),
        train_positives=info["train_positives_presence"],
        validation_positives=info["validation_positives"],
        train_by_source=counts,
    ).sort_values("ci_low raw", ascending=False)
    header = "train_by_source order: " + "/".join(key06.VALIDATION_SOURCES)
    return header + "\n\n" + markdown(wide.reset_index(), 3)


def section_labels(output: Path) -> str:
    """
    Show the 14 labels probed the same way, for reference.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.

    Returns
    -------
    str
        Markdown.
    """
    table = pd.read_csv(output / "statements" / "statements.csv")
    labels = table[table["kind"] == "label"]
    wide = labels.pivot_table(index="feature", columns="space", values=["auroc", "ci_low", "ci_high"])
    wide.columns = [f"{value} {space}" for value, space in wide.columns]
    return markdown(wide.reset_index(), 3)


def section_notes(output: Path) -> str:
    """
    Show the analyst's recorded readings of the review sheets.

    Parameters
    ----------
    output : Path
        KEY-06 output directory.

    Returns
    -------
    str
        Markdown.
    """
    path = output / "review" / "notes.csv"
    if not path.exists():
        return "No notes recorded."
    return markdown(pd.read_csv(path, keep_default_na=False).drop(columns="written"))


SECTIONS = {
    "integrity": section_integrity,
    "variance": section_variance,
    "components": section_components,
    "reverse": section_reverse,
    "residual": section_residual,
    "solutions": section_solutions,
    "clusters-stable": lambda output: section_clusters(output, "stable"),
    "clusters-candidates": lambda output: section_clusters(output, "candidates"),
    "statements-stable": lambda output: section_statements_of(output, "stable"),
    "statements-candidates": lambda output: section_statements_of(output, "candidates"),
    "cross-encoder": section_cross_encoder,
    "outliers": section_outliers,
    "statements": lambda output: section_statements(output, True),
    "statements-one-family": lambda output: section_statements(output, False),
    "labels": section_labels,
    "notes": section_notes,
}


def main(argv: list[str] | None = None) -> None:
    """
    Print one section or all.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--section", choices=sorted(SECTIONS))
    parser.add_argument("--output", type=Path, default=key06.OUTPUT)
    args = parser.parse_args(argv)
    for name, function in SECTIONS.items():
        if args.section in (None, name):
            print(f"## {name}\n\n{function(args.output)}\n")


if __name__ == "__main__":
    main()
