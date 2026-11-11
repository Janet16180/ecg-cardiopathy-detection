"""
Print the tables of the KEY-05 report from its outputs.

Every number of ``docs/key05-condition-keys-results-2026-11-11.md`` is printed by one section of this script.
It computes no metric: it reads the files under ``outputs/key05_condition_keys_20261111/``, selects rows,
takes differences, medians and counts of stored values. Run
``PYTHONPATH=. .venv/bin/python -m scripts.reports.key05_tables --section <name>``; without ``--section``
every section is printed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/key05_condition_keys_20261111"
SINGLE = ("K1", "K2", "K3", "K4", "K5", "K6")
K7 = ("K7-m2", "K7-m4", "K7-m8")
SHORT = {
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
RELATED = (
    ("complete_rbbb", "incomplete_rbbb"),
    ("incomplete_rbbb", "complete_rbbb"),
    ("sinus_tachycardia", "af_or_flutter"),
    ("af_or_flutter", "sinus_tachycardia"),
    ("lafb", "left_axis_deviation"),
    ("left_axis_deviation", "lafb"),
    ("lvh", "st_depression"),
    ("lvh", "t_wave_abnormality"),
    ("st_depression", "lvh"),
    ("t_wave_abnormality", "lvh"),
)


def markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    """
    Format a table as Markdown.

    Parameters
    ----------
    frame : pd.DataFrame
        Table.
    digits : int
        Decimals of floats.

    Returns
    -------
    str
        Markdown table.
    """

    def cell(value: object) -> str:
        if isinstance(value, float | np.floating):
            return "" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def ci(value: float, low: float, high: float, signed: bool = False) -> str:
    """
    Format a value with its interval.

    Parameters
    ----------
    value, low, high : float
        Value and 95% interval.
    signed : bool
        Show the sign.

    Returns
    -------
    str
        ``value (low, high)``.
    """
    form = "{:+.4f}" if signed else "{:.4f}"
    return f"{form.format(value)} ({form.format(low)}, {form.format(high)})"


def read_json(name: str) -> dict:
    """
    Read a JSON output.

    Parameters
    ----------
    name : str
        File name under the output directory.

    Returns
    -------
    dict
        Content.
    """
    return json.loads((OUT / name).read_text())


def selection() -> dict:
    """
    Read the selection, from ``selection.json``.

    Returns
    -------
    dict
        ``selected_single``, ``selected_k7`` and ``metric``.
    """
    return read_json("selection.json")


def section_integrity() -> None:
    """Integrity checks."""
    checks = read_json("integrity.json")
    print(json.dumps(checks, indent=2))


def section_constructions() -> None:
    """Both macro AUROCs of every construction on validation."""
    table = pd.read_csv(OUT / "constructions.csv")
    rows = [
        {
            "construction": r.model,
            "selection metric": ci(r.selection, r.selection_ci_low, r.selection_ci_high),
            "minus P0": ci(
                r.selection_minus_P0, r.selection_minus_P0_ci_low, r.selection_minus_P0_ci_high, True
            ),
            "pooled macro": ci(r.pooled, r.pooled_ci_low, r.pooled_ci_high),
            "pooled minus P0": ci(
                r.pooled_minus_P0, r.pooled_minus_P0_ci_low, r.pooled_minus_P0_ci_high, True
            ),
        }
        for r in table.itertuples(index=False)
    ]
    print(markdown(pd.DataFrame(rows)))


def condition_wide(metric: str, models: tuple[str, ...]) -> pd.DataFrame:
    """
    Per-condition AUROC of several models side by side.

    Parameters
    ----------
    metric : str
        ``within`` or ``pooled``.
    models : tuple[str, ...]
        Model ids.

    Returns
    -------
    pd.DataFrame
        One row per condition.
    """
    table = pd.read_csv(OUT / "conditions.csv")
    wide = table.pivot(index="condition", columns="model", values=metric).loc[list(SHORT), list(models)]
    wide.index = [SHORT[c] for c in wide.index]
    return wide.reset_index(names="condition")


def section_conditions() -> None:
    """Within-source per-condition AUROC of every construction."""
    print(markdown(condition_wide("within", (*SINGLE, *K7, "N1", "P0")), 3))


def section_conditions_pooled() -> None:
    """Pooled per-condition AUROC of every construction."""
    print(markdown(condition_wide("pooled", (*SINGLE, *K7, "N1", "P0")), 3))


def section_conditions_selected() -> None:
    """Compare the selected single key and K7 with P0 per condition, within-source, with intervals."""
    choice = selection()
    table = pd.read_csv(OUT / "conditions.csv")
    rows = []
    for condition in SHORT:
        row = {"condition": SHORT[condition]}
        for model in (choice["selected_single"], choice["selected_k7"], "N1"):
            r = table[(table.model == model) & (table.condition == condition)].iloc[0]
            row[model] = f"{r.within:.3f}"
            row[f"{model} minus P0"] = ci(
                r.within_minus_P0, r.within_minus_P0_ci_low, r.within_minus_P0_ci_high, True
            )
        row["P0"] = f"{table[(table.model == 'P0') & (table.condition == condition)].iloc[0].within:.3f}"
        rows.append(row)
    print(markdown(pd.DataFrame(rows)))


def section_average_precision() -> None:
    """Average precision per condition: pooled, and the mean over the sources where the condition enters."""
    table = pd.read_csv(OUT / "per_source.csv")
    models = [*SINGLE, *K7, "N1", "P0"]
    pooled = table[table.source == "pooled"].pivot(
        index="condition", columns="model", values="average_precision"
    )
    within = table[(table.source != "pooled") & table.enters_macro]
    within = within.groupby(["condition", "model"])["average_precision"].mean().unstack("model")
    for name, wide in (("pooled", pooled), ("mean within source", within)):
        wide = wide.loc[list(SHORT), models]
        wide.index = [SHORT[c] for c in wide.index]
        print(name)
        print(markdown(wide.reset_index(names="condition"), 3))
        print(markdown(wide.mean().rename("mean over conditions").rename_axis("model").reset_index(), 3))
        print()


def section_per_source() -> None:
    """Macro AUROC of every construction within each validation source."""
    table = pd.read_csv(OUT / "per_source.csv")
    table = table[table.enters_macro]
    macro = table.groupby(["source", "model"])["auroc"].mean().unstack("model")
    counts = table[table.model == "P0"].groupby("source").size().rename("conditions")
    print(markdown(macro[[*SINGLE, *K7, "N1", "P0"]].join(counts).reset_index(), 3))


def section_ladder() -> None:
    """Macro gains of each ladder step, for every K7 variant."""
    table = pd.read_csv(OUT / "ladder.csv")
    macro = table[table.condition == "macro"]
    rows = [
        {"K7": r.k7, "step": r.step, "metric": r.metric, "gain": ci(r.gain, r.ci_low, r.ci_high, True)}
        for r in macro.itertuples(index=False)
    ]
    print(markdown(pd.DataFrame(rows)))


def section_ladder_conditions() -> None:
    """Within-source per-condition gains of the ladder with the selected K7."""
    k7 = selection()["selected_k7"]
    table = pd.read_csv(OUT / "ladder.csv")
    table = table[(table.k7 == k7) & (table.metric == "within") & (table.condition != "macro")]
    wide = table.assign(
        text=[ci(g, lo, hi, True) for g, lo, hi in zip(table.gain, table.ci_low, table.ci_high, strict=True)]
    )
    wide = wide.pivot(index="condition", columns="step", values="text")
    steps = list(dict.fromkeys(table.step))
    wide = wide.loc[list(SHORT), steps]
    wide.index = [SHORT[c] for c in wide.index]
    print(markdown(wide.reset_index(names="condition")))
    gains = table.pivot(index="condition", columns="step", values="gain").loc[list(SHORT), steps]
    largest = pd.DataFrame(
        {
            "condition": [SHORT[c] for c in gains.index],
            "largest gain": gains.idxmax(axis=1).to_numpy(),
            "largest loss": gains.idxmin(axis=1).to_numpy(),
        }
    )
    print()
    print(markdown(largest))


def section_cosine_p0() -> None:
    """Cosine of each key with P0's weight vector."""
    table = pd.read_csv(OUT / "cosine_p0.csv")
    wide = table.pivot(index="condition", columns="construction", values="cosine").loc[list(SHORT)]
    wide = wide[[*SINGLE, *K7]]
    wide.index = [SHORT[c] for c in wide.index]
    print(markdown(wide.reset_index(names="condition"), 3))
    print()
    print(markdown(wide.median().rename("median").rename_axis("construction").reset_index(), 3))


def section_pairs_related() -> None:
    """Print the related pairs for the selected single key, raw and projected, with P0."""
    model = selection()["selected_single"]
    table = pd.read_csv(OUT / "pairs.csv")
    rows = []
    for key, target in RELATED:
        part = table[(table.key == key) & (table.target == target)]
        if part.empty:
            rows.append(
                {"key A": SHORT[key], "target B": SHORT[target], "B positives among A negatives": "<100"}
            )
            continue
        raw = part[(part.model == model) & ~part.projected].iloc[0]
        projected = part[(part.model == model) & part.projected].iloc[0]
        p0 = part[part.model == "P0"].iloc[0]
        rows.append(
            {
                "key A": SHORT[key],
                "target B": SHORT[target],
                "B positives among A negatives": raw.positives,
                f"{model} pooled": f"{raw.auroc:.3f}",
                f"{model} within": f"{raw.auroc_within:.3f}",
                f"{model} projected within": f"{projected.auroc_within:.3f}",
                "P0 within": f"{p0.auroc_within:.3f}",
            }
        )
    print(markdown(pd.DataFrame(rows)))


def section_pairs_summary() -> None:
    """How many pairs each model's keys respond to, raw and projected."""
    table = pd.read_csv(OUT / "pairs.csv")
    rows = []
    for (model, projected), part in table.groupby(["model", "projected"]):
        values = part.auroc_within.dropna()
        rows.append(
            {
                "model": model,
                "projected": projected,
                "pairs": len(part),
                "with within AUROC": len(values),
                "above 0.70": int((values > 0.70).sum()),
                "below 0.30": int((values < 0.30).sum()),
                "median abs(AUROC - 0.5)": float(np.median(np.abs(values - 0.5))),
            }
        )
    print(markdown(pd.DataFrame(rows), 3))


def section_pairs_top() -> None:
    """Print the pairs the selected single key responds to most, raw."""
    model = selection()["selected_single"]
    table = pd.read_csv(OUT / "pairs.csv")
    part = table[(table.model == model) & ~table.projected].dropna(subset=["auroc_within"])
    part = (
        part.assign(distance=(part.auroc_within - 0.5).abs())
        .sort_values("distance", ascending=False)
        .head(15)
    )
    projected = table[(table.model == model) & table.projected].set_index(["key", "target"])["auroc_within"]
    p0 = table[table.model == "P0"].set_index(["key", "target"])["auroc_within"]
    rows = [
        {
            "key A": SHORT[r.key],
            "target B": SHORT[r.target],
            "positives": r.positives,
            "within": f"{r.auroc_within:.3f}",
            "projected": f"{projected[(r.key, r.target)]:.3f}",
            "P0": f"{p0[(r.key, r.target)]:.3f}",
        }
        for r in part.itertuples(index=False)
    ]
    print(markdown(pd.DataFrame(rows)))


def section_projected_own() -> None:
    """Own within-source AUROC of each single key before and after projecting the other 13 keys out."""
    table = pd.read_csv(OUT / "projected_own_auroc.csv")
    wide = table.pivot_table(index="condition", columns=["model", "keys"], values="within").loc[list(SHORT)]
    rows = []
    for model in SINGLE:
        change = wide[(model, "projected")] - wide[(model, "raw")]
        rows.append(
            {
                "model": model,
                "median change": float(change.median()),
                "min": float(change.min()),
                "max": float(change.max()),
                "macro raw": float(wide[(model, "raw")].mean()),
                "macro projected": float(wide[(model, "projected")].mean()),
            }
        )
    print(markdown(pd.DataFrame(rows)))
    model = selection()["selected_single"]
    per = pd.DataFrame(
        {
            "condition": [SHORT[c] for c in wide.index],
            "raw": wide[(model, "raw")].to_numpy(),
            "projected": wide[(model, "projected")].to_numpy(),
        }
    )
    print()
    print(markdown(per, 3))


def section_sources() -> None:
    """Source classifier, cosines of keys with source directions and between-source score variance."""
    summary = read_json("sources.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "iterations"}, indent=2))
    cosines = pd.read_csv(OUT / "source_cosines.csv")
    cosines["abs"] = cosines.cosine.abs()
    largest = cosines.groupby(["construction", "condition", "centre"])["abs"].max().groupby("construction")
    print(
        markdown(
            pd.DataFrame({"median of max abs cosine": largest.median(), "max": largest.max()})
            .loc[[*SINGLE, *K7]]
            .reset_index(),
            3,
        )
    )
    share = pd.read_csv(OUT / "source_variance_share.csv")
    wide = share.pivot(index="condition", columns="model", values="between_source_share").loc[list(SHORT)]
    wide.index = [SHORT[c] for c in wide.index]
    print()
    print(markdown(wide[[*SINGLE, *K7, "N1", "P0"]].reset_index(names="condition"), 3))
    print()
    print(markdown(wide.median().rename("median share").rename_axis("model").reset_index(), 3))


def section_sources_selected() -> None:
    """Cosine of the selected single key with each source direction."""
    model = selection()["selected_single"]
    cosines = pd.read_csv(OUT / "source_cosines.csv")
    wide = cosines[cosines.construction == model].pivot(index="condition", columns="source", values="cosine")
    wide = wide.loc[list(SHORT)]
    wide.index = [SHORT[c] for c in wide.index]
    print(markdown(wide.reset_index(names="condition"), 3))


def section_pooled_within() -> None:
    """Pooled minus within-source per-condition AUROC of every construction."""
    table = pd.read_csv(OUT / "conditions.csv")
    table = table.assign(difference=table.pooled - table.within)
    wide = table.pivot(index="condition", columns="model", values="difference").loc[list(SHORT)]
    wide.index = [SHORT[c] for c in wide.index]
    print(markdown(wide[[*SINGLE, *K7, "N1", "P0"]].reset_index(names="condition"), 3))


def reference_summary(metric: str) -> pd.DataFrame:
    """
    Median and 5-95% range of the reference-count curves.

    Parameters
    ----------
    metric : str
        ``within`` or ``pooled``.

    Returns
    -------
    pd.DataFrame
        One row per variant, count and condition.
    """
    table = pd.read_csv(OUT / "references.csv")
    grouped = table.groupby(["variant", "count", "condition"])[metric]
    return pd.DataFrame(
        {"median": grouped.median(), "p05": grouped.quantile(0.05), "p95": grouped.quantile(0.95)}
    ).reset_index()


def count_order(values: pd.Series) -> pd.Series:
    """
    Sort key of reference counts, ``all`` last.

    Parameters
    ----------
    values : pd.Series
        Counts as text.

    Returns
    -------
    pd.Series
        Integers.
    """
    return values.map(lambda v: 10**9 if str(v) == "all" else int(v))


def section_references() -> None:
    """Macro over conditions of the median within-source AUROC per variant and count."""
    summary = reference_summary("within")
    macro = summary.groupby(["variant", "count"])[["median", "p05", "p95"]].mean().reset_index()
    macro = macro.sort_values(["variant", "count"], key=lambda s: count_order(s) if s.name == "count" else s)
    print(markdown(macro))


def section_references_conditions() -> None:
    """Per-condition median within-source AUROC (5-95%) of K2 and K4-all at each count."""
    summary = reference_summary("within")
    for variant in ("K2", "K4-all", "K4-drawn"):
        part = summary[summary.variant == variant]
        text = part.assign(
            cell=[
                f"{m:.3f} ({lo:.3f}-{hi:.3f})"
                for m, lo, hi in zip(part["median"], part.p05, part.p95, strict=True)
            ]
        )
        wide = text.pivot(index="condition", columns="count", values="cell").loc[list(SHORT)]
        wide = wide[sorted(wide.columns, key=lambda v: 10**9 if str(v) == "all" else int(v))]
        wide.index = [SHORT[c] for c in wide.index]
        print(variant)
        print(markdown(wide.reset_index(names="condition")))
        print()


def section_references_reach() -> None:
    """Smallest count whose median within-source AUROC is within 0.01 of the key built on all records."""
    summary = reference_summary("within")
    rows = []
    for (variant, condition), part in summary.groupby(["variant", "condition"]):
        full = float(part[part["count"].astype(str) == "all"]["median"].iloc[0])
        drawn = (
            part[part["count"].astype(str) != "all"]
            .assign(n=lambda f: f["count"].astype(int))
            .sort_values("n")
        )
        reached = drawn[drawn["median"] >= full - 0.01]
        rows.append(
            {
                "variant": variant,
                "condition": SHORT[condition],
                "all": full,
                "smallest n within 0.01": int(reached.n.iloc[0]) if len(reached) else "not reached",
            }
        )
    table = pd.DataFrame(rows).pivot(index="condition", columns="variant", values="smallest n within 0.01")
    print(markdown(table.loc[[SHORT[c] for c in SHORT]].reset_index()))


def section_stability() -> None:
    """Cosines among bootstrap keys and with the full key."""
    table = pd.read_csv(OUT / "stability_cosines.csv")
    summary = table.groupby("construction")[
        ["pairwise_median", "pairwise_p05", "to_full_median", "to_full_p05"]
    ]
    print(
        markdown(
            summary.min()
            .loc[list(SINGLE)]
            .reset_index()
            .rename(columns=lambda c: f"min {c}" if c != "construction" else c)
        )
    )
    model = selection()["selected_single"]
    part = table[table.construction == model].set_index("condition").loc[list(SHORT)]
    part.index = [SHORT[c] for c in part.index]
    print()
    print(markdown(part.drop(columns="construction").reset_index(names="condition"), 4))


def section_stability_auroc() -> None:
    """Spread of validation AUROC over the 200 bootstrap keys."""
    table = pd.read_csv(OUT / "stability_auroc.csv")
    grouped = table.groupby(["construction", "condition"])["within"]
    spread = (grouped.quantile(0.95) - grouped.quantile(0.05)).rename("p95 - p05").reset_index()
    print(
        markdown(
            spread.groupby("construction")["p95 - p05"].agg(["median", "max"]).loc[list(SINGLE)].reset_index()
        )
    )
    macro = table.groupby(["construction", "draw"])["within"].mean().groupby("construction")
    print()
    print(
        markdown(
            pd.DataFrame(
                {
                    "macro p05": macro.quantile(0.05),
                    "macro median": macro.median(),
                    "macro p95": macro.quantile(0.95),
                }
            )
            .loc[list(SINGLE)]
            .reset_index()
        )
    )


def section_leave_family_out() -> None:
    """Keys built without a family against keys built with it, on that family's validation records."""
    table = pd.read_csv(OUT / "leave_family_out.csv")
    summary = table.groupby(["construction", "family"]).agg(
        conditions=("difference", "size"),
        mean_difference=("difference", "mean"),
        min_difference=("difference", "min"),
        below_zero=("ci_high", lambda s: int((s < 0).sum())),
    )
    print(markdown(summary.reset_index(), 4))
    model = selection()["selected_single"]
    part = table[table.construction == model]
    rows = [
        {
            "family": r.family,
            "condition": SHORT[r.condition],
            "with": f"{r.with_:.3f}",
            "without": f"{r.without:.3f}",
            "difference": ci(r.difference, r.ci_low, r.ci_high, True),
            "train positives without": r.train_positives_without,
        }
        for r in part.rename(columns={"with": "with_"}).itertuples(index=False)
    ]
    print()
    print(markdown(pd.DataFrame(rows)))


def section_measurements() -> None:
    """Spearman correlation of the selected key and of P0 with the defining measurements, by source."""
    model = selection()["selected_single"]
    table = pd.read_csv(OUT / "measurement_spearman.csv")
    part = table[table.model.isin([model, "P0"])]
    summary = part.groupby(["condition", "measure", "unreliable", "model"]).agg(
        sources=("spearman", "size"),
        median=("spearman", "median"),
        low=("spearman", "min"),
        high=("spearman", "max"),
        records=("records", "sum"),
    )
    summary = summary.unstack("model")
    order = [c for c in SHORT if c in summary.index.get_level_values(0)]
    summary = summary.loc[order]
    flat = pd.DataFrame(
        {
            "condition": [SHORT[i[0]] for i in summary.index],
            "measure": [i[1] for i in summary.index],
            "unreliable": [i[2] for i in summary.index],
            "sources": summary[("sources", model)].to_numpy(),
            "records": summary[("records", model)].to_numpy(),
            f"{model} median rho": summary[("median", model)].to_numpy(),
            f"{model} range": [
                f"{a:.2f} to {b:.2f}"
                for a, b in zip(summary[("low", model)], summary[("high", model)], strict=True)
            ],
            "P0 median rho": summary[("median", "P0")].to_numpy(),
            "P0 range": [
                f"{a:.2f} to {b:.2f}"
                for a, b in zip(summary[("low", "P0")], summary[("high", "P0")], strict=True)
            ],
        }
    )
    print(markdown(flat, 3))


def section_measurements_constructions() -> None:
    """Median over sources of the absolute Spearman correlation, every construction."""
    table = pd.read_csv(OUT / "measurement_spearman.csv")
    table = table[~table.unreliable]
    median = table.groupby(["condition", "measure", "model"])["spearman"].median().unstack("model")
    print(
        markdown(
            median[[*SINGLE, *K7, "P0"]]
            .abs()
            .median()
            .rename("median abs rho")
            .rename_axis("model")
            .reset_index(),
            3,
        )
    )


def section_review() -> None:
    """Measurement facts of the review records of the selected key."""
    records = pd.read_csv(OUT / "review/records.csv")
    rows = []
    for (condition, kind), part in records.groupby(["condition", "kind"], sort=False):
        rows.append(
            {
                "condition": SHORT[condition],
                "kind": kind,
                "records": len(part),
                "sources": ", ".join(f"{s} {n}" for s, n in part.source.value_counts().items()),
                "method B not ok": int((part.status_b != "ok").sum()),
                "with a flat lead": int((part.n_flat_leads > 0).sum()),
            }
        )
    print(markdown(pd.DataFrame(rows)))
    reading = OUT / "review/reading.csv"
    if reading.exists():
        print()
        print(markdown(pd.read_csv(reading)))


BOUNDS = {
    "sinus_tachycardia": ("hr", "above 100 per minute", lambda v: v > 100),
    "sinus_bradycardia": ("hr", "below 60 per minute", lambda v: v < 60),
    "sinus_arrhythmia": ("rr_range_dominant_b", "RR range above 120 ms (unreliable)", lambda v: v > 120),
    "af_or_flutter": ("rr_nrmssd_b", "normalized RMSSD at least 0.10 (unreliable)", lambda v: v >= 0.10),
    "first_degree_av_block": ("pr", "PR above 200 ms (unreliable)", lambda v: v > 200),
    "complete_rbbb": ("qrs", "QRS at least 120 ms", lambda v: v >= 120),
    "incomplete_rbbb": ("qrs", "QRS 110-119 ms", lambda v: (v >= 110) & (v < 120)),
    "lafb": ("axis", "axis -45 to -90 degrees", lambda v: (v <= -45) & (v >= -90)),
    "left_axis_deviation": ("axis", "axis -30 to -90 degrees", lambda v: (v <= -30) & (v >= -90)),
    "lvh": ("sokolow_mv", "Sokolow-Lyon at least 3.5 mV", lambda v: v >= 3.5),
    "st_depression": ("st_min_mv", "lowest ST at most -0.05 mV", lambda v: v <= -0.05),
    "t_wave_abnormality": ("t_min_mv", "lowest upright-lead T below 0.05 mV", lambda v: v < 0.05),
    "mi_any": ("q_max_ms", "longest Q at least 30 ms", lambda v: v >= 30),
}


def section_review_measured() -> None:
    """Count, per review sheet, the records whose measurement meets the condition's measured bound."""
    records = pd.read_csv(OUT / "review/records.csv")
    rows = []
    for (condition, kind), part in records.groupby(["condition", "kind"], sort=False):
        if condition not in BOUNDS:
            rows.append(
                {"condition": SHORT[condition], "kind": kind, "bound": "none", "measured": "", "meets": ""}
            )
            continue
        measure, words, rule = BOUNDS[condition]
        values = part[measure].dropna()
        rows.append(
            {
                "condition": SHORT[condition],
                "kind": kind,
                "bound": f"{measure}: {words}",
                "measured": len(values),
                "meets": int(rule(values).sum()),
                "min": float(values.min()),
                "max": float(values.max()),
                "QRS min-max": f"{part.qrs.min():.0f}-{part.qrs.max():.0f}",
                "QRS >= 120": int((part.qrs >= 120).sum()),
            }
        )
    print(markdown(pd.DataFrame(rows)))


def section_selection() -> None:
    """Print the committed selection."""
    print(json.dumps(selection(), indent=2))


def section_test() -> None:
    """Print the one test read."""
    result = read_json("test.json")
    rows = []
    for role in result["roles"]:
        for name, value in role["macro"].items():
            rows.append(
                {
                    "role": role["name"],
                    "model": name,
                    "conditions": len(role["entering"]),
                    "macro AUROC": ci(value["value"], *value["ci"]),
                }
            )
        for name in ("single", "k7"):
            d = role[f"{name}_minus_P0"]
            rows.append(
                {
                    "role": role["name"],
                    "model": f"{name} minus P0",
                    "conditions": len(role["entering"]),
                    "macro AUROC": ci(d["difference"], d["ci_low"], d["ci_high"], True),
                }
            )
    print(markdown(pd.DataFrame(rows)))
    print()
    print(
        json.dumps(
            {
                "primary": result["primary"],
                "selection": result["selection"],
                "support": result["support_test_roles"],
                "entering": {r["name"]: r["entering"] for r in result["roles"]},
                "records": {r["name"]: [r["records"], r["groups"], r["redraws"]] for r in result["roles"]},
            },
            indent=2,
        )
    )


def section_test_conditions() -> None:
    """Per-condition reading of the test read, without a decision."""
    table = pd.read_csv(OUT / "test_conditions.csv")
    rows = [
        {
            "role": r.role,
            "model": r.model,
            "condition": SHORT[r.condition],
            "positives": r.positives,
            "AUROC": f"{r.auroc:.3f}",
            "P0": f"{r.auroc_P0:.3f}",
            "minus P0": ci(r.difference, r.ci_low, r.ci_high, True),
            "reading": r.reading,
        }
        for r in table.itertuples(index=False)
    ]
    print(markdown(pd.DataFrame(rows)))


SECTIONS: dict[str, Callable[[], None]] = {
    name.removeprefix("section_").replace("_", "-"): value
    for name, value in dict(globals()).items()
    if name.startswith("section_")
}


def main() -> None:
    """Print one or all sections."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--section", choices=sorted(SECTIONS))
    args = parser.parse_args()
    for name, function in SECTIONS.items():
        if args.section in (None, name):
            print(f"## {name}\n")
            function()
            print()


if __name__ == "__main__":
    main()
