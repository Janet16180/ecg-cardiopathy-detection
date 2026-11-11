"""Print the tables of the MLC-03 signal-quality report from its outputs, and write the proposed-rule counts.

Nothing is measured here: the script reads the flag tables, visual-check tallies, quantiles, pipeline and
near-duplicate outputs written by ``scripts.experiments.run_mlc03_signal_quality``, counts and formats.

Example
-------
PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_signal_tables
PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_signal_tables --section rules
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ecg_experiment.eda.signals import LEADS
from ecg_experiment.mlc03_signal_flags import FLAGS
from ecg_experiment.mlc03_signal_quality import INDEPENDENT
from ecg_experiment.mlc03_signal_runs import MANIFEST, OUTPUT

REVIEW = OUTPUT.parent / "review/signal"
STATUSES = ("ok", "sin_latidos", "derivacion_plana")
NEAR = OUTPUT / "near_duplicates"
MEASUREMENT_STATUS = OUTPUT.parents[1] / "mlc02_condition_models_20261110/features/measurements_status.csv"
SOURCES = ("ptbxl", "ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra", "sph")
FINAL_REVIEW = {"reversal_la_ra": "reversal_la_ra__v2", "reversal_neutral": "reversal_neutral__v2",
                "mains_interference": "mains_interference__v2", "hf_noise": "hf_noise__v2",
                "rhythm_unmeasurable": "rhythm_unmeasurable__v2"}
RULES = (
    ("R1", "exclude record", "missing_samples", None),
    ("R2", "exclude record", "adc_rail", None),
    ("R3", "exclude record", "edge_hold", None),
    ("R4", "mark lead unusable", "flat_lead", "flat_lead__leads"),
    ("R5", "mark lead unusable", "near_flat_lead", "near_flat_lead__leads"),
    ("R6", "mark lead unusable", "contact_loss", "contact_loss__leads"),
    ("R7", "mark derived lead unusable", "identity_violation", None),
    ("R8", "exclude record", "reversal_la_ra", None),
    ("R9", "exclude record", "reversal_neutral", None),
    ("R10", "exclude record", "amplitude_high", None),
)


def markdown(frame: pd.DataFrame) -> str:
    """
    Format a table as Markdown.

    Parameters
    ----------
    frame : pd.DataFrame
        Table.

    Returns
    -------
    str
        Markdown text.
    """
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def flag_table(path: str) -> pd.DataFrame:
    """
    Count flagged included records per flag and source (all roles of the file summed).

    Parameters
    ----------
    path : str
        Counts CSV name under the signal output directory.

    Returns
    -------
    pd.DataFrame
        One row per flag, one column per source, and a total.
    """
    counts = pd.read_csv(OUTPUT / path)
    pivot = counts.groupby(["flag", "source"])["flagged"].sum().unstack(fill_value=0)
    pivot = pivot.reindex([*FLAGS, "any_flag"])[[s for s in SOURCES if s in pivot.columns]]
    records = counts.drop_duplicates(["source", "role"]).groupby("source")["records"].sum()
    pivot.loc["records"] = records.reindex(pivot.columns).to_numpy()
    pivot["total"] = pivot.sum(axis=1)
    return pivot.reset_index()


def tally_table() -> pd.DataFrame:
    """
    Collect the visual-check result of every review directory.

    Returns
    -------
    pd.DataFrame
        One row per review: drawn, showing the defect, share, controls viewed and showing it, reliable.
    """
    rows = []
    for path in sorted(REVIEW.glob("*/tally.json")):
        tally = json.loads(path.read_text())
        verdicts = pd.read_csv(path.with_name("tally.csv"))
        flagged = verdicts[verdicts["group"] == "flagged"]
        viewed = int((flagged["defect_visible"] != "not reviewed").sum())
        shown = tally["flagged_showing_defect"]
        rows.append({"review": tally["review"], "flagged drawn": tally["flagged_drawn"], "viewed": viewed,
                     "show the defect": shown, "share of viewed": f"{shown / max(viewed, 1):.2f}",
                     "controls viewed": tally["unflagged_viewed"],
                     "controls with defect": tally["unflagged_showing_defect"],
                     "reliable": tally["reliable"]})
    return pd.DataFrame(rows)


def rule_masks(flags: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Record masks of the proposed rules.

    Parameters
    ----------
    flags : pd.DataFrame
        Flags table (train and validation, or the test roles).

    Returns
    -------
    dict[str, np.ndarray]
        Flag or derived name to boolean mask; ``amplitude_high`` is the high side of ``amplitude_outlier``.
    """
    masks = {name: flags[name].to_numpy(dtype=bool) for name in FLAGS}
    masks["amplitude_high"] = (flags["amplitude_ratio"].to_numpy(dtype=float) >= 4.0)
    return masks


def lead_count(flags: pd.DataFrame, column: str | None, mask: np.ndarray) -> int:
    """
    Count the lead-records a lead rule would mark.

    Parameters
    ----------
    flags : pd.DataFrame
        Flags table.
    column : str | None
        Column of blamed leads, or None for a record rule.
    mask : np.ndarray
        Records the rule affects.

    Returns
    -------
    int
        Blamed leads summed over the affected records (0 for a record rule).
    """
    if column is None:
        return 0
    return int(sum(len([lead for lead in str(text).split(";") if lead]) for text in flags.loc[mask, column]))


def rule_counts() -> pd.DataFrame:
    """
    Count the records each proposed rule affects, per role and source, on included records of every role.

    Returns
    -------
    pd.DataFrame
        Long table: ``rule``, ``action``, ``flag``, ``role``, ``source``, ``records`` (denominator),
        ``affected`` and ``leads`` (lead-records for lead rules).
    """
    frames = [pd.read_parquet(OUTPUT / "flags_train_validation.parquet")]
    if (OUTPUT / "flags_test_roles.parquet").exists():
        frames.append(pd.read_parquet(OUTPUT / "flags_test_roles.parquet"))
    rows = []
    for frame in frames:
        frame = frame[frame["included"]].reset_index(drop=True)
        masks = rule_masks(frame)
        for (role, source), group in frame.groupby(["role", "source"]):
            positions = group.index.to_numpy()
            for rule, action, flag, column in RULES:
                mask = np.zeros(len(frame), dtype=bool)
                mask[positions] = masks[flag][positions]
                rows.append({"rule": rule, "action": action, "flag": flag, "role": role, "source": source,
                             "records": len(group), "affected": int(mask.sum()),
                             "leads": lead_count(frame, column, mask)})
            union = np.zeros(len(frame), dtype=bool)
            for _, action, flag, _ in RULES:
                if action.startswith("exclude"):
                    union[positions] |= masks[flag][positions]
            rows.append({"rule": "R1-R3, R8-R10", "action": "exclude record (union)", "flag": "",
                         "role": role, "source": source, "records": len(group), "affected": int(union.sum()),
                         "leads": 0})
    return pd.DataFrame(rows)


def lead_distribution(flag: str) -> pd.DataFrame:
    """
    Blamed leads of a lead flag on included train and validation records.

    Parameters
    ----------
    flag : str
        Flag with a ``__leads`` column.

    Returns
    -------
    pd.DataFrame
        Records per lead and source.
    """
    flags = pd.read_parquet(OUTPUT / "flags_train_validation.parquet")
    flags = flags[flags["included"] & flags[flag]]
    rows = []
    for source, group in flags.groupby("source"):
        counts = dict.fromkeys(LEADS, 0)
        for text in group[f"{flag}__leads"]:
            for lead in filter(None, str(text).split(";")):
                counts[lead] += 1
        rows.append({"source": source, **counts})
    return pd.DataFrame(rows)


def quantile_table(measure: str) -> pd.DataFrame:
    """
    Per-source quantiles of one record summary.

    Parameters
    ----------
    measure : str
        Summary name of ``record_summaries``.

    Returns
    -------
    pd.DataFrame
        One row per source.
    """
    table = pd.read_csv(OUTPUT / "measure_quantiles_train_validation.csv")
    table = table[table["measure"] == measure].drop(columns="measure")
    return table.round(4)


def measurement_status() -> pd.DataFrame:
    """
    Status of the MLC-02 measurement extractor on included train and validation records, per flag.

    Returns
    -------
    pd.DataFrame
        One row per flag: flagged records and the share of each extractor status among them.
    """
    status = pd.read_csv(MEASUREMENT_STATUS, usecols=["record_key", "status"])
    flags = pd.read_parquet(OUTPUT / "flags_train_validation.parquet")
    flags = flags[flags["included"]].merge(status, on="record_key", how="left", validate="one_to_one")
    rows = []
    for flag in [*FLAGS, "any_flag"]:
        group = flags[flags[flag]]
        shares = group["status"].value_counts(normalize=True)
        rows.append({"flag": flag, "records": len(group),
                     **{name: f"{shares.get(name, 0.0):.3f}" for name in STATUSES}})
    unflagged = flags[~flags["any_flag"]]["status"].value_counts(normalize=True)
    rows.append({"flag": "no flag", "records": int((~flags["any_flag"]).sum()),
                 **{name: f"{unflagged.get(name, 0.0):.3f}" for name in STATUSES}})
    return pd.DataFrame(rows)


def long_records() -> pd.DataFrame:
    """
    Count defects beyond the first 10 s of the long train and validation recordings.

    Returns
    -------
    pd.DataFrame
        One row per source: long records, median and largest duration, records with missing samples, with
        rail samples only after 10 s, with a lead constant only after 10 s, with any constant run of 1 s or
        more, and with every lead held at the end for 150 ms or more.
    """
    full = pd.read_parquet(OUTPUT / "full_records_train_validation.parquet")
    window = pd.read_parquet(OUTPUT / "measures_train_validation.parquet",
                             columns=["record_key", "rail_samples"])
    flags = pd.read_parquet(OUTPUT / "flags_train_validation.parquet", columns=["record_key", "source"])
    frame = full.merge(window, on="record_key").merge(flags, on="record_key")
    rows = []
    for source, group in frame.groupby("source"):
        seconds = group["full_samples"] / 500.0
        late_rail = (group["full_rail_samples"] > 0) & (group["rail_samples"] == 0)
        rows.append({"source": source, "long records": len(group),
                     "median s": round(float(seconds.median()), 1), "max s": round(float(seconds.max()), 1),
                     "missing samples": int((group["full_missing_samples"] > 0).sum()),
                     "rail only after 10 s": int(late_rail.sum()),
                     "lead constant only after 10 s": int((group["after10_flat_leads"] > 0).sum()),
                     "constant run >= 1 s": int((group["full_const_run_max"] >= 500).sum()),
                     "all-lead end hold >= 150 ms": int((group["full_tail_run_min"] >= 75).sum())})
    return pd.DataFrame(rows)


def near_train() -> pd.DataFrame:
    """
    Count train and validation candidate pairs of both searches by score band.

    Returns
    -------
    pd.DataFrame
        One row per search and band: pairs with the same first-10-s hash, other pairs, and other pairs
        across train and validation.
    """
    dense = pd.read_parquet(NEAR / "dense_pairs_train_validation.parquet")
    shifted = pd.read_parquet(NEAR / "shifted_pairs_train_validation.parquet")
    shifted = shifted[shifted["overlap_s"] >= 4.0].rename(columns={"correlation": "score"})
    bands = [0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.01]
    rows = []
    for search, frame in (("zero_lag", dense), ("shifted", shifted)):
        frame = frame.assign(band=pd.cut(frame["score"], bands, right=False))
        for band, group in frame.groupby("band", observed=False):
            other = group[~group["same_hash"]]
            crossing = int((other["role_a"] != other["role_b"]).sum())
            rows.append({"search": search, "score band": str(band),
                         "same hash": int(group["same_hash"].sum()), "other pairs": len(other),
                         "other pairs across train and validation": crossing})
    return pd.DataFrame(rows)


def lead_amplitude() -> pd.DataFrame:
    """
    Low percentiles of every lead's 1-99 percentile range (``amp_robust``) over included train and validation.

    Constant leads are left out.

    Returns
    -------
    pd.DataFrame
        One row per percentile (0.01%, 0.1%, 1%, 5%, 50%), one column per lead, in mV.
    """
    measures = pd.read_parquet(OUTPUT / "measures_train_validation.parquet",
                               columns=["record_key", *[f"{name}__{lead}" for name in ("amp_robust", "flat")
                                                        for lead in LEADS]])
    flags = pd.read_parquet(OUTPUT / "flags_train_validation.parquet", columns=["record_key", "included"])
    measures = measures[flags.set_index("record_key").loc[measures["record_key"], "included"].to_numpy()]
    rows = []
    for percentile in (0.01, 0.1, 1.0, 5.0, 50.0):
        row = {"percentile": percentile}
        for lead in LEADS:
            values = measures.loc[measures[f"flat__{lead}"] == 0, f"amp_robust__{lead}"]
            row[lead] = round(float(np.percentile(values, percentile)), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def paced_spikes() -> pd.DataFrame:
    """
    Paced-rhythm labels among train and validation records with a 2 mV spike, isolated or not.

    Only the space-C paced-rhythm column of train and validation rows is read (labels of the test roles are
    never opened); the spike measures come from the measures table.

    Returns
    -------
    pd.DataFrame
        One row per spike measure: records with a spike of at least 2 mV and how many carry the paced label
        (positive, negative, undefined).
    """
    manifest = pd.read_csv(MANIFEST, usecols=["record_key", "role", "included", "c_paced_rhythm"],
                           dtype={"included": str}, low_memory=False)
    manifest = manifest[manifest["role"].isin(["train", "validation"]) & (manifest["included"] == "True")]
    columns = [f"spike__{lead}" for lead in LEADS] + [f"spike_isolated__{lead}" for lead in INDEPENDENT]
    measures = pd.read_parquet(OUTPUT / "measures_train_validation.parquet", columns=["record_key", *columns])
    frame = measures.merge(manifest, on="record_key", validate="one_to_one")
    rows = []
    for name, leads in (("spike", LEADS), ("spike_isolated", INDEPENDENT)):
        spiky = frame[[f"{name}__{lead}" for lead in leads]].max(axis=1) >= 2.0
        paced = frame.loc[spiky, "c_paced_rhythm"]
        rows.append({"measure": name, "records with a 2 mV spike": int(spiky.sum()),
                     "paced positive": int((paced == 1).sum()), "paced negative": int((paced == 0).sum()),
                     "paced undefined": int(paced.isna().sum()),
                     "paced positive in all records": int((frame["c_paced_rhythm"] == 1).sum())})
    return pd.DataFrame(rows)


def first_version_pools() -> pd.DataFrame:
    """
    Included flagged records of the first and second version of the tightened flags, from review receipts.

    Returns
    -------
    pd.DataFrame
        One row per review directory with ``flagged_pool``.
    """
    rows = []
    for path in sorted(REVIEW.glob("*/receipt.json")):
        receipt = json.loads(path.read_text())
        if "flagged_pool" in receipt:
            rows.append({"review": path.parent.name, "seed": receipt["seed"],
                         "flagged included records": receipt["flagged_pool"]})
    return pd.DataFrame(rows)


def near_test_maximum() -> pd.DataFrame:
    """
    Candidate pairs and their highest score per search and role pair, across roles with a test role.

    Returns
    -------
    pd.DataFrame
        ``search``, ``role_a``, ``role_b``, ``candidates``, ``highest score`` and ``same hash``.
    """
    pairs = pd.read_csv(OUTPUT / "near_duplicates_test_roles_pairs.csv",
                        usecols=["search", "role_a", "role_b", "score", "same_hash"])
    table = pairs.groupby(["search", "role_a", "role_b"]).agg(
        candidates=("score", "size"), highest=("score", "max"), same_hash=("same_hash", "sum")).reset_index()
    return table.rename(columns={"highest": "highest score", "same_hash": "same hash"}).round(4)


def rules_section() -> str:
    """
    Write ``proposed_rule_counts.csv`` and format the rule counts.

    Returns
    -------
    str
        Markdown: records affected per rule, role and source, then lead-records per lead rule and role.
    """
    counts = rule_counts()
    counts.to_csv(OUTPUT / "proposed_rule_counts.csv", index=False)
    wide = counts.pivot_table(index=["rule", "action", "flag"], columns=["role", "source"], values="affected",
                              aggfunc="sum", fill_value=0)
    leads = counts[counts["leads"] > 0].groupby(["rule", "role"])["leads"].sum().reset_index()
    return markdown(wide.reset_index()) + "\n\n" + markdown(leads)


def pipelines_section() -> str:
    """
    Format the pipeline-effect tables.

    Returns
    -------
    str
        Markdown of the summary, the edge ringing by source and the missing-sample spread.
    """
    names = ("pipeline_effects_summary", "edge_ringing_by_source", "missing_sample_spread")
    tables = [markdown(pd.read_csv(OUTPUT / f"pipeline_effects/{name}.csv").round(3)) for name in names]
    return "\n\n".join(tables)


def optional_table(name: str) -> str:
    """
    Format a CSV of the signal outputs if it exists.

    Parameters
    ----------
    name : str
        File name under the signal output directory.

    Returns
    -------
    str
        Markdown, or a note that the file is absent.
    """
    path = OUTPUT / name
    return markdown(pd.read_csv(path)) if path.exists() else f"({name} not written yet)"


def main(argv: list[str] | None = None) -> None:
    """
    Print the requested sections and write the rule counts.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--measure", default="max_baseline_ptp", help="quantiles: summary name")
    parser.add_argument("--flag", default="flat_lead", help="leads: flag name")
    sections = {
        "flags": lambda args: markdown(flag_table("flag_counts_train_validation.csv")),
        "flags-test": lambda args: (markdown(flag_table("flag_counts_test_roles.csv"))
                                    if (OUTPUT / "flag_counts_test_roles.csv").exists() else "(not run yet)"),
        "tallies": lambda args: markdown(tally_table()),
        "rules": lambda args: rules_section(),
        "leads": lambda args: markdown(lead_distribution(args.flag)),
        "quantiles": lambda args: markdown(quantile_table(args.measure)),
        "pipelines": lambda args: pipelines_section(),
        "near": lambda args: optional_table("near_duplicates_test_roles_counts.csv"),
        "measurement-status": lambda args: markdown(measurement_status()),
        "long-records": lambda args: markdown(long_records()),
        "near-train": lambda args: markdown(near_train()),
        "lead-amplitude": lambda args: markdown(lead_amplitude()),
        "paced-spikes": lambda args: markdown(paced_spikes()),
        "pools": lambda args: markdown(first_version_pools()),
        "near-test-max": lambda args: markdown(near_test_maximum()),
    }
    parser.add_argument("--section", choices=tuple(sections))
    args = parser.parse_args(argv)
    default = [name for name in sections if name not in ("leads", "quantiles")]
    chosen = [args.section] if args.section else default
    for name in chosen:
        print(f"\n## {name}\n\n{sections[name](args)}")


if __name__ == "__main__":
    main()
