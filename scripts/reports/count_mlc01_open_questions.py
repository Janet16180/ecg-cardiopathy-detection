"""Count final MLC-01 records touched by questions that Amendment 1 left open; no label is changed.

Writes ``<dataset>/supplementary/open_questions.csv`` (records per question, source and role) and its receipt.
Run from the repository root with ``PYTHONPATH=.``:
``.venv/bin/python -m scripts.reports.count_mlc01_open_questions --dataset <build directory>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecg_experiment import mlc01_dataset, mlc_labels
from ecg_experiment.files import sha256_file, write_json_atomic

ST_T = mlc_labels.ST_T_NONSPECIFIC_OR_ISCHEMIC
STE = mlc_labels.CONDITIONS["st_elevation"]["positive"]
STD_SCP_OTHER = frozenset({"INJAS", "INJAL", "INJLA", "INJIN", "INJIL", "DIG", "ANEUR", "EL"})
TWA_AHA = frozenset({"146", "147"})


def questions(source: str, record: object, labels: pd.Series) -> dict[str, bool]:
    """
    Which open questions concern one final record.

    Parameters
    ----------
    source : str
        Record source.
    record : object
        The record's parsed codes.
    labels : pd.Series
        The record's manifest row.

    Returns
    -------
    dict[str, bool]
        One flag per question; each flag marks a record whose state is defined and would change.
    """
    vocabulary = mlc_labels.VOCABULARY[source]
    scp = set(record) if vocabulary == "scp" else set()
    sph = vocabulary == "aha"
    bases = {base for base, _ in record} if sph else set()
    st_t = not sph and mlc_labels.listed(ST_T[vocabulary], record, vocabulary)
    twa_negative = labels["b_t_wave_abnormality"] == 0
    return {
        "st_elevation_negative_with_nonspecific_or_ischemic_st_t": labels["c_st_elevation"] == 0 and st_t,
        "right_axis_negative_with_empty_or_axr_heart_axis": labels["c_right_axis_deviation"] == 0
        and bool(scp & {"axis:missing", "axis:AXR"}),
        "ptbxl_st_depression_negative_with_injury_digitalis_aneurysm_or_electrolyte":
            labels["b_st_depression"] == 0 and bool(scp & STD_SCP_OTHER),
        "cpsc_2018_af_or_flutter_negative_partial_annotation": source == "cpsc_2018"
        and labels["b_af_or_flutter"] == 0,
        "sph_t_wave_negative_with_st_deviation_without_direction": sph and twa_negative
        and mlc_labels.aha_match("145=", record),
        "sph_t_wave_negative_with_any_st_deviation": sph and twa_negative and "145" in bases
        and not bases & TWA_AHA,
    }


def main() -> None:
    """Write the counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.dataset / "manifest.csv", index_col="record_key", low_memory=False,
                           dtype={"record": str, "raw_codes": str, "heart_axis": str})
    manifest["heart_axis"] = manifest["heart_axis"].fillna("")
    final = manifest[manifest["included"]]
    rows = []
    for key, row in final.iterrows():
        record = mlc01_dataset.parsed_codes(row["source"], row["raw_codes"], row["heart_axis"])
        rows.append({"record_key": key, **questions(row["source"], record, row.astype(object))})
    flags = pd.DataFrame(rows).set_index("record_key").astype(bool)
    joined = final[["source", "role"]].join(flags)
    table = joined.groupby(["source", "role"])[list(flags.columns)].sum().reset_index()
    long = table.melt(id_vars=["source", "role"], var_name="question", value_name="records")
    long = long[long["records"] > 0].sort_values(["question", "source", "role"])
    out = args.dataset / "supplementary"
    out.mkdir(exist_ok=True)
    long.to_csv(out / "open_questions.csv", index=False)
    write_json_atomic(out / "open_questions.json", {
        "manifest_sha256": sha256_file(args.dataset / "manifest.csv"),
        "script_sha256": sha256_file(Path(__file__)),
        "open_questions_sha256": sha256_file(out / "open_questions.csv")}, sort_keys=True)
    print(long.to_string(index=False))


if __name__ == "__main__":
    main()
