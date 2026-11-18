"""Count the final MLC-01 records affected by mapping questions the protocol left open; no label is changed.

Writes ``<dataset>/supplementary/label_doubts.csv``: for each question, the records of the final dataset per
source and role that a different ruling would move. Run from the repository root with ``PYTHONPATH=.``:
``.venv/bin/python -m scripts.reports.count_mlc01_label_doubts``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecg_experiment import ROOT, mlc01_dataset, mlc_labels
from ecg_experiment.files import sha256_file, write_json_atomic

DATASET = ROOT / "outputs/mlc01_condition_dataset_20261110"
TWA_SCP = mlc_labels.CONDITIONS["t_wave_abnormality"]["positive"]["scp"]
ISCHEMIC_SCP = mlc_labels.CONDITIONS["ischemic_st_t"]["positive"]["scp"]
ISCHEMIC_SNOMED = mlc_labels.CONDITIONS["ischemic_st_t"]["positive"]["snomed"]
TWA_SNOMED = mlc_labels.CONDITIONS["t_wave_abnormality"]["positive"]["snomed"]
PVC_SNOMED = mlc_labels.CONDITIONS["premature_ventricular_complex"]["positive"]["snomed"]
VT_SNOMED = frozenset({"425856008", "164895002"})
LBBB = mlc_labels.CONDITIONS["complete_lbbb"]["positive"]
IRBBB = mlc_labels.CONDITIONS["incomplete_rbbb"]["positive"]


def questions(source: str, record: object, axis: str) -> dict[str, bool]:
    """
    Which open questions concern one record.

    Parameters
    ----------
    source : str
        Record source.
    record : object
        The record's parsed codes.
    axis : str
        PTB-XL ``heart_axis``.

    Returns
    -------
    dict[str, bool]
        One flag per question.
    """
    vocabulary = mlc_labels.VOCABULARY[source]
    scp = record if vocabulary == "scp" else {}
    snomed = record if vocabulary == "snomed" else frozenset()
    return {
        "ptbxl_heart_axis_missing_counted_lad_negative": vocabulary == "scp" and axis == "",
        "ptbxl_heart_axis_axl_counted_lad_negative": vocabulary == "scp" and axis == "AXL",
        "ptbxl_nst_without_std_counted_st_depression_negative": "NST_" in scp and "STD_" not in scp,
        "ptbxl_hvolt_without_lvh_or_vclvh_counted_lvh_negative": "HVOLT" in scp
        and not {"LVH", "VCLVH"} & set(scp),
        "ptbxl_ischemic_without_t_statement_counted_twa_negative": bool(ISCHEMIC_SCP & set(scp))
        and not TWA_SCP & set(scp),
        "snomed_ischemia_without_t_code_counted_twa_negative": bool(snomed & ISCHEMIC_SNOMED)
        and not snomed & TWA_SNOMED and not snomed & {"428750005", "55930002"},
        "snomed_ventricular_tachycardia_without_pvc_counted_pvc_negative": bool(snomed & VT_SNOMED)
        and not snomed & PVC_SNOMED,
        "lbbb_with_incomplete_rbbb_not_a_listed_pair": all(
            mlc_labels.listed(codes[vocabulary], record, vocabulary) for codes in (LBBB, IRBBB)),
    }


def main() -> None:
    """Write the counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    manifest = pd.read_csv(args.dataset / "manifest.csv", index_col="record_key", low_memory=False,
                           dtype={"record": str, "raw_codes": str, "heart_axis": str})
    manifest["heart_axis"] = manifest["heart_axis"].fillna("")
    final = manifest[manifest["included"]]
    rows = []
    for key, source, raw, axis in zip(final.index, final["source"], final["raw_codes"], final["heart_axis"],
                                      strict=True):
        flags = questions(source, mlc01_dataset.parsed_codes(source, raw, axis), axis)
        rows.append({"record_key": key, **flags})
    flags = pd.DataFrame(rows).set_index("record_key")
    joined = final[["source", "role"]].join(flags)
    table = joined.groupby(["source", "role"])[list(flags.columns)].sum().reset_index()
    long = table.melt(id_vars=["source", "role"], var_name="question", value_name="records")
    long = long[long["records"] > 0].sort_values(["question", "source", "role"])
    out = args.dataset / "supplementary"
    out.mkdir(exist_ok=True)
    long.to_csv(out / "label_doubts.csv", index=False)
    write_json_atomic(out / "label_doubts.json", {
        "manifest_sha256": sha256_file(args.dataset / "manifest.csv"),
        "script_sha256": sha256_file(Path(__file__)),
        "label_doubts_sha256": sha256_file(out / "label_doubts.csv")}, sort_keys=True)
    print(long.to_string(index=False))


if __name__ == "__main__":
    main()
