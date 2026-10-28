"""Raw baseline admission preserves original labels, roles and difficult evaluation windows."""

import pandas as pd

from ecg_experiment.measurement_screening042 import patient_role
from ecg_experiment.raw_role_bootstrap import admitted_tables


def test_administrative_baseline_retains_sph_quality_failures_and_excludes_calibration_positives():
    ptb = pd.DataFrame({
        "ecg_id": list(range(1, 8)), "patient_id": list(range(11, 18)),
        "strat_fold": [1, 1, 8, 8, 9, 9, 1], "target": [0, 1, 0, 1, 0, 1, 1],
        "flat_leads": [0, 0, 0, 0, 0, 0, 1], "filename_hr": "stem",
    }).set_index("ecg_id")
    patients = {role: [] for role in ("calibration", "evaluation")}
    for index in range(10000):
        key = "S" + str(index)
        patients[patient_role(key)].append(key)
        if min(map(len, patients.values())) >= 103:
            break
    rows = []
    for role, keys in patients.items():
        for index, key in enumerate(keys[:103]):
            positive = index == 102
            rows.append({"ecg_id": role + str(index).zfill(3), "patient_id": key, "role": role,
                         "primary": int(positive), "aha_codes": "160" if positive else "1",
                         "signal_sha256": role + str(index), "quality_fail": True})
    other = pd.DataFrame(rows).set_index("ecg_id").sort_index()
    ptb_rows, sph_rows, evidence = admitted_tables(ptb, other)
    assert len(ptb_rows) == 6
    assert len(sph_rows) == 205
    assert sph_rows.quality_fail.all()
    assert sph_rows.loc[sph_rows.role.eq("calibration"), "target"].eq(0).all()
    assert sph_rows.loc[sph_rows.role.eq("evaluation"), "target"].sum() == 1
    assert evidence.loc[evidence.role.eq("calibration"), "primary"].sum() == 1
    assert evidence.use_evaluation.all()
    assert len(set(sph_rows.patient_id)) == len(sph_rows)
