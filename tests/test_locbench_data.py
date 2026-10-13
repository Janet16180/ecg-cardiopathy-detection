import numpy as np
import pandas as pd

from ecg_experiment.ecgad import build_split
from ecg_experiment.locbench_data import parse_noise_leads
from ecg_experiment.waveforms import LEADS


def test_parse_noise_leads_reads_names_ranges_and_all() -> None:
    assert parse_noise_leads(" , I-AVR,  ") == ("I", "II", "III", "aVR")
    assert parse_noise_leads(" , alles") == LEADS
    assert parse_noise_leads("V1,V2") == ("V1", "V2")
    assert parse_noise_leads(" , v6") == ("V6",)
    assert parse_noise_leads("Kontaktprobleme aVL ???") == ("aVL",)
    assert parse_noise_leads(" , mittel,  ") is None
    assert parse_noise_leads(float("nan")) is None


def test_build_split_keeps_patients_together_and_marks_duplicate() -> None:
    match = pd.DataFrame(
        {
            "bench_index": np.arange(8),
            "ecg_id": [1, 2, 3, 4, 5, 6, 6, 7],
            "patient_id": [10, 10, 11, 12, 13, 14, 14, 15],
        }
    )
    labeled = np.array([True, False, True, False, True, False, False, True])
    table = build_split(match, labeled, seed=1)
    assert table.loc[6, "partition"] == "duplicate"
    scored = table[table.partition != "duplicate"]
    assert scored.groupby("patient_id").partition.nunique().max() == 1
    assert set(scored.partition) == {"development", "holdout"}
