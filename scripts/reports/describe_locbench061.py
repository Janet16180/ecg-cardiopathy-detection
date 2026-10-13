"""Descriptive readings of the Experiment 061 outputs used in its results report; writes descriptives.json."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.files import write_json_atomic
from ecg_experiment.locbench import START, STOP, build_benchmark
from ecg_experiment.waveforms import LEADS

OUTPUT = ROOT / "outputs/experiment061_locbench_v1"


def main() -> None:
    """Compute QRS-prior coverage of ECGAD marks, noisy-lead shares and clean-normal amplitude scores."""
    bench = build_benchmark()
    prior = np.load(OUTPUT / "tables/qrs_prior/ecgad_maps.npy", mmap_mode="r")
    inside = np.asarray(prior[:, 0, START:STOP]) >= 1
    marks = bench.ecgad_labels[:, :, START:STOP]
    marked = bench.ecgad["marked"].to_numpy()
    limb = np.isin(LEADS, LEADS[:6])
    records = pd.read_csv(OUTPUT / "tables/amplitude/records.csv").set_index("ecg_id")
    columns = [f"lead_{lead}" for lead in LEADS]
    result = {
        "qrs_prior_share_of_scored_ecgad_samples": float(inside.mean()),
        "share_of_marked_lead_samples_inside_qrs_prior": float(
            (marks & inside[:, None, :])[marked].sum() / marks[marked].sum()
        ),
        "noise_set_lead_share": dict(
            zip(LEADS, bench.noise_leads.mean(axis=0).round(4).tolist(), strict=True)
        ),
        "noise_set_limb_only_share": float((bench.noise_leads[:, ~limb].sum(axis=1) == 0).mean()),
        "noise_set_mean_named_leads": float(bench.noise_leads.sum(axis=1).mean()),
        "clean_normal_median_amplitude_lead_score_mV": records.loc[bench.clean_normals["ecg_id"], columns]
        .median()
        .round(4)
        .to_dict(),
    }
    write_json_atomic(OUTPUT / "descriptives.json", result)
    print(result)


if __name__ == "__main__":
    main()
