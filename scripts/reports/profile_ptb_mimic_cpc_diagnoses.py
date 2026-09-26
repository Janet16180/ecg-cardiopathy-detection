"""Run frozen CPC diagnosis geometry checks on PTB training and MIMIC ECGs."""

from __future__ import annotations

from ecg_experiment.ptb_mimic_cpc_diagnosis_geometry import run

if __name__ == "__main__":
    report = run()
    print(report["ptb_diagnosis_neighbor_report"])
    print(report["ptb_unsupervised_clusters"])
    print(report["mimic_ptb_fitted_pca_machine_summary_proximity"])
