"""Run the frozen MIMIC CPC cluster/machine-label association audit."""

from __future__ import annotations

from ecg_experiment.mimic_cpc_cluster_label_profile import run


if __name__ == "__main__":
    report = run()
    print(report["group_summary"])
    print(report["proximity"])
