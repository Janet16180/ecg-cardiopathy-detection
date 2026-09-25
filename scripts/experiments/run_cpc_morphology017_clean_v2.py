"""Version 2 clean Experiment 017 entry point with CPU checkpoint comparison."""

from ecg_experiment.morphology_clean import main
from ecg_experiment.morphology_clean_inputs import ROOT

OUTPUT = ROOT / "outputs/experiment017_clean_replication_v2"
EXTRA_CODE = (
    "scripts/experiments/run_cpc_morphology017_clean_v2.py",
    "docs/experiment-017-clean-replication-v2.md",
)

if __name__ == "__main__":
    main(output=OUTPUT, extra_code=EXTRA_CODE, roundtrip_device="cpu")
