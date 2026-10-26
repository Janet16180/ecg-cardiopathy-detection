"""Run the frozen local NTT-07 readiness stage without model inference or human labels."""

from ecg_experiment import ROOT
from ecg_experiment.files import write_json_atomic
from ecg_experiment.localization.blinded_packet import aggregate_receipt, run_packet


def main() -> None:
    """Create the private packet and publish only a safe aggregate receipt locally."""
    output = ROOT / "outputs/ntt07_blinded_review"
    result = run_packet(output)
    write_json_atomic(
        ROOT / "reports/localization/ntt07-blinded-review/aggregate.json",
        aggregate_receipt(result, output / "result.json"),
    )
    print(f"Technical readiness: {result['technical_readiness_passed']}; clinical annotations: 0")


if __name__ == "__main__":
    main()
