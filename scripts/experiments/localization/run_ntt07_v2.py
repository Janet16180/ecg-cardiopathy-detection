"""Execute the separate physical-layout repair for the unchanged NTT-07 cohort."""

from ecg_experiment.localization.blinded_packet_v2 import run_packet

if __name__ == "__main__":
    result = run_packet()
    print(f"Repaired technical readiness: {result['technical_readiness_passed']}; clinical annotations: 0")
