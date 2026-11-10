"""Run the frozen reference ECG audit."""

from ecg_experiment.reference_ecg_audit import run

if __name__ == "__main__":
    result = run()
    print(result["waveform_review"])
