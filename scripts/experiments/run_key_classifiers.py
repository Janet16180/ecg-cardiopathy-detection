"""Run the prospectively frozen matched condition-classifier study."""

from ecg_experiment.key_classifier_study import run

if __name__ == "__main__":
    result = run()
    print(result["primary"])
    print(result["decision"])
