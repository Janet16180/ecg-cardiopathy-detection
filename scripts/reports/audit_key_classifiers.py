"""Audit saved KEY-04 models, predictions, folds and reference draws."""

from ecg_experiment.key_classifier_audit import run

if __name__ == "__main__":
    print(run())
