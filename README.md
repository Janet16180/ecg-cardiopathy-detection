# ECG Cardiopathy Detection

Time series analysis of electrocardiogram (ECG) recordings for automatic detection of cardiopathies.

## Problem

The dataset contains 2000 ECG time series, of which only about 1% carry a reliable label. This makes
the task a semi-supervised / low-label-regime problem rather than a standard supervised
classification one: the main challenge is learning useful representations of the signal without
depending on labels, and then transferring those representations to a classifier trained on the
small labeled subset.

## Approach

Planned line of work:

1. Exploratory analysis of the ECG signals (sampling rate, length, noise, class balance in the
   labeled subset).
2. Preprocessing: filtering, normalization, segmentation into beats or fixed-length windows.
3. Self-supervised pretraining with contrastive learning on the full unlabeled set.
4. Fine-tuning / linear probing on the labeled subset.
5. Evaluation with metrics suited to imbalanced data (AUROC, AUPRC, per-class recall).

## Stack

- Python 3.10+
- PyTorch
- scikit-learn
- NumPy / pandas / SciPy
- matplotlib

## Repository status

Early stage. Code and data pipeline are not published yet.

## Data

The ECG dataset is not included in this repository. Place raw recordings under `data/raw/` locally.

## Team

- José Emiliano Luna López
- Andre Nicolai Gutierrez Bautista
- Clara Janet Rivera Medina

Sponsor: Gerardo Jesús Camacho González, TEC, Computer Science Department

## Project context

Master's project at Tecnológico de Monterrey. Status: accepted.

## License

Apache License 2.0. See [LICENSE](LICENSE).
