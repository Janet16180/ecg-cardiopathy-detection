"""Reproducible source-stratified 25k selection from the frozen ECG cohort."""

from __future__ import annotations

from collections import Counter

import numpy as np


def select_indices(sources: list[str], size: int, seed: int) -> np.ndarray:
    """Sample exact source-proportional quotas without replacing cohort rows."""
    if not 0 < size <= len(sources):
        raise ValueError("Subset size must fit the source cohort")
    counts = Counter(sources)
    names = sorted(counts)
    exact = {name: size * counts[name] / len(sources) for name in names}
    quotas = {name: int(exact[name]) for name in names}
    remainder_order = sorted(names, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in remainder_order[:size - sum(quotas.values())]:
        quotas[name] += 1

    rng = np.random.default_rng(seed)
    groups = np.asarray(sources)
    selected = [rng.choice(np.flatnonzero(groups == name), size=quotas[name], replace=False)
                for name in names]
    indices = np.concatenate(selected).astype(np.int64)
    rng.shuffle(indices)
    if len(np.unique(indices)) != size:
        raise ValueError("Subset selection contains duplicates")
    return indices


def exposure_order(selected: np.ndarray, exposures: int, seed: int) -> np.ndarray:
    """Cycle through fresh seeded permutations until the fixed budget is met."""
    if len(selected) == 0 or exposures <= 0:
        raise ValueError("Subset and exposure budget must be positive")
    rng = np.random.default_rng(seed)
    epochs = [rng.permutation(selected) for _ in range((exposures + len(selected) - 1) // len(selected))]
    return np.concatenate(epochs)[:exposures]
