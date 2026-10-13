"""Build the local LocBench waveform cache (PTB-XL folds 1-8 and the ECGAD records)."""

from __future__ import annotations

from ecg_experiment.locbench_cache import CACHE_DIR, build_cache, open_cache


def main() -> None:
    """Build the cache and print its size."""
    build_cache()
    signals, index = open_cache()
    print(CACHE_DIR, signals.shape, index.strat_fold.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
