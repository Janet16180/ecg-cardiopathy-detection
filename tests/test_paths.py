from pathlib import Path

import pytest

from ecg_experiment import ROOT, paths
from ecg_experiment.paths import from_stored, to_stored


@pytest.fixture
def repository(monkeypatch, tmp_path):
    root = tmp_path / "clone"
    (root / "data" / "raw").mkdir(parents=True)
    monkeypatch.setattr(paths, "ROOT", root)
    return root


def test_stored_form_is_relative_posix_and_round_trips():
    path = ROOT / "data" / "raw" / "ptb-xl" / "1.0.3"
    stored = to_stored(path)
    assert stored == "data/raw/ptb-xl/1.0.3"
    assert from_stored(stored) == path


def test_stored_form_does_not_depend_on_working_directory(monkeypatch, repository, tmp_path):
    monkeypatch.chdir(repository / "data")
    assert to_stored(Path("raw") / ".." / "raw") == "data/raw"
    monkeypatch.chdir(tmp_path)
    assert from_stored("data/raw") == repository / "data" / "raw"


def test_symlinked_directories_keep_their_repository_name(repository, tmp_path):
    shared = tmp_path / "shared-storage"
    shared.mkdir()
    (repository / "data" / "processed").symlink_to(shared, target_is_directory=True)
    assert to_stored(repository / "data" / "processed" / "cache") == "data/processed/cache"


def test_paths_outside_the_repository_are_rejected(repository, tmp_path):
    with pytest.raises(ValueError, match="is not in the subpath"):
        to_stored(tmp_path / "elsewhere")


def test_legacy_absolute_paths_move_under_the_current_root(repository):
    legacy = "/home/JanetRivera/ecg-cardiopathy-detection/data/raw/mimic-iv-ecg/1.0"
    assert from_stored(legacy) == repository / "data" / "raw" / "mimic-iv-ecg" / "1.0"


def test_other_absolute_paths_are_unchanged(repository):
    assert from_stored("/mnt/shared/checkpoints/model.pt") == Path("/mnt/shared/checkpoints/model.pt")
