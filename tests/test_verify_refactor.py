"""Maintenance verification must reject corrupted history and source drift."""

import hashlib
import json
import tarfile

import pytest

from scripts.verify_refactor import verify


def make_snapshot(root):
    source = root / "ecg_experiment/model.py"
    source.parent.mkdir()
    source.write_text("original = True\n")
    pause = root / "outputs/refactor_pause"
    pause.mkdir(parents=True)
    archive = pause / "source_before_refactor.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="ecg_experiment/model.py")
    (pause / "source_hashes.json").write_text(json.dumps({
        "ecg_experiment/model.py": hashlib.sha256(source.read_bytes()).hexdigest(),
    }))
    (pause / "pause.json").write_text(json.dumps({
        "source_archive": archive.relative_to(root).as_posix(),
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }))
    (root / "docs").mkdir()
    (root / "docs/experiment-queue.json").write_text(json.dumps({
        "scheduling": {"state": "paused_by_user"},
    }))
    return source, archive


def test_preserved_sources_and_pause_are_verified(tmp_path):
    source, _ = make_snapshot(tmp_path)
    result = verify(tmp_path)
    assert result["archive_entries_verified"] == 1
    assert result["training_launched"] is False
    source.write_text("changed = True\n")
    with pytest.raises(ValueError, match="Frozen scientific source changed"):
        verify(tmp_path)


def test_archive_corruption_is_rejected(tmp_path):
    _, archive = make_snapshot(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        verify(tmp_path)
