"""Test archive membership without accessing public ECGs or the network."""

import io
import zipfile

import pytest

from ecg_experiment.ssl_synthetic_ludb_acquisition import EXTENSIONS, admitted_members


def make_archive(extra=None, missing=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for record in range(1, 201):
            for extension in EXTENSIONS:
                name = f"release/data/{record}.{extension}"
                if name != missing:
                    archive.writestr(name, b"artificial")
        if extra:
            archive.writestr(extra, b"unadmitted")
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def test_exact_manifest_excludes_unrequested_files():
    with make_archive(extra="release/unrequested.txt") as archive:
        selected = admitted_members(archive)
        assert len(selected) == 2800
        assert "data/200.v6" in selected


def test_missing_file_stops_acquisition():
    with make_archive(missing="release/data/1.i") as archive, pytest.raises(ValueError, match="lacks"):
        admitted_members(archive)


def test_unsafe_path_stops_acquisition():
    with make_archive(extra="release/../unsafe") as archive, pytest.raises(ValueError, match="Unsafe"):
        admitted_members(archive)
