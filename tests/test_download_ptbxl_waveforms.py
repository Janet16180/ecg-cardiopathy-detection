"""PTB-XL waveform downloads verify, resume, repair and aggregate failures."""

import csv
import hashlib
import zipfile
from pathlib import Path

import pytest

from ecg_experiment.downloads import parse_checksums
from scripts.download_ptbxl_waveforms import download, validate_file, waveform_paths

RELATIVE = Path("records100/00000/00001_lr")
HEA = RELATIVE.with_suffix(".hea")
DAT = RELATIVE.with_suffix(".dat")


def _write_checksums(source: Path, paths: list[Path]) -> None:
    lines = [f"{hashlib.sha256((source / path).read_bytes()).hexdigest()}  ./{path.as_posix()}\n"
             for path in paths]
    (source / "SHA256SUMS.txt").write_text("".join(lines))


def _write_record(source: Path, relative: Path, dat_size: int = 12 * 10 * 100 * 2) -> None:
    (source / relative).parent.mkdir(parents=True, exist_ok=True)
    (source / relative.with_suffix(".hea")).write_text(f"{relative.name} 12 100 1000\n")
    (source / relative.with_suffix(".dat")).write_bytes(bytes(dat_size))


def _write_metadata(target: Path, relatives: list[Path]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with (target / "ptbxl_database.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename_lr", "filename_hr"])
        writer.writeheader()
        for relative in relatives:
            writer.writerow({"filename_lr": str(relative), "filename_hr": "records500/00000/00001_hr"})


@pytest.fixture
def mirror(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_record(source, RELATIVE)
    _write_checksums(source, [HEA, DAT])
    _write_metadata(target, [RELATIVE])
    return source, target


def test_download_verify_resume_and_repair(mirror: tuple[Path, Path]) -> None:
    source, target = mirror
    url = source.as_uri()
    first = download(target, workers=2, retries=0, base_url=url)
    assert first == {"records": 1, "files_downloaded": 2, "files_verified_existing": 0}
    second = download(target, workers=2, retries=0, base_url=url)
    assert second["files_verified_existing"] == 2
    (target / DAT).write_bytes(b"broken")
    third = download(target, workers=2, retries=0, base_url=url)
    assert third["files_downloaded"] == 1
    assert not (target / DAT.with_suffix(".dat.part")).exists()


def test_integrity_and_manifest_required(mirror: tuple[Path, Path]) -> None:
    source, target = mirror
    (source / DAT).write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="Integrity check failed"):
        download(target, workers=1, retries=0, base_url=source.as_uri())
    assert not (target / DAT).exists()
    assert not (target / DAT.with_suffix(".dat.part")).exists()

    (target / "SHA256SUMS.txt").write_text("0" * 64 + "  ./unrelated.dat\n")
    with pytest.raises(ValueError, match="lacks 2 requested files"):
        download(target, workers=1, retries=0, base_url=source.as_uri())


def test_failed_records_are_aggregated_after_others_finish(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    relatives = [Path(f"records100/00000/0000{number}_lr") for number in (1, 2, 3)]
    for relative in relatives:
        _write_record(source, relative)
    _write_checksums(source, [path for relative in relatives
                              for path in (relative.with_suffix(".hea"), relative.with_suffix(".dat"))])
    (source / relatives[0].with_suffix(".dat")).write_bytes(b"corrupt")
    (source / relatives[2].with_suffix(".dat")).unlink()
    _write_metadata(target, relatives)

    with pytest.raises(RuntimeError, match="Failed 2 records") as error:
        download(target, workers=1, retries=0, base_url=source.as_uri())
    assert "00001_lr" in str(error.value)
    assert "00003_lr" in str(error.value)
    assert (target / relatives[1].with_suffix(".dat")).is_file()


def test_header_contract_is_checked_before_checksum(tmp_path: Path) -> None:
    header = tmp_path / "00001_lr.hea"
    header.write_text("00001_lr 12 100 1000\n")
    digest = hashlib.sha256(header.read_bytes()).hexdigest()
    assert validate_file(header, "records100/00000/00001_lr.hea", 100, digest)
    assert not validate_file(header, "records100/00000/00001_lr.hea", 500, digest)
    assert not validate_file(header, "records100/00000/00001_lr.hea", 100, "0" * 64)
    header.write_text("00001_lr 12\n")
    assert not validate_file(header, "records100/00000/00001_lr.hea", 100,
                             hashlib.sha256(header.read_bytes()).hexdigest())
    assert not validate_file(tmp_path / "missing.hea", "records100/00000/missing.hea", 100, digest)


def test_unsafe_path_and_checksum_parser(mirror: tuple[Path, Path]) -> None:
    _, target = mirror
    with (target / "ptbxl_database.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename_lr"])
        writer.writeheader()
        writer.writerow({"filename_lr": "records100/../escape"})
    with pytest.raises(ValueError, match="Unsafe"):
        waveform_paths(target / "ptbxl_database.csv", 100)
    with pytest.raises(ValueError, match="Invalid SHA256SUMS line"):
        parse_checksums("not a checksum")
    single_space = "a" * 64 + " LICENSE.txt"
    assert parse_checksums(single_space) == {"LICENSE.txt": "a" * 64}


def test_archive_extracts_only_requested_verified_files(mirror: tuple[Path, Path]) -> None:
    source, target = mirror
    archive_path = source / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in (HEA, DAT):
            archive.write(source / path, f"ptb-xl-1.0.3/{path.as_posix()}")
        archive.writestr("ptb-xl-1.0.3/other.txt", "not requested")
    result = download(target, archive=True, base_url=source.as_uri(), archive_url=archive_path.as_uri())
    assert result["files_downloaded"] == 2
    assert not (target / "other.txt").exists()
