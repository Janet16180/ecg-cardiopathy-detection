"""Training recovery rejects held-out folds and paths before opening waveforms."""

import csv
import hashlib
from pathlib import Path

import pytest

from ecg_experiment import ptb_training_recovery as recovery


@pytest.mark.parametrize(
    ("fold", "waveform_path", "message"),
    [(9, "records500/00000/00001_hr", "Held-out PTB fold"),
     (1, "../outside", "Unsafe PTB training waveform path")],
)
def test_recovery_rejects_unapproved_waveforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fold: int, waveform_path: str, message: str,
) -> None:
    """A validated manifest digest cannot override waveform access boundaries."""
    manifest = tmp_path / recovery.MANIFEST
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ecg_id", "patient_id", "filename_hr"])
        writer.writerows((str(index), "1.0", waveform_path) for index in range(1, 17419))
    monkeypatch.setattr(recovery, "EXPECTED_MANIFEST_MD5", hashlib.md5(manifest.read_bytes()).hexdigest())
    metadata = tmp_path / recovery.RAW / "ptbxl_database.csv"
    metadata.parent.mkdir(parents=True)
    with metadata.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ecg_id", "patient_id", "filename_hr", "strat_fold"])
        writer.writerow(["1", "1.0", waveform_path, str(fold)])

    def forbidden_read(*args: object) -> None:
        raise AssertionError("Rejected input reached waveform decoding")

    monkeypatch.setattr(recovery, "read_record", forbidden_read)
    with pytest.raises(ValueError, match=message):
        recovery.recover(tmp_path, Path("audit"), 1, 1)
    assert not (tmp_path / "audit").exists()
