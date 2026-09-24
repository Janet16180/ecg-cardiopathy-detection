import csv
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.download_ptbxl_waveforms import download, parse_checksums, waveform_paths


class DownloadWaveformsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.target = self.root / "target"
        relative = Path("records100/00000/00001_lr")
        (self.source / relative).parent.mkdir(parents=True)
        (self.target / relative).parent.mkdir(parents=True)
        self.hea = relative.with_suffix(".hea")
        self.dat = relative.with_suffix(".dat")
        (self.source / self.hea).write_text("00001_lr 12 100 1000\n")
        (self.source / self.dat).write_bytes(bytes(12 * 10 * 100 * 2))
        lines = []
        for path in (self.hea, self.dat):
            digest = hashlib.sha256((self.source / path).read_bytes()).hexdigest()
            lines.append(f"{digest}  ./{path.as_posix()}\n")
        (self.source / "SHA256SUMS.txt").write_text("".join(lines))
        with (self.target / "ptbxl_database.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename_lr", "filename_hr"])
            writer.writeheader()
            writer.writerow({"filename_lr": str(relative), "filename_hr": "records500/00000/00001_hr"})

    def test_download_verify_resume_and_repair(self):
        url = self.source.as_uri()
        first = download(self.target, workers=2, retries=0, base_url=url)
        self.assertEqual(first, {"records": 1, "files_downloaded": 2, "files_verified_existing": 0})
        second = download(self.target, workers=2, retries=0, base_url=url)
        self.assertEqual(second["files_verified_existing"], 2)
        (self.target / self.dat).write_bytes(b"broken")
        third = download(self.target, workers=2, retries=0, base_url=url)
        self.assertEqual(third["files_downloaded"], 1)
        self.assertFalse((self.target / self.dat.with_suffix(".dat.part")).exists())

    def test_integrity_and_manifest_required(self):
        (self.source / self.dat).write_bytes(b"broken")
        with self.assertRaisesRegex(RuntimeError, "Integrity check failed"):
            download(self.target, workers=1, retries=0, base_url=self.source.as_uri())
        self.assertFalse((self.target / self.dat).exists())
        self.assertFalse((self.target / self.dat.with_suffix(".dat.part")).exists())

        (self.target / "SHA256SUMS.txt").write_text("0" * 64 + "  ./unrelated.dat\n")
        with self.assertRaisesRegex(ValueError, "lacks 2 requested files"):
            download(self.target, workers=1, retries=0, base_url=self.source.as_uri())

    def test_unsafe_path_and_checksum_parser(self):
        with (self.target / "ptbxl_database.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename_lr"])
            writer.writeheader()
            writer.writerow({"filename_lr": "records100/../escape"})
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            waveform_paths(self.target / "ptbxl_database.csv", 100)
        with self.assertRaises(ValueError):
            parse_checksums("not a checksum")
        single_space = "a" * 64 + " LICENSE.txt"
        self.assertEqual(parse_checksums(single_space), {"LICENSE.txt": "a" * 64})

    def test_archive_extracts_only_requested_verified_files(self):
        archive_path = self.source / "sample.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for path in (self.hea, self.dat):
                archive.write(self.source / path, f"ptb-xl-1.0.3/{path.as_posix()}")
            archive.writestr("ptb-xl-1.0.3/other.txt", "not requested")
        result = download(self.target, archive=True, base_url=self.source.as_uri(),
                          archive_url=archive_path.as_uri())
        self.assertEqual(result["files_downloaded"], 2)
        self.assertFalse((self.target / "other.txt").exists())


if __name__ == "__main__":
    unittest.main()
