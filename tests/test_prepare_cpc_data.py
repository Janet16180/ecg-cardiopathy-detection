import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import resample_poly

from scripts.data.prepare_cpc_data import (
    build_cache, read_locked_prefix, read_ptb_rows, resample_halves,
    verify_selected_files,
)


class CpcDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_half_resampling_keeps_boundary_independent(self):
        raw = np.zeros((12, 5000), dtype=np.float32)
        raw[:, 2499] = 1
        actual = resample_halves(raw, 250)
        self.assertEqual(actual.shape, (12, 2500))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual[:, 1250:], 0)
        np.testing.assert_allclose(actual[:, :1250],
                                   resample_poly(raw[:, :2500], 1, 2, axis=1))
        self.assertEqual(resample_halves(raw, 100).shape, (12, 1000))
        with self.assertRaisesRegex(ValueError, "finite"):
            resample_halves(np.full((12, 5000), np.nan, dtype=np.float32), 250)

    def test_locked_prefix_stops_before_partial_patient(self):
        official = self.root / "record_list.csv"
        rows = []
        for patient, count in (("10000001", 3), ("10000002", 2), ("10000003", 2)):
            for offset in range(count):
                study = str(40000000 + int(patient[-1]) * 10 + offset)
                rows.append({"subject_id": patient, "study_id": study, "file_name": study,
                             "ecg_time": "unused", "path": f"files/p1000/p{patient}/s{study}/{study}"})
        with official.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        from scripts.prepare_mimic_ssl import read_patients, select_patients, selection_hash
        patients = read_patients(official)
        parent_rows, _ = select_patients(patients, 42, 7)
        parent = self.root / "parent"
        parent.mkdir()
        (parent / "selection.json").write_text(json.dumps({
            "seed": 42, "max_records": 7,
            "record_list_sha256": hashlib.sha256(official.read_bytes()).hexdigest(),
            "selection_sha256": selection_hash(parent_rows, 42, 7, hashlib.sha256(official.read_bytes()).hexdigest())}))
        with (parent / "selected_records.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("subject_id", "study_id", "path"))
            writer.writerows(parent_rows)
        chosen, subjects, _, _ = read_locked_prefix(parent, 4, 42, official)
        self.assertLessEqual(len(chosen), 4)
        self.assertEqual(len(set(row[0] for row in chosen)), len(subjects))
        for subject in subjects:
            self.assertEqual(sum(row[0] == subject for row in chosen), len(patients[subject]))
        with self.assertRaisesRegex(ValueError, "prefix"):
            (parent / "selected_records.csv").write_text("subject_id,study_id,path\nwrong,wrong,wrong\n")
            read_locked_prefix(parent, 4, 42, official)

    def test_official_hash_verification_fails_closed(self):
        raw = self.root / "raw"
        name = "files/p1000/p10000001/s40000001/40000001"
        for relative, content in (("record_list.csv", b"list"), ("LICENSE.txt", b"license"),
                                  (name + ".hea", b"header"), (name + ".dat", b"data")):
            path = raw / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        sums = raw / "SHA256SUMS.txt"
        with sums.open("w") as stream:
            for relative in ("record_list.csv", "LICENSE.txt", name + ".hea", name + ".dat"):
                stream.write(f"{hashlib.sha256((raw / relative).read_bytes()).hexdigest()} {relative}\n")
        selected = [("10000001", "40000001", name)]
        self.assertEqual(verify_selected_files(selected, raw, sums, raw / "record_list.csv")
                         ["verified_records"], 1)
        (raw / (name + ".dat")).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_selected_files(selected, raw, sums, raw / "record_list.csv")

    def test_cache_rows_ids_and_resume_validation(self):
        row = {"ecg_id": "1", "patient_id": "10.0", "source": "ptbxl", "split": "train",
               "raw_dir": str(self.root), "filename_hr": "record"}
        mimic = {"ecg_id": "mimic:40000001", "patient_id": "mimic:10000001",
                 "raw_dir": str(self.root), "filename_hr": "mimic"}
        metadata = {"selection_sha256": "selection", "manifest_sha256": "manifest"}
        raw = np.zeros((12, 5000), dtype=np.float32)
        with patch("scripts.data.prepare_cpc_data.read_record", return_value=raw) as reader:
            info = build_cache([row], [mimic], {"all_train_ssl.csv": "hash"}, metadata,
                               self.root / "cache", 250)
            self.assertEqual(reader.call_count, 2)
        cache = self.root / "cache"
        self.assertEqual(info["shape"], [2, 12, 2500])
        self.assertEqual(np.load(cache / "signals.npy", mmap_mode="r").shape, (2, 12, 2500))
        self.assertEqual(np.load(cache / "ecg_ids.npy").tolist(), ["1", "mimic:40000001"])
        with patch("scripts.data.prepare_cpc_data.read_record", side_effect=AssertionError("redecoded")):
            build_cache([row], [mimic], {"all_train_ssl.csv": "hash"}, metadata, cache, 250)
        with self.assertRaisesRegex(ValueError, "different inputs"):
            build_cache([row], [mimic], {"all_train_ssl.csv": "changed"}, metadata, cache, 250)

    def test_cache_resumes_at_flushed_checkpoint(self):
        rows = [{"ecg_id": str(index), "patient_id": str(index), "source": "ptbxl",
                 "split": "train", "raw_dir": str(self.root), "filename_hr": str(index)}
                for index in range(101)]
        metadata = {"selection_sha256": "selection", "manifest_sha256": "manifest"}
        raw = np.zeros((12, 5000), dtype=np.float32)
        count = 0

        def fail_after_checkpoint(*_):
            nonlocal count
            count += 1
            if count == 101:
                raise RuntimeError("interrupted")
            return raw

        cache = self.root / "cache"
        with patch("scripts.data.prepare_cpc_data.read_record", side_effect=fail_after_checkpoint):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                build_cache(rows, [], {}, metadata, cache, 250)
        self.assertEqual(json.loads((cache / "progress.json").read_text())["completed_rows"], 100)
        with patch("scripts.data.prepare_cpc_data.read_record", return_value=raw) as reader:
            build_cache(rows, [], {}, metadata, cache, 250)
            self.assertEqual(reader.call_count, 1)
        self.assertEqual(np.load(cache / "signals.npy", mmap_mode="r").shape, (101, 12, 2500))


if __name__ == "__main__":
    unittest.main()
