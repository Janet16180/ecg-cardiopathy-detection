import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import wfdb

from ecg_experiment.mimic import (
    audit, check_waveform, lock_selection, read_patients, required_checksums,
    select_patients, selection_hash, signal_hash,
)


class PrepareMimicTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_patient_selection_stable_whole_and_locked(self):
        path = self.root / "record_list.csv"
        rows = []
        for patient, count in (("10000001", 2), ("10000002", 3), ("10000003", 1)):
            for number in range(count):
                study = f"{40000000 + int(patient[-1]) * 10 + number:08d}"
                rows.append({"subject_id": patient, "study_id": study, "file_name": study,
                             "ecg_time": "ignored",
                             "path": f"files/p1000/p{patient}/s{study}/{study}"})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(reversed(rows))
        patients = read_patients(path)
        selected, subjects = select_patients(patients, seed=42, max_records=4)
        self.assertLessEqual(len(selected), 4)
        self.assertEqual({r[0] for r in selected}, set(subjects))
        for subject in subjects:
            self.assertEqual(sum(r[0] == subject for r in selected), len(patients[subject]))
        digest = selection_hash(selected, 42, 4, "a" * 64)
        self.assertEqual(digest, selection_hash(selected, 42, 4, "a" * 64))
        self.assertNotEqual(digest, selection_hash(selected, 43, 4, "a" * 64))
        output = self.root / "out"
        lock_selection(output, selected, digest, 42, 4, "a" * 64)
        lock_selection(output, selected, digest, 42, 4, "a" * 64)
        self.assertEqual(json.loads((output / "selection.json").read_text())["selection_sha256"], digest)
        with self.assertRaisesRegex(ValueError, "another patient selection"):
            lock_selection(output, selected, digest, 43, 4, "a" * 64)

    def test_manifest_requires_official_hashes(self):
        name = "files/p1000/p10000001/s40000001/40000001"
        sums = self.root / "SHA256SUMS.txt"
        sums.write_text("".join(f"{'a'*64} {file}\n" for file in
                                ("LICENSE.txt", "record_list.csv", name + ".hea", name + ".dat")))
        self.assertEqual(len(required_checksums(sums, {name})), 4)
        sums.write_text(sums.read_text().replace(name + ".dat", "other.dat"))
        with self.assertRaisesRegex(ValueError, "lacks 1 required"):
            required_checksums(sums, {name})

    def test_waveform_contract_reorders_leads_and_rejects_bad_rate(self):
        directory = self.root / "waveforms"
        directory.mkdir()
        names = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
        data = np.tile(np.arange(12, dtype=np.float64), (5000, 1)) / 200
        wfdb.wrsamp("sample", fs=500, units=["mV"] * 12, sig_name=names,
                    p_signal=data, fmt=["16"] * 12, adc_gain=[200] * 12,
                    baseline=[0] * 12, write_dir=str(directory))
        canonical = check_waveform(directory, "sample")
        self.assertEqual(canonical.shape, (12, 5000))
        self.assertAlmostEqual(float(canonical[4, 0]), 5 / 200)
        self.assertAlmostEqual(float(canonical[5, 0]), 4 / 200)
        self.assertTrue(np.isfinite(canonical).all())
        original = (directory / "sample.hea").read_text()
        (directory / "sample.hea").write_text(original.replace("sample 12 500 5000", "sample 12 250 5000"))
        with self.assertRaisesRegex(ValueError, "Expected 500 Hz"):
            check_waveform(directory, "sample")

    def test_exact_dedup_and_resume(self):
        rows = [("10000001", "40000001", "files/p1000/p10000001/s40000001/40000001"),
                ("10000002", "40000002", "files/p1000/p10000002/s40000002/40000002"),
                ("10000003", "40000003", "files/p1000/p10000003/s40000003/40000003")]
        output = self.root / "out"
        output.mkdir()
        first = np.ones((12, 5000), dtype=np.float32)
        other = np.zeros((12, 5000), dtype=np.float32)
        with patch("ecg_experiment.mimic.check_waveform", side_effect=[first, first, other]) as read:
            accepted, reasons = audit(rows, self.root, output, "selection", {signal_hash(other)})
            self.assertEqual(read.call_count, 3)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(reasons["exact_duplicate_mimic"], 1)
        self.assertEqual(reasons["exact_duplicate_ptbxl"], 1)
        with patch("ecg_experiment.mimic.check_waveform", side_effect=AssertionError("redecoded")):
            resumed, again = audit(rows, self.root, output, "selection", {signal_hash(other)})
        self.assertEqual(resumed, accepted)
        self.assertEqual(again, reasons)
        with self.assertRaisesRegex(ValueError, "different patient selection"):
            audit(rows, self.root, output, "other", {signal_hash(other)})


if __name__ == "__main__":
    unittest.main()
