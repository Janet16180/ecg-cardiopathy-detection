import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.prepare_ptbxl import prepare


class PreparePTBXLTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = self.root / "metadata"
        self.metadata.mkdir()
        with (self.metadata / "scp_statements.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["", "diagnostic", "diagnostic_class"])
            writer.writeheader()
            writer.writerows([
                {"": "NORM", "diagnostic": "1.0", "diagnostic_class": "NORM"},
                {"": "MI", "diagnostic": "1.0", "diagnostic_class": "MI"},
                {"": "AFIB", "diagnostic": "", "diagnostic_class": ""},
                {"": "SR", "diagnostic": "", "diagnostic_class": ""},
            ])
        self.rows = [self.row(i, f"p{i}", str(i % 8 + 1), "{'MI': 0.0}" if i % 2 else "{'NORM': 0.0, 'SR': 0.0}")
                     for i in range(1, 11)]
        self.rows += [
            self.row(11, "p1", "2", "{'MI': 0.0}"),
            self.row(12, "p1", "2", "{'NORM': 100.0, 'AFIB': 0.0}"),
            self.row(13, "p2", "3", "{'AFIB': 0.0}"),
            self.row(14, "v", "9", "{'MI': 0.0}"),
            self.row(15, "v2", "9", "{'NORM': 100.0, 'AFIB': 0.0}"),
            self.row(16, "t", "10", "{'NORM': 0.0}"),
            self.row(17, "t2", "10", "{'AFIB': 100.0}"),
        ]
        self.write_database()

    @staticmethod
    def row(ecg_id, patient_id, fold, codes):
        return {"ecg_id": str(ecg_id), "patient_id": patient_id, "strat_fold": fold,
                "scp_codes": codes, "report": "private diagnosis", "filename_lr": f"lr/{ecg_id}",
                "filename_hr": f"hr/{ecg_id}"}

    def write_database(self):
        with (self.metadata / "ptbxl_database.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def read(self, path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_manifests_hide_labels_and_keep_patient_selection(self):
        output = self.root / "one"
        summary = prepare(self.metadata, output, 0.1, 42)
        labeled = self.read(output / "labeled_train.csv")
        unlabeled = self.read(output / "unlabeled_train.csv")
        ssl = self.read(output / "all_train_ssl.csv")
        validation = self.read(output / "validation.csv")
        test = self.read(output / "test.csv")
        audit = {r["ecg_id"]: r for r in self.read(output / "audit" / "metadata.csv")}

        self.assertEqual(summary["selected_eligible_patients"], 1)
        self.assertAlmostEqual(summary["actual_eligible_patient_fraction"], 0.1)
        self.assertEqual({r["ecg_id"] for r in labeled} | {r["ecg_id"] for r in unlabeled},
                         {r["ecg_id"] for r in ssl})
        self.assertEqual(len(ssl), 13)
        self.assertEqual([r["ecg_id"] for r in validation], ["14"])
        self.assertEqual([r["ecg_id"] for r in test], ["16"])
        self.assertEqual(audit["12"]["target"], "")
        self.assertEqual(audit["12"]["reason"], "norm_with_other_codes")
        self.assertEqual(audit["13"]["target"], "")
        self.assertEqual(audit["1"]["target"], "1")
        self.assertEqual(audit["2"]["target"], "0")
        self.assertEqual(summary["excluded_unresolved_validation_records"], 1)
        self.assertEqual(summary["excluded_unresolved_test_records"], 1)
        for name in ("unlabeled_train.csv", "all_train_ssl.csv"):
            self.assertEqual(set(self.read(output / name)[0]),
                             {"ecg_id", "patient_id", "filename_lr", "filename_hr"})
            self.assertNotIn("private diagnosis", (output / name).read_text())
        self.assertEqual(set(labeled[0]), {"ecg_id", "patient_id", "filename_lr", "filename_hr", "target"})
        self.assertEqual(set(json.loads((output / "summary.json").read_text())["splits"]),
                         {"all_train_ssl", "labeled_train", "unlabeled_train", "validation", "test"})

        # Every resolved recording of a chosen patient is labeled; unresolved stays unlabeled.
        selected = {r["patient_id"] for r in labeled}
        self.assertTrue(all(r["patient_id"] in selected for r in labeled))
        self.assertTrue(all(audit[r["ecg_id"]]["target"] == "" or
                            r["patient_id"] not in selected for r in unlabeled))

    def test_repeatability_and_fraction(self):
        first = self.root / "first"
        second = self.root / "second"
        prepare(self.metadata, first, 0.2, 42)
        prepare(self.metadata, second, 0.2, 42)
        self.assertEqual((first / "labeled_train.csv").read_bytes(),
                         (second / "labeled_train.csv").read_bytes())
        self.assertEqual(len({r["patient_id"] for r in self.read(first / "labeled_train.csv")}), 2)
        with self.assertRaises(FileExistsError):
            prepare(self.metadata, first, 0.2, 42)
        with self.assertRaises(ValueError):
            prepare(self.metadata, self.root / "bad", 0, 42)

    def test_patient_fold_conflict_and_duplicate_record_rejected(self):
        self.rows.append(self.row(18, "p1", "9", "{'MI': 100.0}"))
        self.write_database()
        with self.assertRaisesRegex(ValueError, "spans official folds"):
            prepare(self.metadata, self.root / "conflict")
        self.rows[-1]["patient_id"] = "new"
        self.rows[-1]["ecg_id"] = "1"
        self.write_database()
        with self.assertRaisesRegex(ValueError, "Duplicate ecg_id"):
            prepare(self.metadata, self.root / "duplicate")


if __name__ == "__main__":
    unittest.main()
