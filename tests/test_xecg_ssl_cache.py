"""Verify interrupted cache preparation preserves identities and waveform rows."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.data.prepare_xecg_ssl import prepare


class SSLCacheTests(unittest.TestCase):
    def test_interrupted_prefix_resumes_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            rows = []
            for i in range(130):
                for suffix in (".hea", ".dat"):
                    (raw / f"{i}{suffix}").write_bytes(str(i).encode())
                rows.append(dict(ecg_id=str(i), patient_id=str(i), source="ptbxl",
                                 split="train", raw_dir=str(raw), filename_hr=str(i)))
            requested = (rows, {"fixture": "stable"})

            def read(_, relative):
                return np.full((12, 5000), int(relative) + 1, dtype=np.float32)

            def interrupted(directory, relative):
                if relative == "128":
                    raise RuntimeError("simulated interruption")
                return read(directory, relative)

            with patch("scripts.data.prepare_xecg_ssl.selected_rows", return_value=requested), \
                 patch("scripts.data.prepare_xecg_ssl.read_record", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    prepare(root / "resumed", workers=1)
            with patch("scripts.data.prepare_xecg_ssl.selected_rows", return_value=requested), \
                 patch("scripts.data.prepare_xecg_ssl.read_record", side_effect=read):
                actual = prepare(root / "resumed", workers=1)
                expected = prepare(root / "fresh", workers=1)
                np.testing.assert_array_equal(np.load(root / "resumed/views.npy"),
                                              np.load(root / "fresh/views.npy"))
                self.assertEqual(actual["views_sha256"], expected["views_sha256"])
                self.assertEqual(actual["ecg_ids"], [str(i) for i in range(130)])
                self.assertTrue(actual["all_train_only"])
            with patch("scripts.data.prepare_xecg_ssl.selected_rows", return_value=(rows, {"fixture": "changed"})):
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    prepare(root / "resumed", workers=1)


if __name__ == "__main__":
    unittest.main()
