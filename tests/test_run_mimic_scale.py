"""Orchestrator guards against incomplete outputs and mismatched update budgets."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.experiments.finetune_pretrained import check_adaptation_budget
from scripts.experiments.run_mimic_scale import ensure_stage, wait_for_artifact


class MimicScaleRunnerTest(unittest.TestCase):
    def test_bounded_checkpoint_is_accepted_only_at_exact_budget(self):
        bounded = {"max_updates": 7000, "batch_size": 32, "epochs": 14}
        check_adaptation_budget(bounded, [{"updates": 7000, "seen_examples": 224000}])
        for history in ([], [{"updates": 6999, "seen_examples": 223968}],
                        [{"updates": 7000, "seen_examples": 223968}]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                check_adaptation_budget(bounded, history)
        check_adaptation_budget({"epochs": 2}, [{}, {}])
        with self.assertRaises(ValueError):
            check_adaptation_budget({"epochs": 2}, [{}])

    def test_existing_incomplete_stage_is_not_relaunched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            stage.mkdir()
            (stage / "history.json").write_text("[]")
            with mock.patch("scripts.experiments.run_mimic_scale.run_stage") as launch:
                with self.assertRaisesRegex(RuntimeError, "Incomplete existing"):
                    ensure_stage(root, "ptb_adaptation", stage, ["python"], lambda: {},
                                 resumable=True)
                launch.assert_not_called()

    def test_reused_wait_pid_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "metrics.json"
            with mock.patch("scripts.experiments.run_mimic_scale.process_cmdline",
                            return_value="python another_task.py"):
                with self.assertRaisesRegex(RuntimeError, "no longer identifies"):
                    wait_for_artifact(4607, "finetune_pretrained", artifact, lambda: None)

    def test_finetune_stage_resumes_from_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            stage.mkdir()
            (stage / "resume.pt").write_bytes(b"checkpoint")
            (stage / "history.json").write_text("[]")
            with mock.patch("scripts.experiments.run_mimic_scale.run_stage", return_value={"ok": True}) as launch:
                result = ensure_stage(root, "ptb_finetune", stage, ["python"], lambda: {},
                                      resumable=True, completion_marker="metrics.json")
            self.assertEqual(result, {"ok": True})
            self.assertEqual(launch.call_args.args[2], ["python", "--resume"])

    def test_partial_finetune_metrics_do_not_block_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            stage.mkdir()
            (stage / "resume.pt").write_bytes(b"checkpoint")
            (stage / "metrics.json").write_text("{}")
            with mock.patch("scripts.experiments.run_mimic_scale.run_stage", return_value={}) as launch:
                ensure_stage(root, "ptb_finetune", stage, ["python"],
                             mock.Mock(side_effect=FileNotFoundError),
                             resumable=True, completion_marker="metrics.json")
            self.assertEqual(launch.call_args.args[2], ["python", "--resume"])


if __name__ == "__main__":
    unittest.main()
