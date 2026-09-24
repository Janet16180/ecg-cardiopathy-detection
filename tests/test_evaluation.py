import json
import unittest

from ecg_experiment.evaluation import (
    metrics,
    patient_bootstrap,
    scenario_ppv,
    select_threshold,
)


class EvaluationTests(unittest.TestCase):
    def test_threshold_inclusive_ties_and_impossible_selection(self):
        self.assertEqual(select_threshold([1, 1, 1, 0], [0.9, 0.5, 0.5, 0.8], 2 / 3), 0.5)
        self.assertEqual(select_threshold([1, 1, 0], [0.9, 0.2, 0.8], 0.5), 0.9)
        with self.assertRaisesRegex(ValueError, "both classes"):
            select_threshold([0, 0], [0.1, 0.2])
        with self.assertRaises(ValueError):
            select_threshold([0, 1], [0.1, float("nan")])

    def test_metrics_counts_calibration_and_json_safety(self):
        result = metrics([1, 0, 1, 0], [0.8, 0.7, 0.3, 0.2], 0.5)
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]),
                         (1, 1, 1, 1))
        self.assertEqual(result["sensitivity"], 0.5)
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["brier"], 0.265)
        self.assertAlmostEqual(result["ece_10_bins"], 0.45)
        json.dumps(result, allow_nan=False)

        none_denominators = metrics([0, 0], [0.0, 0.0], 1.0)
        self.assertIsNone(none_denominators["sensitivity"])
        self.assertIsNone(none_denominators["precision"])
        self.assertIsNone(none_denominators["auroc"])
        self.assertIsNone(none_denominators["average_precision"])
        self.assertIsNone(none_denominators["f1"])
        json.dumps(none_denominators, allow_nan=False)

    def test_patient_bootstrap_deterministic_and_clustered(self):
        y = [1, 1, 0, 0, 1, 0]
        prob = [0.8, 0.7, 0.2, 0.1, 0.6, 0.3]
        patients = ["a", "a", "b", "b", "c", "c"]
        first = patient_bootstrap(y, prob, patients, 0.5, repeats=100, seed=7)
        second = patient_bootstrap(y, prob, patients, 0.5, repeats=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"auroc", "average_precision", "sensitivity",
                                      "specificity", "precision", "brier"})
        self.assertEqual(first["sensitivity"], [1.0, 1.0])
        json.dumps(first, allow_nan=False)

        # Every cluster is one class; single-class resamples are skipped, not scored.
        sparse = patient_bootstrap([1, 0], [0.9, 0.1], ["a", "b"], 0.5,
                                   repeats=20, seed=1)
        self.assertEqual(sparse["auroc"], [1.0, 1.0])

    def test_scenario(self):
        result = scenario_ppv(0.9, 0.95, 0.01)
        self.assertAlmostEqual(result["expected_tp_per_1000"], 9)
        self.assertAlmostEqual(result["expected_fp_per_1000"], 49.5)
        self.assertAlmostEqual(result["expected_referrals_per_1000"], 58.5)
        self.assertAlmostEqual(result["ppv"], 9 / 58.5)
        self.assertIsNone(scenario_ppv(0, 1, 0)["ppv"])


if __name__ == "__main__":
    unittest.main()
