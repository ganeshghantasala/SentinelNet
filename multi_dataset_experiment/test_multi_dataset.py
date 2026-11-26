"""
multi_dataset_experiment/test_multi_dataset.py

Automated Test Suite for Multi-Dataset Benchmarking & 2025 IEEE Baseline Verification.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data_loaders import load_nsl_kdd_binary, load_unsw_nb15_binary
from baseline_xgboost_2025 import PaperBaselineXGBoost2025
from adaptive_sentinelnet_binary import AdaptiveSentinelNetBinary


class TestMultiDatasetBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
        cls.results_dir = os.path.join(SCRIPT_DIR, "results")

    def test_01_nsl_kdd_binary_loading(self):
        """Test 1: Ingestion and preprocessing of NSL-KDD binary dataset."""
        data = load_nsl_kdd_binary(base_dir=self.base_dir)
        self.assertEqual(data["dataset_name"], "NSL-KDD")
        self.assertEqual(data["stats"]["train_samples"], 125973)
        self.assertEqual(data["stats"]["test_samples"], 22544)
        self.assertEqual(data["X_train_full"].shape[1], 41)
        self.assertEqual(data["X_train_enh"].shape[1], 10)
        self.assertEqual(data["X_train_basic"].shape[1], 4)
        self.assertEqual(set(np.unique(data["y_train"])), {0, 1})
        self.assertEqual(set(np.unique(data["y_test"])), {0, 1})

    def test_02_unsw_nb15_binary_loading(self):
        """Test 2: Ingestion and preprocessing of UNSW-NB15 binary dataset."""
        data = load_unsw_nb15_binary(base_dir=self.base_dir)
        self.assertEqual(data["dataset_name"], "UNSW-NB15")
        self.assertEqual(data["stats"]["train_samples"], 175341)
        self.assertEqual(data["stats"]["test_samples"], 82332)
        self.assertEqual(data["X_train_full"].shape[1], 42)
        self.assertEqual(data["X_train_basic"].shape[1], 6)
        self.assertEqual(set(np.unique(data["y_train"])), {0, 1})
        self.assertEqual(set(np.unique(data["y_test"])), {0, 1})

    def test_03_paper_baseline_xgb_feature_selection(self):
        """Test 3: 2025 IEEE paper baseline Top-K selection and predictions."""
        # Use quick synthetic subset to verify model mechanics
        np.random.seed(42)
        X_dummy = np.random.randn(200, 20)
        y_dummy = np.random.randint(0, 2, size=200)

        model = PaperBaselineXGBoost2025(top_k=5, n_estimators=10, random_state=42)
        model.fit(X_dummy, y_dummy)

        self.assertEqual(len(model.selected_indices), 5)
        self.assertEqual(len(model.feature_importances_), 20)

        preds = model.predict(X_dummy, use_fs=True)
        self.assertEqual(len(preds), 200)
        self.assertTrue(set(preds).issubset({0, 1}))

        eval_res = model.evaluate(X_dummy, y_dummy, use_fs=True)
        self.assertIn("accuracy", eval_res)
        self.assertIn("precision", eval_res)
        self.assertIn("recall", eval_res)
        self.assertIn("macro_f1", eval_res)
        self.assertIn("fpr", eval_res)

    def test_04_adaptive_sentinelnet_binary_switching(self):
        """Test 4: SentinelNet dual-mode adaptive switching and uncertainty routing."""
        np.random.seed(42)
        X_enh = np.random.randn(100, 10)
        X_bas = X_enh[:, :4]
        y = np.random.randint(0, 2, size=100)

        sentinel = AdaptiveSentinelNetBinary(confidence_threshold=0.65, n_estimators=10, random_state=42)
        sentinel.fit(X_enh, X_bas, y)

        # Half complete, half missing
        missing_mask = np.zeros(100, dtype=bool)
        missing_mask[50:] = True

        res = sentinel.predict_adaptive(X_enh, X_bas, missing_mask=missing_mask)
        self.assertEqual(len(res["predictions"]), 100)
        self.assertEqual(np.sum(res["modes"] == "Enhanced"), 50)
        self.assertEqual(np.sum(res["modes"] == "Basic"), 50)
        self.assertTrue(set(res["decision_statuses"]).issubset({"ACCEPTED", "REVIEW REQUIRED"}))

    def test_05_benchmark_results_artifacts_integrity(self):
        """Test 5: Verification of all generated experimental result CSVs and markdown."""
        expected_files = [
            "nsl_kdd_comparison.csv",
            "unsw_nb15_comparison.csv",
            "combined_summary.csv",
            "missingness_stress_test.csv",
            "uncertainty_evaluation.csv",
            "metadata.json",
            "MULTI_DATASET_REPORT.md"
        ]
        for fn in expected_files:
            fp = os.path.join(self.results_dir, fn)
            self.assertTrue(os.path.exists(fp), f"Missing artifact: {fp}")
            self.assertGreater(os.path.getsize(fp), 0, f"Empty artifact: {fp}")

        df_comb = pd.read_csv(os.path.join(self.results_dir, "combined_summary.csv"))
        self.assertEqual(len(df_comb), 14) # 7 models x 2 datasets (Full-XGB, Top15-Gain, ANOVA, Lasso, Sentinel-Enh, Sentinel-Bas, Adaptive)
        self.assertIn("accuracy", df_comb.columns)
        self.assertIn("dataset", df_comb.columns)

        df_miss = pd.read_csv(os.path.join(self.results_dir, "missingness_stress_test.csv"))
        self.assertEqual(len(df_miss), 10) # 5 missing levels x 2 datasets
        self.assertIn("delta_acc_vs_fs_nan", df_miss.columns)
        self.assertIn("delta_acc_vs_fs_imp", df_miss.columns)


if __name__ == "__main__":
    unittest.main()
