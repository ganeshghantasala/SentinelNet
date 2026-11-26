"""
multi_dataset_experiment/adaptive_sentinelnet_binary.py

Adaptive Dual-Mode Network Flow IDS (Binary Normal vs Attack):
Implements SentinelNet's feature-adaptive switching and uncertainty-aware decision routing.

Architecture:
- Basic Mode: Trained exclusively on robust core flow telemetry.
- Enhanced Mode: Trained on extended contextual network telemetry.
- Dynamic Switching: When optional contextual features are absent/degraded,
  traffic is classified via Basic Mode instead of failing or relying on fragile imputation.
- Uncertainty Routing: Predictions with maximum estimated probability < threshold
  are routed to 'REVIEW REQUIRED' for human analyst verification.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


class AdaptiveSentinelNetBinary:
    def __init__(self, confidence_threshold=0.65, n_estimators=100, random_state=42):
        self.confidence_threshold = confidence_threshold
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.enhanced_model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
        self.basic_model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)

        self.basic_indices = None
        self.enh_indices = None
        self.train_medians_enh = None
        self.train_time_enh = 0.0
        self.train_time_basic = 0.0

    def fit(self, X_train_enh, X_train_basic, y_train, basic_indices=None, enh_indices=None):
        self.basic_indices = basic_indices
        self.enh_indices = enh_indices
        self.train_medians_enh = np.median(X_train_enh, axis=0)

        # 1. Fit Enhanced Model
        t0 = time.time()
        self.enhanced_model.fit(X_train_enh, y_train)
        self.train_time_enh = time.time() - t0

        # 2. Fit Basic Model
        t1 = time.time()
        self.basic_model.fit(X_train_basic, y_train)
        self.train_time_basic = time.time() - t1

        return self

    def predict_adaptive(self, X_test_enh, X_test_basic, missing_mask=None):
        """
        Dynamically selects Enhanced or Basic model depending on missing_mask per flow.
        """
        n_samples = len(X_test_enh)
        if missing_mask is None:
            missing_mask = np.zeros(n_samples, dtype=bool)

        preds = np.zeros(n_samples, dtype=int)
        confs = np.zeros(n_samples, dtype=float)
        probas = np.zeros((n_samples, 2), dtype=float)
        modes = np.empty(n_samples, dtype=object)

        # Complete records -> Enhanced Mode
        enh_mask = ~missing_mask
        if np.any(enh_mask):
            p_enh = self.enhanced_model.predict_proba(X_test_enh[enh_mask])
            preds[enh_mask] = np.argmax(p_enh, axis=1)
            confs[enh_mask] = np.max(p_enh, axis=1)
            probas[enh_mask] = p_enh
            modes[enh_mask] = "Enhanced"

        # Missing records -> Basic Mode
        if np.any(missing_mask):
            p_bas = self.basic_model.predict_proba(X_test_basic[missing_mask])
            preds[missing_mask] = np.argmax(p_bas, axis=1)
            confs[missing_mask] = np.max(p_bas, axis=1)
            probas[missing_mask] = p_bas
            modes[missing_mask] = "Basic"

        decision_status = np.where(confs >= self.confidence_threshold, "ACCEPTED", "REVIEW REQUIRED")

        return {
            "predictions": preds,
            "confidences": confs,
            "probabilities": probas,
            "modes": modes,
            "decision_statuses": decision_status
        }

    def evaluate_adaptive(self, X_test_enh, X_test_basic, y_test, missing_mask=None):
        t0 = time.time()
        res = self.predict_adaptive(X_test_enh, X_test_basic, missing_mask=missing_mask)
        infer_time = time.time() - t0

        preds = res["predictions"]
        dec_status = res["decision_statuses"]
        accepted_mask = (dec_status == "ACCEPTED")

        cm_overall = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm_overall.ravel()

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Accepted subset metrics (Uncertainty routing guardrail)
        accepted_coverage = float(np.mean(accepted_mask))
        if np.any(accepted_mask):
            acc_accepted = accuracy_score(y_test[accepted_mask], preds[accepted_mask])
            macro_f1_accepted = f1_score(y_test[accepted_mask], preds[accepted_mask], average='macro', zero_division=0)
        else:
            acc_accepted = 0.0
            macro_f1_accepted = 0.0

        return {
            "model_name": "Adaptive SentinelNet (Dual-Mode)",
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "fpr": round(float(fpr), 4),
            "coverage": round(float(accepted_coverage), 4),
            "accepted_accuracy": round(float(acc_accepted), 4),
            "accepted_macro_f1": round(float(macro_f1_accepted), 4),
            "train_time_sec": round(float(self.train_time_enh + self.train_time_basic), 3),
            "infer_time_sec": round(float(infer_time), 4),
            "confusion_matrix": cm_overall.tolist()
        }
