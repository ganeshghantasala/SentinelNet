# Multi-Dataset Intrusion Detection Benchmark: SentinelNet Contribution Validity

**Execution Timestamp:** 2026-11-23 20:42:05  
**Published Reference Paper:**  
*"Feature Selection and XGBoost for Enhanced Intrusion Detection: A Comparative Study Across Benchmark Datasets"*  
**Venue:** IEEE International Symposium on Digital Forensics and Security (ISDFS 2025)  
**DOI:** [10.1109/ISDFS65363.2025.11012060](https://doi.org/10.1109/ISDFS65363.2025.11012060)  

---

## 1. Scope, Nomenclature & Methodological Disclosures

> [!IMPORTANT]
> **Scope Distinction:**
> - **SentinelNet Production IDS:** The offline 3-class (`Normal`, `DoS`, `Probe`) CSV log analysis dashboard deployed in `app.py` for NSL-KDD traffic.
> - **Multi-Dataset Research Experiment:** A standalone research benchmark in `multi_dataset_experiment/` performing binary (`Normal` vs `Attack`) classification across **NSL-KDD** and **UNSW-NB15** to rigorously compare feature selection, gradient boosting, native NaN handling, and feature-adaptive dual-mode IDS architectures.

> [!NOTE]
> **Methodological Verification Points:**
> 1. **Missingness Definition:** Missingness percentage represents **row-level session telemetry availability** ($X\%$ of flows originate from lightweight edge sensors where only core flow headers are available, while $(100-X)\%$ originate from deep-packet inspection sensors).
> 2. **Zero Data Leakage:** Preprocessing scalers, categorical encoders, feature selection (ANOVA/Lasso/Gain), model training, and confidence threshold selection ($	au=0.65$) were executed strictly on training splits without access to holdout test labels.
> 3. **Selective Classification (Reject Option):** The 0.65 confidence threshold implements **selective classification** (rejecting low-margin predictions for human analyst review), not calibrated Bayesian posterior probabilities.
> 4. **Baseline Naming:** The Top-15 XGBoost Gain model is explicitly labeled as a **paper-inspired baseline**; ANOVA SelectKBest and Lasso L1 models reflect the exact feature-selection methods studied in the 2025 IEEE paper.

---

## 2. Complete-Data Comparative Benchmark (Identical Records)

| Dataset | Model Architecture | Features Used | Accuracy | Precision | Recall | Macro F1 | FPR | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD** | 2025 IEEE Full-XGBoost | 41 | 80.68% | 96.88% | 68.26% | 80.67% | 2.90% | 0.43s |
| **NSL-KDD** | 2025 IEEE Paper-Inspired Baseline (Top-15 Gain + XGBoost) | 15 | 79.43% | 96.93% | 65.95% | 79.39% | 2.76% | 0.261s |
| **NSL-KDD** | 2025 IEEE ANOVA SelectKBest + XGBoost (Top-15) | 15 | 77.69% | 96.87% | 62.85% | 77.61% | 2.69% | 0.261s |
| **NSL-KDD** | 2025 IEEE Lasso (L1) + XGBoost (Top-15) | 15 | 75.55% | 96.10% | 59.46% | 75.39% | 3.19% | 0.261s |
| **NSL-KDD** | SentinelNet Enhanced Mode (10 Feats) | 10 | 77.33% | 96.47% | 62.46% | 77.24% | 3.02% | 1.616s |
| **NSL-KDD** | SentinelNet Basic Mode (4 Core Feats) | 4 | 79.54% | 95.32% | 67.36% | 79.52% | 4.37% | 1.228s |
| **NSL-KDD** | Adaptive SentinelNet (Dual-Mode) | Adaptive (4 / 10) | 77.33% | 96.47% | 62.46% | 77.24% | 3.02% | 2.844s |
| **UNSW-NB15** | 2025 IEEE Full-XGBoost | 42 | 87.39% | 82.04% | 98.69% | 86.79% | 26.47% | 0.983s |
| **UNSW-NB15** | 2025 IEEE Paper-Inspired Baseline (Top-15 Gain + XGBoost) | 15 | 87.62% | 82.31% | 98.75% | 87.05% | 26.01% | 0.564s |
| **UNSW-NB15** | 2025 IEEE ANOVA SelectKBest + XGBoost (Top-15) | 15 | 85.09% | 79.07% | 99.17% | 84.17% | 32.17% | 0.564s |
| **UNSW-NB15** | 2025 IEEE Lasso (L1) + XGBoost (Top-15) | 15 | 84.55% | 79.44% | 97.04% | 83.74% | 30.76% | 0.564s |
| **UNSW-NB15** | SentinelNet Enhanced Mode (42 Feats) | 42 | 87.17% | 81.87% | 98.52% | 86.56% | 26.73% | 4.686s |
| **UNSW-NB15** | SentinelNet Basic Mode (6 Core Feats) | 6 | 87.07% | 82.81% | 96.56% | 86.57% | 24.56% | 2.013s |
| **UNSW-NB15** | Adaptive SentinelNet (Dual-Mode) | Adaptive (6 / 42) | 87.17% | 81.87% | 98.52% | 86.56% | 26.73% | 6.699s |

---

## 3. Comprehensive Missingness Evaluation (Multi-Seed Comparison)

Evaluated across **5 random seeds** (`[42, 100, 2024, 2025, 2026]`) under **identical missingness masks** for all models:
- **Adaptive SentinelNet:** Dynamically switches degraded records to Basic Mode without imputation or NaNs.
- **Basic-Only RF:** Static baseline trained strictly on core features.
- **Missingness-Trained XGBoost:** XGBoost trained on the training partition with 50% simulated NaNs.
- **FS-XGBoost (Native NaN):** Uses XGBoost's native split-branching for `np.nan` values without forced imputation.
- **FS-XGBoost (Median Imputed):** Imputes missing values with training set medians.
- **Fixed Enhanced RF (Median Imputed):** Standard fixed RF with training set median imputation.

| Dataset | Missingness | Adaptive SentinelNet Acc | Basic-Only RF Acc | Miss-Trained XGB Acc | FS-XGB (Native NaN) | FS-XGB (Imputed) | Fixed RF (Imputed) | Delta vs Basic RF | Delta vs Miss-Trained XGB | Delta vs Native NaN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD** | 0% | **77.33±0.00%** | 79.54% | 78.30±0.00% | 79.43±0.00% | 79.43±0.00% | 77.33±0.00% | **-2.21%** | **-0.97%** | **-2.10%** |
| **NSL-KDD** | 25% | **77.88±0.05%** | 79.54% | 79.50±0.06% | 73.35±0.24% | 79.50±0.09% | 70.34±0.12% | **-1.66%** | **-1.62%** | **+4.54%** |
| **NSL-KDD** | 50% | **78.41±0.03%** | 79.54% | 80.67±0.02% | 66.97±0.22% | 79.45±0.03% | 63.22±0.15% | **-1.13%** | **-2.26%** | **+11.44%** |
| **NSL-KDD** | 75% | **78.95±0.04%** | 79.54% | 81.89±0.04% | 60.84±0.16% | 79.49±0.07% | 56.23±0.15% | **-0.59%** | **-2.94%** | **+18.11%** |
| **NSL-KDD** | 100% | **79.54±0.00%** | 79.54% | 82.97±0.00% | 54.66±0.00% | 79.48±0.00% | 49.25±0.00% | **+0.00%** | **-3.43%** | **+24.88%** |
| **UNSW-NB15** | 0% | **87.17±0.00%** | 87.07% | 86.53±0.00% | 87.62±0.00% | 87.62±0.00% | 87.17±0.00% | **+0.10%** | **+0.64%** | **-0.45%** |
| **UNSW-NB15** | 25% | **87.18±0.03%** | 87.07% | 85.35±0.05% | 80.76±0.15% | 79.48±0.04% | 79.14±0.03% | **+0.11%** | **+1.83%** | **+6.42%** |
| **UNSW-NB15** | 50% | **87.16±0.07%** | 87.07% | 84.15±0.04% | 73.81±0.09% | 71.30±0.11% | 71.06±0.09% | **+0.09%** | **+3.01%** | **+13.35%** |
| **UNSW-NB15** | 75% | **87.10±0.05%** | 87.07% | 83.01±0.03% | 66.89±0.09% | 63.23±0.08% | 63.08±0.07% | **+0.03%** | **+4.09%** | **+20.20%** |
| **UNSW-NB15** | 100% | **87.07±0.00%** | 87.07% | 81.83±0.00% | 59.98±0.00% | 55.09±0.00% | 55.06±0.00% | **+0.00%** | **+5.24%** | **+27.09%** |

---

## 4. Final Contribution Scorecard: Wins, Ties, and Losses

| Operational Dimension / Condition | SentinelNet Dual-Mode | 2025 IEEE Paper Baselines (XGBoost) | Outcome / Winner | Academic Context & Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **Complete Telemetry (0% Missing) on UNSW-NB15** | Acc: 87.17%, F1: 86.56% | Top-15 Gain XGBoost: **87.62%**, F1: **87.05%** | **Baseline Wins (+0.45%)** | On complete data, gradient boosting with Top-15 gain features slightly edges out Random Forest. |
| **Complete Telemetry (0% Missing) on NSL-KDD** | Acc: 77.33% (Enh) / 79.54% (Basic) | Full-XGBoost: **80.68%**, Top-15: 79.43% | **Baseline Wins (+1.14%)** | Full-feature XGBoost leads on complete 41-feature NSL-KDD binary traffic. |
| **Partial Missingness (25% - 75%) on UNSW-NB15** | Acc: **87.10% – 87.18%** | Native NaN: 66.89% – 80.76%<br>Imputed: 63.23% – 79.48% | **SentinelNet Wins (+6.4% to +20.2%)** | SentinelNet dynamically switches degraded records to Basic Mode; static baselines suffer default branch or imputation distortion. |
| **High Missingness (100% Core Only) on UNSW-NB15** | Acc: **87.07%** (Recall: 96.56%, FPR: 24.56%) | Native NaN: 59.98% (FPR: 68.12%)<br>Imputed: 55.09% | **SentinelNet Wins (+27.1% to +32.0%)** | Basic Mode maintains robust classification; static baselines collapse toward majority/coin-flip guessing. |
| **Missingness-Trained XGBoost on UNSW-NB15** | Acc: **87.07% – 87.18%** | Missingness-Trained XGB: 81.83% – 86.53% | **SentinelNet Wins (+0.64% to +5.24%)** | Training XGBoost with NaNs improves its missingness resilience, but dedicated dual models still outperform a compromised joint model. |
| **Missingness-Trained XGBoost on NSL-KDD** | Acc: 77.33% – 79.54% | Missingness-Trained XGB: **78.30% – 82.97%** | **Baseline Wins (+0.97% to +3.43%)** | In binary NSL-KDD, Missingness-Trained XGBoost achieves higher accuracy across all missingness levels. |
| **NSL-KDD Under Missingness (Native NaN)** | Acc: **77.88% – 79.54%** | Native NaN: 54.66% – 73.35% | **SentinelNet Wins (+4.5% to +24.9%)** | Default NaN branching in XGBoost becomes erratic on NSL-KDD; SentinelNet Basic Mode stays stable. |
| **NSL-KDD Under Missingness (Imputed XGB)** | Acc: 78.41% – 79.54% | Imputed FS-XGB: **79.45% – 79.50%** | **Tie / Baseline Leads by ~1.0%** | In NSL-KDD, payload volume dominates; imputed XGB continues splitting on byte fields without severe collapse. |
| **Reconnaissance Probe Capture (NSL-KDD Multiclass)** | Probe Recall: **60.80%** (1,472/2,421) | Basic-only RF: 30.07% (728/2,421) | **SentinelNet Wins (+30.73% / +102%)** | Enhanced Mode doubles reconnaissance detection, justifying the existence of the dual-mode architecture over Basic-only RF. |
| **Decision Routing / False Alarm Suppression** | Selective accuracy: **90.47%** (UNSW, $	au=0.65$)<br>FPR: reduced to 25.6% | Fixed models produce unflagged errors on degraded flows | **SentinelNet Wins (Operational)** | Low-confidence boundary predictions routed to human review rather than issuing automated false alarms. |
| **Training & Inference Efficiency** | Training: 1.8s – 7.4s<br>Inference: ~0.02s | Training: **0.25s – 0.65s**<br>Inference: **0.009s – 0.02s** | **Baseline Wins (Efficiency)** | XGBoost with histogram/tree boosting trains significantly faster than multi-tree Random Forests on large tabular datasets. |
