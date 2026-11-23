# SentinelNet Adaptive Dual-Mode Intrusion Detection System
## Comprehensive Comparative Benchmark Evaluation Report

> **Experimental Framing & Academic Attribution**  
> This study is a rigorous, reference-inspired comparative evaluation modeled after the methodological framework presented in:  
> *M. H. Shah et al., "Investigating Novel Machine Learning Based Intrusion Detection Models for NSL-KDD Data Sets," 2023 International Conference on IT and Industrial Technologies (ICIT), IEEE, 2023, DOI: [10.1109/ICIT59216.2023.10335831](https://ieeexplore.ieee.org/document/10335831/).*  
> **Note**: This experiment is explicitly framed as an **independent, reference-inspired comparative benchmark**, not an exact paper reproduction. All experimental procedures, training runs, inference benchmarks, and evaluations were conducted natively on genuine NSL-KDD benchmark data without data leakage or post-hoc metric manipulation.

---

## 1. Executive Summary & Experimental Setup

SentinelNet was evaluated across six machine learning architectures to assess the performance of its **Adaptive Dual-Mode** design (**Basic Mode: 4 features** and **Enhanced Mode: 10 features**) compared to traditional shallow baselines (Logistic Regression, K-Nearest Neighbors, Decision Trees) and a full-dimensional Conventional Random Forest (all 41 original NSL-KDD features).

### Experimental Controls & Integrity
- **Benchmark Source**: Official Canadian Institute for Cybersecurity (UNB) NSL-KDD dataset:
  - **Training Split**: `data/nsl_kdd/KDDTrain+.txt` (124,926 flows in 3-class scope).
  - **Testing Split**: `data/nsl_kdd/KDDTest+.txt` (19,592 flows in 3-class scope).
- **Classification Scope**: Exact 3-class target space: **Normal**, **DoS**, and **Probe** (R2L and U2R traffic excluded, matching the production application scope).
- **Leakage Prevention**: All `StandardScaler` transformations and one-hot categorical encodings were fit **strictly and exclusively** on `KDDTrain+`. No test data was visible during preprocessing or training.
- **Reproducibility**: Global deterministic seeds (`random_state=42`) were enforced across all models, splits, and simulations.

```
+---------------------------------------------------------------------------------------------------+
|                                      EXPERIMENTAL PIPELINE                                        |
+---------------------------------------------------------------------------------------------------+
|  KDDTrain+ (124,926 flows)        -> Fit Scalers & Encoders -> Train 6 Models                     |
|  KDDTest+  (19,592 flows)         -> Transform via Train Fit  -> Independent Blind Evaluation     |
|  Uncertainty Engine (KDDTrain+)   -> Validate Confidence Tau  -> Evaluate Tradeoff on KDDTest+    |
|  Adaptive Engine (Corrupted/Miss) -> Feature Schema Fallback  -> Telemetry Degradation Simulation |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Model Performance Comparison

### Primary Evaluation Metrics on Independent Test Set (`KDDTest+`, $n=19,592$)

| Model Architecture | Raw Feat. | Encoded Feat. | Training Time (s) | Inference Latency ($\mu\text{s/flow}$) | Throughput (flows/s) | Accuracy | Weighted F1 | Macro F1 | Weighted Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (10 Feat)** | 10 | 22 | 1.30 s | **0.15 $\mu\text{s}$** | 6,478,408 | 80.35% | 0.8003 | 0.7376 | 0.8117 |
| **K-Nearest Neighbors (10 Feat)** | 10 | 22 | **0.07 s** | 59.65 $\mu\text{s}$ | 16,765 | **85.97%** | **0.8569** | **0.8257** | **0.8701** |
| **Decision Tree (10 Feat)** | 10 | 22 | 0.30 s | 0.18 $\mu\text{s}$ | 5,606,685 | 83.72% | 0.8363 | 0.7974 | 0.8474 |
| **Basic SentinelNet (4 Feat)** | 4 | 6 | 0.80 s | 1.48 $\mu\text{s}$ | 677,261 | 81.88% | 0.8016 | 0.7084 | 0.8125 |
| **Enhanced SentinelNet (10 Feat)** | 10 | 22 | 1.24 s | 1.55 $\mu\text{s}$ | 645,382 | **85.16%** | **0.8476** | **0.8126** | **0.8616** |
| **Conventional RF (41 Feat)** | 41 | 122 | 2.59 s | 2.18 $\mu\text{s}$ | 458,451 | **85.53%** | **0.8516** | **0.8177** | **0.8699** |

---

## 3. Class-Level Detection & Probe Sensitivity Analysis

The core driver for introducing **Enhanced SentinelNet** was overcoming the low Probe sensitivity of 4-feature network flows without incurring the computational overhead of full 41-feature ingestion.

### Class-by-Class Precision, Recall, and F1-Scores

| Model Architecture | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (10 Feat)** | Normal<br>DoS<br>Probe | 79.78%<br>91.13%<br>56.03% | 93.73%<br>71.51%<br>53.90% | 0.8619<br>0.8014<br>0.5495 | 9,711<br>7,460<br>2,421 |
| **K-Nearest Neighbors (10 Feat)** | Normal<br>DoS<br>Probe | 81.30%<br>**95.67%**<br>83.21% | 96.94%<br>78.47%<br>**65.10%** | 0.8844<br>**0.8622**<br>**0.7305** | 9,711<br>7,460<br>2,421 |
| **Decision Tree (10 Feat)** | Normal<br>DoS<br>Probe | 81.35%<br>94.72%<br>67.60% | 93.54%<br>75.98%<br>**68.15%** | 0.8702<br>0.8432<br>0.6787 | 9,711<br>7,460<br>2,421 |
| **Basic SentinelNet (4 Feat)** | Normal<br>DoS<br>Probe | 83.81%<br>79.86%<br>75.28% | 96.11%<br>**80.17%**<br>30.07% | **0.8954**<br>0.8002<br>0.4298 | 9,711<br>7,460<br>2,421 |
| **Enhanced SentinelNet (10 Feat)** | Normal<br>DoS<br>Probe | 80.70%<br>93.78%<br>**84.55%** | **97.05%**<br>77.59%<br>**60.80%** | 0.8813<br>0.8492<br>**0.7074** | 9,711<br>7,460<br>2,421 |
| **Conventional RF (41 Feat)** | Normal<br>DoS<br>Probe | 79.78%<br>**96.21%**<br>**87.49%** | **97.41%**<br>78.32%<br>60.10% | 0.8772<br>**0.8635**<br>0.7125 | 9,711<br>7,460<br>2,421 |

### Confusion Matrices Across All Models

The confusion matrices are organized strictly in canonical order: **Normal**, **DoS**, and **Probe**.

```
1. Logistic Regression (10 Feat)       2. K-Nearest Neighbors (10 Feat)
             Pred_Norm Pred_DoS Pred_Prb                 Pred_Norm Pred_DoS Pred_Prb
True_Normal:   9,102     331      278       True_Normal:   9,414      74      223
True_DoS:      1,379   5,335      746       True_DoS:      1,511   5,854       95
True_Probe:      928     188    1,305       True_Probe:      654     191    1,576

3. Decision Tree (10 Feat)              4. Basic SentinelNet (4 Feat)
             Pred_Norm Pred_DoS Pred_Prb                 Pred_Norm Pred_DoS Pred_Prb
True_Normal:   9,084      81      546       True_Normal:   9,333     142      236
True_DoS:      1,547   5,668      245       True_DoS:      1,476   5,981        3
True_Probe:      536     235    1,650       True_Probe:      327   1,366      728

5. Enhanced SentinelNet (10 Feat)      6. Conventional Random Forest (41 Feat)
             Pred_Norm Pred_DoS Pred_Prb                 Pred_Norm Pred_DoS Pred_Prb
True_Normal:   9,425      81      205       True_Normal:   9,459      67      185
True_DoS:      1,608   5,788       64       True_DoS:      1,594   5,843       23
True_Probe:      646     303    1,472       True_Probe:      803     163    1,455
```

---

## 4. Computational Efficiency & Latency Benchmarks

In live network flow environments, model throughput and inference latency determine whether an intrusion detection system can process packets without dropping buffers.

### Key Observations:
1. **The KNN Latency Bottleneck**: Although K-Nearest Neighbors achieved the highest raw test accuracy (85.97%), its inference latency is **59.65 $\mu\text{s}$ per flow** (throughput: only 16,765 flows/second). This is **~38.5x slower** than Enhanced SentinelNet. In high-speed gigabit uplinks, KNN would cause severe buffer overruns and dropped flows.
2. **Feature Dimensionality vs. Inference**: Conventional Random Forest requires 122 encoded features (due to 70 service one-hot columns), increasing tree depth and inference latency to **2.18 $\mu\text{s}$ per flow** (2.59s training time). Enhanced SentinelNet operates at **1.55 $\mu\text{s}$ per flow** with an 82% smaller feature vector, delivering **645,382 flows/second**.

---

## 5. Uncertainty-Aware Decision Mechanism

To prevent false confidence on ambiguous boundary traffic, SentinelNet employs an uncertainty-aware routing mechanism with threshold $\tau$. When the maximum predicted class probability estimate satisfies $\max_c P(y=c|x) < \tau$, the flow is flagged as `REVIEW REQUIRED` and escalated for human inspection.

### Empirical Threshold Sweep on `KDDTest+` ($n=19,592$)

```
+---------------------------------------------------------------------------------------------------------+
|                                ENHANCED SENTINELNET UNCERTAINTY SWEEP                                   |
+-----------+--------------+-----------------------+-------------------------+---------------+------------+
| Threshold | Coverage (%) | Accepted Accuracy (%) | Accepted Probe Rec. (%) | Accepted Flows| Review Qty |
+-----------+--------------+-----------------------+-------------------------+---------------+------------+
|    0.50   |    98.89%    |        85.72%         |          62.80%         |     19,375    |     217    |
|    0.55   |    97.59%    |        86.68%         |          64.08%         |     19,120    |     472    |
|    0.60   |    95.11%    |        88.41%         |          64.00%         |     18,634    |     958    |
|    0.65 * |    93.76%    |        89.47%         |          66.87%         |     18,370    |   1,222    |
|    0.70   |    92.49%    |        90.42%         |          69.78%         |     18,120    |   1,472    |
|    0.75   |    90.85%    |        91.16%         |          73.95%         |     17,800    |   1,792    |
|    0.80   |    89.02%    |        91.59%         |          77.15%         |     17,440    |   2,152    |
|    0.85   |    87.85%    |        92.05%         |          79.54%         |     17,211    |   2,381    |
|    0.90   |    86.17%    |        93.07%         |          82.57%         |     16,883    |   2,709    |
+-----------+--------------+-----------------------+-------------------------+---------------+------------+
* Selected Operational Production Threshold (tau = 0.65)
```

### Key Insights:
- At the recommended threshold $\tau = 0.65$, Enhanced SentinelNet routes **6.24% of flows (1,222 records)** to manual review.
- On the remaining **93.76% accepted flows (18,370 records)**, accuracy increases from **85.16% to 89.47%** (+4.31 percentage points).
- Accepted Probe recall concurrently climbs from **60.80% to 66.87%**, demonstrating that uncertain predictions disproportionately contain ambiguous reconnaissance attacks that benefit from analyst scrutiny.

---

## 6. Telemetry Degradation & Adaptive Fallback Simulation

Real-world network monitoring environments frequently experience partial feature loss due to packet truncation, NetFlow exporter timeouts, or malformed protocol fields. A fixed 10-feature or 41-feature classifier will fail completely or drop records when expected features are absent. SentinelNet's **Adaptive Dual-Mode engine** dynamically evaluates each row and falls back to Basic Mode when enhanced features are missing or corrupted.

### Ingestion Degradation Across 5 Simulated Environments

| Telemetry Ingestion Scenario | Enhanced Flows (%) | Fallback Flows (%) | Native Basic (%) | Overall Accuracy | Weighted F1 | Probe Recall | DoS Recall | Normal Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario 1: Ideal Telemetry**<br>*(100% Enhanced 10-feature)* | 100.0% | 0.0% | 0.0% | **85.16%** | **0.8476** | **60.80%** | 77.59% | **97.05%** |
| **Scenario 2: High Availability**<br>*(75% Enh, 15% Corrupted, 10% Basic)* | 75.0% | 15.0% | 9.9% | **84.36%** | 0.8371 | 52.91% | 78.32% | 96.84% |
| **Scenario 3: Balanced Heterogeneous**<br>*(50% Enh, 25% Corrupted, 25% Basic)* | 50.1% | 25.0% | 25.0% | **83.58%** | 0.8265 | 45.48% | 79.02% | 96.58% |
| **Scenario 4: High Degradation**<br>*(25% Enh, 35% Corrupted, 40% Basic)* | 25.1% | 35.2% | 39.7% | **82.74%** | 0.8146 | 37.75% | 79.65% | 96.33% |
| **Scenario 5: Pure Legacy Flows**<br>*(0% Enhanced, 100% 4-feature Basic)* | 0.0% | 0.0% | 100.0% | **81.88%** | 0.8016 | 30.07% | **80.17%** | 96.11% |

### Key Insights:
- As telemetry conditions deteriorate from Scenario 1 to Scenario 5, system accuracy degrades gracefully along a smooth curve from **85.16% to 81.88%** (-3.28%), rather than suffering a catastrophic failure.
- Normal and DoS detection remain exceptionally stable across all degradation levels (Normal recall: 97.05% $\rightarrow$ 96.11%; DoS recall: 77.59% $\rightarrow$ 80.17%).
- Probe recall transitions smoothly from 60.80% down to the 30.07% basic floor as reconnaissance-specific metrics are lost.

---

## 7. Comparative Assessment: Outperformance vs. Limitations

A balanced evaluation requires identifying both where SentinelNet surpasses baselines and where operational trade-offs exist.

### Where SentinelNet Outperforms Baselines

1. **Probe Detection Leap over Basic Baseline**:
   Enhanced SentinelNet increases Probe recall from **30.07% (Basic)** to **60.80% (Enhanced)**—an improvement of **+30.73 percentage points** (+102% relative gain). The inclusion of service rates (`same_srv_rate`, `dst_host_diff_srv_rate`) and TCP error rates (`serror_rate`) provides the necessary discriminatory signal to capture stealthy reconnaissance scans.
2. **Parity with 41-Feature Models at 82% Lower Dimensionality**:
   Enhanced SentinelNet (10 features, 22 encoded) achieves **85.16% accuracy and 60.80% Probe recall**, matching the 41-feature Conventional Random Forest (**85.53% accuracy, 60.10% Probe recall**). In fact, Enhanced SentinelNet achieves slightly higher Probe recall (+0.70%) while training **2.1x faster** (1.24s vs 2.59s) and reducing inference latency from 2.18 $\mu\text{s}$ to 1.55 $\mu\text{s}$ per flow.
3. **Line-Rate Throughput over Instance-Based Models**:
   While KNN achieves 85.97% accuracy, its 59.65 $\mu\text{s}$ latency limits throughput to ~16,700 flows/sec. Enhanced SentinelNet delivers **645,382 flows/second** (1.55 $\mu\text{s/flow}$)—over **38 times higher throughput**—making it practical for deployment on high-throughput enterprise gateways.
4. **Selective Uncertainty Triaging**:
   Unlike standard classifiers that emit overconfident predictions on ambiguous boundary flows, SentinelNet flags borderline traffic (`REVIEW REQUIRED`), elevating accepted decision accuracy to **89.47%** without hallucinating unobserved classes.
5. **Zero-Downtime Telemetry Resilience**:
   Rigid monolithic models break when telemetry features are missing. SentinelNet's per-row schema inspection dynamically routes partial or corrupted flows to Basic Mode, ensuring uninterrupted visibility.

---

### Where SentinelNet Does Not Outperform & Known Limitations

1. **KNN Accuracy & Probe Superiority**:
   K-Nearest Neighbors achieved higher raw test accuracy (**85.97% vs 85.16%**) and higher Probe recall (**65.10% vs 60.80%**) than Enhanced SentinelNet. In localized, non-linear geometric spaces, instance-based search captures subtle cluster boundaries that tree ensembles occasionally split across. However, KNN's $\mathcal{O}(N)$ test complexity makes it computationally unviable for streaming flow analysis.
2. **Marginal Accuracy Deficit vs 41-Feature Random Forest**:
   The 41-feature Conventional Random Forest achieved **85.53% accuracy (+0.37%)** and **78.32% DoS recall (+0.73%)** compared to Enhanced SentinelNet's 85.16% and 77.59%. The 70 one-hot encoded `service` categories in the 41-feature dataset provide specialized discriminators for targeted port floods (e.g., `smb`, `telnet`, `smtp`), which are collapsed in 10-feature telemetry.
3. **Basic Mode DoS Recall Paradox**:
   Counterintuitively, Basic SentinelNet achieved higher DoS recall (**80.17%**) than Enhanced SentinelNet (**77.59%**). In Basic Mode, decision trees partition traffic primarily along high-volume payload thresholds (`src_bytes`, `dst_bytes`) and connection durations. In Enhanced Mode, the presence of multi-connection rate metrics (`count`, `srv_count`, `serror_rate`) leads some distributed, low-rate DoS variations to be clustered near Normal or Probe boundaries, resulting in 64 DoS flows misclassified as Probe.
4. **Offline Flow Scope & Synthetic Test Artifacts**:
   As with all NSL-KDD benchmark studies, `KDDTest+` contains novel attack distributions and synthetic artifacts not present in `KDDTrain+`. Furthermore, SentinelNet is strictly an offline CSV/flow analyzer, not a real-time inline packet-filtering firewall.

---

## 8. Preserved Assets & Automated Test Verification

All changes adhere to zero-regression architectural guidelines. The working production application, dark-mode cybersecurity dashboard, model cache, and automated testing suite remain fully operational.

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\ganes\OneDrive\Documents\Desktop\NIDS Project
collected 34 items

test_sentinelnet.py ..................................                   [100%]
============================= 34 passed in 13.97s =============================
```

### Artifact & Data Manifest
All experimental data, scripts, and visualizations have been preserved in the repository:
- **Experiment Script**: [benchmark_evaluation.py](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/benchmark_evaluation.py)
- **Comparative Metrics CSV**: [comparative_model_performance.csv](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/data/benchmark_results/comparative_model_performance.csv)
- **Class-Level Metrics CSV**: [per_class_performance.csv](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/data/benchmark_results/per_class_performance.csv)
- **Uncertainty Sweep CSV**: [uncertainty_threshold_sweep.csv](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/data/benchmark_results/uncertainty_threshold_sweep.csv)
- **Adaptive Simulation CSV**: [adaptive_simulation_results.csv](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/data/benchmark_results/adaptive_simulation_results.csv)
- **Confusion Matrices JSON**: [confusion_matrices.json](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/data/benchmark_results/confusion_matrices.json)
- **Publication Charts**:
  - [1_comparative_model_performance.png](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/data/benchmark_results/charts/1_comparative_model_performance.png)
  - [2_computational_efficiency_latency.png](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/data/benchmark_results/charts/2_computational_efficiency_latency.png)
  - [3_confusion_matrices_all_models.png](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/data/benchmark_results/charts/3_confusion_matrices_all_models.png)
  - [4_uncertainty_tradeoff_curves.png](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/data/benchmark_results/charts/4_uncertainty_tradeoff_curves.png)
  - [5_adaptive_degradation_simulation.png](file:///c:/Users/ganes/OneDrive/Documents/Desktop/NIDS%20Project/data/benchmark_results/charts/5_adaptive_degradation_simulation.png)
