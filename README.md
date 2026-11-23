# SentinelNet – Adaptive Dual-Mode Machine Learning Network Flow IDS

## Quick start (simplified interface)

1. Extract the project and open its folder in CMD.
2. Run `python -m pip install -r requirements.txt` if dependencies are not installed.
3. Run `python app.py` and open `http://127.0.0.1:5000`.
4. Choose a CSV, click **Analyze Network Logs**, or click **Try sample data** for a quick demonstration.
5. View predictions and download the complete annotated CSV report. Expand **Model performance & contribution** for benchmark comparisons and **Previous scans** for local history.

The web interface is `templates/simple.html`. The existing trained Random Forest architecture and dataset have not been changed. Missing optional metadata is rendered as JSON null, preventing browser JSON parsing failures. The basic upload fields are `src_bytes`, `dst_bytes`, `duration`, and `protocol_type`.

The simplified **Model performance & contribution** section presents only (1) the full-feature RF versus Enhanced SentinelNet and (2) the published-paper-inspired CART versus adaptive SentinelNet, with an interactive selector for five measured missing-feature scenarios. The default is 50% Basic-only records. The comparison makes complete-data accuracy limitations visible. The reference is Ingre, Yadav & Soni (2017), DOI 10.1007/978-3-319-63645-0_23. Reproducible scripts and all recorded experimental results are in `paper_comparison/`. The original IEEE reference below remains related work, not the source of the new CART comparison.

### Measured contribution: adaptation to missing fields

Run `python evaluate_missing_features.py` to repeat the new stress test. Training uses the filtered KDDTrain+ records; evaluation uses the same 19,592 KDDTest+ records for both models. With the six additional fields missing for a reproducible random subset, SentinelNet switches to Basic mode; the fixed Enhanced RF instead uses medians and the most frequent flag from training data. Results are in `data/benchmark_results/missing_features_comparison.csv`. At approximately 50% missing availability, adaptive accuracy was 83.42% and fixed imputation accuracy was 70.92%, a 12.50 percentage point gain. These are local experiments under simulated missingness, not a reproduction of the IEEE paper or proof of performance on modern live traffic. The cited IEEE paper is a related candidate, Shah et al. (2023), DOI 10.1109/ICIT59216.2023.10335831.

The legacy architecture and test documentation below is retained for reference; manual flow input and the telemetry/log dashboards are not exposed on the simplified main screen.

## 1. Project Objective

**SentinelNet** is a benchmark-trained, feature-adaptive, uncertainty-aware Network Flow Telemetry Analysis and Machine Learning Intrusion Detection System (IDS). Trained on the genuine Canadian Institute for Cybersecurity **NSL-KDD** benchmark dataset, it ingests network flow CSV logs and classifies traffic into three primary categories:

1. **Normal** – Legitimate, expected network communications.
2. **DoS (Denial of Service)** – Volumetric and resource-exhaustion traffic floods.
3. **Probe (Reconnaissance)** – Network port sweeps, host address discovery, and vulnerability probes.

> **Operational Scope Notice**: SentinelNet is an **offline CSV log analysis application**, not a live packet-capturing or automatic inline attack-blocking firewall. It operates on exported flow records to provide explainable machine learning threat classification, model probability estimates, uncertainty routing, categorical severity rankings, and downloadable annotated audit reports.

---

## 2. Adaptive Dual-Mode Architecture

SentinelNet incorporates a **feature-adaptive dual-mode detection engine** designed to improve reconnaissance probe detection while maintaining backward compatibility with basic 4-feature flow CSV logs:

```
+----------------------------------------------------------------------------------------------------+
|                                      Client Browser (Dashboard UI)                                 |
|  - Upload & Analyze (Adaptive CSV Ingestion)          - Model Benchmark & Evaluation Dashboard     |
|  - Single Flow Inspector (Dual-Mode & Uncertainty)    - Telemetry Scan Audit History               |
|  - XSS Hardened: Pure DOM textContent / createElement rendering (Zero Unsafe innerHTML)            |
+----------------------------------------------------------------------------------------------------+
                                                  ^
                                                  | JSON REST API & Multipart File Uploads
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                 SentinelNet Flask Backend (app.py)                                 |
|  - Automatic Model Selection & Feature Detection Engine                                            |
|  - Defensive Row-Level Validation & Fallback Logging                                               |
|  - Uncertainty Thresholding Engine (Decision Status: ACCEPTED vs REVIEW REQUIRED)                  |
|  - Cache Compatibility Verification (Software Version + Dataset SHA-256 Hashes)                   |
|  - Matplotlib Agg Base64 Analytics Charts (Dual-Mode Comparison, Confusion Matrices, Importances)  |
|  - Persistent Audit Logging & Scan History (JSONL)                                                 |
+----------------------------------------------------------------------------------------------------+
                                      /                                    \
               Basic Flow Features   /                                      \  Enhanced Flow Features
          (src_bytes, dst_bytes,    /                                        \ (+ count, srv_count, flag,
           duration, protocol_type)/                                          \  rates)
                                  v                                            v
+--------------------------------------------------+  +----------------------------------------------+
|             BASIC MODE (4 Features)              |  |          ENHANCED MODE (10 Features)         |
|  - Standard 6-Feature Encoded Representation     |  |  - 22-Feature Encoded Representation         |
|  - Scaler fitted on 3 numerical features         |  |  - Scaler fitted on 8 numerical features     |
|  - Production Model: Random Forest (100 trees)   |  |  - Production Model: Random Forest (100 tr)  |
|  - Baseline KDDTest+ Acc: 81.88%                 |  |  - Enhanced KDDTest+ Acc: 85.16%             |
|  - Probe Recall: 30.07%                          |  |  - Probe Recall: 60.80% (Double!)            |
|  - Threshold: 0.65 -> 98.75% coverage            |  |  - Threshold: 0.65 -> 93.76% coverage        |
+--------------------------------------------------+  +----------------------------------------------+
                                  \                                            /
                                   \                                          /
                                    v                                        v
+----------------------------------------------------------------------------------------------------+
|                                    NSL-KDD Benchmark Datasets                                      |
|  - KDDTrain+.txt (125,973 raw -> 124,926 3-class train flows, strictly excludes R2L & U2R)         |
|  - KDDTest+.txt  (22,544 raw  -> 19,592 3-class independent test flows, never used for training)   |
|  - data/cache/sentinelnet_model_cache.joblib (Dual-model pipeline cache with SHA-256 metadata)     |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Automatic Model Selection & Fallback Protocol

SentinelNet dynamically determines the optimal detection mode for every ingested CSV file and individual record without manual reconfiguration:

1. **Enhanced Mode Activation**:
   - If the uploaded CSV contains all 10 enhanced feature headers (`src_bytes`, `dst_bytes`, `duration`, `protocol_type`, `count`, `srv_count`, `flag`, `same_srv_rate`, `serror_rate`, `dst_host_diff_srv_rate`) and an individual record contains valid values, it is classified using the Enhanced Random Forest model.
2. **Basic Mode Compatibility**:
   - If only the 4 basic features are present (`src_bytes`, `dst_bytes`, `duration`, `protocol_type`), SentinelNet processes the batch in Basic Mode.
3. **Defensive Row-Level Fallback**:
   - If enhanced columns are only partially present in the CSV header, or if an individual row contains invalid enhanced values (e.g. non-integer `count`, out-of-bounds rate, or unsupported `flag`), SentinelNet evaluates that row in Basic Mode using its valid basic features.
   - The specific fallback reason is explicitly recorded in `fallback_reason` and exposed in both the UI preview and the exported CSV report.
4. **Row Rejection**:
   - If required basic features are missing, non-numeric, negative, or contain an unsupported protocol, the row is rejected entirely. SentinelNet **never fabricates** missing flow features.

---

## 4. Uncertainty-Aware Decisions (Manual Review Routing)

To prevent blind over-reliance on ambiguous predictions, SentinelNet implements a **confidence-based uncertainty routing mechanism**:

- **Operating Confidence Threshold**: Evaluated and selected on `KDDTrain+` 80/20 stratified validation data only:
  - Threshold = `0.65` (65.0% maximum model probability estimate).
- **Decision Status Values**:
  - `ACCEPTED`: Maximum model probability estimate $\ge 0.65$. The automated decision is accepted.
  - `REVIEW REQUIRED`: Maximum model probability estimate $< 0.65$. Telemetry features lie in ambiguous decision boundaries between classes and are flagged for security analyst verification.
- **Architectural Guardrails**:
  - `REVIEW REQUIRED` is an operational decision routing status, **not** a fourth learned attack class.
  - The model retains its predicted class (`Normal`, `DoS`, `Probe`) and exact probability estimates.
  - Uncertainty routing does not claim guaranteed detection of novel unknown zero-day attacks.

---

## 5. Independent Experimental Benchmark Comparison (KDDTest+)

Both production models were trained strictly on `KDDTrain+` (124,926 records across 3 classes) and evaluated against the completely separate, unaugmented `KDDTest+` benchmark (19,592 records across 3 classes).

### Side-by-Side Performance Comparison:

| Evaluation Dimension | Basic Mode (4 Features) | Enhanced Mode (10 Features) | Measured Delta | Operational Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **Features Used** | `duration`, `protocol_type`, `src_bytes`, `dst_bytes` (6 encoded) | + `count`, `srv_count`, `flag`, `same_srv_rate`, `serror_rate`, `dst_host_diff_srv_rate` (22 encoded) | +6 NSL-KDD features | Deeper multi-dimensional flow context |
| **Independent Test Accuracy** | **81.88%** | **85.16%** | **+3.28%** | Higher overall classification accuracy |
| **Weighted F1-Score** | **80.16%** | **84.76%** | **+4.60%** | Superior multi-class balance |
| **Macro F1-Score** | **70.84%** | **81.26%** | **+10.42%** | Major lift across under-represented classes |
| **Normal Recall** | **96.11%** | **97.05%** | **+0.94%** | Preserves low false-alarm rate on benign flows |
| **DoS Recall** | **80.17%** | **77.59%** | -2.58% | Comparable volumetric exhaustion detection |
| **Probe Recall** | **30.07%** (728 / 2,421) | **60.80%** (1,472 / 2,421) | **+30.73% (+102%)** | **Detects over double the reconnaissance scans** |
| **Automatic Coverage** | **98.75%** (19,348 flows) | **93.76%** (18,370 flows) | -4.99% | Filters out ambiguous low-certainty flows |
| **Accepted Accuracy** | **82.37%** | **89.47%** | **+7.10%** | **Near 90% accuracy on automatically accepted flows** |
| **Manual Review Required** | **244 flows** (1.25%) | **1,222 flows** (6.24%) | +978 flows | Directs uncertain boundary flows to analysts |

### Confusion Matrices on KDDTest+ (Order: Normal, DoS, Probe):

#### Basic Mode Matrix:
```
                 Predicted Normal   Predicted DoS   Predicted Probe
True Normal           9,333              142              236
True DoS              1,476            5,981                3
True Probe              327            1,366              728
```

#### Enhanced Mode Matrix:
```
                 Predicted Normal   Predicted DoS   Predicted Probe
True Normal           9,425               81              205
True DoS              1,608            5,788               64
True Probe              646              303            1,472  <-- Doubled Probe Capture!
```

---

## 6. Preprocessing & Input Validation Rules

### Numerical Preprocessing:
- Numerical features are standardized using `StandardScaler` fitted strictly on `KDDTrain+`.
- Preprocessing scalers are **never** fitted on test data or uploaded CSVs (zero data leakage).

### Feature Validation Rules:

| Feature Name | Mode | Allowed Data Type & Bounds | Validation Rule |
| :--- | :--- | :--- | :--- |
| `src_bytes` | Basic & Enhanced | Float $\ge 0$, finite, not NaN | Session-level payload volume (supports $>65,535$ bytes) |
| `dst_bytes` | Basic & Enhanced | Float $\ge 0$, finite, not NaN | Session-level return volume |
| `duration` | Basic & Enhanced | Float $\ge 0$, finite, not NaN | Connection duration in seconds |
| `protocol_type`| Basic & Enhanced | Categorical string | Strictly `TCP`, `UDP`, or `ICMP` (case-insensitive) |
| `count` | Enhanced Only | Integer $\ge 0$, finite | Connections to same destination host in past 2s |
| `srv_count` | Enhanced Only | Integer $\ge 0$, finite | Connections to same service in past 2s |
| `flag` | Enhanced Only | Categorical string | Supported NSL-KDD statuses: `SF`, `S0`, `REJ`, `RSTR`, `RSTO`, `S1`, `SH`, `S2`, `RSTOS0`, `S3`, `OTH` |
| `same_srv_rate`| Enhanced Only | Float $0.0 \le x \le 1.0$ | Percentage of connections to the same service |
| `serror_rate` | Enhanced Only | Float $0.0 \le x \le 1.0$ | Percentage of connections with SYN errors |
| `dst_host_diff_srv_rate` | Enhanced Only | Float $0.0 \le x \le 1.0$ | Percentage of connections to different services on dst |

---

## 7. Model Cache Verification & Fingerprinting

The pipeline cache (`data/cache/sentinelnet_model_cache.joblib`) serializes both trained models, scalers, and validation benchmarks alongside strict security metadata:

- `sklearn_version`: Pinned scikit-learn dependency version.
- `numpy_version` & `pandas_version`: Dependency versions.
- `feature_schema`: Canonical basic and enhanced feature column lists.
- `target_classes`: Strictly `['Normal', 'DoS', 'Probe']`.
- `confidence_thresholds`: Operating thresholds (`basic`: 0.65, `enhanced`: 0.65).
- `train_dataset_sha256`: SHA-256 cryptographic hash of `KDDTrain+.txt`.
- `test_dataset_sha256`: SHA-256 cryptographic hash of `KDDTest+.txt`.

If scikit-learn changes, the feature schema changes, or either dataset file is modified, the cache is automatically invalidated and regenerated via a fresh dual-model training run.

---

## 8. Automated Test Suite (36 Unit Tests)

SentinelNet includes a comprehensive automated test suite covering all functional, architectural, security, and machine learning components:

```bash
python -m unittest test_sentinelnet.py
```

### Test Coverage Summary:
- **Dataset Loading & Integrity**: Tests 1-2 (Record counts and file existence).
- **Three-Class Scope & Mapping**: Tests 3-4 (Normal, DoS, Probe mapping; strict exclusion of R2L and U2R).
- **Feature Preprocessing**: Test 5 (Scaler dimensions and one-hot encoding).
- **Multi-Model Benchmark Validation**: Test 6 (Logistic Regression, KNN, Random Forest on 80/20 train split).
- **Independent Baseline Evaluation**: Test 7 (KDDTest+ baseline metrics).
- **CSV Ingestion & Defensive Validation**: Tests 8-12 (Valid CSV, missing columns, non-numeric values, unsupported protocols, negative numbers).
- **Batch Predictions & Explanations**: Tests 13, 22 (Prediction attributes, plain-language explanations, low-confidence warnings).
- **Probability Terminology & Risk Mapping**: Test 14 (Model probability estimates, uncalibrated tree voting, deterministic severity mapping).
- **Annotated Reports & Scan History**: Tests 15, 16 (Annotated CSV export and audit tracking).
- **Web UI & REST Endpoints**: Test 17 (Single-page dashboard and API responses).
- **Confusion Matrix Canonical Ordering**: Test 18 (Consistent Normal, DoS, Probe matrix and per-class metrics).
- **XSS Immunity**: Test 19 (Sanitization of malicious HTML/JS in CSV fields and filenames).
- **Cache Compatibility & Invalidation**: Tests 20, 21 (Rejection of version or fingerprint mismatches).
- **Fresh Pipeline Execution**: Test 23 (Forced retrain pipeline without cache reliance).
- **Enhanced Mode Evaluation**: Test 24 (Validation of $\ge 84\%$ accuracy and $>60\%$ Probe recall).
- **Uncertainty Decision Status**: Test 25 (Verification of `ACCEPTED` vs `REVIEW REQUIRED`).
- **Automatic Model Selection**: Tests 26, 27 (Automatic routing between Enhanced and Basic modes).
- **Defensive Fallback Handling**: Tests 28, 29 (Partial header fallback and row-level invalid enhanced fallback with detailed logging).
- **Invalid Basic Feature Rejection**: Test 30 (Rejection of invalid basic rows even when enhanced columns exist).
- **Enhanced Validation Rules**: Test 31 (Integer checks on counts, [0, 1] bounds on rates, supported flag vocabulary).
- **Single Packet Dual-Mode API**: Test 32 (Interactive flow testing across both modes).
- **Sample CSV Endpoints**: Test 33 (Serving downloadable Basic and Enhanced sample files).
- **Annotated Export Verification**: Test 34 (Verification of `detection_mode`, `decision_status`, and `fallback_reason` columns in exported CSV).
- **Strict JSON Serialization**: Test 35 (Defensive null-handling for missing optional metadata in previews).
- **Simplified Interface & Benchmark API**: Test 36 (Validation of the simplified interface and local benchmark summary endpoints).

---

## 9. How to Run SentinelNet

### Prerequisites:
- Python 3.10+ with `pip`.

### 1. Installation:
```bash
pip install -r requirements.txt
```

### 2. Verify Datasets:
Ensure genuine NSL-KDD benchmark text files are present:
```
data/nsl_kdd/KDDTrain+.txt
data/nsl_kdd/KDDTest+.txt
```
If missing, run:
```bash
python download_data.py
```

### 3. Generate Sample CSVs:
```bash
python generate_sample.py
```
Produces:
- `sample_traffic_basic.csv` (4-feature schema + metadata)
- `sample_traffic_enhanced.csv` (10-feature schema + metadata)

### 4. Run Automated Test Suite:
```bash
python -m unittest test_sentinelnet.py
```

### 5. Launch Application:
```bash
python app.py
```
Navigate in your browser to:
```
http://127.0.0.1:5000
```
