"""
SentinelNet: Comprehensive Comparative Benchmark Evaluation
Reference-Inspired Comparative Study:
  "Investigating Novel Machine Learning Based Intrusion Detection Models for NSL-KDD Data Sets"
  (IEEE ICIT 2023, DOI: 10.1109/ICIT59216.2023.10335831)

This script conducts a rigorous, empirical comparative evaluation across:
1. Logistic Regression (10 features)
2. K-Nearest Neighbors (10 features)
3. Decision Tree (10 features)
4. Conventional Random Forest (all 41 NSL-KDD features)
5. Basic SentinelNet (4 features)
6. Enhanced SentinelNet (10 features)

Evaluates:
- Training and inference latency (efficiency trade-offs)
- Multi-class accuracy, macro F1, weighted F1, per-class metrics, confusion matrices
- Uncertainty-aware decision thresholding (ACCEPTED vs REVIEW REQUIRED)
- Telemetry feature degradation and adaptive fallback behavior
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

# Suppress wmic warning from joblib on Windows sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

# Headless matplotlib configuration
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from app import (
    NSL_KDD_COLUMNS, map_nsl_label, TARGET_CLASSES,
    BASIC_NUMERICAL_COLS, BASIC_FEATURE_COLUMNS,
    ENHANCED_NUMERICAL_COLS, ENHANCED_FEATURE_COLUMNS,
    SUPPORTED_PROTOCOLS, SUPPORTED_FLAGS
)

# Output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "benchmark_results")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# 41-feature numerical attributes
ALL_41_NUMERICAL_COLS = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]


def load_datasets():
    """Loads genuine NSL-KDD benchmark datasets and applies canonical 3-class filtering."""
    train_path = os.path.join(BASE_DIR, "data", "nsl_kdd", "KDDTrain+.txt")
    test_path = os.path.join(BASE_DIR, "data", "nsl_kdd", "KDDTest+.txt")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing NSL-KDD files at {train_path} or {test_path}")

    print("[+] Loading raw NSL-KDD datasets...")
    df_tr_raw = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    df_te_raw = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

    df_tr_raw['mapped_cat'] = df_tr_raw['label'].apply(map_nsl_label)
    df_te_raw['mapped_cat'] = df_te_raw['label'].apply(map_nsl_label)

    # Strictly filter to Normal, DoS, Probe (excluding R2L and U2R)
    df_tr = df_tr_raw[df_tr_raw['mapped_cat'].isin(TARGET_CLASSES)].copy()
    df_te = df_te_raw[df_te_raw['mapped_cat'].isin(TARGET_CLASSES)].copy()

    print(f"    - Filtered KDDTrain+ records: {len(df_tr):,} (Normal: {(df_tr['mapped_cat']=='Normal').sum():,}, DoS: {(df_tr['mapped_cat']=='DoS').sum():,}, Probe: {(df_tr['mapped_cat']=='Probe').sum():,})")
    print(f"    - Filtered KDDTest+ records:  {len(df_te):,} (Normal: {(df_te['mapped_cat']=='Normal').sum():,}, DoS: {(df_te['mapped_cat']=='DoS').sum():,}, Probe: {(df_te['mapped_cat']=='Probe').sum():,})")

    return df_tr, df_te


def build_feature_matrices(df_tr, df_te):
    """
    Constructs standardized feature representations:
      1. Basic (4 raw features -> 6 encoded)
      2. Enhanced (10 raw features -> 22 encoded)
      3. Full 41 features (41 raw features -> 122 encoded)
    All StandardScalers are fitted strictly on KDDTrain+.
    """
    print("[+] Engineering feature representations...")

    # --- 1. Basic 4 features ---
    def prep_basic(df):
        proto_norm = df['protocol_type'].astype(str).str.strip().str.upper()
        return pd.DataFrame({
            'src_bytes': df['src_bytes'].astype(float),
            'duration': df['duration'].astype(float),
            'dst_bytes': df['dst_bytes'].astype(float),
            'proto_TCP': (proto_norm == 'TCP').astype(int),
            'proto_UDP': (proto_norm == 'UDP').astype(int),
            'proto_ICMP': (proto_norm == 'ICMP').astype(int)
        })[BASIC_FEATURE_COLUMNS]

    X_b_tr = prep_basic(df_tr)
    X_b_te = prep_basic(df_te)
    scaler_b = StandardScaler()
    X_b_tr[BASIC_NUMERICAL_COLS] = scaler_b.fit_transform(X_b_tr[BASIC_NUMERICAL_COLS])
    X_b_te[BASIC_NUMERICAL_COLS] = scaler_b.transform(X_b_te[BASIC_NUMERICAL_COLS])

    # --- 2. Enhanced 10 features ---
    def prep_enh(df):
        features = pd.DataFrame()
        for nc in ENHANCED_NUMERICAL_COLS:
            features[nc] = df[nc].astype(float)
        proto_norm = df['protocol_type'].astype(str).str.strip().str.upper()
        for p in SUPPORTED_PROTOCOLS:
            features[f'proto_{p}'] = (proto_norm == p).astype(int)
        flag_norm = df['flag'].astype(str).str.strip().str.upper()
        for fl in SUPPORTED_FLAGS:
            features[f'flag_{fl}'] = (flag_norm == fl).astype(int)
        return features[ENHANCED_FEATURE_COLUMNS]

    X_e_tr = prep_enh(df_tr)
    X_e_te = prep_enh(df_te)
    scaler_e = StandardScaler()
    X_e_tr[ENHANCED_NUMERICAL_COLS] = scaler_e.fit_transform(X_e_tr[ENHANCED_NUMERICAL_COLS])
    X_e_te[ENHANCED_NUMERICAL_COLS] = scaler_e.transform(X_e_te[ENHANCED_NUMERICAL_COLS])

    # --- 3. Full 41 features ---
    services_vocab = sorted(list(df_tr['service'].astype(str).str.strip().unique()))

    def prep_41(df):
        num_dict = {nc: df[nc].astype(float) for nc in ALL_41_NUMERICAL_COLS}
        proto_norm = df['protocol_type'].astype(str).str.strip().str.upper()
        proto_dict = {f'proto_{p}': (proto_norm == p).astype(int) for p in SUPPORTED_PROTOCOLS}
        flag_norm = df['flag'].astype(str).str.strip().str.upper()
        flag_dict = {f'flag_{fl}': (flag_norm == fl).astype(int) for fl in SUPPORTED_FLAGS}
        srv_norm = df['service'].astype(str).str.strip()
        srv_dict = {f'srv_{srv}': (srv_norm == srv).astype(int) for srv in services_vocab}

        # Consolidate in single concatenated DataFrame to avoid fragmentation
        all_cols = {**num_dict, **proto_dict, **flag_dict, **srv_dict}
        return pd.DataFrame(all_cols)

    X_41_tr = prep_41(df_tr)
    X_41_te = prep_41(df_te)
    scaler_41 = StandardScaler()
    X_41_tr[ALL_41_NUMERICAL_COLS] = scaler_41.fit_transform(X_41_tr[ALL_41_NUMERICAL_COLS])
    X_41_te[ALL_41_NUMERICAL_COLS] = scaler_41.transform(X_41_te[ALL_41_NUMERICAL_COLS])

    y_train = df_tr['mapped_cat'].values
    y_test = df_te['mapped_cat'].values

    print(f"    - Basic feature space:     {X_b_tr.shape[1]} features")
    print(f"    - Enhanced feature space:  {X_e_tr.shape[1]} features")
    print(f"    - Full 41-feature space:   {X_41_tr.shape[1]} features")

    return (X_b_tr, X_b_te), (X_e_tr, X_e_te), (X_41_tr, X_41_te), y_train, y_test


def evaluate_models(basic_data, enh_data, full_data, y_train, y_test):
    """Trains and benchmarks all 6 comparative architectures on independent KDDTest+."""
    X_b_tr, X_b_te = basic_data
    X_e_tr, X_e_te = enh_data
    X_41_tr, X_41_te = full_data

    model_registry = {
        "Logistic Regression (10 Feat)": {
            "clf": LogisticRegression(max_iter=1000, random_state=42),
            "X_tr": X_e_tr, "X_te": X_e_te, "feat_count": 10, "enc_count": X_e_tr.shape[1],
            "family": "Linear Baseline"
        },
        "K-Nearest Neighbors (10 Feat)": {
            "clf": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "X_tr": X_e_tr, "X_te": X_e_te, "feat_count": 10, "enc_count": X_e_tr.shape[1],
            "family": "Instance-based Baseline"
        },
        "Decision Tree (10 Feat)": {
            "clf": DecisionTreeClassifier(random_state=42),
            "X_tr": X_e_tr, "X_te": X_e_te, "feat_count": 10, "enc_count": X_e_tr.shape[1],
            "family": "Single Tree Baseline"
        },
        "Basic SentinelNet (4 Feat)": {
            "clf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "X_tr": X_b_tr, "X_te": X_b_te, "feat_count": 4, "enc_count": X_b_tr.shape[1],
            "family": "Ensemble (SentinelNet Basic)"
        },
        "Enhanced SentinelNet (10 Feat)": {
            "clf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "X_tr": X_e_tr, "X_te": X_e_te, "feat_count": 10, "enc_count": X_e_tr.shape[1],
            "family": "Ensemble (SentinelNet Enhanced)"
        },
        "Conventional Random Forest (41 Feat)": {
            "clf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "X_tr": X_41_tr, "X_te": X_41_te, "feat_count": 41, "enc_count": X_41_tr.shape[1],
            "family": "Full-Dimensionality Ensemble"
        }
    }

    results = []
    per_class_rows = []
    confusion_matrices = {}
    fitted_models = {}

    print("\n[+] Executing model training and independent test evaluation...")

    for name, cfg in model_registry.items():
        clf = cfg["clf"]
        X_tr = cfg["X_tr"]
        X_te = cfg["X_te"]

        print(f"    --> Training: {name} (N={len(X_tr):,}, D={cfg['enc_count']})...")
        t_fit_start = time.perf_counter()
        clf.fit(X_tr, y_train)
        fit_time = time.perf_counter() - t_fit_start

        print(f"        Evaluating on KDDTest+ (N={len(X_te):,})...")
        t_inf_start = time.perf_counter()
        preds = clf.predict(X_te)
        inf_time = time.perf_counter() - t_inf_start

        latency_per_sample_us = (inf_time / len(X_te)) * 1_000_000

        # Predict probabilities if supported
        probas = None
        if hasattr(clf, "predict_proba"):
            probas = clf.predict_proba(X_te)

        fitted_models[name] = {"clf": clf, "preds": preds, "probas": probas, "inf_time": inf_time, "fit_time": fit_time}

        acc = float(accuracy_score(y_test, preds))
        f1_w = float(f1_score(y_test, preds, average='weighted', zero_division=0))
        f1_m = float(f1_score(y_test, preds, average='macro', zero_division=0))
        prec_w = float(precision_score(y_test, preds, average='weighted', zero_division=0))
        rec_w = float(recall_score(y_test, preds, average='weighted', zero_division=0))

        rep = classification_report(y_test, preds, labels=TARGET_CLASSES, target_names=TARGET_CLASSES, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=TARGET_CLASSES)
        confusion_matrices[name] = cm.tolist()

        res_row = {
            "Model": name,
            "Family": cfg["family"],
            "Raw Features": cfg["feat_count"],
            "Encoded Features": cfg["enc_count"],
            "Training Time (s)": round(fit_time, 3),
            "Inference Time (s)": round(inf_time, 3),
            "Latency (us/flow)": round(latency_per_sample_us, 2),
            "Throughput (flows/s)": round(len(X_te) / inf_time, 1) if inf_time > 0 else 0,
            "Accuracy": round(acc, 4),
            "Weighted F1": round(f1_w, 4),
            "Macro F1": round(f1_m, 4),
            "Weighted Precision": round(prec_w, 4),
            "Weighted Recall": round(rec_w, 4),
            "Normal Recall": round(rep['Normal']['recall'], 4),
            "DoS Recall": round(rep['DoS']['recall'], 4),
            "Probe Recall": round(rep['Probe']['recall'], 4),
            "Normal F1": round(rep['Normal']['f1-score'], 4),
            "DoS F1": round(rep['DoS']['f1-score'], 4),
            "Probe F1": round(rep['Probe']['f1-score'], 4)
        }
        results.append(res_row)

        for c in TARGET_CLASSES:
            per_class_rows.append({
                "Model": name,
                "Class": c,
                "Precision": round(rep[c]['precision'], 4),
                "Recall": round(rep[c]['recall'], 4),
                "F1-Score": round(rep[c]['f1-score'], 4),
                "Support": int(rep[c]['support'])
            })

        print(f"        [Done] Acc={acc*100:.2f}%, F1_w={f1_w:.4f}, Probe_Rec={rep['Probe']['recall']*100:.2f}%, Fit={fit_time:.2f}s, Latency={latency_per_sample_us:.1f}us")

    df_results = pd.DataFrame(results)
    df_per_class = pd.DataFrame(per_class_rows)

    return df_results, df_per_class, confusion_matrices, fitted_models


def evaluate_uncertainty_routing(fitted_models, y_test):
    """Evaluates confidence threshold sweeps on Basic and Enhanced SentinelNet."""
    print("\n[+] Evaluating Uncertainty-Aware Decision Thresholds (Sweep: 0.50 to 0.90)...")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    sweep_rows = []
    for mode_name, key in [("Basic SentinelNet (4 Feat)", "Basic SentinelNet (4 Feat)"),
                           ("Enhanced SentinelNet (10 Feat)", "Enhanced SentinelNet (10 Feat)")]:
        m_info = fitted_models[key]
        probas = m_info["probas"]
        preds = m_info["preds"]
        confidences = np.max(probas, axis=1)

        for t in thresholds:
            accepted_mask = confidences >= t
            coverage = float(accepted_mask.mean())
            acc_count = int(accepted_mask.sum())
            rev_count = int((~accepted_mask).sum())

            if acc_count > 0:
                acc_accepted = float(accuracy_score(y_test[accepted_mask], preds[accepted_mask]))
                f1_w_acc = float(f1_score(y_test[accepted_mask], preds[accepted_mask], average='weighted', zero_division=0))
                rep_acc = classification_report(y_test[accepted_mask], preds[accepted_mask], labels=TARGET_CLASSES, target_names=TARGET_CLASSES, output_dict=True, zero_division=0)
                probe_rec_acc = float(rep_acc['Probe']['recall'])
            else:
                acc_accepted, f1_w_acc, probe_rec_acc = 0.0, 0.0, 0.0

            sweep_rows.append({
                "Architecture": mode_name,
                "Threshold": round(t, 2),
                "Coverage (%)": round(coverage * 100, 2),
                "Accepted Accuracy (%)": round(acc_accepted * 100, 2),
                "Accepted Weighted F1": round(f1_w_acc, 4),
                "Accepted Probe Recall (%)": round(probe_rec_acc * 100, 2),
                "Accepted Count": acc_count,
                "Manual Review Count": rev_count,
                "Manual Review (%)": round((rev_count / len(y_test)) * 100, 2)
            })

    df_sweep = pd.DataFrame(sweep_rows)
    return df_sweep


def simulate_telemetry_degradation(fitted_models, y_test):
    """
    Simulates network flow feature degradation scenarios to evaluate SentinelNet's
    adaptive model selection and graceful degradation capability.
    """
    print("\n[+] Simulating Telemetry Degradation & Adaptive Model Selection...")
    preds_e = fitted_models["Enhanced SentinelNet (10 Feat)"]["preds"]
    preds_b = fitted_models["Basic SentinelNet (4 Feat)"]["preds"]

    N = len(y_test)
    np.random.seed(42)
    rand_floats = np.random.rand(N)

    scenarios = [
        ("Scenario 1: Ideal Enhanced Telemetry", 1.00, 0.00, 0.00, "100% full enhanced 10-feature availability"),
        ("Scenario 2: High Enhanced Availability", 0.75, 0.15, 0.10, "75% enhanced, 15% corrupted/fallback, 10% basic-only"),
        ("Scenario 3: Balanced Heterogeneous Ingest", 0.50, 0.25, 0.25, "50% enhanced, 25% corrupted/fallback, 25% basic-only"),
        ("Scenario 4: High Feature Degradation", 0.25, 0.35, 0.40, "25% enhanced, 35% corrupted/fallback, 40% basic-only"),
        ("Scenario 5: Pure Legacy Basic Flows", 0.00, 0.00, 1.00, "0% enhanced availability (100% 4-feature legacy flows)")
    ]

    sim_rows = []
    for sc_name, p_enh, p_fb, p_bas, desc in scenarios:
        sim_preds = np.empty(N, dtype=object)
        sim_modes = np.empty(N, dtype=object)

        for i in range(N):
            r = rand_floats[i]
            if r < p_enh:
                sim_modes[i] = "Enhanced"
                sim_preds[i] = preds_e[i]
            elif r < (p_enh + p_fb):
                sim_modes[i] = "Basic (Fallback)"
                sim_preds[i] = preds_b[i]
            else:
                sim_modes[i] = "Basic (Native)"
                sim_preds[i] = preds_b[i]

        acc = float(accuracy_score(y_test, sim_preds))
        f1_w = float(f1_score(y_test, sim_preds, average='weighted', zero_division=0))
        f1_m = float(f1_score(y_test, sim_preds, average='macro', zero_division=0))
        rep = classification_report(y_test, sim_preds, labels=TARGET_CLASSES, target_names=TARGET_CLASSES, output_dict=True, zero_division=0)

        enh_count = int((sim_modes == "Enhanced").sum())
        fb_count = int((sim_modes == "Basic (Fallback)").sum())
        bas_count = int((sim_modes == "Basic (Native)").sum())

        sim_rows.append({
            "Scenario": sc_name,
            "Description": desc,
            "Enhanced Flows (%)": round((enh_count / N) * 100, 1),
            "Fallback Flows (%)": round((fb_count / N) * 100, 1),
            "Basic Native Flows (%)": round((bas_count / N) * 100, 1),
            "Overall Accuracy (%)": round(acc * 100, 2),
            "Weighted F1": round(f1_w, 4),
            "Macro F1": round(f1_m, 4),
            "Probe Recall (%)": round(rep['Probe']['recall'] * 100, 2),
            "DoS Recall (%)": round(rep['DoS']['recall'] * 100, 2),
            "Normal Recall (%)": round(rep['Normal']['recall'] * 100, 2)
        })

    df_sim = pd.DataFrame(sim_rows)
    return df_sim


def generate_comparative_charts(df_results, df_sweep, df_sim, confusion_matrices):
    """Renders and saves high-resolution, dark-themed comparative charts."""
    print("\n[+] Generating high-resolution publication charts in data/benchmark_results/charts/...")

    plt.style.use('dark_background')
    palette = {
        "blue": "#58a6ff",
        "green": "#3fb950",
        "orange": "#d29922",
        "red": "#f85149",
        "purple": "#bc8cff",
        "teal": "#39c5bb",
        "bg": "#161b22",
        "border": "#30363d",
        "text": "#e6edf3",
        "muted": "#8b949e"
    }

    # Chart 1: Comparative Model Performance Across Key Dimensions
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=palette["bg"])
    ax.set_facecolor(palette["bg"])

    models = df_results["Model"].tolist()
    short_names = [
        "Logistic Reg (10F)",
        "KNN (10F)",
        "Decision Tree (10F)",
        "Basic SentinelNet (4F)",
        "Enhanced SentinelNet (10F)",
        "Conventional RF (41F)"
    ]

    metrics = ["Accuracy", "Weighted F1", "Macro F1", "Probe Recall"]
    colors = [palette["blue"], palette["teal"], palette["purple"], palette["orange"]]
    x = np.arange(len(models))
    w = 0.20

    for i, m in enumerate(metrics):
        vals = [df_results.loc[df_results["Model"] == name, m].values[0] * 100 for name in models]
        rects = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=colors[i], alpha=0.9, edgecolor=palette["border"])
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha='center', va='bottom',
                        color=palette["text"], fontsize=6.5, fontfamily='monospace')

    ax.set_ylabel("Percentage (%)", color=palette["muted"], fontsize=10)
    ax.set_title("Comparative Benchmark Evaluation on NSL-KDD (Independent KDDTest+, n=19,592)",
                 color=palette["text"], fontsize=12, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, color=palette["text"], fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 115)
    ax.tick_params(colors=palette["muted"], labelsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.3, color=palette["border"], axis='y')
    for s in ax.spines.values(): s.set_color(palette["border"])
    ax.legend(frameon=True, facecolor=palette["bg"], edgecolor=palette["border"],
              labelcolor=palette["text"], fontsize=9, ncol=4, loc='upper left')

    plt.tight_layout()
    chart1_path = os.path.join(CHARTS_DIR, "1_comparative_model_performance.png")
    fig.savefig(chart1_path, dpi=160, facecolor=palette["bg"], edgecolor='none')
    plt.close(fig)

    # Chart 2: Computational Efficiency & Latency Trade-off
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=palette["bg"])
    ax1.set_facecolor(palette["bg"])
    ax2.set_facecolor(palette["bg"])

    y_pos = np.arange(len(models))
    fit_times = df_results["Training Time (s)"].tolist()
    latencies = df_results["Latency (us/flow)"].tolist()

    bars1 = ax1.barh(y_pos, fit_times, color=palette["blue"], alpha=0.85, edgecolor=palette["border"], height=0.55)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(short_names, color=palette["text"], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Training Time (seconds on 124,926 flows)", color=palette["muted"], fontsize=9)
    ax1.set_title("Model Training Overhead", color=palette["text"], fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.3, color=palette["border"], axis='x')
    for s in ax1.spines.values(): s.set_color(palette["border"])
    for bar in bars1:
        v = bar.get_width()
        ax1.annotate(f"{v:.2f}s", xy=(v, bar.get_y() + bar.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", ha='left', va='center',
                     color=palette["text"], fontsize=8, fontfamily='monospace')

    bars2 = ax2.barh(y_pos, latencies, color=palette["green"], alpha=0.85, edgecolor=palette["border"], height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Inference Latency (microseconds / flow)", color=palette["muted"], fontsize=9)
    ax2.set_title("Real-Time Flow Ingestion Latency", color=palette["text"], fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.3, color=palette["border"], axis='x')
    for s in ax2.spines.values(): s.set_color(palette["border"])
    for bar in bars2:
        v = bar.get_width()
        ax2.annotate(f"{v:.1f} us", xy=(v, bar.get_y() + bar.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", ha='left', va='center',
                     color=palette["text"], fontsize=8, fontfamily='monospace')

    plt.tight_layout()
    chart2_path = os.path.join(CHARTS_DIR, "2_computational_efficiency_latency.png")
    fig.savefig(chart2_path, dpi=160, facecolor=palette["bg"], edgecolor='none')
    plt.close(fig)

    # Chart 3: Confusion Matrices for All 6 Models (2x3 Grid)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), facecolor=palette["bg"])
    axes = axes.flatten()

    for idx, (name, cm_list) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        ax.set_facecolor(palette["bg"])
        cm_arr = np.array(cm_list)

        im = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(short_names[idx], color=palette["text"], fontsize=10, fontweight='bold', pad=8)
        ticks = np.arange(len(TARGET_CLASSES))
        ax.set_xticks(ticks)
        ax.set_xticklabels(TARGET_CLASSES, color=palette["text"], fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(TARGET_CLASSES, color=palette["text"], fontsize=8)
        ax.set_ylabel("True Class", color=palette["muted"], fontsize=8.5)
        ax.set_xlabel("Predicted Class", color=palette["muted"], fontsize=8.5)

        thresh = cm_arr.max() / 2.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                val = cm_arr[i, j]
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        color="white" if val > thresh else palette["muted"],
                        fontfamily='monospace', fontsize=8.5, fontweight='bold')

        for s in ax.spines.values(): s.set_color(palette["border"])

    fig.suptitle("Independent Test Confusion Matrices (Order: Normal, DoS, Probe; n=19,592)",
                 color=palette["text"], fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    chart3_path = os.path.join(CHARTS_DIR, "3_confusion_matrices_all_models.png")
    fig.savefig(chart3_path, dpi=160, facecolor=palette["bg"], edgecolor='none')
    plt.close(fig)

    # Chart 4: Uncertainty Trade-off Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=palette["bg"])
    ax1.set_facecolor(palette["bg"])
    ax2.set_facecolor(palette["bg"])

    sweep_b = df_sweep[df_sweep["Architecture"].str.contains("Basic")]
    sweep_e = df_sweep[df_sweep["Architecture"].str.contains("Enhanced")]

    # Plot Coverage vs Threshold
    ax1.plot(sweep_b["Threshold"], sweep_b["Coverage (%)"], marker='o', color=palette["blue"], label="Basic Mode Coverage", linewidth=2)
    ax1.plot(sweep_e["Threshold"], sweep_e["Coverage (%)"], marker='s', color=palette["green"], label="Enhanced Mode Coverage", linewidth=2)
    ax1.axvline(x=0.65, color=palette["orange"], linestyle='--', alpha=0.8, label="Operating Threshold (0.65)")
    ax1.set_xlabel("Confidence Threshold", color=palette["muted"], fontsize=9.5)
    ax1.set_ylabel("Automatic Classification Coverage (%)", color=palette["muted"], fontsize=9.5)
    ax1.set_title("Automated Coverage vs Confidence Threshold", color=palette["text"], fontsize=11, fontweight='bold')
    ax1.set_ylim(50, 105)
    ax1.grid(True, linestyle=':', alpha=0.3, color=palette["border"])
    for s in ax1.spines.values(): s.set_color(palette["border"])
    ax1.legend(frameon=True, facecolor=palette["bg"], edgecolor=palette["border"], labelcolor=palette["text"], fontsize=8.5)

    # Plot Accepted Accuracy vs Threshold
    ax2.plot(sweep_b["Threshold"], sweep_b["Accepted Accuracy (%)"], marker='o', color=palette["blue"], label="Basic Accepted Accuracy", linewidth=2)
    ax2.plot(sweep_e["Threshold"], sweep_e["Accepted Accuracy (%)"], marker='s', color=palette["green"], label="Enhanced Accepted Accuracy", linewidth=2)
    ax2.axvline(x=0.65, color=palette["orange"], linestyle='--', alpha=0.8, label="Operating Threshold (0.65)")
    ax2.set_xlabel("Confidence Threshold", color=palette["muted"], fontsize=9.5)
    ax2.set_ylabel("Accuracy on Accepted Flows (%)", color=palette["muted"], fontsize=9.5)
    ax2.set_title("Accepted Accuracy vs Confidence Threshold", color=palette["text"], fontsize=11, fontweight='bold')
    ax2.set_ylim(75, 96)
    ax2.grid(True, linestyle=':', alpha=0.3, color=palette["border"])
    for s in ax2.spines.values(): s.set_color(palette["border"])
    ax2.legend(frameon=True, facecolor=palette["bg"], edgecolor=palette["border"], labelcolor=palette["text"], fontsize=8.5)

    plt.tight_layout()
    chart4_path = os.path.join(CHARTS_DIR, "4_uncertainty_tradeoff_curves.png")
    fig.savefig(chart4_path, dpi=160, facecolor=palette["bg"], edgecolor='none')
    plt.close(fig)

    # Chart 5: Telemetry Degradation & Adaptive Ingestion Simulation
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=palette["bg"])
    ax.set_facecolor(palette["bg"])

    sc_labels = [s.split(":")[0] for s in df_sim["Scenario"].tolist()]
    x_sim = np.arange(len(sc_labels))
    w_sim = 0.35

    r_acc = ax.bar(x_sim - w_sim / 2, df_sim["Overall Accuracy (%)"], w_sim, label="Overall System Accuracy", color=palette["blue"], alpha=0.9, edgecolor=palette["border"])
    r_prb = ax.bar(x_sim + w_sim / 2, df_sim["Probe Recall (%)"], w_sim, label="Probe Recall", color=palette["orange"], alpha=0.9, edgecolor=palette["border"])

    for rect in r_acc:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom',
                    color=palette["text"], fontsize=7.5, fontfamily='monospace')
    for rect in r_prb:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom',
                    color=palette["orange"], fontsize=7.5, fontfamily='monospace', fontweight='bold')

    ax.set_ylabel("Percentage (%)", color=palette["muted"], fontsize=9.5)
    ax.set_title("Adaptive Dual-Mode Performance Under Progressive Telemetry Degradation",
                 color=palette["text"], fontsize=11, fontweight='bold', pad=12)
    ax.set_xticks(x_sim)
    ax.set_xticklabels(sc_labels, color=palette["text"], fontsize=9)
    ax.set_ylim(0, 110)
    ax.tick_params(colors=palette["muted"], labelsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.3, color=palette["border"], axis='y')
    for s in ax.spines.values(): s.set_color(palette["border"])
    ax.legend(frameon=True, facecolor=palette["bg"], edgecolor=palette["border"], labelcolor=palette["text"], fontsize=9)

    plt.tight_layout()
    chart5_path = os.path.join(CHARTS_DIR, "5_adaptive_degradation_simulation.png")
    fig.savefig(chart5_path, dpi=160, facecolor=palette["bg"], edgecolor='none')
    plt.close(fig)

    print("[+] All comparative charts successfully generated and saved.")


def save_reports_and_csvs(df_results, df_per_class, df_sweep, df_sim, confusion_matrices):
    """Exports structured evaluation CSVs and confusion matrices JSON."""
    print("[+] Exporting evaluation CSV data...")

    results_csv_path = os.path.join(OUTPUT_DIR, "comparative_model_performance.csv")
    df_results.to_csv(results_csv_path, index=False)

    per_class_csv_path = os.path.join(OUTPUT_DIR, "per_class_performance.csv")
    df_per_class.to_csv(per_class_csv_path, index=False)

    sweep_csv_path = os.path.join(OUTPUT_DIR, "uncertainty_threshold_sweep.csv")
    df_sweep.to_csv(sweep_csv_path, index=False)

    sim_csv_path = os.path.join(OUTPUT_DIR, "adaptive_simulation_results.csv")
    df_sim.to_csv(sim_csv_path, index=False)

    cm_json_path = os.path.join(OUTPUT_DIR, "confusion_matrices.json")
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(confusion_matrices, f, indent=2)

    print(f"    - Model Performance CSV:    {results_csv_path}")
    print(f"    - Per-Class Performance CSV: {per_class_csv_path}")
    print(f"    - Uncertainty Sweep CSV:     {sweep_csv_path}")
    print(f"    - Telemetry Simulation CSV:  {sim_csv_path}")
    print(f"    - Confusion Matrices JSON:   {cm_json_path}")


def main():
    print("================================================================================")
    print(" SentinelNet – Benchmark Evaluation & Comparative Empirical Study")
    print(" Reference-Inspired Study: IEEE ICIT 2023 (Shah et al., DOI: 10.1109/ICIT59216.2023.10335831)")
    print("================================================================================")

    # 1. Load datasets
    df_tr, df_te = load_datasets()

    # 2. Build feature representations
    basic_data, enh_data, full_data, y_train, y_test = build_feature_matrices(df_tr, df_te)

    # 3. Benchmark all models
    df_results, df_per_class, confusion_matrices, fitted_models = evaluate_models(
        basic_data, enh_data, full_data, y_train, y_test
    )

    # 4. Evaluate uncertainty thresholding
    df_sweep = evaluate_uncertainty_routing(fitted_models, y_test)

    # 5. Simulate telemetry degradation & adaptive selection
    df_sim = simulate_telemetry_degradation(fitted_models, y_test)

    # 6. Save data artifacts
    save_reports_and_csvs(df_results, df_per_class, df_sweep, df_sim, confusion_matrices)

    # 7. Generate publication charts
    generate_comparative_charts(df_results, df_sweep, df_sim, confusion_matrices)

    print("\n================================================================================")
    print(" SUMMARY COMPARATIVE RESULTS TABLE (KDDTest+, n=19,592)")
    print("================================================================================")
    disp_cols = ["Model", "Raw Features", "Training Time (s)", "Latency (us/flow)", "Accuracy", "Weighted F1", "Macro F1", "Probe Recall"]
    print(df_results[disp_cols].to_string(index=False))

    print("\n================================================================================")
    print(" [DONE] All comparative benchmark experiments completed successfully.")
    print("================================================================================")


if __name__ == "__main__":
    main()
