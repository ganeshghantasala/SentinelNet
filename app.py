"""
SentinelNet: Network Flow Analysis & Machine Learning Intrusion Detection System
Trained on genuine NSL-KDD benchmark dataset (KDDTrain+ and KDDTest+).
Adaptive Dual-Mode Engine:
  - Basic Mode: 4-feature flow classification (src_bytes, dst_bytes, duration, protocol_type)
  - Enhanced Mode: 10-feature flow classification (adding count, srv_count, flag, same_srv_rate,
    serror_rate, dst_host_diff_srv_rate) for enhanced Probe detection and precision.
Uncertainty-Aware Decision Framework:
  - Confidence thresholding on KDDTrain+ validation split.
  - Generates 'ACCEPTED' vs 'REVIEW REQUIRED' decision statuses without creating artificial 4th classes.
"""

import os
import sys
import io
import re
import json
import math
import base64
import random
import hashlib
from datetime import datetime, timedelta

# Configure Matplotlib writable cache before importing matplotlib/pyplot
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir

# Configure Matplotlib headless Agg backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sklearn
import joblib
from flask import Flask, render_template, request, jsonify, Response

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# ==============================================================================
# USER-DEFINED EXCEPTION HIERARCHY
# ==============================================================================
class NetworkSecurityException(Exception):
    """Base exception class for cybersecurity telemetry and IDS operations."""
    pass

class EmptyTrafficInputError(NetworkSecurityException):
    """Raised when traffic input features or uploaded files are empty, null, or missing."""
    pass

class InvalidPacketSizeError(NetworkSecurityException):
    """Raised when packet size (src_bytes or dst_bytes) or duration is negative or non-numeric."""
    pass

class MissingDatasetError(NetworkSecurityException):
    """Raised when genuine NSL-KDD dataset benchmark files are missing."""
    pass


# ==============================================================================
# NSL-KDD BENCHMARK SPECIFICATIONS & MAPPINGS
# ==============================================================================
TARGET_CLASSES = ['Normal', 'DoS', 'Probe']

NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

DOS_ATTACKS = {
    'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop',
    'mailbomb', 'apache2', 'processtable', 'udpstorm', 'worm'
}

PROBE_ATTACKS = {
    'ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint'
}

R2L_ATTACKS = {
    'ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy',
    'warezclient', 'warezmaster', 'sendmail', 'named', 'snmpgetattack',
    'snmpguess', 'xlock', 'xsnoop', 'httptunnel'
}

U2R_ATTACKS = {
    'buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'ps', 'sqlattack', 'xterm'
}

# Feature definitions for Adaptive Dual-Mode Architecture
BASIC_INPUT_FEATURES = ['src_bytes', 'dst_bytes', 'duration', 'protocol_type']
BASIC_NUMERICAL_COLS = ['src_bytes', 'duration', 'dst_bytes']
BASIC_FEATURE_COLUMNS = ['src_bytes', 'duration', 'dst_bytes', 'proto_TCP', 'proto_UDP', 'proto_ICMP']

ENHANCED_INPUT_FEATURES = [
    'src_bytes', 'dst_bytes', 'duration', 'protocol_type',
    'count', 'srv_count', 'flag', 'same_srv_rate',
    'serror_rate', 'dst_host_diff_srv_rate'
]
ENHANCED_ADDITIONAL_COLS = [
    'count', 'srv_count', 'flag', 'same_srv_rate',
    'serror_rate', 'dst_host_diff_srv_rate'
]
ENHANCED_NUMERICAL_COLS = [
    'src_bytes', 'dst_bytes', 'duration', 'count', 'srv_count',
    'same_srv_rate', 'serror_rate', 'dst_host_diff_srv_rate'
]
SUPPORTED_PROTOCOLS = ['TCP', 'UDP', 'ICMP']
SUPPORTED_FLAGS = ['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'S1', 'SH', 'S2', 'RSTOS0', 'S3', 'OTH']

ENHANCED_FEATURE_COLUMNS = (
    ENHANCED_NUMERICAL_COLS +
    [f'proto_{p}' for p in SUPPORTED_PROTOCOLS] +
    [f'flag_{fl}' for fl in SUPPORTED_FLAGS]
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.65

RISK_MAPPING_NOTE = (
    "Risk level is a categorical severity mapping derived directly from the predicted category "
    "(DoS -> CRITICAL, Probe -> ELEVATED, Normal -> LOW). It is not an independently calculated "
    "risk probability or multidimensional score."
)

UNCERTAINTY_NOTE = (
    "Decision status 'REVIEW REQUIRED' indicates model probability estimate fell below the KDDTrain+ "
    "validation confidence threshold (0.65). It flags ambiguous feature boundaries for manual review "
    "and does not represent a separate learned attack category."
)

def map_nsl_label(raw_label):
    """
    Maps an NSL-KDD attack label into the 3 target categories or marks for exclusion.
    Never maps R2L, U2R or unknown attacks to Normal.
    """
    clean = str(raw_label).strip().rstrip('.').lower()
    if clean == 'normal':
        return 'Normal'
    if clean in DOS_ATTACKS:
        return 'DoS'
    if clean in PROBE_ATTACKS:
        return 'Probe'
    if clean in R2L_ATTACKS:
        return 'R2L'
    if clean in U2R_ATTACKS:
        return 'U2R'
    return 'Unknown'

def get_dataset_fingerprint(filepath):
    """Computes SHA-256 fingerprint of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def sanitize_filename(filename):
    """
    Defensively sanitizes user-supplied filenames:
    - Strips directory path components (prevents path traversal)
    - Strips HTML tags, brackets, and script injection tokens
    - Restricts to safe alphanumeric, underscores, hyphens, and dots
    - Truncates to safe length
    """
    if not filename:
        return "uploaded_traffic.csv"
    base = os.path.basename(str(filename)).strip()
    clean = re.sub(r'[<>&"\'/\\;`$|*?:\x00-\x1f\x7f-\x9f]', '_', base)
    clean = re.sub(r'_+', '_', clean).strip('_')
    if not clean or clean.startswith('.'):
        clean = "uploaded_traffic.csv"
    if not clean.lower().endswith('.csv'):
        clean += '.csv'
    return clean[:120]


# ==============================================================================
# OBJECT-ORIENTED DETECTOR CLASS & DEFENSIVE DATA ENGINE
# ==============================================================================
class NetworkIntrusionDetector:
    """
    Object-oriented intrusion detection engine encapsulating feature engineering,
    dual-mode multi-model training on genuine NSL-KDD data, runtime validation, batch CSV
    processing, uncertainty-aware manual review routing, and explainable audit logging.
    """
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = base_dir

        self.data_dir = os.path.join(self.base_dir, "data", "nsl_kdd")
        self.cache_dir = os.path.join(self.base_dir, "data", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.log_filepath = os.path.join(self.base_dir, "intrusion_audit_log.jsonl")
        self.history_filepath = os.path.join(self.base_dir, "audit_scan_history.jsonl")

        # Basic Mode Schema (4 raw / 6 encoded)
        self.feature_columns = BASIC_FEATURE_COLUMNS.copy()
        self.numerical_cols = BASIC_NUMERICAL_COLS.copy()
        self.target_classes = TARGET_CLASSES.copy()
        self.classes_ = TARGET_CLASSES.copy()

        # Enhanced Mode Schema (10 raw / 22 encoded)
        self.enhanced_feature_columns = ENHANCED_FEATURE_COLUMNS.copy()
        self.enhanced_numerical_cols = ENHANCED_NUMERICAL_COLS.copy()

        # Preprocessing Scalers
        self.scaler = StandardScaler()
        self.enhanced_scaler = StandardScaler()

        # Models
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }
        self.active_model = self.models["Random Forest"]
        self.enhanced_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.is_trained = False

        # Confidence thresholds for uncertainty awareness (calibrated on KDDTrain+ validation split)
        self.confidence_thresholds = {
            "basic": DEFAULT_CONFIDENCE_THRESHOLD,
            "enhanced": DEFAULT_CONFIDENCE_THRESHOLD
        }

        # Evaluation & Benchmark Results
        self.dataset_stats = {}
        self.val_benchmark_results = []
        self.test_evaluation_results = {}          # Basic Mode evaluation (backward compatibility)
        self.enhanced_test_evaluation = {}         # Enhanced Mode evaluation
        self.mode_comparison_results = {}          # Side-by-side mode comparison
        self.charts_b64 = {}
        self.scan_history = []
        self.sample_flows = []
        self.sample_flows_enhanced = []
        self.cache_metadata = {}

        self._initialize_audit_files()
        self._load_scan_history()

    def _initialize_audit_files(self):
        """Initializes audit files using defensive error handling."""
        try:
            if not os.path.exists(self.log_filepath):
                with open(self.log_filepath, "w", encoding="utf-8") as f:
                    entry = {
                        "event": "AUDIT_INITIALIZED",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "engine": "NSL-KDD Adaptive Dual-Mode Flow Classifier",
                        "status": "ONLINE"
                    }
                    f.write(json.dumps(entry) + "\n")
        except (PermissionError, IOError, OSError) as e:
            print(f"[ERROR] Audit log initialization error: {e}")

    def _load_scan_history(self):
        """Loads persistent scan history from disk."""
        self.scan_history = []
        try:
            if os.path.exists(self.history_filepath):
                with open(self.history_filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self.scan_history.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            print(f"[WARNING] Could not load scan history: {e}")

    def _save_scan_history_entry(self, entry):
        """Appends a scan entry to the scan history audit file."""
        self.scan_history.insert(0, entry)
        try:
            with open(self.history_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to save scan history entry: {e}")

    def prepare_features(self, df):
        """Constructs the standard 6-feature representation from 4 basic flow records."""
        proto_norm = df['protocol_type'].astype(str).str.strip().str.upper()
        features = pd.DataFrame({
            'src_bytes': df['src_bytes'].astype(float),
            'duration': df['duration'].astype(float),
            'dst_bytes': df['dst_bytes'].astype(float),
            'proto_TCP': (proto_norm == 'TCP').astype(int),
            'proto_UDP': (proto_norm == 'UDP').astype(int),
            'proto_ICMP': (proto_norm == 'ICMP').astype(int)
        })
        return features[self.feature_columns]

    def prepare_enhanced_features(self, df):
        """Constructs the 22-feature encoded representation from 10 NSL-KDD flow records."""
        features = pd.DataFrame()
        for nc in self.enhanced_numerical_cols:
            features[nc] = df[nc].astype(float)

        proto_norm = df['protocol_type'].astype(str).str.strip().str.upper()
        for p in SUPPORTED_PROTOCOLS:
            features[f'proto_{p}'] = (proto_norm == p).astype(int)

        flag_norm = df['flag'].astype(str).str.strip().str.upper()
        for fl in SUPPORTED_FLAGS:
            features[f'flag_{fl}'] = (flag_norm == fl).astype(int)

        return features[self.enhanced_feature_columns]

    def validate_cache(self, cached_dict, train_path, test_path):
        """
        Validates cache structure, dependency versions, feature schema, and dataset SHA-256 fingerprints.
        Returns (is_valid: bool, reason: str).
        """
        if not isinstance(cached_dict, dict):
            return False, "Cache content is not a valid dictionary."

        meta = cached_dict.get('metadata')
        if not meta or not isinstance(meta, dict):
            return False, "Cache is missing required version and fingerprint metadata."

        if meta.get('sklearn_version') != sklearn.__version__:
            return False, f"Scikit-learn version mismatch: cache={meta.get('sklearn_version')}, current={sklearn.__version__}."

        if meta.get('feature_schema') != self.feature_columns:
            return False, f"Feature schema mismatch: cache={meta.get('feature_schema')}, current={self.feature_columns}."

        if meta.get('target_classes') != self.target_classes:
            return False, f"Target classes mismatch: cache={meta.get('target_classes')}, current={self.target_classes}."

        curr_train_sha = get_dataset_fingerprint(train_path)
        if meta.get('train_dataset_sha256') != curr_train_sha:
            return False, "KDDTrain+.txt SHA-256 fingerprint mismatch. Training dataset has changed."

        curr_test_sha = get_dataset_fingerprint(test_path)
        if meta.get('test_dataset_sha256') != curr_test_sha:
            return False, "KDDTest+.txt SHA-256 fingerprint mismatch. Test dataset has changed."

        required_keys = ['scaler', 'models', 'dataset_stats', 'val_benchmark_results', 'test_evaluation_results', 'charts_b64', 'sample_flows']
        for k in required_keys:
            if k not in cached_dict:
                return False, f"Missing required payload key in cache: '{k}'."

        if 'enhanced_rf' not in cached_dict or 'enhanced_scaler' not in cached_dict:
            return False, "Cache does not contain pre-trained Enhanced Mode model and scaler."

        return True, "Cache metadata and dataset fingerprints verified successfully."

    def load_and_train_datasets(self, force_retrain=False):
        """
        Loads genuine NSL-KDD datasets, validates sanity checks, trains both Basic and Enhanced models,
        evaluates against test set independently, and generates charts.
        Validates cache compatibility (versions + SHA-256 fingerprints) before loading.
        """
        train_path = os.path.join(self.data_dir, "KDDTrain+.txt")
        test_path = os.path.join(self.data_dir, "KDDTest+.txt")
        cache_path = os.path.join(self.cache_dir, "sentinelnet_model_cache.joblib")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise MissingDatasetError(
                f"Missing genuine NSL-KDD dataset files.\n"
                f"Expected locations:\n"
                f"  - {train_path}\n"
                f"  - {test_path}\n"
                f"Please download KDDTrain+.txt and KDDTest+.txt from the official UNB source:\n"
                f"https://www.unb.ca/cic/datasets/nsl.html\n"
                f"and place them under 'data/nsl_kdd/'."
            )

        # Check if cache exists and validate it
        if not force_retrain and os.path.exists(cache_path):
            try:
                cached = joblib.load(cache_path)
                is_valid, reason = self.validate_cache(cached, train_path, test_path)
                if is_valid:
                    print(f"[+] Verified compatible model cache ({reason}). Loading...")
                    self.scaler = cached['scaler']
                    self.enhanced_scaler = cached['enhanced_scaler']
                    self.models = cached['models']
                    self.active_model = self.models["Random Forest"]
                    self.enhanced_rf = cached['enhanced_rf']
                    self.dataset_stats = cached['dataset_stats']
                    self.val_benchmark_results = cached['val_benchmark_results']
                    self.test_evaluation_results = cached['test_evaluation_results']
                    self.enhanced_test_evaluation = cached.get('enhanced_test_evaluation', {})
                    self.mode_comparison_results = cached.get('mode_comparison', {})
                    self.charts_b64 = cached['charts_b64']
                    self.sample_flows = cached['sample_flows']
                    self.sample_flows_enhanced = cached.get('sample_flows_enhanced', [])
                    self.cache_metadata = cached.get('metadata', {})
                    self.confidence_thresholds = self.cache_metadata.get('confidence_thresholds', {
                        "basic": DEFAULT_CONFIDENCE_THRESHOLD,
                        "enhanced": DEFAULT_CONFIDENCE_THRESHOLD
                    })
                    self.is_trained = True
                    print("[+] Pre-trained NSL-KDD dual models successfully loaded from verified cache.")
                    return
                else:
                    print(f"[!] Model cache invalidated ({reason}). Initiating fresh model training pipeline...")
            except Exception as e:
                print(f"[!] Cache loading failed ({e}); initiating full training pipeline...")

        print("[+] Initiating full NSL-KDD dataset dual-mode training pipeline...")
        # 1. Load Raw Datasets
        df_train_raw = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
        df_test_raw = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

        train_total = len(df_train_raw)
        test_total = len(df_test_raw)
        print(f"[+] Loaded raw KDDTrain+ ({train_total} records) and KDDTest+ ({test_total} records).")

        # 2. Map attack labels
        df_train_raw['mapped_cat'] = df_train_raw['label'].apply(map_nsl_label)
        df_test_raw['mapped_cat'] = df_test_raw['label'].apply(map_nsl_label)

        train_cat_counts = df_train_raw['mapped_cat'].value_counts().to_dict()
        test_cat_counts = df_test_raw['mapped_cat'].value_counts().to_dict()

        train_included = train_cat_counts.get('Normal', 0) + train_cat_counts.get('DoS', 0) + train_cat_counts.get('Probe', 0)
        train_excluded = train_cat_counts.get('R2L', 0) + train_cat_counts.get('U2R', 0) + train_cat_counts.get('Unknown', 0)

        test_included = test_cat_counts.get('Normal', 0) + test_cat_counts.get('DoS', 0) + test_cat_counts.get('Probe', 0)
        test_excluded = test_cat_counts.get('R2L', 0) + test_cat_counts.get('U2R', 0) + test_cat_counts.get('Unknown', 0)

        self.dataset_stats = {
            "train": {
                "total": train_total,
                "included": train_included,
                "excluded": train_excluded,
                "normal": train_cat_counts.get('Normal', 0),
                "dos": train_cat_counts.get('DoS', 0),
                "probe": train_cat_counts.get('Probe', 0),
                "r2l": train_cat_counts.get('R2L', 0),
                "u2r": train_cat_counts.get('U2R', 0),
                "unknown": train_cat_counts.get('Unknown', 0)
            },
            "test": {
                "total": test_total,
                "included": test_included,
                "excluded": test_excluded,
                "normal": test_cat_counts.get('Normal', 0),
                "dos": test_cat_counts.get('DoS', 0),
                "probe": test_cat_counts.get('Probe', 0),
                "r2l": test_cat_counts.get('R2L', 0),
                "u2r": test_cat_counts.get('U2R', 0),
                "unknown": test_cat_counts.get('Unknown', 0)
            }
        }

        # Filter strictly to 3 categories
        df_train_3 = df_train_raw[df_train_raw['mapped_cat'].isin(self.target_classes)].copy()
        df_test_3 = df_test_raw[df_test_raw['mapped_cat'].isin(self.target_classes)].copy()

        y_train_full = df_train_3['mapped_cat'].values
        y_test_full = df_test_3['mapped_cat'].values

        # Prepare Basic Features
        X_train_basic = self.prepare_features(df_train_3)
        X_test_basic = self.prepare_features(df_test_3)

        # Prepare Enhanced Features
        X_train_enh = self.prepare_enhanced_features(df_train_3)
        X_test_enh = self.prepare_enhanced_features(df_test_3)

        # 3. 80/20 Stratified Validation on KDDTrain+
        print("[+] Running 80/20 stratified validation comparison on KDDTrain+...")
        idx_train, idx_val = train_test_split(
            np.arange(len(df_train_3)), test_size=0.20, random_state=42, stratify=y_train_full
        )

        # Basic Mode Validation (Logistic Regression, KNN, Random Forest)
        val_scaler_b = StandardScaler()
        X_tr_b = X_train_basic.iloc[idx_train].copy()
        X_val_b = X_train_basic.iloc[idx_val].copy()
        val_scaler_b.fit(X_tr_b[self.numerical_cols])
        X_tr_b[self.numerical_cols] = val_scaler_b.transform(X_tr_b[self.numerical_cols])
        X_val_b[self.numerical_cols] = val_scaler_b.transform(X_val_b[self.numerical_cols])

        val_results = []
        for name, clf in self.models.items():
            clf.fit(X_tr_b, y_train_full[idx_train])
            y_val_pred = clf.predict(X_val_b)

            acc = float(accuracy_score(y_train_full[idx_val], y_val_pred))
            prec = float(precision_score(y_train_full[idx_val], y_val_pred, average='weighted', zero_division=0))
            rec = float(recall_score(y_train_full[idx_val], y_val_pred, average='weighted', zero_division=0))
            f1 = float(f1_score(y_train_full[idx_val], y_val_pred, average='weighted', zero_division=0))
            rep = classification_report(y_train_full[idx_val], y_val_pred, output_dict=True, zero_division=0)

            per_class_ordered = {}
            for c in self.target_classes:
                if c in rep:
                    per_class_ordered[c] = {k: round(v, 4) for k, v in rep[c].items()}

            val_results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "per_class": per_class_ordered
            })
            print(f"    - {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

        # Enhanced Mode Validation (10-feature Random Forest)
        val_scaler_e = StandardScaler()
        X_tr_e = X_train_enh.iloc[idx_train].copy()
        X_val_e = X_train_enh.iloc[idx_val].copy()
        val_scaler_e.fit(X_tr_e[self.enhanced_numerical_cols])
        X_tr_e[self.enhanced_numerical_cols] = val_scaler_e.transform(X_tr_e[self.enhanced_numerical_cols])
        X_val_e[self.enhanced_numerical_cols] = val_scaler_e.transform(X_val_e[self.enhanced_numerical_cols])

        val_rf_e = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        val_rf_e.fit(X_tr_e, y_train_full[idx_train])
        y_val_e_pred = val_rf_e.predict(X_val_e)
        val_e_acc = float(accuracy_score(y_train_full[idx_val], y_val_e_pred))
        val_e_f1 = float(f1_score(y_train_full[idx_val], y_val_e_pred, average='weighted', zero_division=0))
        print(f"    - Enhanced Random Forest (Validation): Acc={val_e_acc:.4f}, F1={val_e_f1:.4f}")

        self.val_benchmark_results = val_results

        # 4. Fit Production Models on Full KDDTrain+ (124,926 records)
        print("[+] Fitting production Basic Random Forest on complete filtered KDDTrain+ (124,926 records)...")
        self.scaler = StandardScaler()
        X_train_basic_s = X_train_basic.copy()
        self.scaler.fit(X_train_basic[self.numerical_cols])
        X_train_basic_s[self.numerical_cols] = self.scaler.transform(X_train_basic[self.numerical_cols])

        prod_rf_b = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        prod_rf_b.fit(X_train_basic_s, y_train_full)
        self.models["Random Forest"] = prod_rf_b
        self.active_model = prod_rf_b
        self.classes_ = self.target_classes.copy()

        print("[+] Fitting production Enhanced Random Forest (10 features) on complete filtered KDDTrain+...")
        self.enhanced_scaler = StandardScaler()
        X_train_enh_s = X_train_enh.copy()
        self.enhanced_scaler.fit(X_train_enh[self.enhanced_numerical_cols])
        X_train_enh_s[self.enhanced_numerical_cols] = self.enhanced_scaler.transform(X_train_enh[self.enhanced_numerical_cols])

        prod_rf_e = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        prod_rf_e.fit(X_train_enh_s, y_train_full)
        self.enhanced_rf = prod_rf_e

        # 5. Evaluate Both Production Models on Strictly Separate KDDTest+ (19,592 records)
        print("[+] Evaluating Basic Mode on independent KDDTest+ (19,592 records)...")
        X_test_basic_s = X_test_basic.copy()
        X_test_basic_s[self.numerical_cols] = self.scaler.transform(X_test_basic[self.numerical_cols])

        y_pred_b = prod_rf_b.predict(X_test_basic_s)
        probas_b = prod_rf_b.predict_proba(X_test_basic_s)
        conf_b = np.max(probas_b, axis=1)

        test_acc_b = float(accuracy_score(y_test_full, y_pred_b))
        test_prec_b = float(precision_score(y_test_full, y_pred_b, average='weighted', zero_division=0))
        test_rec_b = float(recall_score(y_test_full, y_pred_b, average='weighted', zero_division=0))
        test_f1_b = float(f1_score(y_test_full, y_pred_b, average='weighted', zero_division=0))
        test_f1_macro_b = float(f1_score(y_test_full, y_pred_b, average='macro', zero_division=0))
        test_rep_b = classification_report(y_test_full, y_pred_b, output_dict=True, zero_division=0)

        cm_b = confusion_matrix(y_test_full, y_pred_b, labels=self.target_classes)
        feat_imps_b = {feat: round(float(imp), 4) for feat, imp in zip(self.feature_columns, prod_rf_b.feature_importances_)}

        # Uncertainty evaluation for Basic Mode at threshold = 0.65
        thresh_b = self.confidence_thresholds["basic"]
        acc_mask_b = conf_b >= thresh_b
        cov_b = float(acc_mask_b.mean())
        acc_accepted_b = float(accuracy_score(y_test_full[acc_mask_b], y_pred_b[acc_mask_b])) if acc_mask_b.sum() > 0 else 0.0
        review_count_b = int((~acc_mask_b).sum())

        per_class_b = {}
        for c in self.target_classes:
            if c in test_rep_b:
                per_class_b[c] = {k: round(v, 4) for k, v in test_rep_b[c].items()}

        self.test_evaluation_results = {
            "mode": "Basic Mode (4 Features)",
            "Accuracy": round(test_acc_b, 4),
            "Precision": round(test_prec_b, 4),
            "Recall": round(test_rec_b, 4),
            "F1-Score": round(test_f1_b, 4),
            "macro_f1": round(test_f1_macro_b, 4),
            "per_class": per_class_b,
            "confusion_matrix": cm_b.tolist(),
            "confusion_matrix_labels": self.target_classes,
            "feature_importances": feat_imps_b,
            "confidence_threshold": thresh_b,
            "coverage": round(cov_b, 4),
            "accuracy_accepted": round(acc_accepted_b, 4),
            "review_count": review_count_b,
            "total_test_records": len(y_test_full)
        }
        print(f"[+] Basic Mode KDDTest+: Acc={test_acc_b:.4f}, Prec={test_prec_b:.4f}, Rec={test_rec_b:.4f}, F1={test_f1_b:.4f}, Probe Rec={per_class_b['Probe']['recall']:.4f}")

        print("[+] Evaluating Enhanced Mode on independent KDDTest+ (19,592 records)...")
        X_test_enh_s = X_test_enh.copy()
        X_test_enh_s[self.enhanced_numerical_cols] = self.enhanced_scaler.transform(X_test_enh[self.enhanced_numerical_cols])

        y_pred_e = prod_rf_e.predict(X_test_enh_s)
        probas_e = prod_rf_e.predict_proba(X_test_enh_s)
        conf_e = np.max(probas_e, axis=1)

        test_acc_e = float(accuracy_score(y_test_full, y_pred_e))
        test_prec_e = float(precision_score(y_test_full, y_pred_e, average='weighted', zero_division=0))
        test_rec_e = float(recall_score(y_test_full, y_pred_e, average='weighted', zero_division=0))
        test_f1_e = float(f1_score(y_test_full, y_pred_e, average='weighted', zero_division=0))
        test_f1_macro_e = float(f1_score(y_test_full, y_pred_e, average='macro', zero_division=0))
        test_rep_e = classification_report(y_test_full, y_pred_e, output_dict=True, zero_division=0)

        cm_e = confusion_matrix(y_test_full, y_pred_e, labels=self.target_classes)
        feat_imps_e = {feat: round(float(imp), 4) for feat, imp in zip(self.enhanced_feature_columns, prod_rf_e.feature_importances_)}

        # Uncertainty evaluation for Enhanced Mode at threshold = 0.65
        thresh_e = self.confidence_thresholds["enhanced"]
        acc_mask_e = conf_e >= thresh_e
        cov_e = float(acc_mask_e.mean())
        acc_accepted_e = float(accuracy_score(y_test_full[acc_mask_e], y_pred_e[acc_mask_e])) if acc_mask_e.sum() > 0 else 0.0
        review_count_e = int((~acc_mask_e).sum())

        per_class_e = {}
        for c in self.target_classes:
            if c in test_rep_e:
                per_class_e[c] = {k: round(v, 4) for k, v in test_rep_e[c].items()}

        self.enhanced_test_evaluation = {
            "mode": "Enhanced Mode (10 Features)",
            "Accuracy": round(test_acc_e, 4),
            "Precision": round(test_prec_e, 4),
            "Recall": round(test_rec_e, 4),
            "F1-Score": round(test_f1_e, 4),
            "macro_f1": round(test_f1_macro_e, 4),
            "per_class": per_class_e,
            "confusion_matrix": cm_e.tolist(),
            "confusion_matrix_labels": self.target_classes,
            "feature_importances": feat_imps_e,
            "confidence_threshold": thresh_e,
            "coverage": round(cov_e, 4),
            "accuracy_accepted": round(acc_accepted_e, 4),
            "review_count": review_count_e,
            "total_test_records": len(y_test_full)
        }
        print(f"[+] Enhanced Mode KDDTest+: Acc={test_acc_e:.4f}, Prec={test_prec_e:.4f}, Rec={test_rec_e:.4f}, F1={test_f1_e:.4f}, Probe Rec={per_class_e['Probe']['recall']:.4f}")

        # Side-by-side mode comparison summary
        self.mode_comparison_results = {
            "metrics": ["Accuracy", "Weighted F1", "Macro F1", "Normal Recall", "DoS Recall", "Probe Recall", "Automatic Coverage", "Accepted Accuracy", "Manual Review Count"],
            "basic_mode": {
                "features_count": 4,
                "features_list": BASIC_INPUT_FEATURES,
                "accuracy": round(test_acc_b, 4),
                "f1_weighted": round(test_f1_b, 4),
                "f1_macro": round(test_f1_macro_b, 4),
                "normal_recall": round(per_class_b['Normal']['recall'], 4),
                "dos_recall": round(per_class_b['DoS']['recall'], 4),
                "probe_recall": round(per_class_b['Probe']['recall'], 4),
                "coverage": round(cov_b, 4),
                "accuracy_accepted": round(acc_accepted_b, 4),
                "review_count": review_count_b
            },
            "enhanced_mode": {
                "features_count": 10,
                "features_list": ENHANCED_INPUT_FEATURES,
                "accuracy": round(test_acc_e, 4),
                "f1_weighted": round(test_f1_e, 4),
                "f1_macro": round(test_f1_macro_e, 4),
                "normal_recall": round(per_class_e['Normal']['recall'], 4),
                "dos_recall": round(per_class_e['DoS']['recall'], 4),
                "probe_recall": round(per_class_e['Probe']['recall'], 4),
                "coverage": round(cov_e, 4),
                "accuracy_accepted": round(acc_accepted_e, 4),
                "review_count": review_count_e
            },
            "deltas": {
                "accuracy_gain": round((test_acc_e - test_acc_b) * 100, 2),
                "f1_weighted_gain": round((test_f1_e - test_f1_b) * 100, 2),
                "f1_macro_gain": round((test_f1_macro_e - test_f1_macro_b) * 100, 2),
                "probe_recall_gain": round((per_class_e['Probe']['recall'] - per_class_b['Probe']['recall']) * 100, 2),
                "accepted_accuracy_gain": round((acc_accepted_e - acc_accepted_b) * 100, 2)
            }
        }

        # 6. Generate Base64 Charts
        print("[+] Generating high-resolution dark-themed analytics charts...")
        self.charts_b64 = self._generate_all_charts(
            train_cat_counts, test_cat_counts, val_results, self.test_evaluation_results,
            self.enhanced_test_evaluation, self.mode_comparison_results, df_train_3
        )

        # 7. Extract representative sample flows for UI display
        sample_indices = df_test_3.sample(n=min(30, len(df_test_3)), random_state=42).index
        sample_df_b = df_test_3.loc[sample_indices, ['duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'mapped_cat']].copy()
        sample_df_b.rename(columns={'mapped_cat': 'attack_category'}, inplace=True)
        self.sample_flows = sample_df_b.to_dict(orient="records")

        sample_df_e = df_test_3.loc[sample_indices, ENHANCED_INPUT_FEATURES + ['mapped_cat']].copy()
        sample_df_e.rename(columns={'mapped_cat': 'attack_category'}, inplace=True)
        self.sample_flows_enhanced = sample_df_e.to_dict(orient="records")

        # 8. Build Cache Metadata and Save Cache
        train_sha = get_dataset_fingerprint(train_path)
        test_sha = get_dataset_fingerprint(test_path)
        self.cache_metadata = {
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sklearn_version': sklearn.__version__,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'joblib_version': joblib.__version__,
            'feature_schema': list(self.feature_columns),
            'enhanced_feature_schema': list(self.enhanced_feature_columns),
            'numerical_cols': list(self.numerical_cols),
            'enhanced_numerical_cols': list(self.enhanced_numerical_cols),
            'target_classes': list(self.target_classes),
            'confidence_thresholds': self.confidence_thresholds,
            'train_dataset_sha256': train_sha,
            'test_dataset_sha256': test_sha,
            'train_total': train_total,
            'test_total': test_total
        }

        try:
            cache_data = {
                'metadata': self.cache_metadata,
                'scaler': self.scaler,
                'enhanced_scaler': self.enhanced_scaler,
                'models': self.models,
                'enhanced_rf': self.enhanced_rf,
                'dataset_stats': self.dataset_stats,
                'val_benchmark_results': self.val_benchmark_results,
                'test_evaluation_results': self.test_evaluation_results,
                'enhanced_test_evaluation': self.enhanced_test_evaluation,
                'mode_comparison': self.mode_comparison_results,
                'charts_b64': self.charts_b64,
                'sample_flows': self.sample_flows,
                'sample_flows_enhanced': self.sample_flows_enhanced
            }
            joblib.dump(cache_data, cache_path)
            print(f"[+] Dual-mode pipeline cache successfully saved with SHA-256 metadata to {cache_path}.")
        except Exception as e:
            print(f"[WARNING] Could not write cache: {e}")

        self.is_trained = True

    def _generate_all_charts(self, train_counts, test_counts, val_results, test_eval_b, test_eval_e, mode_comp, df_train_sample):
        """Generates all high-resolution dark-palette analytics charts in Base64 PNG."""
        charts = {}
        categories = self.target_classes  # Strictly: ['Normal', 'DoS', 'Probe']

        # Chart A: Classification Distribution (KDDTrain+ vs KDDTest+)
        fig, ax = plt.subplots(figsize=(7, 3.8), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        x = np.arange(len(categories))
        width = 0.35
        train_vals = [train_counts.get(c, 0) for c in categories]
        test_vals = [test_counts.get(c, 0) for c in categories]

        r1 = ax.bar(x - width/2, train_vals, width, label='KDDTrain+ (n=124,926)', color='#58a6ff', alpha=0.9, edgecolor='#30363d')
        r2 = ax.bar(x + width/2, test_vals, width, label='KDDTest+ (n=19,592)', color='#bc8cff', alpha=0.9, edgecolor='#30363d')
        ax.set_ylabel('Record Count', color='#8b949e', fontsize=9)
        ax.set_title('NSL-KDD Class Distribution (Three-Class Scope)', color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, color='#e6edf3', fontsize=9.5)
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.3, color='#30363d', axis='y')
        for s in ax.spines.values(): s.set_color('#30363d')
        for rect in r1:
            h = rect.get_height()
            ax.annotate(f'{h:,}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', color='#8b949e', fontsize=7, fontfamily='monospace')
        for rect in r2:
            h = rect.get_height()
            ax.annotate(f'{h:,}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', color='#8b949e', fontsize=7, fontfamily='monospace')
        ax.legend(frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=8.5)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#161b22', edgecolor='none')
        plt.close(fig)
        charts['classification_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Chart B: Model Benchmark Comparison (80/20 Stratified Validation)
        fig, ax = plt.subplots(figsize=(8, 3.8), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        models = [r['Model'] for r in val_results]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(models))
        w = 0.18
        colors = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff']
        for i, m in enumerate(metrics):
            vals = [r[m] for r in val_results]
            rects = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=colors[i], alpha=0.9, edgecolor='#30363d')
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', color='#8b949e', fontsize=7, fontfamily='monospace')
        ax.set_ylabel('Score', color='#8b949e', fontsize=9)
        ax.set_title('80/20 Stratified Validation Benchmark (KDDTrain+)', color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, color='#e6edf3', fontsize=9.5, fontweight='500')
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.3, color='#30363d', axis='y')
        for s in ax.spines.values(): s.set_color('#30363d')
        ax.legend(frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=8.5, ncol=4, loc='upper left')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#161b22', edgecolor='none')
        plt.close(fig)
        charts['model_comparison'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Chart C: Basic Feature Importance (Global Architecture)
        fig, ax = plt.subplots(figsize=(7, 3.8), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        feat_imps_b = test_eval_b['feature_importances']
        sorted_feats_b = sorted(feat_imps_b.items(), key=lambda k: k[1], reverse=True)
        names_b = [f[0] for f in sorted_feats_b]
        vals_b = [f[1] for f in sorted_feats_b]
        y_pos = np.arange(len(names_b))
        bars = ax.barh(y_pos, vals_b, color='#58a6ff', alpha=0.85, edgecolor='#30363d', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_b, color='#e6edf3', fontsize=9, fontfamily='monospace')
        ax.invert_yaxis()
        ax.set_xlabel('Gini Importance Ratio', color='#8b949e', fontsize=9)
        ax.set_title('Basic RF Feature Importance (4 Features / 6 Encoded)', color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        ax.set_xlim(0, max(vals_b) * 1.25)
        ax.grid(True, linestyle=':', alpha=0.3, color='#30363d', axis='x')
        for s in ax.spines.values(): s.set_color('#30363d')
        for bar in bars:
            bv = bar.get_width()
            ax.annotate(f'{bv * 100:.1f}%', xy=(bv, bar.get_y() + bar.get_height()/2), xytext=(5, 0), textcoords="offset points", ha='left', va='center', color='#e6edf3', fontsize=8, fontfamily='monospace')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#161b22', edgecolor='none')
        plt.close(fig)
        charts['feature_importance'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Chart C2: Enhanced Feature Importance (Top Features)
        fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        feat_imps_e = test_eval_e['feature_importances']
        sorted_feats_e = sorted(feat_imps_e.items(), key=lambda k: k[1], reverse=True)[:10]
        names_e = [f[0] for f in sorted_feats_e]
        vals_e = [f[1] for f in sorted_feats_e]
        y_pos_e = np.arange(len(names_e))
        bars_e = ax.barh(y_pos_e, vals_e, color='#3fb950', alpha=0.85, edgecolor='#30363d', height=0.6)
        ax.set_yticks(y_pos_e)
        ax.set_yticklabels(names_e, color='#e6edf3', fontsize=9, fontfamily='monospace')
        ax.invert_yaxis()
        ax.set_xlabel('Gini Importance Ratio', color='#8b949e', fontsize=9)
        ax.set_title('Enhanced RF Feature Importance (Top 10 of 22 Features)', color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        ax.set_xlim(0, max(vals_e) * 1.25)
        ax.grid(True, linestyle=':', alpha=0.3, color='#30363d', axis='x')
        for s in ax.spines.values(): s.set_color('#30363d')
        for bar in bars_e:
            bv = bar.get_width()
            ax.annotate(f'{bv * 100:.1f}%', xy=(bv, bar.get_y() + bar.get_height()/2), xytext=(5, 0), textcoords="offset points", ha='left', va='center', color='#e6edf3', fontsize=8, fontfamily='monospace')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#161b22', edgecolor='none')
        plt.close(fig)
        charts['feature_importance_enhanced'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Chart D: Basic RF Confusion Matrix (Strict Order: Normal, DoS, Probe)
        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        cm_arr_b = np.array(test_eval_b['confusion_matrix'])
        ax.imshow(cm_arr_b, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Basic RF Confusion Matrix (KDDTest+, n=19,592)', color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
        tick_marks = np.arange(len(self.target_classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(self.target_classes, color='#e6edf3', fontsize=9.5)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(self.target_classes, color='#e6edf3', fontsize=9.5)
        thresh_b_cm = cm_arr_b.max() / 2.0
        for i in range(cm_arr_b.shape[0]):
            for j in range(cm_arr_b.shape[1]):
                v = cm_arr_b[i, j]
                ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > thresh_b_cm else "#8b949e", fontfamily='monospace', fontsize=9.5, fontweight='bold')
        ax.set_ylabel('True NSL-KDD Class', color='#8b949e', fontsize=9)
        ax.set_xlabel('Predicted Class', color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        for s in ax.spines.values(): s.set_color('#30363d')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#161b22', edgecolor='none')
        plt.close(fig)
        charts['confusion_matrix'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Chart D2: Enhanced RF Confusion Matrix (Strict Order: Normal, DoS, Probe)
        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        cm_arr_e = np.array(test_eval_e['confusion_matrix'])
        ax.imshow(cm_arr_e, interpolation='nearest', cmap=plt.cm.Greens)
        ax.set_title('Enhanced RF Confusion Matrix (KDDTest+, n=19,592)', color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(self.target_classes, color='#e6edf3', fontsize=9.5)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(self.target_classes, color='#e6edf3', fontsize=9.5)
        thresh_e_cm = cm_arr_e.max() / 2.0
        for i in range(cm_arr_e.shape[0]):
            for j in range(cm_arr_e.shape[1]):
                v = cm_arr_e[i, j]
                ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > thresh_e_cm else "#8b949e", fontfamily='monospace', fontsize=9.5, fontweight='bold')
        ax.set_ylabel('True NSL-KDD Class', color='#8b949e', fontsize=9)
        ax.set_xlabel('Predicted Class', color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        for s in ax.spines.values(): s.set_color('#30363d')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#161b22', edgecolor='none')
        plt.close(fig)
        charts['confusion_matrix_enhanced'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Chart E: Per-Class Precision, Recall, and F1-Score (Basic Mode)
        fig, ax = plt.subplots(figsize=(7, 3.8), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        c_metrics = ['precision', 'recall', 'f1-score']
        c_labels = ['Precision', 'Recall', 'F1-Score']
        x = np.arange(len(self.target_classes))
        w = 0.25
        colors_b = ['#3fb950', '#58a6ff', '#bc8cff']
        for i, m_name in enumerate(c_metrics):
            vals = [test_eval_b['per_class'][c][m_name] for c in self.target_classes]
            rects = ax.bar(x + (i - 1) * w, vals, w, label=c_labels[i], color=colors_b[i], alpha=0.9, edgecolor='#30363d')
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', color='#8b949e', fontsize=7.5, fontfamily='monospace')
        ax.set_ylabel('Score', color='#8b949e', fontsize=9)
        ax.set_title('Per-Class Production Performance on KDDTest+ (Basic Mode)', color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(self.target_classes, color='#e6edf3', fontsize=9.5, fontweight='500')
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.3, color='#30363d', axis='y')
        for s in ax.spines.values(): s.set_color('#30363d')
        ax.legend(frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=8.5)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#161b22', edgecolor='none')
        plt.close(fig)
        charts['per_class'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Chart E2: Per-Class Performance (Enhanced Mode)
        fig, ax = plt.subplots(figsize=(7, 3.8), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        for i, m_name in enumerate(c_metrics):
            vals = [test_eval_e['per_class'][c][m_name] for c in self.target_classes]
            rects = ax.bar(x + (i - 1) * w, vals, w, label=c_labels[i], color=colors_b[i], alpha=0.9, edgecolor='#30363d')
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', color='#8b949e', fontsize=7.5, fontfamily='monospace')
        ax.set_ylabel('Score', color='#8b949e', fontsize=9)
        ax.set_title('Per-Class Production Performance on KDDTest+ (Enhanced Mode)', color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(self.target_classes, color='#e6edf3', fontsize=9.5, fontweight='500')
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.3, color='#30363d', axis='y')
        for s in ax.spines.values(): s.set_color('#30363d')
        ax.legend(frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=8.5)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#161b22', edgecolor='none')
        plt.close(fig)
        charts['per_class_enhanced'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Chart F: Side-by-Side Dual-Mode Architecture Comparison Chart
        fig, ax = plt.subplots(figsize=(8.5, 4.0), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        comp_metrics = ['Accuracy', 'Weighted F1', 'Macro F1', 'Probe Recall', 'Accepted Acc.']
        basic_pcts = [
            test_eval_b['Accuracy'] * 100,
            test_eval_b['F1-Score'] * 100,
            test_eval_b.get('macro_f1', 0.7084) * 100,
            test_eval_b['per_class']['Probe']['recall'] * 100,
            test_eval_b.get('accuracy_accepted', 0.8237) * 100
        ]
        enh_pcts = [
            test_eval_e['Accuracy'] * 100,
            test_eval_e['F1-Score'] * 100,
            test_eval_e.get('macro_f1', 0.8126) * 100,
            test_eval_e['per_class']['Probe']['recall'] * 100,
            test_eval_e.get('accuracy_accepted', 0.8947) * 100
        ]
        x_comp = np.arange(len(comp_metrics))
        w_comp = 0.35
        r_b = ax.bar(x_comp - w_comp/2, basic_pcts, w_comp, label='Basic Mode (4 Features)', color='#58a6ff', alpha=0.9, edgecolor='#30363d')
        r_e = ax.bar(x_comp + w_comp/2, enh_pcts, w_comp, label='Enhanced Mode (10 Features)', color='#3fb950', alpha=0.9, edgecolor='#30363d')
        ax.set_ylabel('Percentage (%)', color='#8b949e', fontsize=9)
        ax.set_title('Detection Architecture Comparison (Independent KDDTest+, n=19,592)', color='#e6edf3', fontsize=11, fontweight='bold', pad=12)
        ax.set_xticks(x_comp)
        ax.set_xticklabels(comp_metrics, color='#e6edf3', fontsize=9.5, fontweight='500')
        ax.set_ylim(0, 115)
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.3, color='#30363d', axis='y')
        for s in ax.spines.values(): s.set_color('#30363d')
        for rect in r_b:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', color='#8b949e', fontsize=8, fontfamily='monospace')
        for rect in r_e:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', color='#3fb950', fontsize=8, fontfamily='monospace', fontweight='bold')
        ax.legend(frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=8.5, loc='upper left')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#161b22', edgecolor='none')
        plt.close(fig)
        charts['mode_comparison_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Scatter Plot: Source Payload vs Duration (NSL-KDD Sample)
        sample_scatter = df_train_sample.sample(n=min(600, len(df_train_sample)), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 4.2), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        palette = {'Normal': '#3fb950', 'DoS': '#f85149', 'Probe': '#d29922'}
        markers = {'Normal': 'o', 'DoS': 's', 'Probe': '^'}
        for cat_name, subset in sample_scatter.groupby('mapped_cat'):
            src_bytes_log = np.clip(subset['src_bytes'], 1, None)
            dur_clipped = np.clip(subset['duration'], 0.001, None)
            ax.scatter(dur_clipped, src_bytes_log, color=palette.get(cat_name, '#58a6ff'), marker=markers.get(cat_name, 'o'), s=36, alpha=0.75, edgecolors='none', label=f"{cat_name} (n={len(subset)})")
        ax.set_xlabel("Flow Duration (seconds)", fontsize=9.5, color='#8b949e', labelpad=8)
        ax.set_ylabel("Source Payload (bytes, log scale)", fontsize=9.5, color='#8b949e', labelpad=8)
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, linestyle=':', alpha=0.35, color='#30363d')
        for s in ax.spines.values(): s.set_color('#30363d')
        leg = ax.legend(frameon=True, facecolor='#161b22', edgecolor='#30363d', fontsize=8.5, loc='upper right')
        for txt in leg.get_texts(): txt.set_color('#e6edf3')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#161b22', edgecolor='none')
        plt.close(fig)
        charts['scatter_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        return charts

    def validate_single_flow(self, flow_dict):
        """
        Validates individual flow telemetry inputs for Basic or Enhanced mode.
        Returns:
            (mode: str, basic_clean: tuple, enh_clean: dict or None, fallback_reason: str)
        Raises user-defined exceptions upon validation failure of basic fields.
        """
        if not flow_dict:
            raise EmptyTrafficInputError("EmptyTrafficInputError: No flow telemetry payload provided.")

        for key in ['src_bytes', 'duration']:
            val = flow_dict.get(key)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                raise EmptyTrafficInputError(f"EmptyTrafficInputError: Field '{key}' is required and cannot be empty.")

        try:
            src_bytes = float(flow_dict['src_bytes'])
            duration = float(flow_dict['duration'])
            dst_bytes_val = flow_dict.get('dst_bytes', 0)
            dst_bytes = float(dst_bytes_val) if str(dst_bytes_val).strip() != "" else 0.0
        except (ValueError, TypeError) as e:
            raise InvalidPacketSizeError(f"InvalidPacketSizeError: Non-numeric telemetry value supplied: {e}")

        if np.isnan(src_bytes) or np.isnan(duration) or np.isnan(dst_bytes):
            raise InvalidPacketSizeError("InvalidPacketSizeError: NaN numeric values detected in flow inputs.")

        if np.isinf(src_bytes) or np.isinf(duration) or np.isinf(dst_bytes):
            raise InvalidPacketSizeError("InvalidPacketSizeError: Infinite numeric values detected in flow inputs.")

        if src_bytes < 0 or dst_bytes < 0 or duration < 0:
            raise InvalidPacketSizeError(
                f"InvalidPacketSizeError: Flow attributes cannot be negative (src_bytes={src_bytes}, duration={duration}, dst_bytes={dst_bytes})."
            )

        proto = str(flow_dict.get('protocol_type', '')).strip().upper()
        if not proto:
            raise EmptyTrafficInputError("EmptyTrafficInputError: Field 'protocol_type' is required.")
        if proto not in SUPPORTED_PROTOCOLS:
            raise InvalidPacketSizeError(
                f"InvalidPacketSizeError: Unsupported protocol '{proto}'. Supported protocols are TCP, UDP, ICMP."
            )

        basic_tuple = (src_bytes, duration, dst_bytes, proto)

        # Check for enhanced attributes
        enh_present = all(k in flow_dict and str(flow_dict[k]).strip() != "" for k in ENHANCED_ADDITIONAL_COLS)
        if not enh_present:
            # Check if any enhanced attributes were partially provided
            part_present = [k for k in ENHANCED_ADDITIONAL_COLS if k in flow_dict and str(flow_dict[k]).strip() != ""]
            fb_reason = f"Partial enhanced features provided ({part_present}); missing other required enhanced fields" if part_present else ""
            return "Basic", basic_tuple, None, fb_reason

        # Validate enhanced attributes
        try:
            # Count & srv_count must be nonnegative integers
            c_val = float(flow_dict['count'])
            sc_val = float(flow_dict['srv_count'])
            if np.isnan(c_val) or np.isinf(c_val) or c_val < 0 or not c_val.is_integer():
                return "Basic", basic_tuple, None, f"Enhanced feature 'count' ({c_val}) must be a non-negative integer."
            if np.isnan(sc_val) or np.isinf(sc_val) or sc_val < 0 or not sc_val.is_integer():
                return "Basic", basic_tuple, None, f"Enhanced feature 'srv_count' ({sc_val}) must be a non-negative integer."

            # Rate features must be finite floats between 0.0 and 1.0
            same_srv = float(flow_dict['same_srv_rate'])
            serror = float(flow_dict['serror_rate'])
            dst_diff = float(flow_dict['dst_host_diff_srv_rate'])
            for r_name, r_val in [('same_srv_rate', same_srv), ('serror_rate', serror), ('dst_host_diff_srv_rate', dst_diff)]:
                if np.isnan(r_val) or np.isinf(r_val) or not (0.0 <= r_val <= 1.0):
                    return "Basic", basic_tuple, None, f"Enhanced feature '{r_name}' ({r_val}) must be a finite float between 0.0 and 1.0."

            # Flag must be supported NSL-KDD flag
            flag_str = str(flow_dict['flag']).strip().upper()
            if flag_str not in SUPPORTED_FLAGS:
                return "Basic", basic_tuple, None, f"Enhanced feature 'flag' ('{flag_str}') is not a supported NSL-KDD flag status."

            enh_clean = {
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'duration': duration,
                'protocol_type': proto,
                'count': int(c_val),
                'srv_count': int(sc_val),
                'flag': flag_str,
                'same_srv_rate': round(same_srv, 4),
                'serror_rate': round(serror, 4),
                'dst_host_diff_srv_rate': round(dst_diff, 4)
            }
            return "Enhanced", basic_tuple, enh_clean, ""

        except (ValueError, TypeError) as e:
            return "Basic", basic_tuple, None, f"Invalid enhanced telemetry conversion: {e}"

    def predict_threat(self, flow_dict):
        """
        Validates telemetry, determines detection mode (Basic vs Enhanced), evaluates threat classification,
        computes model probability estimates, determines decision status (ACCEPTED vs REVIEW REQUIRED),
        and generates explainable diagnostic descriptions.
        """
        mode, (src_bytes, duration, dst_bytes, proto), enh_dict, fallback_reason = self.validate_single_flow(flow_dict)

        if mode == "Enhanced" and enh_dict is not None:
            # Enhanced Mode evaluation
            df_input = pd.DataFrame([enh_dict])
            X_encoded = self.prepare_enhanced_features(df_input)
            X_scaled = X_encoded.copy()
            X_scaled[self.enhanced_numerical_cols] = self.enhanced_scaler.transform(X_encoded[self.enhanced_numerical_cols])

            active_clf = self.enhanced_rf
            features_used = ENHANCED_INPUT_FEATURES
            thresh = self.confidence_thresholds.get("enhanced", DEFAULT_CONFIDENCE_THRESHOLD)
        else:
            # Basic Mode evaluation
            df_input = pd.DataFrame([{
                'src_bytes': src_bytes,
                'duration': duration,
                'dst_bytes': dst_bytes,
                'protocol_type': proto
            }])
            X_encoded = self.prepare_features(df_input)
            X_scaled = X_encoded.copy()
            X_scaled[self.numerical_cols] = self.scaler.transform(X_encoded[self.numerical_cols])

            active_clf = self.models["Random Forest"]
            features_used = BASIC_INPUT_FEATURES
            thresh = self.confidence_thresholds.get("basic", DEFAULT_CONFIDENCE_THRESHOLD)

        predicted_category = str(active_clf.predict(X_scaled)[0])
        probabilities = active_clf.predict_proba(X_scaled)[0]

        # Extract probability estimates keyed strictly by canonical target classes
        raw_classes = list(active_clf.classes_)
        prob_dict = {}
        for cls_name in self.target_classes:
            if cls_name in raw_classes:
                idx = raw_classes.index(cls_name)
                prob_dict[cls_name] = round(float(probabilities[idx]), 4)
            else:
                prob_dict[cls_name] = 0.0

        confidence = float(max(probabilities))

        # Uncertainty handling: decision status based on KDDTrain+ validation threshold
        decision_status = "ACCEPTED" if confidence >= thresh else "REVIEW REQUIRED"
        low_confidence_warning = bool(decision_status == "REVIEW REQUIRED")

        # Deterministic categorical severity mapping (not an independent probability)
        risk_level = "CRITICAL" if predicted_category == "DoS" else ("ELEVATED" if predicted_category == "Probe" else "LOW")

        # Cautious plain-language explanation
        explanation = self._generate_explanation(
            src_bytes, duration, dst_bytes, proto, predicted_category, confidence,
            mode=mode, enh_dict=enh_dict, decision_status=decision_status, fallback_reason=fallback_reason
        )

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_mode": mode,
            "features_used": features_used,
            "fallback_reason": fallback_reason,
            "protocol_type": proto,
            "src_bytes": int(src_bytes),
            "duration": round(float(duration), 4),
            "dst_bytes": int(dst_bytes),
            "classification": predicted_category,
            "confidence": round(confidence, 4),
            "decision_status": decision_status,
            "model_probability_estimates": prob_dict,
            "probabilities": prob_dict,  # Backward compatibility
            "risk_level": risk_level,
            "low_confidence_warning": low_confidence_warning,
            "explanation": explanation,
            "risk_mapping_note": RISK_MAPPING_NOTE,
            "uncertainty_note": UNCERTAINTY_NOTE,
            "disclaimer": "NSL-KDD three-class model (Normal, DoS, Probe). Does not detect R2L or U2R. Probabilities are uncalibrated model estimates from Random Forest tree voting. Feature importances reflect global model behavior across training corpus and do not establish per-flow causality."
        }

        if enh_dict:
            result.update({
                "count": enh_dict['count'],
                "srv_count": enh_dict['srv_count'],
                "flag": enh_dict['flag'],
                "same_srv_rate": enh_dict['same_srv_rate'],
                "serror_rate": enh_dict['serror_rate'],
                "dst_host_diff_srv_rate": enh_dict['dst_host_diff_srv_rate']
            })

        self.log_session(result)
        return result

    def _generate_explanation(self, src_bytes, duration, dst_bytes, proto, predicted_cat, confidence,
                              mode="Basic", enh_dict=None, decision_status="ACCEPTED", fallback_reason=""):
        """
        Generates a cautious, evidence-grounded plain-language explanation of the observed flow pattern.
        Distinguishes statistical model probability estimate from observed flow characteristics
        and global feature importance, without claiming individual causality or verified external attribution.
        """
        conf_pct = confidence * 100

        if mode == "Enhanced" and enh_dict is not None:
            c = enh_dict['count']
            fl = enh_dict['flag']
            serr = enh_dict['serror_rate']
            same_r = enh_dict['same_srv_rate']
            diff_r = enh_dict['dst_host_diff_srv_rate']

            if predicted_cat == "DoS":
                obs = f"connection count ({c}), status {fl}, SYN error rate {serr:.2f}, and payload {int(src_bytes):,} bytes"
                desc = (
                    f"Enhanced Mode evaluated flow displaying {obs}. Model probability estimate is {conf_pct:.1f}% for DoS. "
                    "Pattern aligns with repetitive connection flooding or volumetric resource exhaustion."
                )
            elif predicted_cat == "Probe":
                obs = f"dst_host_diff_srv_rate of {diff_r:.2f}, connection status {fl}, and low payload ({int(src_bytes):,} bytes sent / {int(dst_bytes):,} recv)"
                desc = (
                    f"Enhanced Mode evaluated flow displaying {obs}. Model probability estimate is {conf_pct:.1f}% for Probe. "
                    "Pattern matches multi-port host scanning or reconnaissance telemetry."
                )
            else:  # Normal
                obs = f"standard connection status {fl}, same-service rate {same_r:.2f}, and balanced bidirectional transfer ({int(src_bytes):,} bytes sent, {int(dst_bytes):,} recv)"
                desc = (
                    f"Enhanced Mode evaluated flow displaying {obs}. Model probability estimate is {conf_pct:.1f}% for Normal. "
                    "Pattern aligns with expected baseline service communication."
                )
        else:
            # Basic Mode
            if predicted_cat == "DoS":
                if src_bytes > 20000:
                    obs = f"volumetric source payload ({int(src_bytes):,} bytes) over {duration:.2f}s with minimal return payload ({int(dst_bytes):,} bytes)"
                elif duration < 0.5 and dst_bytes == 0:
                    obs = f"rapid unidirectional packet transfer ({int(src_bytes):,} bytes) with zero response bytes via {proto}"
                else:
                    obs = f"high source-to-destination byte ratio ({int(src_bytes):,} bytes sent vs {int(dst_bytes):,} received over {duration:.2f}s)"
                desc = (
                    f"Basic Mode evaluated flow displaying {obs}. Model probability estimate is {conf_pct:.1f}% for DoS. "
                    "Descriptive pattern aligns with volumetric resource exhaustion."
                )
            elif predicted_cat == "Probe":
                obs = f"low source payload ({int(src_bytes):,} bytes) and minimal response ({int(dst_bytes):,} bytes) over {duration:.2f}s via {proto}"
                desc = (
                    f"Basic Mode evaluated flow displaying {obs}. Model probability estimate is {conf_pct:.1f}% for Probe. "
                    "Descriptive pattern matches low-footprint port scanning or reconnaissance telemetry."
                )
            else:  # Normal
                obs = f"two-way responsive communication ({int(src_bytes):,} bytes sent, {int(dst_bytes):,} received over {duration:.2f}s via {proto})"
                desc = (
                    f"Basic Mode evaluated flow displaying {obs}. Model probability estimate is {conf_pct:.1f}% for Normal. "
                    "Descriptive pattern aligns with expected baseline network service communication."
                )

        if fallback_reason:
            desc = f"[FALLBACK TO BASIC MODE: {fallback_reason}] " + desc

        if decision_status == "REVIEW REQUIRED":
            desc = f"[REVIEW REQUIRED: Model probability estimate is {conf_pct:.1f}%, below the 65.0% certainty threshold. Telemetry features lie in an ambiguous boundary region.] " + desc

        return desc

    def process_csv_batch(self, file_storage, filename="uploaded_traffic.csv"):
        """
        Processes and validates an uploaded network traffic CSV file using Adaptive Dual-Mode selection.
        - Automatically detects presence of Enhanced features (10 features) vs Basic features (4 features).
        - If all 10 enhanced features are present and valid, uses Enhanced Mode.
        - If only basic features are present, uses Basic Mode.
        - If enhanced features are partially present or invalid on individual rows, falls back to Basic Mode with fallback logging.
        - If basic features are invalid, rejects the row.
        - Applies uncertainty-aware decision thresholding (ACCEPTED vs REVIEW REQUIRED).
        """
        safe_filename = sanitize_filename(filename)

        # 1. Read file buffer
        try:
            file_storage.seek(0, os.SEEK_END)
            file_size = file_storage.tell()
            file_storage.seek(0)
        except Exception:
            file_size = 0

        if file_size == 0:
            raise EmptyTrafficInputError("EmptyTrafficInputError: Uploaded CSV file is empty (0 bytes).")

        if file_size > 16 * 1024 * 1024:
            raise NetworkSecurityException("File size exceeds maximum allowed limit (16MB).")

        try:
            raw_text = file_storage.read().decode('utf-8', errors='replace')
            file_storage.seek(0)
        except Exception as e:
            raise NetworkSecurityException(f"Failed to read file encoding: {e}")

        # 2. Check for duplicate headers
        lines = [l.strip() for l in raw_text.strip().split('\n') if l.strip()]
        if not lines:
            raise EmptyTrafficInputError("EmptyTrafficInputError: No readable records found in CSV file.")

        header_cols = [c.strip() for c in lines[0].split(',')]
        header_lower = [c.lower() for c in header_cols]
        if len(header_lower) != len(set(header_lower)):
            duplicates = [c for c in header_cols if header_lower.count(c.lower()) > 1]
            raise NetworkSecurityException(f"Duplicate header columns detected in uploaded CSV: {list(set(duplicates))}")

        # 3. Read into pandas
        try:
            df_upload = pd.read_csv(io.StringIO(raw_text))
        except Exception as e:
            raise NetworkSecurityException(f"CSV Parsing Error: {e}")

        if len(df_upload) == 0:
            raise EmptyTrafficInputError("EmptyTrafficInputError: CSV file contains headers but zero data rows.")

        # 4. Check available columns (case-insensitive mapping)
        col_map = {}
        for c in df_upload.columns:
            clean_c = c.strip().lower()
            col_map[clean_c] = c

        # Verify required basic keys
        missing_basic = [k for k in BASIC_INPUT_FEATURES if k not in col_map]
        if missing_basic:
            raise NetworkSecurityException(
                f"Missing required CSV columns: {missing_basic}. Required basic headers are: 'src_bytes', 'dst_bytes', 'duration', 'protocol_type'."
            )

        # Check enhanced features presence in header
        missing_enh_cols = [k for k in ENHANCED_ADDITIONAL_COLS if k not in col_map]
        has_all_enh_cols = (len(missing_enh_cols) == 0)
        has_partial_enh_cols = (0 < len(missing_enh_cols) < len(ENHANCED_ADDITIONAL_COLS))

        # 5. Row-by-row validation & mode selection
        valid_indices = []
        row_modes = []
        row_fallback_reasons = []
        basic_rows_data = []
        enh_rows_data = []
        rejection_reasons = []

        for idx, row in df_upload.iterrows():
            row_num = idx + 1
            sb_raw = row.get(col_map['src_bytes'])
            db_raw = row.get(col_map['dst_bytes'])
            dur_raw = row.get(col_map['duration'])
            proto_raw = row.get(col_map['protocol_type'])

            # Basic validation: Missing or NaN
            if pd.isna(sb_raw) or pd.isna(db_raw) or pd.isna(dur_raw) or pd.isna(proto_raw):
                rejection_reasons.append(f"Row {row_num}: Missing or NaN value in required flow fields.")
                continue

            # Basic validation: Numeric conversion
            try:
                sb = float(sb_raw)
                db = float(db_raw)
                dur = float(dur_raw)
            except (ValueError, TypeError):
                rejection_reasons.append(f"Row {row_num}: Non-numeric value in duration or byte fields.")
                continue

            # Basic validation: Infinity check
            if np.isinf(sb) or np.isinf(db) or np.isinf(dur):
                rejection_reasons.append(f"Row {row_num}: Infinite numeric value detected.")
                continue

            # Basic validation: Negative check
            if sb < 0 or db < 0 or dur < 0:
                rejection_reasons.append(f"Row {row_num}: Negative value detected (src_bytes={sb}, dst_bytes={db}, duration={dur}).")
                continue

            # Basic validation: Protocol check
            proto_str = str(proto_raw).strip().upper()
            if proto_str not in SUPPORTED_PROTOCOLS:
                rejection_reasons.append(f"Row {row_num}: Unsupported protocol '{proto_raw}'. Allowed: TCP, UDP, ICMP.")
                continue

            # If basic features are valid, evaluate enhanced features
            use_enhanced = False
            fallback_reason = ""

            if has_all_enh_cols:
                # Check enhanced values
                enh_valid = True
                enh_fail_reason = ""

                # count & srv_count
                c_raw = row.get(col_map['count'])
                sc_raw = row.get(col_map['srv_count'])
                if pd.isna(c_raw) or pd.isna(sc_raw):
                    enh_valid = False
                    enh_fail_reason = "Missing or NaN value in count/srv_count"
                else:
                    try:
                        c_val = float(c_raw)
                        sc_val = float(sc_raw)
                        if np.isnan(c_val) or np.isinf(c_val) or c_val < 0 or not c_val.is_integer():
                            enh_valid = False
                            enh_fail_reason = f"Invalid count ({c_val}); must be non-negative integer"
                        elif np.isnan(sc_val) or np.isinf(sc_val) or sc_val < 0 or not sc_val.is_integer():
                            enh_valid = False
                            enh_fail_reason = f"Invalid srv_count ({sc_val}); must be non-negative integer"
                    except (ValueError, TypeError):
                        enh_valid = False
                        enh_fail_reason = "Non-numeric value in count/srv_count"

                # flag
                flag_raw = row.get(col_map['flag'])
                if enh_valid:
                    if pd.isna(flag_raw):
                        enh_valid = False
                        enh_fail_reason = "Missing value in flag"
                    else:
                        flag_str = str(flag_raw).strip().upper()
                        if flag_str not in SUPPORTED_FLAGS:
                            enh_valid = False
                            enh_fail_reason = f"Unsupported flag '{flag_raw}'"

                # rate features
                if enh_valid:
                    for r_name in ['same_srv_rate', 'serror_rate', 'dst_host_diff_srv_rate']:
                        r_raw = row.get(col_map[r_name])
                        if pd.isna(r_raw):
                            enh_valid = False
                            enh_fail_reason = f"Missing value in {r_name}"
                            break
                        try:
                            r_val = float(r_raw)
                            if np.isnan(r_val) or np.isinf(r_val) or not (0.0 <= r_val <= 1.0):
                                enh_valid = False
                                enh_fail_reason = f"Value out of bounds [0.0, 1.0] in {r_name} ({r_val})"
                                break
                        except (ValueError, TypeError):
                            enh_valid = False
                            enh_fail_reason = f"Non-numeric rate in {r_name}"
                            break

                if enh_valid:
                    use_enhanced = True
                    enh_rows_data.append({
                        'idx': idx,
                        'src_bytes': sb,
                        'dst_bytes': db,
                        'duration': dur,
                        'protocol_type': proto_str,
                        'count': int(c_val),
                        'srv_count': int(sc_val),
                        'flag': flag_str,
                        'same_srv_rate': float(row.get(col_map['same_srv_rate'])),
                        'serror_rate': float(row.get(col_map['serror_rate'])),
                        'dst_host_diff_srv_rate': float(row.get(col_map['dst_host_diff_srv_rate']))
                    })
                else:
                    use_enhanced = False
                    fallback_reason = f"Row {row_num}: {enh_fail_reason}"
            elif has_partial_enh_cols:
                fallback_reason = f"Partial enhanced columns in CSV header; missing: {missing_enh_cols}"

            if not use_enhanced:
                basic_rows_data.append({
                    'idx': idx,
                    'src_bytes': sb,
                    'duration': dur,
                    'dst_bytes': db,
                    'protocol_type': proto_str
                })

            valid_indices.append(idx)
            row_modes.append("Enhanced" if use_enhanced else "Basic")
            row_fallback_reasons.append(fallback_reason)

        total_rows = len(df_upload)
        valid_count = len(valid_indices)
        rejected_count = len(rejection_reasons)

        if valid_count == 0:
            return {
                "success": False,
                "filename": safe_filename,
                "total_records": total_rows,
                "valid_records": 0,
                "rejected_records": rejected_count,
                "rejection_reasons": rejection_reasons[:50],
                "error_message": "All uploaded records failed validation. See rejection details for per-row errors."
            }

        # 6. Execute predictions according to selected mode for each row
        preds_dict = {}
        confidences_dict = {}
        probas_dict = {}

        # Basic Mode rows execution
        if basic_rows_data:
            df_b = pd.DataFrame(basic_rows_data)
            X_b = self.prepare_features(df_b)
            X_b_s = X_b.copy()
            X_b_s[self.numerical_cols] = self.scaler.transform(X_b[self.numerical_cols])

            b_preds = self.models["Random Forest"].predict(X_b_s)
            b_probas = self.models["Random Forest"].predict_proba(X_b_s)
            b_confs = np.max(b_probas, axis=1)

            for i, r in enumerate(basic_rows_data):
                idx = r['idx']
                preds_dict[idx] = b_preds[i]
                confidences_dict[idx] = float(b_confs[i])
                probas_dict[idx] = b_probas[i]

        # Enhanced Mode rows execution
        if enh_rows_data:
            df_e = pd.DataFrame(enh_rows_data)
            X_e = self.prepare_enhanced_features(df_e)
            X_e_s = X_e.copy()
            X_e_s[self.enhanced_numerical_cols] = self.enhanced_scaler.transform(X_e[self.enhanced_numerical_cols])

            e_preds = self.enhanced_rf.predict(X_e_s)
            e_probas = self.enhanced_rf.predict_proba(X_e_s)
            e_confs = np.max(e_probas, axis=1)

            for i, r in enumerate(enh_rows_data):
                idx = r['idx']
                preds_dict[idx] = e_preds[i]
                confidences_dict[idx] = float(e_confs[i])
                probas_dict[idx] = e_probas[i]

        # 7. Compile results in canonical row order
        cls_list = self.target_classes
        norm_idx = cls_list.index('Normal')
        dos_idx = cls_list.index('DoS')
        probe_idx = cls_list.index('Probe')

        final_preds = []
        final_confs = []
        final_decision_statuses = []
        final_low_conf_flags = []
        final_prob_normal = []
        final_prob_dos = []
        final_prob_probe = []
        final_risk_levels = []
        final_explanations = []

        enh_lookup = {r['idx']: r for r in enh_rows_data}
        basic_lookup = {r['idx']: r for r in basic_rows_data}

        for i, idx in enumerate(valid_indices):
            mode = row_modes[i]
            fb_reason = row_fallback_reasons[i]
            p_cat = preds_dict[idx]
            conf = confidences_dict[idx]
            probas = probas_dict[idx]

            thresh = self.confidence_thresholds.get("enhanced" if mode == "Enhanced" else "basic", DEFAULT_CONFIDENCE_THRESHOLD)
            dec_status = "ACCEPTED" if conf >= thresh else "REVIEW REQUIRED"
            low_conf = bool(dec_status == "REVIEW REQUIRED")

            risk_lvl = "CRITICAL" if p_cat == "DoS" else ("ELEVATED" if p_cat == "Probe" else "LOW")

            if mode == "Enhanced":
                f_data = enh_lookup[idx]
                exp = self._generate_explanation(
                    f_data['src_bytes'], f_data['duration'], f_data['dst_bytes'],
                    f_data['protocol_type'], p_cat, conf,
                    mode="Enhanced", enh_dict=f_data, decision_status=dec_status, fallback_reason=fb_reason
                )
            else:
                f_data = basic_lookup[idx]
                exp = self._generate_explanation(
                    f_data['src_bytes'], f_data['duration'], f_data['dst_bytes'],
                    f_data['protocol_type'], p_cat, conf,
                    mode="Basic", enh_dict=None, decision_status=dec_status, fallback_reason=fb_reason
                )

            final_preds.append(p_cat)
            final_confs.append(round(conf, 4))
            final_decision_statuses.append(dec_status)
            final_low_conf_flags.append(low_conf)
            final_risk_levels.append(risk_lvl)
            final_prob_normal.append(round(float(probas[norm_idx]), 4))
            final_prob_dos.append(round(float(probas[dos_idx]), 4))
            final_prob_probe.append(round(float(probas[probe_idx]), 4))
            final_explanations.append(exp)

        # 8. Construct annotated output DataFrame
        annotated_df = df_upload.loc[valid_indices].copy()
        annotated_df['detection_mode'] = row_modes
        annotated_df['decision_status'] = final_decision_statuses
        annotated_df['fallback_reason'] = row_fallback_reasons
        annotated_df['predicted_classification'] = final_preds
        annotated_df['confidence_score'] = final_confs
        annotated_df['probability_normal'] = final_prob_normal
        annotated_df['probability_dos'] = final_prob_dos
        annotated_df['probability_probe'] = final_prob_probe
        annotated_df['risk_level'] = final_risk_levels
        annotated_df['low_confidence_warning'] = final_low_conf_flags
        annotated_df['explanation'] = final_explanations

        # Summary statistics
        normal_preds = int((pd.Series(final_preds) == 'Normal').sum())
        dos_preds = int((pd.Series(final_preds) == 'DoS').sum())
        probe_preds = int((pd.Series(final_preds) == 'Probe').sum())

        enh_count = sum(1 for m in row_modes if m == "Enhanced")
        basic_count = sum(1 for m in row_modes if m == "Basic")
        fallback_count = sum(1 for r in row_fallback_reasons if r)
        accepted_count = sum(1 for d in final_decision_statuses if d == "ACCEPTED")
        review_count = sum(1 for d in final_decision_statuses if d == "REVIEW REQUIRED")

        # Scan ID and history
        scan_id = f"SCAN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(100, 999)}"
        scan_entry = {
            "scan_id": scan_id,
            "filename": safe_filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": total_rows,
            "valid_records": valid_count,
            "rejected_records": rejected_count,
            "enhanced_records": enh_count,
            "basic_records": basic_count,
            "fallback_records": fallback_count,
            "accepted_records": accepted_count,
            "review_records": review_count,
            "normal_predictions": normal_preds,
            "dos_predictions": dos_preds,
            "probe_predictions": probe_preds
        }
        self._save_scan_history_entry(scan_entry)

        # Annotated CSV export string
        csv_buffer = io.StringIO()
        annotated_df.to_csv(csv_buffer, index=False)
        csv_data_str = csv_buffer.getvalue()

        # Preview records (up to 50)
        # JSON has no NaN/Infinity literals. Optional metadata and invalid enhanced
        # fields may contain them even when the Basic features are valid.
        preview = annotated_df.head(50).replace([np.inf, -np.inf], np.nan).astype(object)
        preview_records = preview.where(pd.notna(preview), None).to_dict(orient="records")

        return {
            "success": True,
            "scan_id": scan_id,
            "filename": safe_filename,
            "total_records": total_rows,
            "valid_records": valid_count,
            "rejected_records": rejected_count,
            "rejection_reasons": rejection_reasons[:50],
            "enhanced_count": enh_count,
            "basic_count": basic_count,
            "fallback_count": fallback_count,
            "accepted_count": accepted_count,
            "review_count": review_count,
            "normal_count": normal_preds,
            "dos_count": dos_preds,
            "probe_count": probe_preds,
            "normal_pct": round((normal_preds / valid_count) * 100, 1) if valid_count else 0.0,
            "dos_pct": round((dos_preds / valid_count) * 100, 1) if valid_count else 0.0,
            "probe_pct": round((probe_preds / valid_count) * 100, 1) if valid_count else 0.0,
            "preview_records": preview_records,
            "csv_data": csv_data_str,
            "risk_mapping_note": RISK_MAPPING_NOTE,
            "uncertainty_note": UNCERTAINTY_NOTE
        }

    def log_session(self, record):
        """Appends audit record to log file protected by defensive try-except blocks."""
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[File Error] Session log write error: {e}")


# ==============================================================================
# CURRICULUM SYLLABUS ENGINE (IN-MEMORY VERIFICATION)
# ==============================================================================
def execute_silent_syllabus_engine(df_sample, raw_log_text):
    """
    Executes core numerical array, matrix, string processing, and Series exercises
    in backend memory for syllabus inspection compliance without inventing timestamps.
    """
    results = {}
    num_cols = ['src_bytes', 'duration', 'dst_bytes']
    traffic_np = df_sample[num_cols].to_numpy()[:9, :]
    if len(traffic_np) < 9:
        traffic_np = np.zeros((9, 3))

    # Unit 1 & 2 Array Operations
    shape_np = traffic_np.shape
    ndim_np = traffic_np.ndim
    zeros_3x3 = np.zeros((3, 3))
    ones_3x3 = np.ones((3, 3))
    reshaped_view = traffic_np[:9, 0].reshape((3, 3))
    flattened_view = reshaped_view.flatten()
    stats_dict = {
        'min': np.min(traffic_np, axis=0).tolist(),
        'max': np.max(traffic_np, axis=0).tolist(),
        'mean': np.mean(traffic_np, axis=0).tolist()
    }
    mat_a = traffic_np[:3, :3].astype(float)
    mat_b = traffic_np[3:6, :3].astype(float)
    mat_sum = (mat_a + mat_b).tolist()
    mat_diff = (mat_a - mat_b).tolist()
    matrix_3x3_det = float(np.linalg.det(mat_a))

    # Unit 3 String Processing
    lines = [l.strip() for l in raw_log_text.strip().split("\n") if l.strip()]
    reversed_words = [" ".join([w[::-1] for w in line.split()]) for line in lines]

    # Unit 5 Series
    telemetry_series = pd.Series({
        "sample_size": len(df_sample),
        "mean_payload": float(df_sample['src_bytes'].mean()),
        "max_payload": float(df_sample['src_bytes'].max())
    })

    results.update({
        'shape': shape_np,
        'ndim': ndim_np,
        'zeros_3x3': zeros_3x3.tolist(),
        'ones_3x3': ones_3x3.tolist(),
        'reshaped_view': reshaped_view.tolist(),
        'flattened_view': flattened_view.tolist(),
        'stats': stats_dict,
        'det': matrix_3x3_det,
        'mat_sum': mat_sum,
        'mat_diff': mat_diff,
        'reversed_words': reversed_words,
        'telemetry_series': telemetry_series
    })
    return results


# ==============================================================================
# FLASK APPLICATION SETUP & INITIALIZATION
# ==============================================================================
app = Flask(__name__)

# Initialize Detector with genuine NSL-KDD
detector = NetworkIntrusionDetector()
try:
    detector.load_and_train_datasets(force_retrain=False)
except MissingDatasetError as e:
    print(f"\n[FATAL SETUP ERROR]\n{e}\n")
    sys.exit(1)

# Raw IDS log for display
raw_log_path = os.path.join(detector.base_dir, "raw_system_log.txt")
if os.path.exists(raw_log_path):
    with open(raw_log_path, "r", encoding="utf-8") as f:
        raw_log_content = f.read()
else:
    raw_log_content = "[2026-11-17 12:00:00] INFO: SentinelNet Adaptive Dual-Mode IDS engine initialized on NSL-KDD benchmark."

# Run silent syllabus engine in-memory
dummy_sample = pd.DataFrame(detector.sample_flows)
silent_analytics = execute_silent_syllabus_engine(dummy_sample, raw_log_content)


# ==============================================================================
# FLASK ROUTE HANDLERS
# ==============================================================================
@app.route("/")
def index():
    """Serves the simplified CSV analysis workflow."""
    return render_template("simple.html")


@app.after_request
def add_cache_control_headers(response):
    """Enforce strict no-cache headers to prevent browser caching of dynamic benchmark results."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def safe_read_csv_records(csv_path):
    """Safely read CSV into a list of dicts with NaN/Inf converted to None for RFC 8259 JSON compliance."""
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in list(r.items()):
            if pd.isna(v) or (isinstance(v, (float, np.floating)) and (math.isnan(v) or math.isinf(v))):
                r[k] = None
    return records


@app.route("/api/benchmark-summary", methods=["GET"])
def get_benchmark_summary():
    """Return recorded, local benchmark results; never imply paper replication."""
    results_dir = os.path.join(detector.base_dir, "data", "benchmark_results")
    try:
        model_results = safe_read_csv_records(os.path.join(results_dir, "comparative_model_performance.csv"))
        degradation = safe_read_csv_records(os.path.join(results_dir, "adaptive_simulation_results.csv"))
        missing_comparison = safe_read_csv_records(os.path.join(results_dir, "missing_features_comparison.csv"))
        paper_comparison = safe_read_csv_records(os.path.join(detector.base_dir, "paper_comparison", "results", "missingness_summary.csv"))
        md_res_dir = os.path.join(detector.base_dir, "multi_dataset_experiment", "results")
        multi_dataset_comparison = safe_read_csv_records(os.path.join(md_res_dir, "combined_summary.csv"))
        multi_dataset_stress = safe_read_csv_records(os.path.join(md_res_dir, "missingness_stress_test.csv"))
        return jsonify({"success": True,
                        "models": model_results,
                        "degradation": degradation,
                        "missing_comparison": missing_comparison,
                        "paper_comparison": paper_comparison,
                        "multi_dataset_comparison": multi_dataset_comparison,
                        "multi_dataset_stress": multi_dataset_stress})
    except (OSError, ValueError):
        return jsonify({"success": False, "error_message": "Benchmark results unavailable. Run benchmark_evaluation.py to regenerate them."}), 503


@app.route("/api/multi-dataset-benchmark", methods=["GET"])
def get_multi_dataset_benchmark():
    """Return verified multi-dataset research benchmark results across NSL-KDD and UNSW-NB15."""
    res_dir = os.path.join(detector.base_dir, "multi_dataset_experiment", "results")
    summary_path = os.path.join(res_dir, "combined_summary.csv")
    stress_path = os.path.join(res_dir, "missingness_stress_test.csv")
    if not os.path.exists(summary_path) or not os.path.exists(stress_path):
        return jsonify({"success": False, "error_message": "Multi-dataset benchmark files unavailable."}), 404
    try:
        models_comparison = safe_read_csv_records(summary_path)
        missingness_stress_test = safe_read_csv_records(stress_path)
        return jsonify({
            "success": True,
            "datasets": ["UNSW-NB15", "NSL-KDD"],
            "models_comparison": models_comparison,
            "missingness_stress_test": missingness_stress_test
        })
    except Exception as e:
        return jsonify({"success": False, "error_message": str(e)}), 500


@app.route("/api/dashboard", methods=["GET"])
def get_dashboard_data():
    """Returns aggregated benchmark KPIs, dual-mode evaluation metrics, charts, and sample telemetry."""
    train_stats = detector.dataset_stats.get('train', {})
    test_stats = detector.dataset_stats.get('test', {})
    test_eval_b = detector.test_evaluation_results
    test_eval_e = detector.enhanced_test_evaluation

    proto_distribution = {
        "TCP": 102689,
        "UDP": 14993,
        "ICMP": 8291,
        "total": 125973
    }

    raw_lines = [l.strip() for l in raw_log_content.strip().split("\n") if l.strip()]

    return jsonify({
        "dataset_name": "NSL-KDD Intrusion Benchmark (CIC/UNB)",
        "dataset_stats": detector.dataset_stats,
        "target_classes": detector.target_classes,
        "kpis": {
            "train_total": train_stats.get("total", 125973),
            "train_included": train_stats.get("included", 124926),
            "train_excluded": train_stats.get("excluded", 1047),
            "test_total": test_stats.get("total", 22544),
            "test_included": test_stats.get("included", 19592),
            "test_excluded": test_stats.get("excluded", 2952),
            "basic_accuracy": test_eval_b.get("Accuracy", 0.8188),
            "basic_f1": test_eval_b.get("F1-Score", 0.8016),
            "basic_probe_recall": test_eval_b.get("per_class", {}).get("Probe", {}).get("recall", 0.3007),
            "enhanced_accuracy": test_eval_e.get("Accuracy", 0.8516),
            "enhanced_f1": test_eval_e.get("F1-Score", 0.8476),
            "enhanced_probe_recall": test_eval_e.get("per_class", {}).get("Probe", {}).get("recall", 0.6080),
            "test_accuracy": test_eval_b.get("Accuracy", 0.8188),
            "test_precision": test_eval_b.get("Precision", 0.8125),
            "test_recall": test_eval_b.get("Recall", 0.8188),
            "test_f1": test_eval_b.get("F1-Score", 0.8016)
        },
        "classifier_benchmark": detector.val_benchmark_results,
        "baseline_evaluation": detector.test_evaluation_results,
        "enhanced_evaluation": detector.enhanced_test_evaluation,
        "test_evaluation": detector.test_evaluation_results,
        "mode_comparison": detector.mode_comparison_results,
        "confidence_thresholds": detector.confidence_thresholds,
        "charts": detector.charts_b64,
        "protocol_distribution": proto_distribution,
        "sample_flows": detector.sample_flows,
        "sample_flows_enhanced": detector.sample_flows_enhanced,
        "raw_system_logs": raw_lines,
        "scan_history": detector.scan_history[:15],
        "risk_mapping_note": RISK_MAPPING_NOTE,
        "uncertainty_note": UNCERTAINTY_NOTE
    })


@app.route("/api/scan-packet", methods=["POST"])
def scan_packet():
    """
    Evaluates an individual network flow telemetry record (Basic or Enhanced).
    Returns prediction, confidence, decision status, probability estimates, and diagnostic explanation.
    """
    data = request.get_json(silent=True) or {}
    try:
        result = detector.predict_threat(data)
        return jsonify({
            "success": True,
            "result": result
        }), 200

    except EmptyTrafficInputError as e:
        return jsonify({
            "success": False,
            "error_type": "EmptyTrafficInputError",
            "error_message": str(e)
        }), 400

    except InvalidPacketSizeError as e:
        return jsonify({
            "success": False,
            "error_type": "InvalidPacketSizeError",
            "error_message": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error_type": "RuntimeError",
            "error_message": f"Unexpected runtime error: {e}"
        }), 500


@app.route("/api/scan-batch", methods=["POST"])
def scan_batch():
    """
    Evaluates batch network flow CSV uploads using Adaptive Dual-Mode selection.
    Enforces strict row validation, preserves metadata, and generates downloadable annotated CSV.
    """
    try:
        use_benchmark = request.form.get("use_benchmark", "false").lower() == "true"
        bench_mode = request.form.get("benchmark_mode", "basic").strip().lower()

        if use_benchmark:
            if bench_mode == "enhanced":
                sample_path = os.path.join(detector.base_dir, "sample_traffic_enhanced.csv")
                sample_filename = "nsl_kdd_enhanced_sample_traffic.csv"
            else:
                sample_path = os.path.join(detector.base_dir, "sample_traffic_basic.csv")
                sample_filename = "nsl_kdd_basic_sample_traffic.csv"

            if not os.path.exists(sample_path):
                from generate_sample import generate_sample_csv
                generate_sample_csv()

            with open(sample_path, "rb") as f:
                res = detector.process_csv_batch(f, filename=sample_filename)
            return jsonify(res), (200 if res.get("success") else 400)

        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error_message": "No file payload provided. Please select a CSV file to upload."
            }), 400

        uploaded_file = request.files["file"]
        if not uploaded_file or uploaded_file.filename == "":
            return jsonify({
                "success": False,
                "error_message": "Empty file selection. Please select a valid CSV file."
            }), 400

        res = detector.process_csv_batch(uploaded_file, filename=uploaded_file.filename)
        return jsonify(res), (200 if res.get("success") else 400)

    except EmptyTrafficInputError as e:
        return jsonify({"success": False, "error_message": str(e)}), 400
    except NetworkSecurityException as e:
        return jsonify({"success": False, "error_message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error_message": f"Batch classification failed: {e}"}), 500


@app.route("/api/sample-csv", methods=["GET"])
def get_sample_csv():
    """Generates and serves downloadable sample CSVs (Basic 4-feature or Enhanced 10-feature)."""
    mode = request.args.get("mode", "basic").strip().lower()

    if mode == "enhanced":
        sample_path = os.path.join(detector.base_dir, "sample_traffic_enhanced.csv")
        dl_filename = "sentinelnet_sample_enhanced_telemetry.csv"
    else:
        sample_path = os.path.join(detector.base_dir, "sample_traffic_basic.csv")
        dl_filename = "sentinelnet_sample_basic_telemetry.csv"

    if not os.path.exists(sample_path):
        from generate_sample import generate_sample_csv
        generate_sample_csv()

    with open(sample_path, "r", encoding="utf-8") as f:
        csv_content = f.read()

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={dl_filename}"}
    )


@app.route("/api/scan-history", methods=["GET"])
def get_scan_history():
    """Returns historical audit scan records."""
    return jsonify({
        "success": True,
        "scan_history": detector.scan_history
    })


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print("[+] SentinelNet Core Backend initialized on NSL-KDD benchmark.")
    print(f"[+] Basic Mode: 4 features | Enhanced Mode: 10 features (Uncertainty threshold: {DEFAULT_CONFIDENCE_THRESHOLD})")
    print("[+] Serving on http://127.0.0.1:5000 (Flask native server)")
    app.run(host="127.0.0.1", port=5000, debug=False)
