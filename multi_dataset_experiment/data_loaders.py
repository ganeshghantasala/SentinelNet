"""
multi_dataset_experiment/data_loaders.py

Dataset Loaders & Strict Preprocessing for Multi-Dataset Benchmarking:
- NSL-KDD (CIC/UNB Benchmark)
- UNSW-NB15 (UNSW Canberra Cyber / ACCS Benchmark)

Methodological Guardrails:
1. All scalers and categorical encoders are fitted STRICTLY on training splits only (Zero Data Leakage).
2. Unseen test categories are handled defensively without synthetic interpolation.
3. Targets are standardized to binary classification: 0 = Normal, 1 = Attack.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

NSL_COL_NAMES = [
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

NSL_BASIC_COLS = ['duration', 'protocol_type', 'src_bytes', 'dst_bytes']
NSL_ENHANCED_COLS = [
    'duration', 'protocol_type', 'src_bytes', 'dst_bytes',
    'count', 'srv_count', 'flag', 'same_srv_rate', 'serror_rate', 'dst_host_diff_srv_rate'
]

UNSW_BASIC_COLS = ['dur', 'proto', 'service', 'state', 'sbytes', 'dbytes']


def load_nsl_kdd_binary(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    train_path = os.path.join(base_dir, "data", "nsl_kdd", "KDDTrain+.txt")
    test_path = os.path.join(base_dir, "data", "nsl_kdd", "KDDTest+.txt")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"NSL-KDD benchmark files not found in {os.path.join(base_dir, 'data', 'nsl_kdd')}")

    df_train = pd.read_csv(train_path, header=None, names=NSL_COL_NAMES)
    df_test = pd.read_csv(test_path, header=None, names=NSL_COL_NAMES)

    # Binary targets: 0 = Normal, 1 = Attack
    y_train = (df_train['label'].astype(str).str.strip().str.rstrip('.').str.lower() != 'normal').astype(int).values
    y_test = (df_test['label'].astype(str).str.strip().str.rstrip('.').str.lower() != 'normal').astype(int).values

    X_train_raw = df_train.drop(columns=['label', 'difficulty'])
    X_test_raw = df_test.drop(columns=['label', 'difficulty'])

    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat = encoder.fit_transform(X_train_raw[cat_cols])
    X_test_cat = encoder.transform(X_test_raw[cat_cols])

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_raw[num_cols])
    X_test_num = scaler.transform(X_test_raw[num_cols])

    X_train_full = np.hstack([X_train_num, X_train_cat])
    X_test_full = np.hstack([X_test_num, X_test_cat])
    full_feature_names = np.array(num_cols + cat_cols)

    # Enhanced subset indices (10 features)
    enh_indices = [list(full_feature_names).index(c) for c in NSL_ENHANCED_COLS]
    X_train_enh = X_train_full[:, enh_indices]
    X_test_enh = X_test_full[:, enh_indices]

    # Basic subset indices (4 features)
    basic_indices = [list(full_feature_names).index(c) for c in NSL_BASIC_COLS]
    X_train_basic = X_train_full[:, basic_indices]
    X_test_basic = X_test_full[:, basic_indices]

    stats = {
        "dataset": "NSL-KDD",
        "train_samples": int(len(y_train)),
        "train_normal": int(np.sum(y_train == 0)),
        "train_attack": int(np.sum(y_train == 1)),
        "test_samples": int(len(y_test)),
        "test_normal": int(np.sum(y_test == 0)),
        "test_attack": int(np.sum(y_test == 1)),
        "full_features": len(full_feature_names),
        "enhanced_features": len(enh_indices),
        "basic_features": len(basic_indices)
    }

    return {
        "dataset_name": "NSL-KDD",
        "X_train_full": X_train_full,
        "X_test_full": X_test_full,
        "X_train_enh": X_train_enh,
        "X_test_enh": X_test_enh,
        "X_train_basic": X_train_basic,
        "X_test_basic": X_test_basic,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": full_feature_names,
        "enh_indices": enh_indices,
        "basic_indices": basic_indices,
        "stats": stats
    }


def load_unsw_nb15_binary(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_path = os.path.join(base_dir, "data", "unsw_nb15", "UNSW_NB15_training-set.csv")
    test_path = os.path.join(base_dir, "data", "unsw_nb15", "UNSW_NB15_testing-set.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"UNSW-NB15 benchmark files not found in {os.path.join(base_dir, 'data', 'unsw_nb15')}. Run download_unsw_data.py first.")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y_train = df_train['label'].astype(int).values
    y_test = df_test['label'].astype(int).values

    drop_cols = ['id', 'attack_cat', 'label']
    X_train_raw = df_train.drop(columns=drop_cols)
    X_test_raw = df_test.drop(columns=drop_cols)

    cat_cols = ['proto', 'service', 'state']
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat = encoder.fit_transform(X_train_raw[cat_cols])
    X_test_cat = encoder.transform(X_test_raw[cat_cols])

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_raw[num_cols])
    X_test_num = scaler.transform(X_test_raw[num_cols])

    X_train_full = np.hstack([X_train_num, X_train_cat])
    X_test_full = np.hstack([X_test_num, X_test_cat])
    full_feature_names = np.array(num_cols + cat_cols)

    # Basic subset indices (6 core flow features)
    basic_indices = [list(full_feature_names).index(c) for c in UNSW_BASIC_COLS]
    X_train_basic = X_train_full[:, basic_indices]
    X_test_basic = X_test_full[:, basic_indices]

    stats = {
        "dataset": "UNSW-NB15",
        "train_samples": int(len(y_train)),
        "train_normal": int(np.sum(y_train == 0)),
        "train_attack": int(np.sum(y_train == 1)),
        "test_samples": int(len(y_test)),
        "test_normal": int(np.sum(y_test == 0)),
        "test_attack": int(np.sum(y_test == 1)),
        "full_features": len(full_feature_names),
        "enhanced_features": len(full_feature_names),
        "basic_features": len(basic_indices)
    }

    return {
        "dataset_name": "UNSW-NB15",
        "X_train_full": X_train_full,
        "X_test_full": X_test_full,
        "X_train_enh": X_train_full,
        "X_test_enh": X_test_full,
        "X_train_basic": X_train_basic,
        "X_test_basic": X_test_basic,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": full_feature_names,
        "enh_indices": list(range(len(full_feature_names))),
        "basic_indices": basic_indices,
        "stats": stats
    }
