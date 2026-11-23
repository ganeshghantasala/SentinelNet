import os
import random
import shutil
import pandas as pd
from datetime import datetime, timedelta

COL_NAMES = [
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

DOS_ATTACKS = {'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'apache2', 'processtable', 'udpstorm', 'worm'}
PROBE_ATTACKS = {'ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint'}

def generate_sample_csv():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(base_dir, 'data', 'nsl_kdd', 'KDDTest+.txt')
    df_raw = pd.read_csv(test_path, header=None, names=COL_NAMES)
    
    clean_labels = df_raw['label'].astype(str).str.strip().str.rstrip('.').str.lower()
    
    normal_idx = df_raw[clean_labels == 'normal'].index.tolist()
    dos_idx = df_raw[clean_labels.isin(DOS_ATTACKS)].index.tolist()
    probe_idx = df_raw[clean_labels.isin(PROBE_ATTACKS)].index.tolist()
    
    random.seed(42)
    selected_normal = random.sample(normal_idx, 12)
    selected_dos = random.sample(dos_idx, 8)
    selected_probe = random.sample(probe_idx, 5)
    
    all_indices = selected_normal + selected_dos + selected_probe
    random.shuffle(all_indices)
    
    # 1. Generate Basic Sample CSV (4 basic flow features)
    sample_basic = df_raw.loc[all_indices, ['duration', 'protocol_type', 'src_bytes', 'dst_bytes']].copy()
    
    base_time = datetime(2026, 11, 17, 10, 0, 0)
    timestamps = [(base_time + timedelta(seconds=i*45)).strftime('%Y-%m-%d %H:%M:%S') for i in range(len(all_indices))]
    src_ips = [f"192.168.1.{random.randint(10, 150)}" for _ in range(len(all_indices))]
    dst_ips = [f"10.0.0.{random.randint(1, 20)}" for _ in range(len(all_indices))]
    src_ports = [random.randint(1024, 65530) for _ in range(len(all_indices))]
    dst_ports = [random.choice([80, 443, 22, 53, 8080, 21]) for _ in range(len(all_indices))]
    
    ground_truths = []
    for idx in all_indices:
        l = clean_labels.loc[idx]
        if l == 'normal': ground_truths.append('Normal')
        elif l in DOS_ATTACKS: ground_truths.append('DoS')
        else: ground_truths.append('Probe')
        
    sample_basic['Timestamp'] = timestamps
    sample_basic['src_ip'] = src_ips
    sample_basic['dst_ip'] = dst_ips
    sample_basic['src_port'] = src_ports
    sample_basic['dst_port'] = dst_ports
    sample_basic['attack_category'] = ground_truths
    
    basic_cols = ['Timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol_type', 'duration', 'src_bytes', 'dst_bytes', 'attack_category']
    sample_basic = sample_basic[basic_cols]
    
    out_path_basic = os.path.join(base_dir, 'sample_traffic.csv')
    sample_basic.to_csv(out_path_basic, index=False)
    
    out_path_basic_alias = os.path.join(base_dir, 'sample_traffic_basic.csv')
    sample_basic.to_csv(out_path_basic_alias, index=False)
    print(f"[+] Basic Sample CSV generated: {out_path_basic} with {len(sample_basic)} rows.")
    
    # 2. Generate Enhanced Sample CSV (10 NSL-KDD features)
    enh_features = [
        'duration', 'protocol_type', 'src_bytes', 'dst_bytes',
        'count', 'srv_count', 'flag', 'same_srv_rate',
        'serror_rate', 'dst_host_diff_srv_rate'
    ]
    sample_enh = df_raw.loc[all_indices, enh_features].copy()
    sample_enh['Timestamp'] = timestamps
    sample_enh['src_ip'] = src_ips
    sample_enh['dst_ip'] = dst_ips
    sample_enh['src_port'] = src_ports
    sample_enh['dst_port'] = dst_ports
    sample_enh['attack_category'] = ground_truths
    
    enh_cols = ['Timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port'] + enh_features + ['attack_category']
    sample_enh = sample_enh[enh_cols]
    
    out_path_enh = os.path.join(base_dir, 'sample_traffic_enhanced.csv')
    sample_enh.to_csv(out_path_enh, index=False)
    print(f"[+] Enhanced Sample CSV generated: {out_path_enh} with {len(sample_enh)} rows.")

if __name__ == '__main__':
    generate_sample_csv()
