import os
import urllib.request
import sys

# Official partitioned training and testing sets of UNSW-NB15
# Reference: Moustafa & Slay (2015), UNSW Canberra Cyber / ACCS
URL_TRAIN = "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv"
URL_TEST = "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_testing-set.csv"

TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "unsw_nb15")
os.makedirs(TARGET_DIR, exist_ok=True)

train_dest = os.path.join(TARGET_DIR, "UNSW_NB15_training-set.csv")
test_dest = os.path.join(TARGET_DIR, "UNSW_NB15_testing-set.csv")

def download_file(url, dest, min_expected_lines=80000):
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000_000:
        with open(dest, "r", encoding="utf-8", errors="ignore") as f:
            lines = sum(1 for _ in f)
        print(f"[CACHE] {dest} already exists with {lines:,} lines ({os.path.getsize(dest):,} bytes). Skipping download.")
        return lines

    print(f"Downloading from {url} to {dest}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest)
    
    with open(dest, "r", encoding="utf-8", errors="ignore") as f:
        lines = sum(1 for _ in f)
    print(f"Successfully downloaded {dest}: {lines:,} lines ({os.path.getsize(dest):,} bytes).")
    if lines < min_expected_lines:
        raise RuntimeError(f"{dest} has fewer lines ({lines:,}) than expected ({min_expected_lines:,}).")
    return lines

if __name__ == "__main__":
    print("[*] Initiating UNSW-NB15 benchmark dataset acquisition...")
    train_lines = download_file(URL_TRAIN, train_dest, min_expected_lines=80000)
    test_lines = download_file(URL_TEST, test_dest, min_expected_lines=80000)
    print(f"[+] UNSW-NB15 Dataset Verification Complete:\n    Training Split: {train_lines:,} lines\n    Testing Split:  {test_lines:,} lines")
