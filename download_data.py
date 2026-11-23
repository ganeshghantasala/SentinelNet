import os
import urllib.request

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nsl_kdd")
os.makedirs(TARGET_DIR, exist_ok=True)

train_path = os.path.join(TARGET_DIR, "KDDTrain+.txt")
test_path = os.path.join(TARGET_DIR, "KDDTest+.txt")

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    with open(dest, "r", encoding="utf-8") as f:
        lines = sum(1 for _ in f)
    print(f"Downloaded {dest}: {lines} lines.")
    return lines

if __name__ == "__main__":
    train_lines = download_file(TRAIN_URL, train_path)
    test_lines = download_file(TEST_URL, test_path)
    print(f"Verification complete: Train={train_lines} (expected 125973), Test={test_lines} (expected 22544)")
