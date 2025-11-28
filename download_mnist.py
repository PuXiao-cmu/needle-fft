"""Download MNIST dataset."""

import urllib.request
import os

# Create data directory
os.makedirs("./data", exist_ok=True)

# MNIST URLs (using a mirror that works)
base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

print("Downloading MNIST dataset...")

for filename in files:
    url = base_url + filename
    filepath = os.path.join("./data", filename)

    if os.path.exists(filepath):
        print(f"✓ {filename} already exists")
        continue

    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print(f"  ✓ Downloaded {filename} ({size/1024/1024:.2f} MB)")
    except Exception as e:
        print(f"  ✗ Failed to download {filename}: {e}")

print("\nDone!")
