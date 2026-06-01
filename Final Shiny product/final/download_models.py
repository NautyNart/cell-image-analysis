"""
Run this script once before launching the Shiny app to download model weights.
Usage:
    cd "Final Shiny product/final"
    python3 download_models.py
"""

import os
import gdown

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_FILES = {
    "100px_best_model.pth":    "https://drive.google.com/file/d/1OjqHAYzM00FMlSVYjVJUX8jXjXd3GdHh/view",
    "50px_best_model.pth":     "https://drive.google.com/file/d/1anAAM535rmfz4x2-2aT8ByqK0EIZxKDw/view",
    "masked_best_model.pth":   "https://drive.google.com/file/d/18GHcVPmvCPIpV4NpBuqb7aW2_FRTTOQP/view",
}

def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODEL_FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"✅ {filename} already exists, skipping.")
            continue
        print(f"⬇️  Downloading {filename}...")
        gdown.download(url, dest, quiet=False, fuzzy=True)
        print(f"✅ {filename} downloaded.")

if __name__ == "__main__":
    download_models()
