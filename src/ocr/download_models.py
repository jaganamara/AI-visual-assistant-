"""
Resilient model downloader for EasyOCR.
Uses requests with streaming chunks and retries to prevent WinError 10054 connection resets.
"""

import os
import zipfile
import logging
import requests

from src.utils.device_manager import get_cache_dir

logger = logging.getLogger(__name__)

MODELS = {
    "craft_mlt_25k.pth": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "zip_name": "craft_mlt_25k.zip",
    },
    "english_g2.pth": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
        "zip_name": "english_g2.zip",
    },
}


def download_file_with_resume(url: str, dest_path: str, max_retries: int = 5) -> None:
    """Streams a remote file to disk with retries and browser user-agent."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    temp_path = dest_path + ".tmp"
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {os.path.basename(dest_path)} (attempt {attempt}/{max_retries})...")
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(temp_path, dest_path)
            print(f"Successfully downloaded: {dest_path}")
            return
        except Exception as e:
            print(f"Download attempt {attempt} failed: {e}")
            if attempt == max_retries:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise


def ensure_easyocr_models():
    """Ensures craft_mlt_25k.pth and english_g2.pth are downloaded and unpacked in cache."""
    cache_dir = get_cache_dir("easyocr")
    os.makedirs(cache_dir, exist_ok=True)

    for target_pth, info in MODELS.items():
        pth_path = os.path.join(cache_dir, target_pth)
        if os.path.exists(pth_path) and os.path.getsize(pth_path) > 1000:
            print(f"Model already present: {target_pth}")
            continue

        zip_path = os.path.join(cache_dir, info["zip_name"])
        download_file_with_resume(info["url"], zip_path)

        # Unzip
        print(f"Extracting {info['zip_name']}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(cache_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"Ready: {pth_path}")


if __name__ == "__main__":
    ensure_easyocr_models()
