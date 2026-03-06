"""
Download the Amazon Reviews 2023 Movies_and_TV dataset.
Usage: python data/download.py
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_URL, RAW_DATA_PATH, DATA_DIR


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r[{bar}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)
    else:
        print(f"\r  Downloaded {downloaded / 1e6:.1f} MB", end="", flush=True)


def download(url: str = DATA_URL, dest: str = RAW_DATA_PATH) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"File already exists at {dest}  ({size_mb:.1f} MB) — skipping download.")
        return dest

    print(f"Downloading dataset from:\n  {url}\nto:\n  {dest}\n")
    start = time.time()
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as exc:
        # Clean up partial file
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Download failed: {exc}") from exc

    elapsed = time.time() - start
    size_mb = os.path.getsize(dest) / 1e6
    print(f"\nDone. {size_mb:.1f} MB in {elapsed:.1f}s")
    return dest


if __name__ == "__main__":
    download()
