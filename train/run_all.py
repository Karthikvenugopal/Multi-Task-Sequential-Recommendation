"""
Convenience runner: download → preprocess → train all 3 models → export ONNX.

Usage:
    python train/run_all.py [--skip-download] [--skip-preprocess]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def run(cmd: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Running: {cmd}")
    print(f"{'='*60}\n")
    ret = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if ret.returncode != 0:
        print(f"Command failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        run("python data/download.py")

    if not args.skip_preprocess:
        run("python data/preprocess.py")

    run("python train/train_sasrec.py")
    run("python train/train_shared_bottom.py")
    run("python train/train_mmoe.py")
    run("python train/export_onnx.py")

    print("\n\nAll done! Start the serving layer with:")
    print("  uvicorn serve.app:app --host 0.0.0.0 --port 8000")
    print("\nOpen MLflow UI with:")
    print("  mlflow ui --backend-store-uri mlruns/")


if __name__ == "__main__":
    main()
