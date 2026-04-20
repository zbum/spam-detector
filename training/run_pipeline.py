"""End-to-end pipeline: sample data -> train -> export ONNX -> verify.

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --skip-data   # keep existing CSVs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(step: str, cmd: list[str]) -> None:
    print(f"\n========== [{step}] {' '.join(cmd)} ==========", flush=True)
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        print(f"[error] step '{step}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-data", action="store_true",
                        help="do not regenerate sample CSVs")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_data:
        run("make-sample-data", [py, "make_sample_data.py"])
    else:
        print("[info] skipping sample data generation")

    run("train", [py, "train.py", "--config", args.config])
    run("export-onnx", [py, "export_onnx.py", "--config", args.config])
    run("verify-onnx", [py, "verify_onnx.py", "--config", args.config])

    print("\n[done] pipeline completed successfully")


if __name__ == "__main__":
    main()
