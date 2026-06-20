#!/usr/bin/env python3
"""
scripts/run_benchmarks.py

A unified entry point for running the GraphMind V6 benchmarks.

Features:
  - Smart dataset detection and prompt to download/unzip or run on synthetic.
  - Smart cache detection and prompt to use cached models or retrain from scratch.
  - CLI flags to skip all prompts and run non-interactively.
  - tqdm integration for training and evaluation.

Non-interactive usage (real data, force retrain):
    python scripts/run_benchmarks.py --dataset ubiqlog --retrain

Non-interactive usage (synthetic, cached):
    python scripts/run_benchmarks.py --dataset synthetic --cache
"""

import sys
import os
import urllib.request
import zipfile
import time
import argparse

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import settings
from src.benchmarks.baselines_extra import set_force_retrain as set_force_retrain_extra
from src.models.v6_pipeline import set_force_retrain as set_force_retrain_v6
from src.benchmarks.evaluator_v2 import BenchmarkEvaluatorV2


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraphMind V6 -- Reproducibility & Benchmark Orchestrator"
    )
    parser.add_argument(
        "--dataset",
        choices=["ubiqlog", "synthetic"],
        default=None,
        help="Dataset to use. Skips the interactive dataset prompt.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retrain all models from scratch, ignoring any cached models.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use cached models if available (default behaviour when not retraining).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*80)
    print("   GraphMind V6 -- Reproducibility & Benchmark Orchestrator")
    print("="*80)

    ubiqlog_path = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")
    zip_path = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog+smartphone+lifelogging.zip")

    # ------------------------------------------------------------------ #
    # 1. Dataset selection                                                 #
    # ------------------------------------------------------------------ #

    # --- Non-interactive: --dataset flag was provided ---
    if args.dataset is not None:
        dataset_choice = args.dataset
        if dataset_choice == "ubiqlog" and not os.path.exists(ubiqlog_path):
            print("\n[!] --dataset ubiqlog specified but dataset not found.")
            print(f"    Expected path: {ubiqlog_path}")
            print("\n[*] Downloading real UbiqLog dataset now (non-interactive)...")
            _download_and_extract(zip_path, ubiqlog_path)
        else:
            print(f"\n[+] Dataset: {dataset_choice} (non-interactive mode)")

    # --- Interactive: no --dataset flag ---
    else:
        if not os.path.exists(ubiqlog_path):
            print("\n[!] Real UbiqLog dataset not found at datasets/ubiqlog/UbiqLog4UCI.")
            print("Would you like to run on the Synthetic dataset or download the real dataset?")
            print("  [s] Run benchmarks on Synthetic dataset (fast, no download needed)")
            print("  [d] Download & extract the real UbiqLog dataset (~61MB zip, extracts to ~1.9GB)")
            print("-"*80)
            choice = input("Enter choice [s/d] (default: s): ").strip().lower()

            if choice == 'd':
                dataset_choice = 'ubiqlog'
                _download_and_extract(zip_path, ubiqlog_path)
            else:
                dataset_choice = 'synthetic'
                print("\n[*] Selected Synthetic dataset.")
        else:
            print("\n[+] Real UbiqLog dataset detected.")
            print("Which dataset would you like to use?")
            print("  [1] Real UbiqLog dataset (default)")
            print("  [2] Synthetic dataset")
            print("-"*80)
            choice = input("Enter choice [1/2] (default: 1): ").strip()
            if choice == '2':
                dataset_choice = 'synthetic'
                print("\n[*] Selected Synthetic dataset.")
            else:
                dataset_choice = 'ubiqlog'
                print("\n[*] Selected Real UbiqLog dataset.")

    # ------------------------------------------------------------------ #
    # 2. Model cache / retrain selection                                   #
    # ------------------------------------------------------------------ #

    saved_models_dir = os.path.join(PROJECT_ROOT, "models", "saved")
    cached_files = []
    if os.path.exists(saved_models_dir):
        cached_files = [
            f for f in os.listdir(saved_models_dir)
            if f.endswith(".pt") or f.endswith(".pkl")
        ]

    # --- Non-interactive: --retrain or --cache flag provided ---
    if args.retrain:
        force_retrain = True
        print("\n[*] --retrain flag set: all models will be retrained from scratch.")
    elif args.cache:
        force_retrain = False
        print("\n[*] --cache flag set: cached models will be used where available.")

    # --- Interactive: neither flag provided ---
    else:
        if cached_files:
            print("\n" + "="*80)
            print("CACHE STATUS: Pre-trained models exist in models/saved/")
            for cf in sorted(cached_files)[:8]:
                print(f"  - {cf}")
            if len(cached_files) > 8:
                print(f"  ... and {len(cached_files) - 8} other cache files.")
            print("="*80)
            print("How would you like to handle model training?")
            print("  [c] Use cached models (extremely fast, zero retraining)")
            print("  [r] Retrain all models from scratch (overwrites cache)")
            print("-"*80)
            choice = input("Enter choice [c/r] (default: c): ").strip().lower()
            if choice == 'r':
                force_retrain = True
                print("\n[*] Retraining enabled. Cache will be rebuilt.")
            else:
                force_retrain = False
                print("\n[*] Cache enabled. Pre-trained models will be loaded.")
        else:
            print("\n" + "="*80)
            print("CACHE STATUS: No cached models found.")
            print("="*80)
            print("[*] All models will be trained from scratch and saved to models/saved/.")
            force_retrain = True

    # Apply retrain settings to all subsystems
    set_force_retrain_extra(force_retrain)
    set_force_retrain_v6(force_retrain)

    # ------------------------------------------------------------------ #
    # 3. Run evaluation                                                    #
    # ------------------------------------------------------------------ #
    print("\n" + "="*80)
    print(f"   Executing Evaluator V2 on '{dataset_choice}' dataset")
    if force_retrain:
        print("   Mode: RETRAIN ALL -- all models trained fresh, saved to models/saved/")
    else:
        print("   Mode: CACHED -- pre-trained models loaded from models/saved/")
    print("="*80)

    # Disable Gemma for benchmark neutrality
    os.environ["ENABLE_GEMMA"] = "false"
    settings.ENABLE_GEMMA = False

    t_start = time.perf_counter()
    evaluator = BenchmarkEvaluatorV2(dataset_source=dataset_choice)
    evaluator.load_dataset()
    results = evaluator.run_all()
    evaluator.write_results(results)

    elapsed = time.perf_counter() - t_start
    print("\n" + "="*80)
    print(f"   SUCCESS: Benchmarks completed in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print("="*80)
    print("Output files written:")
    print(f"  - results/benchmark_results_v2.csv")
    print(f"  - results/ablation_results_v2.csv")
    print(f"  - results/statistical_results_v2.csv")
    print(f"  - reports/kpi_summary.json")
    print(f"  - reports/YYYY-MM-DD_benchmark.md")
    print("="*80 + "\n")


def _download_and_extract(zip_path: str, ubiqlog_path: str) -> None:
    """Download and extract the UbiqLog dataset. Raises on failure (no fallback)."""
    os.makedirs(os.path.join(PROJECT_ROOT, "datasets"), exist_ok=True)
    url = "https://archive.ics.uci.edu/static/public/369/ubiqlog+smartphone+lifelogging.zip"
    print(f"\nDownloading dataset from:\n  {url}\n")

    try:
        from tqdm import tqdm

        class TqdmUpTo(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
    except ImportError:
        print("Downloading... (install tqdm for a progress bar)")
        urllib.request.urlretrieve(url, filename=zip_path)

    print("\nExtracting zip archive...")
    extract_dir = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        from tqdm import tqdm
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.infolist()
            for member in tqdm(members, desc="Extracting"):
                zip_ref.extract(member, extract_dir)
    except ImportError:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    if not os.path.exists(ubiqlog_path):
        raise RuntimeError(
            f"Extraction complete but expected path not found: {ubiqlog_path}\n"
            "Check the zip structure and adjust ubiqlog_path accordingly."
        )
    print(f"[+] Dataset ready at: {ubiqlog_path}")


if __name__ == "__main__":
    main()
