"""
scripts/generate_dataset.py

Entry point: generates the 10-user synthetic behavioural dataset.
Run once before any other phase.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_generator import DatasetGenerator

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    """Generate all 10 user datasets."""
    gen = DatasetGenerator()
    gen.generate_all_users()
    print("Dataset generation complete.")
