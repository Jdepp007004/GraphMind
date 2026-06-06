"""
scripts/generate_100_users.py

Generates the extended 100 persona user dataset for GraphMind by cloning and mutating
the 10 base user personas, satisfying Priority 3 of the benchmark system.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_generator import DatasetGenerator

if __name__ == "__main__":
    print("Generating 100-user persona dataset in data/synthetic/users/...")
    DatasetGenerator().generate_100_users()
    print("Generation complete!")
