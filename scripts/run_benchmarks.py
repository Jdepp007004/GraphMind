"""
scripts/run_benchmarks.py

Entry point: runs BenchmarkEvaluator for all users and saves results CSV.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.benchmarks.evaluator import BenchmarkEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """Run full benchmark evaluation."""
    evaluator = BenchmarkEvaluator()
    df = evaluator.run_all()
    evaluator.print_summary_table()
    logger.info(f"Benchmark results saved. Shape: {df.shape}")
