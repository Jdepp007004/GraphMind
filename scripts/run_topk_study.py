"""
scripts/run_topk_study.py

Performs a study of GraphMind's prefetch precision, recall, and F1 score 
across different Top-K recommendation sizes (Top-1, Top-3, Top-5, Top-10, Top-15).
Satisfies Priority 1 of the benchmark system.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.benchmarks.graphmind_policy_runner import GraphMindPolicyRunner

def run_study(user_id: str = "user_00"):
    print(f"=============================================================")
    print(f"             Top-K Prefetch Performance Study ({user_id})")
    print(f"=============================================================")
    print(f"{'Top-K':<8} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 50)
    
    # Load events
    path = os.path.join(settings.USERS_DIR, f"{user_id}.json")
    if not os.path.exists(path):
        print(f"Error: Dataset for {user_id} not found at {path}. Run generation first.")
        return
        
    with open(path) as f:
        events = json.load(f)[:300]  # Evaluated on first 300 events for fast study
        
    for k in [1, 3, 5, 10, 15]:
        runner = GraphMindPolicyRunner(user_id, top_k=k)
        res = runner.run(events)
        p = res["prefetch_precision"]
        r = res["prefetch_recall"]
        f1 = res["prefetch_f1"]
        print(f"Top-{k:<4} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
    print(f"=============================================================")

if __name__ == "__main__":
    run_study()
