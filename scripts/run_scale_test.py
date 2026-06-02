"""
scripts/run_scale_test.py

GraphMind graph scalability smoke/stress test.
"""

import csv
import os
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED
from src.core.graph_engine import BehaviouralGraph


SCALE_USER_COUNTS = [10, 100, 1000, 10000]


def _event(user_id: str, idx: int) -> dict:
    return {
        "timestamp": float(idx),
        "user_id": user_id,
        "app_id": f"com.scale.app{idx % 7}",
        "category": "utility",
        "battery": 80.0 - (idx % 20),
        "time_of_day_bucket": idx % 48,
        "day": idx // 20,
        "weekend": False,
    }


def run_scale_case(user_count: int, events_per_user: int = 5) -> dict:
    EventBus.get_instance().clear_all()
    tracemalloc.start()
    start = time.perf_counter()

    total_nodes = 0
    total_edges = 0
    serialization_time = 0.0
    prediction_time = 0.0

    for user_idx in range(user_count):
        user_id = f"scale_user_{user_idx:05d}"
        graph = BehaviouralGraph(user_id)
        for event_idx in range(events_per_user):
            EventBus.get_instance().publish(TOPIC_APP_LAUNCHED, _event(user_id, event_idx))

        total_nodes += graph.node_count()
        total_edges += graph.edge_count()

        if graph.node_count():
            node_id = next(iter(graph._graph.nodes()))
            pred_start = time.perf_counter()
            graph.get_top_k_next_nodes(node_id, 3, 80.0)
            prediction_time += time.perf_counter() - pred_start

        if user_idx == 0:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "graph.pkl")
                ser_start = time.perf_counter()
                graph.save_to_disk(path)
                serialization_time = time.perf_counter() - ser_start

        EventBus.get_instance().clear_all()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start
    return {
        "user_count": user_count,
        "events_per_user": events_per_user,
        "node_count": total_nodes,
        "edge_count": total_edges,
        "memory_usage_mb": round(peak / (1024 * 1024), 3),
        "serialization_time_ms": round(serialization_time * 1000, 3),
        "prediction_time_ms": round(prediction_time * 1000, 3),
        "elapsed_seconds": round(elapsed, 3),
        "survived": True,
    }


def run_scale_test(output_path: str = None, user_counts=None,
                   events_per_user: int = 5) -> list:
    output_path = output_path or os.path.join(settings.RESULTS_DIR, "scale_test.csv")
    user_counts = user_counts or SCALE_USER_COUNTS
    rows = [run_scale_case(count, events_per_user) for count in user_counts]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    rows = run_scale_test()
    for row in rows:
        print(row)
