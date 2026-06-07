"""
GraphMind Performance Profiler
================================
Lightweight profiling utility for measuring graph engine and prefetch engine performance.
Useful for ensuring on-device inference stays within latency budgets.

Usage:
    python -m src.benchmarks.profiler --quick
    python -m src.benchmarks.profiler --full
"""

import time
import random
import statistics
from dataclasses import dataclass
from typing import Callable


@dataclass
class ProfileResult:
    """Result of a profiling run."""
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def __str__(self) -> str:
        return (
            f"{self.name:<45} "
            f"mean={self.mean_ms:7.3f}ms  "
            f"p95={self.p95_ms:7.3f}ms  "
            f"p99={self.p99_ms:7.3f}ms  "
            f"[{self.iterations} iters]"
        )


def profile(fn: Callable, name: str, iterations: int = 100, warmup: int = 10) -> ProfileResult:
    """
    Profile a callable, returning latency statistics.

    Args:
        fn: Zero-argument callable to profile.
        name: Human-readable name for the operation.
        iterations: Number of timed iterations.
        warmup: Number of warmup iterations (not measured).

    Returns:
        ProfileResult with latency statistics.
    """
    # Warmup
    for _ in range(warmup):
        fn()

    # Measure
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    times_ms.sort()

    return ProfileResult(
        name=name,
        iterations=iterations,
        mean_ms=statistics.mean(times_ms),
        median_ms=statistics.median(times_ms),
        p95_ms=times_ms[int(0.95 * len(times_ms))],
        p99_ms=times_ms[int(0.99 * len(times_ms))],
        min_ms=min(times_ms),
        max_ms=max(times_ms),
    )


def _make_demo_graph(n_apps: int = 50, n_edges: int = 300) -> dict:
    """Create a random Markov graph for profiling."""
    apps = [f"com.app.{i}" for i in range(n_apps)]
    graph: dict[str, dict[str, float]] = {app: {} for app in apps}
    for _ in range(n_edges):
        src = random.choice(apps)
        dst = random.choice(apps)
        if src != dst:
            graph[src][dst] = random.random()
    # Normalize
    for src in graph:
        total = sum(graph[src].values())
        if total > 0:
            graph[src] = {dst: w / total for dst, w in graph[src].items()}
    return graph


def run_quick_profile() -> None:
    """Run a quick profiling suite (suitable for CI)."""
    print("=" * 75)
    print("  GraphMind Performance Profiler — Quick Mode")
    print("=" * 75)
    print()

    apps = [f"com.app.{i}" for i in range(50)]
    graph = _make_demo_graph(50, 300)
    sequence = [random.choice(apps) for _ in range(1000)]

    results = []

    # --- Graph lookup ---
    def graph_lookup():
        app = random.choice(apps)
        return graph.get(app, {})

    results.append(profile(graph_lookup, "Graph neighbor lookup (50-node graph)", 1000))

    # --- Confidence score computation ---
    def confidence_score():
        current = random.choice(apps)
        candidate = random.choice(apps)
        markov_p = graph.get(current, {}).get(candidate, 0.0)
        last_idx = next((i for i in range(len(sequence) - 1, -1, -1) if sequence[i] == candidate), None)
        recency = (0.9 ** (len(sequence) - 1 - last_idx)) if last_idx is not None else 0.0
        freq = sequence.count(candidate) / len(sequence)
        return 0.5 * markov_p + 0.1 * recency + 0.4 * freq

    results.append(profile(confidence_score, "Confidence score computation (1 app)", 1000))

    # --- Full prefetch pass (all candidates) ---
    def full_prefetch_pass():
        current = random.choice(apps)
        scores = {}
        for app in apps:
            if app == current:
                continue
            markov_p = graph.get(current, {}).get(app, 0.0)
            last_idx = next((i for i in range(len(sequence) - 1, -1, -1) if sequence[i] == app), None)
            recency = (0.9 ** (len(sequence) - 1 - last_idx)) if last_idx is not None else 0.0
            freq = sequence.count(app) / len(sequence)
            scores[app] = 0.5 * markov_p + 0.1 * recency + 0.4 * freq
        return [a for a, s in scores.items() if s >= 0.16]

    results.append(profile(full_prefetch_pass, "Full prefetch pass (49 candidates)", 200))

    print(f"{'Operation':<45} {'Mean':>10}  {'P95':>10}  {'P99':>10}  {'Iters':>8}")
    print("-" * 75)
    for r in results:
        print(r)

    print()
    print("  ✅ All operations complete.")
    print("  ℹ️  On Samsung Galaxy A23 (Snapdragon 680), multiply by ~3–5×.")
    print()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "--quick"
    if mode in ("--quick", "-q"):
        run_quick_profile()
    else:
        print("Usage: python -m src.benchmarks.profiler [--quick]")
