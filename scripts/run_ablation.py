"""
GraphMind Confidence Score Ablation Study
==========================================
Runs ablation experiments to quantify the contribution of each component
(Markov probability, recency, frequency) to the overall F1 score.

This script reproduces the ablation study referenced in the paper.
"""

import random
from typing import Generator
from dataclasses import dataclass, field


WEIGHTS_CONFIGS = {
    "Full (0.5+0.1+0.4)": (0.5, 0.1, 0.4),
    "No recency (0.5+0.0+0.5)": (0.5, 0.0, 0.5),
    "No frequency (0.5+0.5+0.0)": (0.5, 0.5, 0.0),
    "No Markov (0.0+0.3+0.7)": (0.0, 0.3, 0.7),
    "Equal (0.33+0.33+0.33)": (0.33, 0.33, 0.33),
    "Markov only (1.0+0.0+0.0)": (1.0, 0.0, 0.0),
    "Frequency only (0.0+0.0+1.0)": (0.0, 0.0, 1.0),
}


@dataclass
class AblationResult:
    """Result of an ablation configuration."""
    config_name: str
    weights: tuple[float, float, float]
    hits: int = 0
    total: int = 0
    predictions_made: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.predictions_made, 1)

    @property
    def precision(self) -> float:
        return self.hits / max(self.predictions_made, 1)

    @property
    def recall(self) -> float:
        return self.hits / max(self.total, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def _generate_sequence(n: int = 500, seed: int = 0) -> list[str]:
    """Generate a synthetic app sequence with realistic transitions."""
    random.seed(seed)
    apps = [f"app_{i}" for i in range(20)]
    # Bias towards a few popular apps
    weights = [10, 8, 6, 5, 4] + [1] * 15
    seq = []
    current = random.choices(apps, weights=weights)[0]
    for _ in range(n):
        seq.append(current)
        # Prefer apps adjacent in the sorted list (simulate session locality)
        idx = apps.index(current)
        candidates = apps[max(0, idx-2):idx] + apps[idx+1:min(len(apps), idx+4)]
        if candidates and random.random() < 0.6:
            current = random.choices(candidates)[0]
        else:
            current = random.choices(apps, weights=weights)[0]
    return seq


def _build_graph(sequence: list[str]) -> dict[str, dict[str, float]]:
    from collections import defaultdict
    counts: dict = defaultdict(lambda: defaultdict(int))
    for i in range(len(sequence) - 1):
        counts[sequence[i]][sequence[i + 1]] += 1
    graph: dict = {}
    for src, dsts in counts.items():
        total = sum(dsts.values())
        graph[src] = {dst: cnt / total for dst, cnt in dsts.items()}
    return graph


def _score(app: str, current: str, graph: dict, history: list[str], weights: tuple) -> float:
    wm, wr, wf = weights
    markov_p = graph.get(current, {}).get(app, 0.0)
    last = next((i for i in range(len(history) - 1, -1, -1) if history[i] == app), None)
    recency = (0.9 ** (len(history) - 1 - last)) if last is not None else 0.0
    freq_val = history.count(app) / max(len(history), 1)
    max_f = max((history.count(a) / max(len(history), 1) for a in set(history)), default=1.0)
    frequency = freq_val / max_f if max_f > 0 else 0.0
    return wm * markov_p + wr * recency + wf * frequency


def run_ablation(threshold: float = 0.16, n_users: int = 10) -> list[AblationResult]:
    """Run ablation study across multiple synthetic users."""
    results = {name: AblationResult(name, weights) for name, weights in WEIGHTS_CONFIGS.items()}

    for user_id in range(n_users):
        seq = _generate_sequence(n=500, seed=user_id * 7 + 13)
        train, test = seq[:400], seq[400:]
        graph = _build_graph(train)
        apps = list(set(seq))

        for name, weights in WEIGHTS_CONFIGS.items():
            r = results[name]
            for i in range(len(test) - 1):
                current = test[i]
                actual = test[i + 1]
                r.total += 1

                scores = {
                    app: _score(app, current, graph, train + test[:i], weights)
                    for app in apps if app != current
                }
                predictions = [app for app, s in scores.items() if s >= threshold]
                if predictions:
                    r.predictions_made += 1
                    if actual in predictions:
                        r.hits += 1

    return list(results.values())


def main():
    print("=" * 65)
    print("  GraphMind V5 — Confidence Score Ablation Study")
    print("=" * 65)
    print()
    print("  Running ablation across 10 synthetic users (500 events each)...")
    print()

    results = run_ablation(threshold=0.16, n_users=10)
    results.sort(key=lambda r: r.f1, reverse=True)

    print(f"  {'Configuration':<35} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Hit Rate':>10}")
    print("  " + "-" * 65)
    for r in results:
        marker = " ← PRODUCTION" if "0.5+0.1+0.4" in r.config_name else ""
        print(
            f"  {r.config_name:<35} {r.f1:8.4f} {r.precision:10.4f} {r.recall:8.4f} "
            f"{r.hit_rate:10.1%}{marker}"
        )

    print()
    best = results[0]
    print(f"  ✅ Best configuration: '{best.config_name}' (F1={best.f1:.4f})")
    print()


if __name__ == "__main__":
    main()
