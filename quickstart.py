#!/usr/bin/env python3
"""
GraphMind Quickstart Script
============================
Runs a minimal demonstration of GraphMindRL V5 on a small synthetic dataset.
This is a self-contained demo that does NOT require the full UbiqLog dataset.

Usage:
    python quickstart.py

Expected runtime: ~30 seconds
"""

import sys
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ── Synthetic demo data ──────────────────────────────────────────────────────

DEMO_APPS = [
    "com.whatsapp",
    "com.instagram.android",
    "com.google.android.youtube",
    "com.google.android.gm",
    "com.spotify.music",
    "com.android.chrome",
    "com.samsung.android.camera",
    "com.twitter.android",
    "com.netflix.mediaclient",
    "com.google.android.maps",
]

def generate_demo_sequence(n: int = 200, seed: int = 42) -> list[str]:
    """Generate a realistic synthetic app usage sequence."""
    random.seed(seed)
    
    # Define transition probabilities (simulate real usage patterns)
    transitions = {
        "com.whatsapp": {"com.instagram.android": 0.3, "com.google.android.youtube": 0.2, "com.android.chrome": 0.3, "com.whatsapp": 0.2},
        "com.instagram.android": {"com.google.android.youtube": 0.4, "com.whatsapp": 0.3, "com.android.chrome": 0.2, "com.twitter.android": 0.1},
        "com.google.android.youtube": {"com.spotify.music": 0.3, "com.whatsapp": 0.3, "com.instagram.android": 0.2, "com.netflix.mediaclient": 0.2},
        "com.google.android.gm": {"com.android.chrome": 0.5, "com.whatsapp": 0.3, "com.google.android.maps": 0.2},
        "com.spotify.music": {"com.whatsapp": 0.4, "com.instagram.android": 0.3, "com.google.android.youtube": 0.3},
        "com.android.chrome": {"com.google.android.gm": 0.3, "com.whatsapp": 0.3, "com.google.android.youtube": 0.2, "com.google.android.maps": 0.2},
        "com.samsung.android.camera": {"com.instagram.android": 0.5, "com.whatsapp": 0.3, "com.google.android.gm": 0.2},
        "com.twitter.android": {"com.android.chrome": 0.4, "com.whatsapp": 0.3, "com.instagram.android": 0.3},
        "com.netflix.mediaclient": {"com.whatsapp": 0.4, "com.spotify.music": 0.3, "com.instagram.android": 0.3},
        "com.google.android.maps": {"com.android.chrome": 0.4, "com.whatsapp": 0.3, "com.google.android.gm": 0.3},
    }
    
    sequence = [random.choice(DEMO_APPS)]
    for _ in range(n - 1):
        current = sequence[-1]
        next_probs = transitions.get(current, {})
        if next_probs:
            apps = list(next_probs.keys())
            probs = list(next_probs.values())
            sequence.append(random.choices(apps, weights=probs, k=1)[0])
        else:
            sequence.append(random.choice(DEMO_APPS))
    
    return sequence


def build_markov_graph(sequence: list[str]) -> dict:
    """Build a weighted Markov transition graph from a sequence."""
    from collections import defaultdict
    
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(len(sequence) - 1):
        counts[sequence[i]][sequence[i + 1]] += 1
    
    # Normalize to probabilities
    graph = {}
    for app, transitions in counts.items():
        total = sum(transitions.values())
        graph[app] = {next_app: count / total for next_app, count in transitions.items()}
    
    return graph


def compute_confidence_score(
    app: str,
    current_app: str,
    graph: dict,
    sequence: list[str],
    weights: tuple = (0.5, 0.1, 0.4),
) -> float:
    """
    Compute confidence score for an app given the current context.
    
    Score = 0.5 × markov_prob + 0.1 × recency_score + 0.4 × frequency_score
    """
    w_markov, w_recency, w_frequency = weights
    
    # Markov transition probability
    markov_prob = graph.get(current_app, {}).get(app, 0.0)
    
    # Recency score (exponential decay from last use)
    last_idx = None
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] == app:
            last_idx = i
            break
    if last_idx is not None:
        steps_ago = len(sequence) - 1 - last_idx
        recency_score = 0.9 ** steps_ago
    else:
        recency_score = 0.0
    
    # Frequency score (normalized historical frequency)
    freq = sequence.count(app) / len(sequence)
    max_freq = max(sequence.count(a) / len(sequence) for a in set(sequence))
    frequency_score = freq / max_freq if max_freq > 0 else 0.0
    
    return w_markov * markov_prob + w_recency * recency_score + w_frequency * frequency_score


def run_demo():
    """Run the GraphMind V5 quickstart demo."""
    print("=" * 60)
    print("  GraphMindRL V5 — Quickstart Demo")
    print("  Samsung EnnovateX AX Hackathon 2025")
    print("=" * 60)
    print()
    
    # Generate synthetic data
    print("📊 Generating synthetic app usage sequence (200 events)...")
    sequence = generate_demo_sequence(n=200)
    print(f"   Apps: {len(set(sequence))} unique apps, {len(sequence)} total events")
    print()
    
    # Build Markov graph
    print("🕸️  Building Markov behaviour graph...")
    graph = build_markov_graph(sequence[:160])  # Train on 80%
    print(f"   Nodes: {len(graph)}  |  Avg out-degree: {sum(len(v) for v in graph.values()) / max(len(graph), 1):.1f}")
    print()
    
    # Evaluate prefetch accuracy
    print("🤖 Evaluating GraphMindRL V5 prefetch engine...")
    threshold = 0.16
    test_sequence = sequence[160:]  # Test on last 20%
    
    hits = 0
    predictions_made = 0
    rl_adjustments = 0
    rolling_hits = []
    
    for i in range(len(test_sequence) - 1):
        current = test_sequence[i]
        actual_next = test_sequence[i + 1]
        
        # Compute scores for all candidate apps
        scores = {
            app: compute_confidence_score(app, current, graph, sequence[:160 + i])
            for app in DEMO_APPS
            if app != current
        }
        
        # Apply threshold
        predictions = [app for app, score in scores.items() if score >= threshold]
        
        if predictions:
            predictions_made += 1
            hit = actual_next in predictions
            if hit:
                hits += 1
            rolling_hits.append(1 if hit else 0)
            
            # RL threshold adaptation
            if len(rolling_hits) >= 20:
                recent_hit_rate = sum(rolling_hits[-20:]) / 20
                if recent_hit_rate > 0.80:
                    threshold = min(threshold + 0.005, 0.50)
                    rl_adjustments += 1
                elif recent_hit_rate < 0.50:
                    threshold = max(threshold - 0.005, 0.05)
                    rl_adjustments += 1
    
    # Results
    hit_rate = hits / max(predictions_made, 1)
    
    print()
    print("=" * 60)
    print("  📈 Demo Results")
    print("=" * 60)
    print(f"  Test events:        {len(test_sequence) - 1}")
    print(f"  Predictions made:   {predictions_made}")
    print(f"  Cache hits:         {hits}")
    print(f"  Hit rate:           {hit_rate:.1%}")
    print(f"  RL adjustments:     {rl_adjustments}")
    print(f"  Final threshold:    {threshold:.4f}")
    print()
    
    print("  🔍 Sample predictions (last 5):")
    current = test_sequence[-2]
    scores = {
        app: compute_confidence_score(app, current, graph, sequence)
        for app in DEMO_APPS
        if app != current
    }
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    for rank, (app, score) in enumerate(top3, 1):
        bar = "█" * int(score * 30)
        status = "✅ PREFETCH" if score >= threshold else "  skip"
        print(f"  {rank}. {app:<40} {score:.4f} {bar} {status}")
    
    print()
    print("  ℹ️  For full benchmark results on 31 real users:")
    print("     python scripts/run_phase11_e.py")
    print()
    print("  ℹ️  For interactive dashboard:")
    print("     cd dashboard && npm run dev")
    print()
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
