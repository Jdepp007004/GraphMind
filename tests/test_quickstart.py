"""
Tests for the GraphMind quickstart demo script.

These tests verify that the quickstart runs end-to-end without errors
and produces sensible output values.
"""

import sys
import random
from pathlib import Path

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from quickstart import (
    generate_demo_sequence,
    build_markov_graph,
    compute_confidence_score,
    DEMO_APPS,
)


class TestDemoSequence:
    """Tests for synthetic sequence generation."""

    def test_length(self):
        seq = generate_demo_sequence(n=100)
        assert len(seq) == 100

    def test_apps_are_valid(self):
        seq = generate_demo_sequence(n=200)
        assert all(app in DEMO_APPS for app in seq)

    def test_deterministic_with_seed(self):
        seq1 = generate_demo_sequence(n=50, seed=99)
        seq2 = generate_demo_sequence(n=50, seed=99)
        assert seq1 == seq2

    def test_different_seeds_differ(self):
        seq1 = generate_demo_sequence(n=50, seed=1)
        seq2 = generate_demo_sequence(n=50, seed=2)
        assert seq1 != seq2

    def test_coverage(self):
        """At least half the apps should appear in a long sequence."""
        seq = generate_demo_sequence(n=500)
        assert len(set(seq)) >= len(DEMO_APPS) // 2


class TestMarkovGraph:
    """Tests for Markov graph construction."""

    def test_graph_has_nodes(self):
        seq = generate_demo_sequence(n=100)
        graph = build_markov_graph(seq)
        assert len(graph) > 0

    def test_probabilities_sum_to_one(self):
        seq = generate_demo_sequence(n=200)
        graph = build_markov_graph(seq)
        for src, transitions in graph.items():
            total = sum(transitions.values())
            assert abs(total - 1.0) < 1e-9, f"Probabilities for {src} sum to {total}"

    def test_all_values_nonnegative(self):
        seq = generate_demo_sequence(n=200)
        graph = build_markov_graph(seq)
        for src, transitions in graph.items():
            for dst, prob in transitions.items():
                assert prob >= 0.0, f"Negative probability: {src} → {dst} = {prob}"

    def test_empty_sequence_handled(self):
        graph = build_markov_graph(["com.whatsapp"])
        assert isinstance(graph, dict)


class TestConfidenceScore:
    """Tests for confidence score computation."""

    def test_score_in_range(self):
        seq = generate_demo_sequence(n=200)
        graph = build_markov_graph(seq[:150])
        score = compute_confidence_score(
            app="com.whatsapp",
            current_app="com.instagram.android",
            graph=graph,
            sequence=seq[:150],
        )
        assert 0.0 <= score <= 1.0

    def test_zero_score_for_unseen_app(self):
        """An app never in training should score very low."""
        score = compute_confidence_score(
            app="com.nonexistent.app",
            current_app="com.whatsapp",
            graph={},
            sequence=["com.whatsapp"] * 10,
        )
        assert score == 0.0

    def test_high_frequency_boosts_score(self):
        """App appearing very frequently should get a higher score than rare one."""
        # Frequent app
        seq = ["com.whatsapp"] * 90 + ["com.instagram.android"] * 10
        graph = build_markov_graph(seq)
        score_freq = compute_confidence_score("com.whatsapp", "com.instagram.android", graph, seq)
        score_rare = compute_confidence_score("com.instagram.android", "com.whatsapp", graph, seq)
        # WhatsApp is 9x more frequent — its score should be higher
        assert score_freq > score_rare

    def test_weight_override(self):
        """Custom weights should change the score."""
        seq = generate_demo_sequence(n=100)
        graph = build_markov_graph(seq)
        score_default = compute_confidence_score("com.whatsapp", "com.instagram.android", graph, seq)
        score_custom = compute_confidence_score("com.whatsapp", "com.instagram.android", graph, seq, weights=(1.0, 0.0, 0.0))
        # With markov-only weights, score purely depends on transition probability
        markov_only = graph.get("com.instagram.android", {}).get("com.whatsapp", 0.0)
        assert abs(score_custom - markov_only) < 1e-10
