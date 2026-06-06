"""
src/experiments/
================
Experimental model variants that were rigorously tested and conclusively
ruled out in favour of the GraphMindRL_V5 confidence-scoring approach.

These files are retained deliberately to demonstrate the research process:
    Idea → Experiment → Failure → Evidence → Final Decision

Models
------
cluster_markov.py
    Cluster-augmented Markov model. Apps grouped into semantic clusters
    (social, entertainment, productivity) as super-nodes. The intent was
    to smooth sparse transitions by sharing statistics across similar apps.
    Result: F1 ≈ Markov-1 (no significant gain). Cluster boundaries in
    UbiqLog are not sharp enough to add signal over raw app-level M1.

context_markov.py
    Time-conditioned Markov model. Maintains separate transition matrices
    per time-of-day band (6-band, 12-band, 24-hour, 48-bucket).
    Intent: capture "morning apps" vs "evening apps" patterns.
    Result: Phase 11C audit showed 94–98% coverage (states ARE seen),
    but conditional distributions add noise rather than signal on 2-month
    datasets. Requires ≥12 months of data for reliable time conditioning.

variable_order_markov.py
    Variable-Order Markov model. Adapts between M1 and M2 per state
    based on bigram confidence. Falls back to M1 when bigram is unseen
    or has fewer than K observations.
    Result: Equivalent to Modified Kneser-Ney (Phase 11D). All variants
    (K=3/5/10) achieved F1 ≈ 0.727–0.728, significantly below the
    confidence-layer approach (F1=0.7745).

Evidence Files
--------------
results/v5_modified_kn.csv         - Phase D: ModKN results
results/v5_all_experiments.csv     - All Phase 3–8 experiments
reports/v5_architecture_verification.md
reports/time_context_coverage_audit.md
reports/v5_optimization_summary.md
"""
