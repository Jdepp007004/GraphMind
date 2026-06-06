# Graph Representation Study (Phase 6)

**Date:** 2026-06-06  
**Source:** `results/v5_graph_study.csv` (124 rows: 4 policies × 31 users)

---

## Results

| Policy | Node Identity | F1 | ΔF1 | Hit Rate |
|--------|-------------|-----|-----|---------|
| **Baseline (GraphMindRL)** | app + recency/freq | **0.7424** | — | 0.9357 |
| Graph_Bigram | (prev_app, app) | 0.7295 | -0.0129 | 0.9297 |
| Graph_NodeApp | app only | 0.7267 | -0.0157 | 0.9380 |
| Graph_NodeAppTime6 | (app, 6-band) | 0.7116 | -0.0308 | 0.9334 |
| Graph_NodeAppTime12 | (app, 12-band) | 0.6918 | -0.0506 | 0.9293 |

---

## Q1: Does graph representation add value over Markov-1?

**Graph_NodeApp F1 = Markov-1 F1 = 0.7267. Identical.**

The graph representation with node=app is architecturally and numerically identical to Markov-1. This confirms the audit finding at full benchmark scale.

**Graph_Bigram (F1=0.7295)** adds +0.0028 over Graph_NodeApp by capturing second-order context in the node identity. It is equivalent to M2_Naive, which also scored 0.7295.

---

## Q2: Is a time-aware graph better?

**No.** Both time-aware graph variants underperform:
- Graph_NodeAppTime6:  F1=0.7116 (−0.031 vs baseline)
- Graph_NodeAppTime12: F1=0.6918 (−0.051 vs baseline)

The pattern is identical to Phase 3: splitting the node space by time bucket fragments the graph too aggressively. Edges that were well-sampled in the `(app → app)` graph become sparse in the `(app, time) → (app, time)` graph.

---

## Q3: Does higher-order graph help?

**Marginally.** Graph_Bigram (node = (prev,app)) achieves F1=0.7295 vs Graph_NodeApp 0.7267 (+0.003). This is statistically negligible (same as M2_Naive).

---

## Summary

| Representation | Equivalent to | F1 | Adds Value? |
|---------------|-------------|-----|------------|
| Node=app | Markov-1 | 0.7267 | No |
| Node=(prev,app) | Markov-2 Naive | 0.7295 | +0.003 (negligible) |
| Node=(app,time6) | TimeAwareM1_6Band | 0.7116 | No, hurts |
| Node=(app,time12) | TimeAwareM1_12Band | 0.6918 | No, hurts |

**GraphMind's real advantage comes from the recency/frequency confidence layer on top of the graph, not from the graph topology itself.**

---
