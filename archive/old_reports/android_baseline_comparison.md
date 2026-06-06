# Android Industry Baseline Comparison

## Overview

This document compares GraphMind's predictive caching approach against Android's
native memory management systems and classical baselines. The goal is to contextualise
GraphMind's benchmark results within the broader Android ecosystem.

---

## Android's Native Memory Management: LMKD

### What LMKD Is

The **Linux Memory Killer Daemon (LMKD)** is Android's primary memory pressure handler.
It runs as a system service and terminates background processes when physical RAM falls
below configurable thresholds.

**LMKD is fundamentally reactive, not predictive.**

| Property | LMKD | GraphMind |
|----------|------|-----------|
| Paradigm | Reactive kill | Predictive pre-warm |
| Trigger | Memory pressure threshold | Behavioural pattern |
| Decision basis | OOM score adj (static priority) | Graph transition probabilities |
| User context | Ignored | Central feature |
| Temporal context | Ignored | Time-of-day + weekday |
| Personalisation | None | Per-user Markov chains |

### Why LMKD Is Not a Prediction Algorithm

LMKD's job is to **free memory when the system is under pressure**. It does not:
- Predict which apps the user is about to launch
- Pre-warm apps into memory
- Learn from user history
- Use time-of-day patterns

GraphMind and LMKD solve **complementary problems**:
- LMKD handles *reactive eviction* when memory is scarce
- GraphMind handles *proactive prefetch* to minimise cold-start latency

> **Important:** We do not claim direct benchmark superiority over LMKD. These are
> different problem formulations evaluated on different metrics.

---

## Baseline Hierarchy

### Tier 1: Reactive Stateless (LMKD-style)

| Method | Description | Latency Saved |
|--------|-------------|--------------|
| LMKD (representative) | Evict lowest OOM score adj | 0 ms (no prediction) |
| LRU | Evict least-recently-used | ~1,991 ms |
| MRU | Evict most-recently-used | ~1,991 ms |

**LMKD-style behavior approximation:** In our simulation, a pure LRU cache with
HOT_SIZE=5 represents the closest analog to LMKD's recency-based priority ordering.
LRU achieves hit rate ~0.928 because the user's most recent apps are frequently
reused — but this is coincidental recency, not prediction.

### Tier 2: Frequency-Based

| Method | Description | Observed F1 |
|--------|-------------|------------|
| LFU | Keep most-frequently-used | ~0.29 |
| Frequency | Static top-k by frequency | ~0.29 |
| RecencyFrequency | α·recency + β·frequency | ~0.29 |

Frequency-based methods capture long-term usage patterns but fail to predict
*sequential context* (the fact that launching Gmail often follows Chrome).

### Tier 3: Predictive — Markov Models

| Method | Training | F1 (observed) | Key Feature |
|--------|----------|--------------|-------------|
| Markov-1 | Personal | 0.727 | Sequential transitions |
| Markov-2 | Personal | 0.730 | 2nd-order context |
| VariableOrderMarkov | Personal | TBD | Laplace + fallback |
| ContextMarkov | Personal | TBD | Time + weekday conditioning |
| ClusterMarkov | Cross-user | TBD | Personal → Cluster → Global |
| GlobalMarkov2 | Cross-user | 0.679 | Population-level patterns |

Markov models explicitly model `P(next_app | previous_apps)`, enabling **true
next-app prediction** rather than reactive cache management.

**Key insight from GlobalMarkov2 result (F1=0.679 vs personal F1=0.727):**
Cross-user patterns do not transfer well. App usage is highly personal.
This validates the per-user personalisation in GraphMind.

### Tier 4: GraphMind Stack

| Method | Key Addition | F1 (observed) |
|--------|-------------|--------------|
| GraphOnly | Graph lookup | 0.727 |
| Graph+Confidence | Recency + freq confidence | 0.741 |
| GraphMindRL (V3) | RL cache budget adaptation | 0.742 |
| RLAdaptiveEnsemble (V4) | RL predictor weighting | TBD |

---

## Predictive vs Reactive: Key Distinction

```
REACTIVE (LMKD, LRU, LFU):
  User launches app → memory pressure → system reacts
  Latency = cold start time (3,000+ ms for cold)
  No prefetch, no prediction

PREDICTIVE (GraphMind):
  User behaviour → graph predicts next apps → pre-warm in background
  Latency = warm/hot start time (274ms hot, 1,301ms warm)
  Potential savings: 2,489ms (cold→hot) to 1,462ms (cold→warm)
```

---

## Measured Latency Savings (Samsung Galaxy A23)

| Tier transition | Latency saved | Notes |
|----------------|--------------|-------|
| Cold → Hot | **~2,489 ms** | Mean cold 2,763ms, hot 274ms |
| Cold → Warm | **~1,462 ms** | Mean cold 2,763ms, warm 1,301ms |
| Warm → Hot | **~1,027 ms** | Incremental improvement |

GraphMindRL (V3) achieves average latency saved of **~2,003 ms per launch** across
31 users, measured against the baseline of all launches being cold starts.

**Practical interpretation:** Users whose frequent apps are pre-warmed experience
2+ seconds faster app launches on average — a perceptible UX improvement.

---

## Memory Efficiency Considerations

GraphMind's prefetch is bounded by HOT_SIZE=5 and WARM_SIZE=15:
- Maximum 5 apps pre-warmed in HOT tier
- Maximum 15 apps kept in WARM tier
- No unbounded memory growth (unlike naive "warm everything" approaches)

This is comparable to Android's recents list management, which keeps
the last N apps in background memory.

---

## Practical Deployment Considerations

| Factor | LMKD | LRU/LFU | Markov | GraphMind |
|--------|------|---------|--------|-----------|
| Runtime overhead | Very low | Very low | Low | Low-Medium |
| Model size | None | None | ~50 KB per user | ~200 KB per user |
| Training required | No | No | Offline | Offline |
| Online adaptation | No | Implicit | No | Yes (RL) |
| Cold-start behaviour | Reactive | Reactive | Random | Global fallback |
| Personalisation | No | Implicit | Full | Full |

**Deployment path:** GraphMind's Markov matrices and graph structures are
small enough (< 1 MB per user) to be stored on-device and updated
incrementally as usage data accumulates.

---

## Conclusion

1. **LMKD is not a competitor** — it manages memory pressure, not prediction.
   GraphMind is complementary: LMKD evicts when needed, GraphMind prefetches proactively.

2. **LRU/LFU are the fair reactive baselines.** They achieve hit rates of ~0.928
   through coincidental recency — not prediction. GraphMind matches or exceeds
   their hit rate while providing 74%+ F1 through genuine transition prediction.

3. **Personalization is essential.** GlobalMarkov2 (cross-user) scores F1=0.679 vs
   personal Markov-1 F1=0.727 — a 4.8-point gap confirming user-specific models
   are required.

4. **GraphMind's competitive advantage is in F1**, not just hit rate. High F1 means
   the system predicts *the right apps* — reducing both false prefetches (wasted
   memory) and misses (cold starts).
