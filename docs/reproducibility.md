# GraphMind V5 — Reproducibility Guide

> **Samsung EnnovateX AX Hackathon 2026 — PS03**
>
> This document provides the exact 5-step process to reproduce **F1 = 0.7745** and all 7
> PS03 KPIs from scratch, starting from raw data.
>
> The official benchmark has been run twice independently and produced identical outputs:
>
> | Run | Date | F1 |
> |-----|------|----|
> | Run 1 | 2026-06-06 09:39 | 0.7745 |
> | Run 2 | 2026-06-06 10:00 | 0.7745 |

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [5-Step Reproduction Process](#5-step-reproduction-process)
3. [Expected Output of Each Step](#expected-output-of-each-step)
4. [Chronological Split Methodology](#chronological-split-methodology)
5. [Statistical Validation Methodology](#statistical-validation-methodology)
6. [Frozen Configuration Reference](#frozen-configuration-reference)
7. [Seed and Determinism Notes](#seed-and-determinism-notes)

---

## System Requirements

| Resource | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.11.9 |
| RAM | 4 GB | 8 GB recommended |
| Disk | 3 GB | UbiqLog raw + processed |
| Time | ~10 minutes | Including data processing |

---

## 5-Step Reproduction Process

### Step 1 — Installation

```bash
git clone https://github.com/Jdepp007004/GraphMind.git
cd GraphMind
python -m venv venv && venv\Scripts\activate  # Windows
# OR: source venv/bin/activate  (macOS/Linux)
pip install -r requirements.txt
cp .env.example .env
```

**Expected**: No errors. `pip install` completes successfully.

---

### Step 2 — Data Preprocessing

Place UbiqLog raw CSV files in `data/raw/`, then:

```bash
python scripts/ubiqlog_transition_pipeline.py
```

**Expected output**:
```
Loaded 35 users, 9,723,451 events.
Filtering: removing users with < 100 transitions...
After filtering: 31 users retained (4 removed).
Extracting transitions (MAX_GAP=3600s)...
Extracted 208,695 transitions.
Chronological split: train=166,956 / val=20,870 / test=20,869
Saved: data/processed/transitions.csv
Done.
```

---

### Step 3 — Run the Official Benchmark

```bash
# Disable Gemma for a clean benchmark run (proves metric neutrality)
set ENABLE_GEMMA=false  # Windows
# export ENABLE_GEMMA=false  (macOS/Linux)

python scripts/run_phase11_e.py
```

**Expected output** (last 20 lines):
```
============================================================
GraphMind RL V5 — Official Benchmark
============================================================
Loading UbiqLog dataset...  ✓ (31 users, 208,695 transitions)
Building behaviour graphs...  ✓
Running evaluation (all policies)...

Policy               F1        Hit Rate   Latency Saved
────────────────────────────────────────────────────────
GraphMindRL_V5      0.7745    93.1%      1847ms
GraphMindRL_V5(t=0.10) 0.7733 93.3%     1849ms
RL_LatencyFocus     0.7539    90.7%      1726ms
GraphMindRL_Base    0.7424    93.6%      2002ms
Graph+Confidence    0.7369    91.8%      1724ms
Markov-2            0.7355    91.4%      1710ms
Markov-1            0.7267    92.4%      1682ms
────────────────────────────────────────────────────────
✅ PRODUCTION RESULT CONFIRMED: F1 = 0.7745
```

---

### Step 4 — Extract and Review KPIs

```bash
python -m src.benchmarks.evaluator_v2 --dataset synthetic
```

**Expected KPI output** (printed to stdout):
```
STABILITY: PASS — 0 issues

══════════════════════════════════════════════════════════════════════════════════
  KPI                                            Target    Achieved    Status
══════════════════════════════════════════════════════════════════════════════════
  Next Context Prediction Accuracy (F1)           ≥0.75      0.7745  ✅ PASS
  Cache Hit Rate (%)                              ≥85%       93.10%  ✅ PASS
  Memory Thrashing Reduction (%)                  ≥50%       100.00%  [PASS]
  App Load Time Improvement (%)                   ≥20%        42.21%  [PASS]
  App Launch Time Improvement (%)                 ≥10%        45.14%  [PASS]
  System Stability (issues)                       = 0            0   ✅ PASS
  Memory Utilisation Efficiency Improvement (%)   ≥30%         0.37%  [FAIL]
══════════════════════════════════════════════════════════════════════════════════
```

The KPI JSON is auto-saved to `reports/kpi_summary.json`.

---

### Step 5 — Run the Verification Hardcheck

```bash
python GRAPHMIND_HARDCHECK.py
```

**Expected final line**:
```
ALL CHECKS PASSED
```

If this passes, the reproduction is confirmed. The official result is:

> **F1 = 0.7745** · p = 0.0115 · Cohen's d = 0.491 · 31 users · Reproducible ✓

---

## Expected Output of Each Step

| Step | Command | Key Output |
|---|---|---|
| 1 — Install | `pip install -r requirements.txt` | No errors |
| 2 — Data | `ubiqlog_transition_pipeline.py` | `31 users, 208,695 transitions` |
| 3 — Benchmark | `run_phase11_e.py` | `F1 = 0.7745 CONFIRMED` |
| 4 — KPIs | `evaluator_v2.py` | `reports/kpi_summary.json` saved |
| 5 — Verify | `GRAPHMIND_HARDCHECK.py` | `ALL CHECKS PASSED` |

---

## Chronological Split Methodology

### Why Chronological?

A **chronological split** is the only correct evaluation methodology for time-series prediction tasks:

- Training data: the first 80% of each user's app-switch transitions (earliest events)
- Validation data: the next 10% (middle period)
- Test data: the final 10% (most recent events)

This mirrors real deployment: the model is trained on historical data and evaluated on future data it has never seen.

### The Data Leakage Problem with Random Splits

If we used a random 80/10/10 split, the training set would contain transitions from **after** the test set events. This is impossible in deployment — you cannot train on events that haven't happened yet. Random splits inflate all metrics by giving the model information it wouldn't have access to in practice.

**Example**: In random splitting, the model might observe that user A opened WhatsApp at 11pm on Day 60 (test event), and the model's training data contains WhatsApp transitions from Day 59 and Day 61. The chronological split prevents this entirely.

### Implementation

```python
# src/data/transition_extractor.py
def chronological_split(transitions, train_ratio=0.80, val_ratio=0.10):
    """
    Split transitions chronologically.

    Transitions must be sorted by timestamp (ascending).
    Returns (train, val, test) — all three are contiguous time windows.
    """
    n = len(transitions)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        transitions[:train_end],
        transitions[train_end:val_end],
        transitions[val_end:],
    )
```

The split is applied **per user** — each user's transitions are split independently. This prevents a user with many transitions from dominating the training set at the expense of their own test window.

---

## Statistical Validation Methodology

### Paired t-Test (n=31 users)

GraphMind V5 is compared against the reference policy (GraphMindRL Baseline) using a **paired t-test**:

**Rationale**: Each of the 31 users appears in both conditions (baseline evaluation and GraphMind V5 evaluation). The paired test accounts for between-user variability and tests whether V5 is better *for the same users*, not just on average across different users.

**Computation**:

```python
# For each user u:
d_u = F1_u(V5) - F1_u(Baseline)

# Paired t-statistic:
t = mean(d) / (std(d) / sqrt(n))   # n = 31

# Two-tailed p-value:
from scipy import stats
t_stat, p_value = stats.ttest_rel(v5_f1_per_user, baseline_f1_per_user)
```

**Results**:

| Statistic | Value | Interpretation |
|---|---|---|
| n (users) | 31 | Sufficient for parametric testing |
| mean(d) | +0.0321 | GraphMind V5 is better for the average user |
| std(d) | 0.0646 | Per-user improvement varies |
| t | 2.681 | Test statistic |
| p | 0.0115 | Significant at α = 0.05 ✓ |

### Effect Size: Cohen's d

```python
cohens_d = mean(d) / std(d)
# = 0.0321 / 0.0646 = 0.491
```

| Cohen's d | Magnitude |
|---|---|
| 0.2 | Small |
| **0.491** | **Medium-to-large ← GraphMind V5** |
| 0.8 | Large |

A Cohen's d of 0.491 means GraphMind V5's improvement is practically meaningful, not just statistically significant. The two-condition requirement (p < 0.05 AND d > 0.2) was enforced for every accepted hypothesis during development.

### Acceptance Criteria

GraphMind V5 adopts a **two-metric acceptance rule** for every experiment:

| Criterion | Threshold | Value for V5 | Pass? |
|---|---|---|---|
| Statistical significance | p < 0.05 | p = 0.0115 | ✓ |
| Effect size | Cohen's d > 0.2 | d = 0.491 | ✓ |
| Direction | ΔF1 > 0 | ΔF1 = +0.0321 | ✓ |

### Bootstrap Confidence Intervals

In addition to the paired t-test, bootstrap confidence intervals (B=10,000 resamplings) are computed for the mean F1 per policy:

```python
# src/benchmarks/statistics.py
def bootstrap_ci(values, n_samples=10000, alpha=0.05):
    boot_means = [np.mean(np.random.choice(values, len(values))) for _ in range(n_samples)]
    return np.percentile(boot_means, [alpha/2*100, (1-alpha/2)*100])
```

---

## Frozen Configuration Reference

The production configuration is frozen in `config/settings.py`. Do not modify these values without a new benchmark:

```python
# Confidence weights (Phase 11A grid-search validated)
PREFETCH_CONFIDENCE_W_TRANSITION = 0.50
PREFETCH_CONFIDENCE_W_RECENCY    = 0.10
PREFETCH_CONFIDENCE_W_FREQUENCY  = 0.40
PREFETCH_CONFIDENCE_W_CONTEXT    = 0.00

# Threshold (Phase 11B+E validated)
PREFETCH_CONFIDENCE_THRESHOLD = 0.16

# Cache sizes
HOT_TIER_CAPACITY  = 5
WARM_TIER_CAPACITY = 15

# Data split
DATASET_TRAIN_RATIO = 0.80
DATASET_VAL_RATIO   = 0.10
DATASET_TEST_RATIO  = 0.10

# Random seed
RANDOM_SEED = 42
```

**Freeze date**: 2026-06-06
**Official result**: F1 = 0.7745, p = 0.0115, Cohen's d = 0.491

---

## Seed and Determinism Notes

All random operations in the benchmark use `RANDOM_SEED = 42`:

| Component | Random use | Fixed by |
|---|---|---|
| Synthetic dataset generation | Shuffling, sampling | `np.random.seed(42)` |
| Bootstrap resampling | Resampling | `np.random.seed(42)` in statistics.py |
| PPO training | Weight initialisation | `torch.manual_seed(42)` |
| Latency simulation | Gaussian noise | `random.seed(42)` in policy runner |

The UbiqLog dataset is **not randomly shuffled** — transitions are processed in the order they appear in the raw CSV files (chronological order is preserved throughout).

**Result**: Running the benchmark twice with the same seed and data produces bit-for-bit identical outputs, as confirmed in two independent runs on 2026-06-06.

---

*Questions about reproducibility? See [docs/installation.md](installation.md) for setup help,*
*or open an issue on [GitHub](https://github.com/Jdepp007004/GraphMind).*
