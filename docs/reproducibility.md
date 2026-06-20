# GraphMind V6 — Reproducibility Guide

Reproducing the **7/7 KPI PASS** result requires only two steps on any machine where the repo is cloned.

---

## TL;DR — One Command (18 min)

```bash
python scripts/run_benchmarks.py --dataset ubiqlog --cache
```

Expected final output:
```
==================================================================================
  KPI                                               Target     Achieved   Status
==================================================================================
  Next Context Prediction Accuracy (F1)             >=0.75       0.9792  [PASS]
  Cache Hit Rate (%)                                 >=85%       97.92%  [PASS]
  Memory Thrashing Reduction (%)                     >=50%      100.00%  [PASS]
  App Load Time Improvement (%)                      >=20%       72.18%  [PASS]
  App Launch Time Improvement (%)                    >=10%       82.20%  [PASS]
  System Stability (issues)                            = 0            0  [PASS]
  Memory Utilisation Efficiency Improvement (%)      >=30%       96.91%  [PASS]
==================================================================================
  Overall: 7/7 KPIs PASS
==================================================================================
```

---

## Step-by-Step

### Step 1: Clone and Install

```bash
git clone https://github.com/Jdepp007004/GraphMind.git
cd GraphMind
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Get the Dataset

**Option A — Dataset already in repo (if `datasets/ubiqlog/UbiqLog4UCI/` exists):**  
Skip this step.

**Option B — Auto-download via CLI:**
```bash
python scripts/run_benchmarks.py
# When prompted: choose [d] to download UbiqLog
```

**Option C — Manual download:**
1. Go to: <!-- [PLACEHOLDER: UBIQLOG_UCI_LINK] https://archive.ics.uci.edu/dataset/508/ubiqlog -->
2. Download the dataset zip
3. Unzip into `datasets/ubiqlog/UbiqLog4UCI/`

### Step 3: Run Benchmarks

```bash
# Option A: Use cached pre-trained models (fastest — 18 min)
python scripts/run_benchmarks.py --dataset ubiqlog --cache

# Option B: Retrain all models from scratch (~43 min)
python scripts/run_benchmarks.py --dataset ubiqlog --retrain

# Option C: Synthetic dataset (no download, ~5 min)
python scripts/run_benchmarks.py --dataset synthetic --cache

# Option D: Interactive mode (prompts for all choices)
python scripts/run_benchmarks.py
```

---

## What Gets Saved

After a successful run:

| File | Contents |
|------|----------|
| `results/benchmark_results_v2.csv` | 14-policy comparison (cache hit rate, F1, latency, thrash) |
| `results/ablation_results_v2.csv` | Ablation study results |
| `results/statistical_results_v2.csv` | p-values, Cohen's d |
| `reports/kpi_summary.json` | PS03 KPI summary |
| `results/reports/YYYY-MM-DD_benchmark.md` | Human-readable benchmark report |
| `models/saved/` | All trained models (per-user + ARIMA/LSTM/Prophet) |

---

## Pre-trained Models (Zero Training)

The repository includes **all 31 per-user V6 Transformer rerankers** plus ARIMA/LSTM/Prophet models in `models/saved/`. When you clone the repo and run `--cache`, no training occurs — results are reproduced in 18 minutes.

| Model file pattern | Description |
|---|---|
| `v6_reranker_ubiqlog_{user_id}.pt` | Per-user EmbeddingTransformerReranker (31 files) |
| `v6_reranker_ubiqlog_{user_id}_meta.pkl` | Vocabulary and metadata (31 files) |
| `arima_ubiqlog.pkl` | ARIMA model for ARIMA baseline |
| `lstm_ubiqlog.pt` + `lstm_ubiqlog.pkl` | LSTM baseline |
| `prophet_ubiqlog.pkl` | Prophet baseline |

---

<!-- [PLACEHOLDER: KPI_SS] Screenshot of the terminal output showing "7/7 KPIs PASS" — add here as ![KPI Output](docs/screenshots/kpi_output.png) -->

---

## Verification

Run the hardcheck script to verify all benchmarks are within tolerance:

```bash
python GRAPHMIND_HARDCHECK.py
```

Expected: `ALL CHECKS PASSED`
