# GraphMind V6 — Reproducibility Guide

> All benchmark results can be reproduced in **three commands**.  
> Pre-trained models are committed to the repository — **no retraining required** for reproduction.

---

## Fastest Path (≈ 18 minutes, no download)

```bash
# 1. Clone and set up
git clone https://github.com/Jdepp007004/GraphMind.git
cd GraphMind
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# 2. Run benchmark using cached models
python scripts/run_benchmarks.py --dataset ubiqlog --cache
```

> Uses the 31 pre-trained V6 Transformer rerankers from `models/saved/`. No retraining.

**Expected output:**
```
══════════════════════════════════════════════
  GraphMind V6 — KPI Summary (UbiqLog, 31 users)
══════════════════════════════════════════════
  Cache Hit Rate:              97.92%   ✅ PASS (target ≥ 85%)
  Next Context Prediction:     97.92%   ✅ PASS (target ≥ 75%)
  Thrashing Reduction vs LRU:  100.00%  ✅ PASS (target ≥ 50%)
  App Load Time Improvement:   72.18%   ✅ PASS (target ≥ 20%)
  App Launch Time Improvement: 82.20%   ✅ PASS (target ≥ 10%)
  System Stability:            0 issues ✅ PASS (target: 0)
  Memory Utilisation Efficiency: 96.91% ✅ PASS (target ≥ 30%)
──────────────────────────────────────────────
  7/7 KPIs PASS
══════════════════════════════════════════════
```

Results are saved to `reports/kpi_summary.json` and `results/benchmark_results_v2.csv`.

---

## Full Retrain from Scratch (≈ 43 minutes)

```bash
python scripts/run_benchmarks.py --dataset ubiqlog --retrain
```

This will:
1. Load UbiqLog dataset from `data/ubiqlog/`
2. Train ARIMA, LSTM, Prophet baselines
3. Train 31 per-user EmbeddingTransformerRerankers
4. Run the full 14-policy evaluation
5. Compute and display KPIs

> Requires UbiqLog dataset downloaded to `data/ubiqlog/`. See [installation.md](installation.md).

---

## Synthetic Dataset (No Download Required, ≈ 5 minutes)

```bash
python scripts/run_benchmarks.py --dataset synthetic --cache
```

Uses the 10-user synthetic dataset. Results will differ numerically from UbiqLog results but all 7 KPIs will pass.

---

## Interactive Mode

```bash
python scripts/run_benchmarks.py
```

Prompts for dataset choice and cache/retrain preference.

---

## Dataset Download (for UbiqLog benchmark)

1. Visit [https://archive.ics.uci.edu/dataset/369](https://archive.ics.uci.edu/dataset/369)
2. Download `UbiqLog4UCI.zip`
3. Extract to `data/ubiqlog/` such that:
   ```
   data/ubiqlog/2_F/...csv files...
   data/ubiqlog/5_F/...csv files...
   ...
   ```

---

## Verifying Results

After benchmark completes:

```bash
# View KPI summary
cat reports/kpi_summary.json

# View full benchmark CSV
python -c "import pandas as pd; df=pd.read_csv('results/benchmark_results_v2.csv'); print(df[['policy','cache_hit_rate','f1']].sort_values('cache_hit_rate', ascending=False).head(5))"
```

---

## Environment

Tested on:
- Python 3.10, 3.11, 3.12
- Windows 11, Ubuntu 22.04
- CPU-only (no GPU required)
- 8 GB RAM (4 GB minimum)
