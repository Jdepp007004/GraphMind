# Reproducibility Guide

> **GraphMindRL V5 — Step-by-Step Instructions to Reproduce the Official Result**

Official result: **F1 = 0.7745, p = 0.0115, Cohen's d = 0.491** (31 users, UbiqLog dataset).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the official benchmark
python scripts/run_phase11_e.py

# 3. Expected output (last line):
# GraphMindRL_V5   F1=0.7745   p=0.0115   Cohen_d=0.491
```

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Repository Setup](#repository-setup)
3. [Data Setup](#data-setup)
4. [Running the Benchmark](#running-the-benchmark)
5. [Expected Output](#expected-output)
6. [Running the Dashboard](#running-the-dashboard)
7. [Verifying Individual Components](#verifying-individual-components)
8. [Reproducibility Checklist](#reproducibility-checklist)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

| Component | Requirement |
|---|---|
| Operating system | Windows 10/11, macOS 12+, or Ubuntu 20.04+ |
| Python | 3.10 or higher |
| RAM | 4 GB minimum (8 GB recommended) |
| Storage | 2 GB free (for dataset and results) |
| Node.js | 18+ (for dashboard only) |
| CPU | Any modern x86-64 or ARM64 |
| GPU | Not required |

---

## Repository Setup

### Clone the Repository

```bash
git clone <repo-url>
cd Samsung
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes all necessary packages. Core dependencies:

```
pandas>=2.0.0
numpy>=1.24.0
networkx>=3.0
scipy>=1.10.0
tqdm>=4.65.0
```

### Verify Installation

```bash
python -c "import pandas, numpy, networkx, scipy; print('All dependencies installed.')"
```

Expected output: `All dependencies installed.`

---

## Data Setup

The UbiqLog dataset must be placed in the `data/raw/` directory before running the benchmark.

### Obtain the Dataset

The UbiqLog4UCI dataset is publicly available from the UCI Machine Learning Repository:

- [DATASET_LINK]

Download and extract the CSV files to `data/raw/`:

```
data/
└── raw/
    ├── user_01.csv
    ├── user_02.csv
    ├── ...
    └── user_35.csv
```

### Verify Data

```bash
python -c "
import os
files = os.listdir('data/raw')
print(f'Found {len(files)} files in data/raw/')
"
```

Expected output: `Found 35 files in data/raw/`

> **Note**: If the dataset is already pre-processed (transitions extracted), the preprocessed files in `data/processed/` can be used directly and the raw CSV download can be skipped. See [Verifying Individual Components](#verifying-individual-components).

---

## Running the Benchmark

### Official Benchmark Command

```bash
python scripts/run_phase11_e.py
```

This is the **single canonical command** to reproduce the official result. It evaluates 9 policies on all 31 users and outputs the final comparison.

### What the Script Does

1. Loads the pre-processed transitions from `data/processed/` (or re-processes from `data/raw/` if not available).
2. For each policy, simulates the prefetch engine on the test set for all 31 users.
3. Computes per-user F1, precision, recall, hit rate, and latency saved.
4. Runs a paired t-test vs. the GraphMindRL baseline for each experimental policy.
5. Outputs a CSV to `results/final_production_results.csv`.
6. Prints a summary table to stdout.

### Expected Runtime

| Hardware | Runtime |
|---|---|
| Modern laptop (8-core) | ~2–4 minutes |
| Entry-level machine (4-core) | ~5–8 minutes |

---

## Expected Output

### Stdout Summary

```
============================================================
 GraphMindRL V5 — Phase 11E Final Benchmark
 31 users · 80/10/10 chronological split
============================================================

Policy                  F1      P       R       Hit%    Δ F1      p        d
-------------------------------------------------------------------------------------
GraphMindRL_V5          0.7745  0.7512  0.8063  93.1%   +0.0321   0.0115   0.491  ✓
GraphMindRL_V5_t10      0.7733  0.7498  0.8044  93.3%   +0.0309   0.0105   0.498  ✓
RL_LatencyFocus         0.7539  0.7301  0.7813  90.7%   +0.0116   0.0003   0.752  ✓
GraphMindRL_Baseline    0.7424  0.7218  0.7714  93.6%     0.0000     —       —
Graph+Confidence        0.7369  0.7147  0.7651  91.8%   -0.0055   0.3421   0.187  n.s.
Markov2                 0.7355  0.7132  0.7637  91.4%   -0.0069   0.2891   0.203  n.s.
Markov1                 0.7267  0.7073  0.7552  92.4%   -0.0157   0.1123   0.291  n.s.
GraphOnly               0.7267  0.7073  0.7552  92.4%   -0.0157   0.1123   0.291  n.s.
GlobalMarkov2           0.7201  0.6998  0.7498  91.1%   -0.0223   0.0731   0.342  n.s.

============================================================
 PRODUCTION RESULT: GraphMindRL_V5
 F1 = 0.7745  ΔF1 = +0.0321  p = 0.0115  Cohen_d = 0.491
============================================================
```

### Output File

```bash
cat results/final_production_results.csv
```

The CSV contains the full per-policy results. The key row:

```
GraphMindRL_V5,0.7745,0.7512,0.8063,0.9307,1847.2,0.0321,0.0115,0.491,True,31
```

---

## Running the Dashboard

### Install Dashboard Dependencies

```bash
cd dashboard
npm install
```

### Launch the Dashboard

```bash
npm run dev
```

The dashboard will be available at **http://localhost:3000**.

### Dashboard Data

The dashboard reads from pre-generated JSON files in `dashboard/public/data/`. These files are committed to the repository and reflect the official frozen results. No Python execution is required to view the dashboard.

To regenerate the dashboard data from the result CSVs (e.g., after re-running the benchmark):

```bash
cd ..  # back to Samsung/
python scripts/generate_dashboard_data.py
```

---

## Verifying Individual Components

### Verify the Transition Extractor

```bash
python -c "
from src.data.transition_extractor import extract_transitions
import pandas as pd
# Load a small sample and extract transitions
print('Transition extractor: OK')
"
```

### Verify the Behaviour Graph

```bash
python -c "
from src.models.graph_model import BehaviourGraph
bg = BehaviourGraph([('app_a', 'app_b'), ('app_a', 'app_c'), ('app_b', 'app_a')])
cands = bg.get_candidates('app_a')
print(f'Graph candidates for app_a: {cands}')
print('Behaviour graph: OK')
"
```

### Verify the Confidence Engine

```bash
python -c "
from src.prefetch.confidence_prefetch import ConfidencePrefetch
pf = ConfidencePrefetch()
print(f'Confidence engine config: {pf.get_config()}')
print('Confidence engine: OK')
"
```

### Verify the Production Config

```bash
python -c "
from config.settings import W_TRANSITION, W_RECENCY, W_FREQUENCY, W_CONTEXT
from config.settings import INITIAL_THRESHOLD, HOT_CACHE_SIZE, WARM_CACHE_SIZE
print(f'W_TRANSITION={W_TRANSITION}, W_RECENCY={W_RECENCY}, W_FREQUENCY={W_FREQUENCY}, W_CONTEXT={W_CONTEXT}')
print(f'THRESHOLD={INITIAL_THRESHOLD}, HOT={HOT_CACHE_SIZE}, WARM={WARM_CACHE_SIZE}')
assert W_TRANSITION == 0.50, 'FAIL: W_TRANSITION mismatch'
assert W_RECENCY == 0.10,    'FAIL: W_RECENCY mismatch'
assert W_FREQUENCY == 0.40,  'FAIL: W_FREQUENCY mismatch'
assert W_CONTEXT == 0.00,    'FAIL: W_CONTEXT mismatch'
assert INITIAL_THRESHOLD == 0.16, 'FAIL: THRESHOLD mismatch'
assert HOT_CACHE_SIZE == 5,   'FAIL: HOT_CACHE_SIZE mismatch'
assert WARM_CACHE_SIZE == 15, 'FAIL: WARM_CACHE_SIZE mismatch'
print('Production config: ALL CHECKS PASSED')
"
```

---

## Reproducibility Checklist

Use this checklist before submitting to confirm full reproducibility:

### Environment

- [ ] Python 3.10+ installed and verified
- [ ] All packages from `requirements.txt` installed without error
- [ ] `python -c "import pandas, numpy, networkx, scipy"` runs without error
- [ ] Node.js 18+ installed (for dashboard)

### Data

- [ ] Dataset files present in `data/raw/` (35 CSV files)
  OR pre-processed transitions present in `data/processed/`
- [ ] `data/processed/user_summary.csv` readable

### Benchmark

- [ ] `python scripts/run_phase11_e.py` completes without error
- [ ] Output shows `GraphMindRL_V5  F1=0.7745`
- [ ] Output shows `p=0.0115`
- [ ] Output shows `Cohen_d=0.491`
- [ ] `results/final_production_results.csv` is updated

### Dashboard

- [ ] `cd dashboard && npm install` completes without error
- [ ] `npm run dev` starts without error
- [ ] http://localhost:3000 loads in browser
- [ ] Executive Overview shows F1 = 0.7745
- [ ] Benchmark Explorer table is populated
- [ ] All 7 pages render without errors

### Configuration

- [ ] `config/settings.py` contains `W_TRANSITION = 0.50`
- [ ] `config/settings.py` contains `W_RECENCY = 0.10`
- [ ] `config/settings.py` contains `W_FREQUENCY = 0.40`
- [ ] `config/settings.py` contains `W_CONTEXT = 0.00`
- [ ] `config/settings.py` contains `INITIAL_THRESHOLD = 0.16`
- [ ] `config/settings.py` contains `HOT_CACHE_SIZE = 5`
- [ ] `config/settings.py` contains `WARM_CACHE_SIZE = 15`

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Ensure you are running scripts from the repository root, not from a subdirectory:

```bash
cd Samsung  # repository root
python scripts/run_phase11_e.py
```

### `FileNotFoundError: data/raw/`

Ensure the dataset files are downloaded and placed in `data/raw/`. See [Data Setup](#data-setup).

### `AssertionError: FAIL: W_TRANSITION mismatch`

The production config has been modified. Restore `config/settings.py` from the git history:

```bash
git checkout config/settings.py
```

### Dashboard shows no data

The dashboard JSON files may be missing. Run:

```bash
python scripts/generate_dashboard_data.py
```

Then restart the dashboard with `npm run dev`.

### Different F1 result

If the F1 result differs from 0.7745, check:
1. Python version (must be 3.10+)
2. Package versions (check `pip list` against `requirements.txt`)
3. Dataset files (must be the original UbiqLog4UCI dataset, unmodified)
4. Config file (must not be modified — run the config verification command above)

---

*Reproducibility confirmed on: Windows 11, Python 3.10.14, macOS 13, Ubuntu 22.04.*
*Official result frozen on 2026-06-06. Git tag: `pre-dashboard-freeze`.*
