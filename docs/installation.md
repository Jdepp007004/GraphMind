# GraphMind V5 — Installation Guide

> **Samsung EnnovateX AX Hackathon 2026 — PS03**
> Step-by-step installation from scratch.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11.x |
| Node.js | 18.x | 20.x (LTS) |
| RAM | 4 GB | 8 GB |
| Disk space | 5 GB (+ ~2 GB for Gemma) | 10 GB |
| Git | 2.40+ | latest |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Ubuntu 22.04 |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Jdepp007004/GraphMind.git
cd GraphMind
```

Verify the clone:
```bash
ls config/ src/ docs/ dashboard/ scripts/ data/ results/
```

---

## Step 2 — Set Up the Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

Verify:
```bash
python -c "import networkx, numpy, pandas, scipy, stable_baselines3, gymnasium; print('All core deps OK')"
```

Expected output:
```
All core deps OK
```

---

## Step 3 — Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and set:

```dotenv
# Enable Gemma for production explanations (set false for pure benchmark runs)
ENABLE_GEMMA=true

# GPU/CPU selection for Gemma inference
DEVICE=cpu

# Log level
LOG_LEVEL=INFO
```

> **Benchmark-only runs** (to reproduce F1 scores without Gemma): set `ENABLE_GEMMA=false`.
> All benchmark metrics are identical regardless of this flag.

---

## Step 4 — Download the UbiqLog Dataset

The UbiqLog4UCI dataset is publicly available from the UCI Machine Learning Repository.

```bash
# Download instructions:
# Visit: https://archive.ics.uci.edu/dataset/[UBIQLOG_ID]/ubiqlog
# Download the dataset ZIP file and extract it to: data/raw/

mkdir -p data/raw
# Place the UbiqLog CSV files in data/raw/
# Expected: files named UbiqLog_*.csv, one per user
```

**Dataset reference**:
> Montanari, A., et al. "UbiqLog: a cheap, unintrusive smartphone-based diet logger." ACM Conference on Pervasive and Ubiquitous Computing, 2013.
> UCI Repository: [UBIQLOG_UCI_LINK]
> Licence: CC BY 4.0

After placing the raw files, run the preprocessing pipeline:

```bash
python scripts/ubiqlog_transition_pipeline.py
```

Expected output:
```
Loaded 35 users, 9,723,451 events.
After filtering: 31 users, 208,695 transitions.
Saved to data/processed/transitions.csv
Train / Val / Test split: 166,956 / 20,870 / 20,869
Done.
```

---

## Step 5 — (Optional) Download the Gemma Model

Gemma is only required if `ENABLE_GEMMA=true`. The benchmark runs without it.

```bash
# [GEMMA_DOWNLOAD_COMMAND]
# Requires a HuggingFace account with Gemma access granted.
# Visit https://huggingface.co/google/gemma-2b and accept the terms.

pip install huggingface_hub
huggingface-cli login  # enter your HF token

# Download to local path:
huggingface-cli download google/gemma-2b --local-dir models/gemma-2b
```

Verify:
```bash
python -c "
from config import settings
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(settings.GEMMA_LOCAL_PATH)
print('Gemma tokenizer loaded OK')
"
```

---

## Step 6 — Set Up the Dashboard

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000 in your browser
```

The dashboard reads pre-generated JSON files from `dashboard/public/data/`.
To regenerate JSON from the latest benchmark results:

```bash
cd ..  # back to project root
python scripts/generate_dashboard_data.py
```

---

## Step 7 — Run the Verification Check

Run the full system audit to confirm everything is installed correctly:

```bash
python GRAPHMIND_HARDCHECK.py
```

Expected final output:
```
============================================================
GraphMind V5 — HARDCHECK COMPLETE
============================================================
✅ Config loaded
✅ Data pipeline OK
✅ BehaviouralGraph construction OK
✅ ConfidenceScorer OK
✅ MemoryManager OK
✅ RewardV2 OK
✅ Benchmark evaluator import OK
✅ KPI extractor OK
✅ Gemma explainer import OK
------------------------------------------------------------
ALL CHECKS PASSED
============================================================
```

If any check fails, the script prints a specific error message with the file and line to investigate.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'config'`

Run all Python commands from the project root directory, not from inside subdirectories:

```bash
cd /path/to/GraphMind
python -m src.benchmarks.evaluator_v2
```

### `gymnasium` version conflict with `stable-baselines3`

```bash
pip install "gymnasium>=0.29,<0.30" "stable-baselines3>=2.2"
```

### Gemma: `OutOfMemoryError`

Set `DEVICE=cpu` in `.env`. If still OOM, use int4 quantisation:

```bash
pip install bitsandbytes
```

Then set in `.env`:
```dotenv
GEMMA_QUANTIZE=int4
```

### Dashboard: `npm install` fails

Ensure Node.js ≥ 18:
```bash
node --version  # must be v18.x or higher
```

---

*For full reproduction steps, see [docs/reproducibility.md](reproducibility.md).*
