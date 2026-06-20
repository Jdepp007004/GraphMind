# GraphMind V6 — Installation Guide

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| RAM | 4 GB minimum (8 GB recommended for Gemma) |
| Disk | ~2 GB (models + dataset) |

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/Jdepp007004/GraphMind.git
cd GraphMind
```

---

## Step 2: Create and Activate a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

---

## Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:
- `torch` — PyTorch for EmbeddingTransformerReranker
- `stable-baselines3` — PPO agent
- `gymnasium` — RL environment
- `networkx` — BehaviouralGraph
- `pandas`, `numpy` — Data processing
- `statsmodels` — ARIMA
- `prophet` — Prophet baseline
- `scipy` — Statistical testing
- `tqdm` — Progress bars

---

## Step 4: Download the UbiqLog Dataset (Optional — for full benchmark)

The UbiqLog dataset is available from UCI Machine Learning Repository under **CC BY 4.0**:

**Link:** [https://archive.ics.uci.edu/dataset/369](https://archive.ics.uci.edu/dataset/369)

Download `UbiqLog4UCI.zip` and extract to `data/ubiqlog/`:

```
data/
└── ubiqlog/
    ├── 2_F/
    ├── 5_F/
    ├── 6_M/
    ...
    └── 34_F/
```

> **If you skip this step**, the benchmark will use the built-in **synthetic dataset** automatically.

---

## Step 5: Run the Benchmark

**Zero-training mode** (uses pre-trained models from `models/saved/` — no download needed):
```bash
python scripts/run_benchmarks.py --dataset ubiqlog --cache
```

**Full retrain from scratch** (~43 minutes):
```bash
python scripts/run_benchmarks.py --dataset ubiqlog --retrain
```

**Synthetic dataset** (no data download needed, ~5 minutes):
```bash
python scripts/run_benchmarks.py --dataset synthetic --cache
```

Expected output: `7/7 KPIs PASS`

---

## Step 6: Launch the Dashboard (Optional)

```bash
cd dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

**Node.js 18+** is required for the dashboard.

---

## Optional: Enable Gemma 2B Explanations

To enable natural language explanation generation:

1. Set up a HuggingFace token (free account at [huggingface.co](https://huggingface.co))
2. Set environment variable:
   ```bash
   export HF_TOKEN=your_token_here   # Linux/macOS
   set HF_TOKEN=your_token_here      # Windows
   ```
3. Set `ENABLE_GEMMA=true` in `.env` (copy from `.env.example`)
4. First run will download Gemma 2B (~5 GB) automatically

> The Gemma layer is **optional** and has **zero effect** on benchmark KPIs.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: torch` | Run `pip install torch` separately |
| `prophet` install fails | Install `pystan` first: `pip install pystan==2.19.1.1` |
| Dashboard `npm install` fails | Ensure Node.js ≥18: `node --version` |
| `CUDA out of memory` | Add `--no-gpu` flag or reduce batch size in `config/settings.py` |
| Dataset not found | Run with `--dataset synthetic` to use built-in data |
