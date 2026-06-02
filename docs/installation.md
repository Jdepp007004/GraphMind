# GraphMind — Installation Guide

## Requirements

- Python 3.11 or 3.12
- 8 GB RAM recommended
- 20 GB disk (Gemma 2B model, if used)
- CUDA GPU optional (CPU training supported)
- Windows 10/11 or Ubuntu 20.04+

## Quick Start (CPU-only, no GPU required)

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/graphmind.git
cd graphmind
```

### 2. Create Virtual Environment

```bash
# Python 3.12 recommended
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install networkx==3.3 numpy pandas scipy gymnasium stable-baselines3 shimmy python-dotenv

# PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# LangGraph and LangChain
pip install langgraph==0.1.14

# Dashboard and visualization
pip install streamlit pyvis plotly apscheduler

# Testing
pip install pytest pytest-cov
```

### 4. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env if needed
# DEVICE=cpu             # or cuda:0 for GPU
# LOG_LEVEL=INFO
# WANDB_API_KEY=         # optional, for W&B logging
```

### 5. Generate Dataset

```bash
python scripts/generate_dataset.py
```

This creates `data/synthetic/users/user_00.json` through `user_09.json`.
Takes ~2 minutes. Idempotent — safe to re-run.

### 6. Train RL Policies

```bash
# Train all 10 users (takes ~30 min on CPU)
python scripts/train_rl.py --all --timesteps 200000

# Quick test: train just user_00 with fewer steps
python scripts/train_rl.py --user user_00 --timesteps 50000
```

### 7. Run Simulations

```bash
# Run simulation for all users
for user in user_00 user_01 user_02 user_03 user_04 user_05 user_06 user_07 user_08 user_09; do
    python scripts/run_simulation.py --user $user
done

# Windows PowerShell
foreach ($i in 0..9) {
    $user = "user_{0:D2}" -f $i
    python scripts/run_simulation.py --user $user
}
```

### 8. Run Benchmarks

```bash
python scripts/run_benchmarks.py
```

Results saved to `results/benchmark_results.csv`.

### 9. Launch Dashboard

```bash
streamlit run src/dashboard/app.py
```

Access at http://localhost:8501

### 10. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific phase
pytest tests/test_phase1_graph.py -v
pytest tests/test_phase2_memory.py -v
pytest tests/test_phase3_rl.py -v
pytest tests/test_phase4_agents.py -v
pytest tests/test_phase5_benchmarks.py -v
```

### 11. Run Hardcheck

```bash
python GRAPHMIND_HARDCHECK.py --phase 1
python GRAPHMIND_HARDCHECK.py --phase 2
python GRAPHMIND_HARDCHECK.py --phase 3
python GRAPHMIND_HARDCHECK.py --phase 4
python GRAPHMIND_HARDCHECK.py --phase 5
python GRAPHMIND_HARDCHECK.py --phase 6
python GRAPHMIND_HARDCHECK.py --verbose
```

## Optional: Gemma 2B Setup

If you have a HuggingFace token with Gemma access:

```bash
export HF_TOKEN=your_token_here

# Install transformers
pip install transformers huggingface_hub

# Download model
python scripts/download_models.py
```

Without Gemma, GraphMind uses rule-based dataset generation and graph-based node prioritization. All hardchecks pass without Gemma.

## Troubleshooting

**ImportError: No module named 'gymnasium'**
```bash
pip install gymnasium shimmy
```

**ModuleNotFoundError: stable_baselines3**
```bash
pip install stable-baselines3
```

**FileNotFoundError: data/synthetic/users/user_00.json**
```bash
python scripts/generate_dataset.py
```

**PPO model not found**
```bash
python scripts/train_rl.py --user user_00 --timesteps 50000
```
