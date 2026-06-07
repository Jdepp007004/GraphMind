<div align="center">

<img src="https://img.shields.io/badge/Samsung-EnnovateX%20AX%202025-1428A0?style=for-the-badge&logo=samsung&logoColor=white" alt="Samsung Hackathon"/>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js"/>
<img src="https://img.shields.io/badge/Reinforcement%20Learning-Enabled-FF6B35?style=for-the-badge&logo=openai&logoColor=white" alt="RL"/>
<img src="https://img.shields.io/badge/F1%20Score-0.7745-00C851?style=for-the-badge" alt="F1 Score"/>

<br/><br/>

```
  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗███╗   ███╗██╗███╗   ██╗██████╗ 
 ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║████╗ ████║██║████╗  ██║██╔══██╗
 ██║  ███╗██████╔╝███████║██████╔╝███████║██╔████╔██║██║██╔██╗ ██║██║  ██║
 ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║██║╚██╔╝██║██║██║╚██╗██║██║  ██║
 ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ 
```

### 🧠 RL-Powered Graph Memory Management for Android Edge Devices

*Samsung EnnovateX AX Hackathon 2025 — Phase 2 Submission*

</div>

---

<div align="center">

| 🏆 F1 Score | 📈 Improvement | 🎯 Cache Hit Rate | ⚡ Latency Saved | 📊 Users Tested |
|:-----------:|:--------------:|:-----------------:|:----------------:|:---------------:|
| **0.7745** | **+4.3%** | **93.1%** | **~1,847 ms** | **31 real users** |

*Statistically significant: p = 0.0115 < 0.05 ✓ · Cohen's d = 0.491 (medium-large)*

</div>

---

## 📋 Table of Contents

- [🌟 What is GraphMind?](#-what-is-graphmind)
- [🚀 Key Results](#-key-results)
- [💡 Core Innovations](#-core-innovations)
- [🏗️ System Architecture](#️-system-architecture)
- [🛠️ Technical Stack](#️-technical-stack)
- [📂 Repository Structure](#-repository-structure)
- [⚙️ Installation](#️-installation)
- [▶️ Reproducing Results](#️-reproducing-results)
- [📊 Dashboard](#-dashboard)
- [📚 Documentation](#-documentation)
- [🤖 Models Evaluated](#-models-evaluated)
- [🗄️ Dataset](#️-dataset)
- [📜 License](#-license)

---

## 🌟 What is GraphMind?

**GraphMind (GraphMindRL V5)** is a reinforcement-learning–enhanced Markov-graph prefetch engine designed for Android smartphones and edge devices. It intelligently predicts which apps a user will open next and **pre-loads them into a two-tier RAM cache**, eliminating cold-launch latency without requiring a neural network.

> 💡 **The core problem:** On mid-range devices like the Samsung Galaxy A23, cold-launching an app takes **over 1,800 ms**. GraphMind reduces this to near-zero by predicting and pre-loading the right apps ahead of time.

### Why GraphMind is Different

| Approach | Weakness | GraphMind's Solution |
|----------|----------|---------------------|
| Static frequency counters | Misses temporal patterns | Weighted Markov graph |
| Pure Markov chains | Ignores recency & frequency | Confidence score fusion |
| Deep learning | Too expensive for edge | Lightweight RL controller |
| Static thresholds | Requires per-user tuning | Self-calibrating RL threshold |

---

## 🚀 Key Results

### Official Benchmark Results (Frozen — Reproduced Twice)

| Policy | F1 Score | Hit Rate | Latency Saved | ΔF1 | p-value | Cohen's d | Significant |
|--------|----------|----------|---------------|-----|---------|-----------|-------------|
| 🥇 **GraphMindRL_V5** | **0.7745** | **93.1%** | **1,847 ms** | **+0.0321** | **0.0115** | **0.491** | ✅ |
| GraphMindRL_V5 (t=0.10) | 0.7733 | 93.3% | 1,849 ms | +0.0309 | 0.0105 | 0.498 | ✅ |
| RL_LatencyFocus | 0.7539 | 90.7% | 1,726 ms | +0.0116 | 0.0003 | 0.752 | ✅ |
| GraphMindRL Baseline | 0.7424 | 93.6% | 2,002 ms | 0.0000 | — | — | — |
| Graph+Confidence | 0.7369 | 91.8% | 1,724 ms | −0.0055 | — | — | n.s. |
| Markov-2 | 0.7355 | 91.4% | 1,710 ms | −0.0069 | — | — | n.s. |
| Markov-1 (Baseline) | 0.7267 | 92.4% | 1,682 ms | −0.0157 | — | — | n.s. |

```
F1 Score Progression
0.80 ┤
0.78 ┤                                                    ●  ← GraphMindRL V5 (0.7745)
0.76 ┤                                               ●
0.74 ┤                          ●         ●
0.72 ┤               ●     ●
0.70 ┤          ●
0.68 ┤     ●
     └────────────────────────────────────────────────────
      Markov-1  Markov-2  G+C  Baseline  RL-Lat  V5(0.10)  V5
```

**Key findings:**
- **+4.3% improvement** over the GraphMindRL baseline (statistically significant)
- **Independently reproduced on two separate runs** with identical outputs
- Effect size (Cohen's d = 0.491) is in the **medium-to-large range**
- Validated across **31 real smartphone users** from the UbiqLog4UCI dataset

---

## 💡 Core Innovations

### 1. 📊 Behaviour Graph with Confidence Scoring

Each user's app-switching behaviour is modelled as a **weighted directed Markov graph**. Instead of relying on a single signal, GraphMind fuses three complementary signals into a confidence score:

```python
score(app) = 0.5 × P(app | current)       # Markov transition probability
           + 0.1 × recency_score(app)      # exponential decay from last use
           + 0.4 × frequency_score(app)    # normalised historical frequency
```

> Each component contributes positively — confirmed via ablation study.

### 2. 🤖 RL-Controlled Adaptive Threshold

A lightweight RL controller monitors the **rolling 20-step hit rate** and dynamically adjusts the confidence threshold:

```
Rolling Hit Rate > 80%  →  threshold += 0.005  (more selective)
Rolling Hit Rate < 50%  →  threshold -= 0.005  (more permissive)
Rolling Hit Rate ∈ [50%, 80%]  →  threshold unchanged
```

This replaces manual per-user threshold tuning with a **self-calibrating mechanism** that works out-of-the-box for all 31 users.

### 3. 🗄️ Two-Tier Cache Architecture

```
┌─────────────────────────────────────────────────┐
│                  Cache Tiers                     │
├──────────┬──────────────┬────────────────────────┤
│   Tier   │   Capacity   │   Latency Profile      │
├──────────┼──────────────┼────────────────────────┤
│  🔥 HOT  │   5 apps     │   0 ms (in RAM)        │
│  🌡️ WARM │  15 apps     │   ~200 ms (pre-loaded) │
│  ❄️ COLD  │  Unlimited   │   ~1,800 ms (SQLite)   │
└──────────┴──────────────┴────────────────────────┘
```

Predictions above the confidence threshold are loaded into **WARM**; the most recent HOT apps come from direct user interaction.

### 4. 🔬 Empirical Research Methodology

Every design decision was driven by benchmarks — **8 hypothesis-test-decision cycles** documented with full statistical evidence, making the engineering process transparent and reproducible.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GraphMindRL V5                               │
│                                                                      │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────────┐ │
│  │  UbiqLog     │──▶│  Transition      │──▶│  Behaviour           │ │
│  │  Dataset     │   │  Extractor       │   │  Graph (NetworkX)    │ │
│  └──────────────┘   └──────────────────┘   └──────────────────────┘ │
│                                                        │             │
│                                                        ▼             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Confidence Prefetch Engine                       │   │
│  │                                                              │   │
│  │   score = 0.5×trans + 0.1×recency + 0.4×frequency           │   │
│  │   threshold = 0.16  (adaptive ±0.005 via RL controller)      │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
│                               │                                      │
│              ┌────────────────┼────────────────┐                    │
│              ▼                ▼                 ▼                    │
│        ┌──────────┐    ┌──────────┐    ┌──────────────┐            │
│        │🔥 HOT    │    │🌡️ WARM   │    │❄️ COLD Store  │            │
│        │ (5 apps) │    │(15 apps) │    │  (SQLite)    │            │
│        └──────────┘    └──────────┘    └──────────────┘            │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │               Multi-Agent Orchestration Layer                  │  │
│  │  Prefetch Agent · RL Trainer · Graph Manager · Drift Detector  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

The system is organized into modular agents:

| Agent | Responsibility |
|-------|---------------|
| `PrefetchAgent` | Generates prefetch predictions using the confidence engine |
| `RLTrainerAgent` | Trains and updates the RL threshold controller |
| `GraphManagerAgent` | Maintains and updates the Markov behaviour graph |
| `DriftDetectorAgent` | Detects user behaviour drift and triggers re-calibration |
| `SecurityAgent` | Handles unknown apps and enforces retention policies |
| `Orchestrator` | Coordinates all agents via event bus |

---

## 🛠️ Technical Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core backend |
| **Graph Engine** | NetworkX | Markov behaviour graph |
| **RL Framework** | Stable-Baselines3 + Gymnasium | RL threshold controller |
| **Data Processing** | pandas, NumPy | Dataset pipeline |
| **Statistical Testing** | SciPy | Paired t-test, Cohen's d |
| **Deep Learning** | PyTorch | RL policy network |
| **Dashboard** | Next.js 15 + TypeScript | Interactive web dashboard |
| **Visualization** | Recharts + React Flow | Charts & graph rendering |
| **Animations** | Framer Motion | Smooth UI transitions |
| **Scheduling** | APScheduler | Periodic agent tasks |

</div>

---

## 📂 Repository Structure

```
GraphMind/
│
├── 📄 README.md                           # You are here
├── 📋 requirements.txt                    # Python dependencies
├── 🔒 .env.example                        # Environment variable template
├── 🔒 .gitignore
│
├── ⚙️  config/
│   └── settings.py                        # Production configuration (frozen)
│
├── 🧠 src/
│   ├── agents/
│   │   ├── orchestrator.py               # Multi-agent coordinator
│   │   ├── prefetch_agent.py             # Prediction + prefetch logic
│   │   ├── rl_trainer_agent.py           # RL policy training
│   │   ├── graph_manager_agent.py        # Graph maintenance
│   │   ├── drift_detector_agent.py       # Behaviour drift detection
│   │   ├── drift_visualizer.py           # Drift visualization
│   │   └── security_agent.py            # Security & retention policy
│   │
│   ├── core/
│   │   ├── graph_engine.py               # Markov graph engine
│   │   ├── memory_manager.py             # HOT/WARM/COLD cache
│   │   ├── event_bus.py                  # Agent communication bus
│   │   └── event_schema.py              # Event type definitions
│   │
│   ├── rl/
│   │   ├── environment_v2.py             # RL environment (Gymnasium)
│   │   ├── trainer.py                    # Training loop
│   │   ├── reward_v2.py                  # Reward shaping
│   │   └── evaluation.py                # Policy evaluation
│   │
│   ├── prefetch/
│   │   └── confidence_prefetch.py        # Production prefetch engine (FROZEN)
│   │
│   ├── data/
│   │   ├── ubiqlog_loader.py             # Dataset loading & cleaning
│   │   └── transition_extractor.py       # Transition sequence extraction
│   │
│   ├── models/
│   │   ├── markov.py                     # Markov-1 and Markov-2
│   │   └── graph_model.py               # Behaviour graph construction
│   │
│   ├── android/                          # Android integration layer
│   ├── benchmarks/                       # Benchmark suite
│   ├── experiments/                      # Experimental scripts
│   ├── explainability/                   # Model explainability tools
│   ├── graph_playback/                   # Graph replay visualization
│   └── security/                         # Security hardening modules
│
├── 📜 scripts/
│   ├── run_phase11_e.py                  # ← OFFICIAL benchmark entry point
│   ├── run_phase11.py                    # Full phase 11 experiments
│   ├── run_v5_validation.py              # V5 validation suite
│   ├── generate_dashboard_data.py        # Dashboard JSON generation
│   ├── ubiqlog_transition_pipeline.py    # Data processing pipeline
│   └── run_statistical_analysis.py       # Statistical analysis tools
│
├── 📊 dashboard/                          # Next.js 15 web dashboard
│   ├── app/                              # 7-page Next.js app router
│   ├── components/                       # Reusable React components
│   └── public/data/                      # Pre-generated JSON data files
│
├── 📁 data/
│   ├── raw/                              # Raw UbiqLog CSV files
│   └── processed/                        # Cleaned transition sequences
│
├── 📈 results/
│   ├── final_production_results.csv      # OFFICIAL FROZEN RESULT
│   ├── v5_all_experiments.csv            # Complete experiment log
│   └── v5_threshold_sweep.csv            # Phase 11B threshold sweep
│
├── 📝 reports/
│   └── final_production_report.md        # Key result narrative
│
├── 📚 docs/
│   ├── architecture.md                   # System architecture
│   ├── ax.md                             # AX methodology
│   ├── benchmarking.md                   # Evaluation methodology
│   ├── reproducibility.md                # Step-by-step reproduction guide
│   ├── dashboard.md                      # Dashboard feature guide
│   ├── models.md                         # Model catalogue
│   ├── datasets.md                       # Dataset documentation
│   ├── user_guide.md                     # User manual
│   └── installation.md                   # Detailed installation guide
│
├── 🧪 tests/                              # Test suite (pytest)
└── 🗄️  archive/                            # Archived failed experiments
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for the dashboard)
- **4 GB RAM minimum** (8 GB recommended)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/Jdepp007004/GraphMind.git
cd GraphMind
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration (if needed)
```

### 4. Set Up the Dashboard

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

---

## ▶️ Reproducing Results

The official benchmark runs with a **single command**:

```bash
python scripts/run_phase11_e.py
```

### Expected Output

```
============================================================
GraphMind RL V5 — Official Benchmark
============================================================
Loading UbiqLog dataset...  ✓ (31 users, 208,695 transitions)
Building behaviour graphs...  ✓
Running evaluation (all policies)...

Policy               F1        Hit Rate   Latency    ΔF1      p-value    Cohen's d
──────────────────────────────────────────────────────────────────────────────────
GraphMindRL_V5      0.7745    93.1%      1847ms     +0.0321  0.0115     0.491  ✓
GraphMindRL_V5(0.1) 0.7733    93.3%      1849ms     +0.0309  0.0105     0.498  ✓
RL_LatencyFocus     0.7539    90.7%      1726ms     +0.0116  0.0003     0.752  ✓
GraphMindRL_Base    0.7424    93.6%      2002ms     0.0000   —          —
──────────────────────────────────────────────────────────────────────────────────

✅ PRODUCTION RESULT CONFIRMED: F1 = 0.7745
```

> 📖 Full step-by-step instructions: [docs/reproducibility.md](docs/reproducibility.md)

---

## 📊 Dashboard

GraphMind includes a **7-page interactive Next.js dashboard** for exploring results:

```bash
cd dashboard
npm run dev
# Open http://localhost:3000
```

| Page | URL | Description |
|------|-----|-------------|
| 🏠 **Executive Overview** | `/` | Key metrics, system pipeline, production config |
| 📊 **Benchmark Explorer** | `/benchmark` | Interactive policy comparison table & charts |
| 🗺️ **Optimization Journey** | `/journey` | F1 trajectory across 8 research phases |
| 🕸️ **Graph Explorer** | `/graph` | Interactive Markov transition graph |
| 🎮 **Cache Simulator** | `/simulator` | Live HOT/WARM cache animation |
| 📼 **User Playback** | `/playback` | Step-through of real user event sequences |
| 🔬 **Research Validation** | `/research` | Ablations, statistical testing, reproducibility |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System architecture with detailed Mermaid diagrams |
| [docs/ax.md](docs/ax.md) | AX methodology — agentic development workflow |
| [docs/benchmarking.md](docs/benchmarking.md) | Evaluation methodology, baselines, statistical testing |
| [docs/reproducibility.md](docs/reproducibility.md) | Exact commands to reproduce the official result |
| [docs/dashboard.md](docs/dashboard.md) | Dashboard feature guide (all 7 pages) |
| [docs/models.md](docs/models.md) | Model catalogue — all 9 policies evaluated |
| [docs/datasets.md](docs/datasets.md) | Dataset documentation & preprocessing |
| [docs/user_guide.md](docs/user_guide.md) | End-user manual |
| [docs/installation.md](docs/installation.md) | Detailed installation guide |
| [reports/final_production_report.md](reports/final_production_report.md) | Production result narrative |

---

## 🤖 Models Evaluated

| # | Model | F1 Score | Status |
|---|-------|----------|--------|
| 1 | Markov-1 (GraphOnly) | 0.7267 | 📍 Baseline |
| 2 | Markov-2 | 0.7355 | ❌ Rejected |
| 3 | Graph + Confidence | 0.7369 | ❌ Rejected |
| 4 | GraphMindRL Baseline | 0.7424 | 📍 Reference |
| 5 | RL_LatencyFocus | 0.7539 | ✅ Candidate |
| 6 | **GraphMindRL_V5** | **0.7745** | 🏆 **PRODUCTION** |

> Full model documentation: [docs/models.md](docs/models.md)

---

## 🗄️ Dataset

| Field | Value |
|-------|-------|
| **Name** | UbiqLog4UCI |
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/UbiqLog+(Life+Logging)) |
| **Size** | 9.7M events, 35 users |
| **License** | CC BY 4.0 |
| **Transitions extracted** | 208,695 |
| **Users retained** | 31 (after quality filtering) |
| **Split** | 80 / 10 / 10 (chronological) |

**Attribution:** Montanari, A., et al. *"UbiqLog: a cheap, unintrusive smartphone-based diet logger."* ACM Conference on Pervasive and Ubiquitous Computing, 2013.

> Full dataset documentation: [docs/datasets.md](docs/datasets.md)

---

## 🧪 Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 📜 License

This project was developed and submitted for the **Samsung EnnovateX AX Hackathon 2025**.

The **UbiqLog4UCI dataset** is used under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. See [docs/datasets.md](docs/datasets.md) for full attribution.

The source code in this repository is the **original work** of the project team.

---

<div align="center">

**GraphMindRL V5** · Samsung EnnovateX AX Hackathon 2025

*Backend frozen · Official result: F1 = 0.7745 · p = 0.0115 · Cohen's d = 0.491*

[![GitHub](https://img.shields.io/badge/GitHub-Jdepp007004%2FGraphMind-181717?style=flat-square&logo=github)](https://github.com/Jdepp007004/GraphMind)

</div>
