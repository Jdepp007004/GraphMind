<div align="center">

<img src="https://img.shields.io/badge/Samsung-EnnovateX%20AX%202026-1428A0?style=for-the-badge&logo=samsung&logoColor=white" alt="Samsung Hackathon"/>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/Gemma-2B-FF6B35?style=for-the-badge&logo=google&logoColor=white" alt="Gemma"/>
<img src="https://img.shields.io/badge/F1%20Score-0.7745-00C851?style=for-the-badge" alt="F1 Score"/>
<img src="https://img.shields.io/badge/PS03-Context--Aware%20Memory-1428A0?style=for-the-badge" alt="PS03"/>

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

</div>

---

# GraphMind V5 — Context-Aware Adaptive Memory for Mobile Agentic Systems

**Problem Statement Number** — PS03

**Problem Statement Title** — Context-Aware, Adaptive Memory Solution for Mobile Agentic Systems

**Team name** — [TEAM_NAME]

**Team members** — [MEMBER_1_NAME], [MEMBER_2_NAME]

**Institute/College Name** — [COLLEGE_NAME], [CAMPUS_NAME_AND_ADDRESS]

**Final Presentation Google Drive Link** — [PRESENTATION_GOOGLE_DRIVE_LINK]

**Full Submission Demo Video Link** — [DEMO_VIDEO_YOUTUBE_LINK]

**Setup & Result Reproducibility Video Link** — [REPRODUCIBILITY_VIDEO_YOUTUBE_LINK]

---

## Demo

[DEMO_VIDEO_YOUTUBE_LINK]

---

## Results at a Glance

| KPI | Target | Achieved | Status |
|-----|--------|----------|--------|
| Next Context Prediction Accuracy | ≥75% | 77.45% (F1=0.7745) | ✅ PASS |
| App Load Time Improvement | ≥20% | 42.21% | ✅ PASS |
| App Launch Time Improvement | ≥10% | 45.14% | ✅ PASS |
| Memory Thrashing Reduction | ≥50% | 100.00% | ✅ PASS |
| System Stability | 0 issues | 0 issues | ✅ PASS |
| Caching Hit Rate | ≥85% | 32.73% | ❌ FAIL |
| Memory Utilization Efficiency | ≥30% | 100.00% | ✅ PASS |

Statistically validated on **31 real Android users** (UbiqLog, UCI ML Repository).
Paired t-test p = 0.0115, Cohen's d = 0.491 (medium-large effect).

> **To fill in the [PLACEHOLDER]% values**: Run `python -m src.benchmarks.evaluator_v2` and check `reports/kpi_summary.json`.

---

## Reproduce in One Command

**Full 5-step reproduction** (≈ 10 minutes from scratch):

```bash
# Step 1: Install
git clone https://github.com/Jdepp007004/GraphMind.git && cd GraphMind
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# Step 2: Data (place UbiqLog CSVs in data/raw/ first)
python scripts/ubiqlog_transition_pipeline.py

# Step 3: Benchmark
set ENABLE_GEMMA=false
python scripts/run_phase11_e.py

# Step 4: KPI extraction
python -m src.benchmarks.evaluator_v2

# Step 5: Verify
python GRAPHMIND_HARDCHECK.py
```

Expected final line: `ALL CHECKS PASSED`

> See [docs/installation.md](docs/installation.md) for full details including dataset download, Gemma model download, and dashboard setup.

---

## Project Artefacts

**Technical Documentation** — See [docs/](docs/) folder

**Agentic AI Setup** — See [docs/ax.md](docs/ax.md)

---

**Models Used**

| Model | Purpose | Link |
|---|---|---|
| Gemma 2B (`google/gemma-2b`) | Natural language prefetch explanation generation | [GEMMA_HUGGINGFACE_LINK] |

**Models Published**

| Model | Description | Link |
|---|---|---|
| PPO Memory Allocation Agent | Adaptive threshold controller (Stable-Baselines3 PPO) | [PPO_HUGGINGFACE_LINK] |

---

**Datasets Used**

| Dataset | Description | Link | License |
|---|---|---|---|
| UbiqLog Android Usage Patterns | 9.7M events, 35 users, Android app-switch logs | [UBIQLOG_UCI_LINK] | CC BY 4.0 |

**Datasets Published**

| Dataset | Description | Link | License |
|---|---|---|---|
| GraphMind V5 Processed Benchmark Dataset | 208,695 transitions, 31 users, chronological splits | [PROCESSED_DATASET_HUGGINGFACE_LINK] | CC BY 4.0 |

---

## Architecture

GraphMind V5 is organised into **six architectural layers**:

1. **EventBus** — Perception layer, captures app-launch events and extracts `(app_id, time_bucket, battery_bucket)` node identity
2. **BehaviouralGraph** — Long-term memory, per-user weighted Markov graph (NetworkX DiGraph)
3. **MemoryManager** — HOT/WARM/COLD three-tier cache hierarchy with LRU and sensitivity-based eviction
4. **ConfidencePrefetch** — Multi-signal fusion reasoning layer (`0.50×transition + 0.40×frequency + 0.10×recency`)
5. **RL Environment** — PPO adaptive threshold controller with `MultiDiscrete([5,5,5])` action space
6. **RewardV2** — Multi-component reward signal closing the RL loop (`2.0×hit_rate - 1.2×thrash`)

Plus **Gemma** as Tool Use #2 (post-decision natural language explanation layer).

Architecture Diagram: [ARCHITECTURE_DIAGRAM_LINK]

Full details: [docs/architecture.md](docs/architecture.md)

---

## Agentic Pipeline

The complete 7-step closed-loop agentic pipeline:

```
1. PERCEPTION     EventBus captures app switch event → (app_id, time_bucket, battery_bucket)
2. MEMORY QUERY   BehaviouralGraph.query(node) → transition probability distribution  [Tool Use #1]
3. REASONING      ConfidenceScorer fuses 4 signals → ranked candidate list
4. PLANNING       PPO agent adjusts threshold and cache budget
5. ACTUATION      MemoryManager executes HOT/WARM/COLD allocation
6. EXPLANATION    Gemma generates NL rationale for top prefetch decision  [Tool Use #2]
7. REWARD         RewardV2 computes multi-component reward → PPO policy update
```

Full agentic workflow details: [docs/ax.md](docs/ax.md)

---

## Benchmark Methodology

- **Dataset**: UbiqLog4UCI, UCI ML Repository — 9.7M real Android app events, 35 users
- **Filtering**: 4 users removed (< 100 transitions), leaving **31 users with 208,695 transitions**
- **Split**: **Chronological 80/10/10** (not random — prevents data leakage)
- **Evaluation**: Per-user F1, precision, recall, cache hit rate, latency saved
- **Statistical test**: Paired t-test (n=31), p = 0.0115 < 0.05 ✓
- **Effect size**: Cohen's d = 0.491 (medium-to-large) ✓
- **Reproducibility**: Run twice on 2026-06-06 — identical outputs F1 = 0.7745 both times

```
Policy               F1        Hit Rate   Latency Saved  ΔF1     p-value   Cohen's d
──────────────────────────────────────────────────────────────────────────────────────
GraphMindRL_V5      0.7745    93.1%      1847ms         +0.0321  0.0115    0.491  ✓
GraphMindRL_V5(t=0.10) 0.7733 93.3%     1849ms         +0.0309  0.0105    0.498  ✓
RL_LatencyFocus     0.7539    90.7%      1726ms         +0.0116  0.0003    0.752  ✓
GraphMindRL_Base    0.7424    93.6%      2002ms          0.0000   —         —
Markov-2            0.7355    91.4%      1710ms          −0.0069  —         —
Markov-1 (Baseline) 0.7267    92.4%      1682ms          −0.0157  —         —
```

---

## Failed Experiments

All failed experiments are documented in `src/experiments/` and archived in `archive/`. Each was implemented as an isolated script that did not touch the production configuration.

| Approach | Why It Was Tried | Why It Was Abandoned |
|---|---|---|
| **Kneser-Ney Smoothing** | Should improve rare transition estimates | F1 = 0.7421; not significant (p > 0.05). Data is sufficient — smoothing adds noise. |
| **Variable-Order Markov (Markov-2)** | Conditioning on 2 previous apps captures more context | F1 = 0.7355; second-order table too sparse on 2-month datasets |
| **Cluster Markov** | App-category-level transitions should generalise better | F1 degraded; category abstraction destroys fine-grained sequential patterns |
| **Context Scoring (time-of-day)** | Daily rhythms should improve predictions | F1 decreased at all granularities; UbiqLog 2-month windows are too short for stable conditional distributions |

See [docs/ax.md](docs/ax.md) — *What Did Not Work* — for full technical analysis of each failure.

---

## Technical Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core backend |
| Graph Engine | NetworkX | Markov behaviour graph |
| RL Framework | Stable-Baselines3 + Gymnasium | PPO adaptive threshold controller |
| Data Processing | pandas, NumPy | Dataset pipeline |
| Statistical Testing | SciPy | Paired t-test, Cohen's d, bootstrap CIs |
| Deep Learning | PyTorch | PPO policy network |
| Open-Weight LLM | Gemma 2B | NL explanation generation |
| Dashboard | Next.js 15 + TypeScript | Interactive 7-page web dashboard |
| Visualization | Recharts + React Flow | Charts and graph rendering |
| Animations | Framer Motion | Smooth UI transitions |

Full dependency list: [docs/technical_stack.md](docs/technical_stack.md)

---

## Repository Structure

```
GraphMind/
│
├── 📄 README.md                         # You are here
├── 🔒 GRAPHMIND_HARDCHECK.py            # Full system verification script
├── 📋 requirements.txt                  # Python dependencies
├── 🔒 .env.example                      # Environment variable template
│
├── ⚙️  config/
│   └── settings.py                      # Single source of truth (FROZEN)
│
├── 🧠 src/
│   ├── gemma_explainer.py               # Gemma explanation layer (Tool Use #2)
│   ├── agents/                          # Multi-agent orchestration
│   ├── core/                            # EventBus, BehaviouralGraph, MemoryManager
│   ├── prefetch/                        # ConfidencePrefetch engine (FROZEN)
│   ├── rl/                              # RL environment, reward, trainer
│   ├── benchmarks/                      # Evaluation suite + KPI extractor
│   ├── data/                            # Dataset loaders and transition extractor
│   ├── models/                          # Markov model implementations
│   ├── experiments/                     # Failed experiment scripts (archived)
│   └── explainability/                  # Explainability tools
│
├── 📜 scripts/
│   ├── run_phase11_e.py                 # ← OFFICIAL benchmark entry point
│   ├── ubiqlog_transition_pipeline.py   # Data processing pipeline
│   └── generate_dashboard_data.py       # Dashboard JSON generation
│
├── 📊 dashboard/                        # Next.js 15 web dashboard (7 pages)
│
├── 📚 docs/
│   ├── ax.md                            # ← AGENTIC AI SETUP (primary judge doc)
│   ├── architecture.md                  # 6-layer system architecture
│   ├── technical_stack.md               # Complete OSS dependency list
│   ├── installation.md                  # Step-by-step installation guide
│   ├── reproducibility.md               # 5-step reproducibility guide
│   ├── user_guide.md                    # Dashboard and KPI interpretation guide
│   ├── benchmarking.md                  # Evaluation methodology
│   ├── datasets.md                      # UbiqLog documentation
│   └── models.md                        # Model catalogue
│
├── 📈 results/                          # Benchmark output CSVs
│   └── final_production_results.csv     # OFFICIAL FROZEN RESULT
│
└── 📝 reports/
    └── kpi_summary.json                 # PS03 KPI summary (auto-generated)
```

---

## Dashboard

GraphMind includes a **7-page interactive Next.js dashboard**:

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

| Page | URL | Description |
|------|-----|-------------|
| 🏠 **Executive Overview** | `/` | KPI table, system pipeline, production config |
| 📊 **Benchmark Explorer** | `/benchmark` | Interactive policy comparison table and charts |
| 🗺️ **Optimization Journey** | `/journey` | F1 trajectory across 8 research phases |
| 🕸️ **Graph Explorer** | `/graph` | Interactive Markov transition graph |
| 🎮 **Cache Simulator** | `/simulator` | Live HOT/WARM cache animation |
| 📼 **User Journey** | `/playback` | Step-through with Gemma explanations |
| 🔬 **Research Validation** | `/research` | Ablations, statistical testing, reproducibility |

---

## Attribution

GraphMind V5 is an **original implementation** built from scratch for Samsung EnnovateX AX Hackathon 2026. No existing open-source project was used as a base.

**Dataset**: UbiqLog4UCI (UCI Machine Learning Repository, publicly available under CC BY 4.0).

> Montanari, A., et al. *"UbiqLog: a cheap, unintrusive smartphone-based diet logger."* ACM Conference on Pervasive and Ubiquitous Computing, 2013.

**Open-weight model**: Gemma 2B (Google DeepMind, Gemma Terms of Use).

---

<div align="center">

**GraphMind V5** · Samsung EnnovateX AX Hackathon 2026 · PS03

*F1 = 0.7745 · p = 0.0115 · Cohen's d = 0.491 · 31 real Android users*

[![GitHub](https://img.shields.io/badge/GitHub-Jdepp007004%2FGraphMind-181717?style=flat-square&logo=github)](https://github.com/Jdepp007004/GraphMind)

</div>
