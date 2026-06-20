# GraphMind V6 — Technical Stack

## Core Language

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | Core backend |

---

## Machine Learning & AI

| Library | Version | Purpose | Link |
|---|---|---|---|
| PyTorch | ≥2.0 | EmbeddingTransformerReranker, LSTM | [pytorch.org](https://pytorch.org) |
| Stable-Baselines3 | ≥2.0 | PPO adaptive threshold controller | [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io) |
| Gymnasium | ≥0.29 | RL environment (PPO) | [gymnasium.farama.org](https://gymnasium.farama.org) |
| statsmodels | ≥0.14 | ARIMA baseline model | [statsmodels.org](https://www.statsmodels.org) |
| Prophet (neuralprophet) | ≥1.1 | Prophet baseline model | [github.com/facebook/prophet](https://github.com/facebook/prophet) |
| scikit-learn | ≥1.3 | Label encoding, metrics | [scikit-learn.org](https://scikit-learn.org) |
| transformers (HuggingFace) | ≥4.40 | Gemma 2B model loading | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) |

---

## Graph & Data Processing

| Library | Version | Purpose | Link |
|---|---|---|---|
| NetworkX | ≥3.2 | BehaviouralGraph (Markov DiGraph) | [networkx.org](https://networkx.org) |
| pandas | ≥2.0 | UbiqLog dataset processing | [pandas.pydata.org](https://pandas.pydata.org) |
| NumPy | ≥1.26 | Numerical computation | [numpy.org](https://numpy.org) |
| SciPy | ≥1.12 | Bootstrap CIs, paired t-test | [scipy.org](https://scipy.org) |

---

## Storage

| Technology | Purpose |
|---|---|
| SQLite (built-in) | Cold-tier graph persistence (`cold_graph.db`) |
| pickle | Model serialisation (ARIMA, metadata) |
| `.pt` (PyTorch) | Transformer reranker model weights |
| `.zip` (SB3) | PPO policy serialisation |

---

## Dashboard

| Library | Version | Purpose | Link |
|---|---|---|---|
| Next.js | 15 | Web framework (7-page dashboard) | [nextjs.org](https://nextjs.org) |
| TypeScript | 5 | Type-safe dashboard code | [typescriptlang.org](https://www.typescriptlang.org) |
| Recharts | ≥2.12 | KPI charts, benchmark comparison | [recharts.org](https://recharts.org) |
| React Flow | ≥11 | Interactive graph visualisation | [reactflow.dev](https://reactflow.dev) |
| Framer Motion | ≥11 | UI animations and transitions | [framer.com/motion](https://www.framer.com/motion/) |
| Tailwind CSS | 3 | Styling | [tailwindcss.com](https://tailwindcss.com) |

---

## Developer Tools

| Tool | Purpose |
|---|---|
| tqdm | Training and evaluation progress bars |
| pytest | Unit testing |
| pyproject.toml | Project metadata and build config |

---

## Open-Weight Model

| Model | Provider | HuggingFace | Licence |
|---|---|---|---|
| Gemma 2B (gemma-2b) | Google DeepMind | [google/gemma-2b](https://huggingface.co/google/gemma-2b) | Gemma Terms of Use |

---

## Dataset

| Dataset | Source | Licence |
|---|---|---|
| UbiqLog4UCI (Android usage patterns) | [UCI ML Repository #369](https://archive.ics.uci.edu/dataset/369) | CC BY 4.0 |

---

## Full `requirements.txt`

```
torch>=2.0.0
stable-baselines3>=2.0.0
gymnasium>=0.29.0
networkx>=3.2
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
prophet>=1.1.0
tqdm>=4.66.0
transformers>=4.40.0
```
