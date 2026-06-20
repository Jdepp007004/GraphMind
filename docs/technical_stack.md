# GraphMind V5 -- Technical Stack

> **Samsung EnnovateX AX Hackathon 2026 -- PS03**
> Complete list of all open-source libraries, frameworks, and models used in GraphMind V5.

---

## Python Environment

| Property | Value |
|---|---|
| **Language** | Python 3.10+ |
| **Recommended** | Python 3.11.x |
| **Virtual environment** | `python -m venv venv` |

---

## Core Python Dependencies

| Library | Version | Purpose | Link |
|---|---|---|---|
| `networkx` | ≥ 3.1 | Markov behaviour graph (directed weighted digraph) | [networkx.org](https://networkx.org/) |
| `numpy` | ≥ 1.24 | Numerical arrays, probability computations | [numpy.org](https://numpy.org/) |
| `pandas` | ≥ 2.0 | Dataset loading, transition extraction | [pandas.pydata.org](https://pandas.pydata.org/) |
| `scipy` | ≥ 1.11 | Paired t-test, effect size (Cohen's d), bootstrap CIs | [scipy.org](https://scipy.org/) |
| `scikit-learn` | ≥ 1.3 | Label encoding, train/test split utilities | [scikit-learn.org](https://scikit-learn.org/) |
| `stable-baselines3` | ≥ 2.2 | PPO reinforcement learning agent | [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/) |
| `gymnasium` | ≥ 0.29 | RL environment interface (OpenAI Gym successor) | [gymnasium.farama.org](https://gymnasium.farama.org/) |
| `torch` | ≥ 2.1 | Neural network backend for PPO policy | [pytorch.org](https://pytorch.org/) |
| `transformers` | ≥ 4.38 | Gemma model loading and inference | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) |
| `python-dotenv` | ≥ 1.0 | `.env` file loading for secrets | [pypi.org/project/python-dotenv](https://pypi.org/project/python-dotenv/) |
| `apscheduler` | ≥ 3.10 | Periodic agent task scheduling | [apscheduler.readthedocs.io](https://apscheduler.readthedocs.io/) |
| `pytest` | ≥ 7.4 | Unit test runner | [pytest.org](https://pytest.org/) |
| `pytest-cov` | ≥ 4.1 | Test coverage reporting | [pytest-cov.readthedocs.io](https://pytest-cov.readthedocs.io/) |

### Pinned versions (from `requirements.txt`)

```
networkx>=3.1
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
stable-baselines3>=2.2
gymnasium>=0.29
torch>=2.1
transformers>=4.38
python-dotenv>=1.0
apscheduler>=3.10
pytest>=7.4
pytest-cov>=4.1
```

---

## Dashboard Dependencies (Node.js)

| Library | Version | Purpose | Link |
|---|---|---|---|
| **Next.js** | 15.x | App Router framework | [nextjs.org](https://nextjs.org/) |
| **TypeScript** | 5.x | Type-safe JavaScript | [typescriptlang.org](https://typescriptlang.org/) |
| **React** | 18.x | Component model | [react.dev](https://react.dev/) |
| **Recharts** | 2.x | Data visualisation (bar, line, scatter charts) | [recharts.org](https://recharts.org/) |
| **@xyflow/react** | 11.x | Interactive Markov graph visualisation | [reactflow.dev](https://reactflow.dev/) |
| **Framer Motion** | 10.x | Smooth UI animations and transitions | [framer.com/motion](https://www.framer.com/motion/) |
| **Tailwind CSS** | 3.x | Utility-first CSS framework | [tailwindcss.com](https://tailwindcss.com/) |

---

## Open-Weight AI Models

### Gemma 2B (google/gemma-2b)

| Property | Value |
|---|---|
| **Model ID** | `google/gemma-2b` (instruction-tuned preferred: `google/gemma-2b-it`) |
| **HuggingFace** | [GEMMA_HUGGINGFACE_LINK] |
| **Parameters** | 2 billion |
| **Architecture** | Decoder-only transformer (Gemma architecture) |
| **Licence** | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| **Use in GraphMind** | Post-decision natural language explanation generation |
| **On-device footprint** | ~1.5 GB (int4 quantised), runs on Galaxy A23 class hardware |
| **Local path** | `models/gemma-2b/` |

**Download command**:
```bash
# [GEMMA_DOWNLOAD_COMMAND]
# Example:
huggingface-cli download google/gemma-2b --local-dir models/gemma-2b
```

---

## In-House Models

### PPO Memory Allocation Agent

| Property | Value |
|---|---|
| **Framework** | Stable-Baselines3 PPO |
| **Action space** | `MultiDiscrete([5, 5, 5])` -- 125 discrete actions |
| **Observation space** | 109-dimensional continuous vector |
| **Training timesteps** | 200,000 (configurable via `settings.PPO_TOTAL_TIMESTEPS`) |
| **Published on HuggingFace** | [PPO_HUGGINGFACE_LINK] (if published) |
| **Local path** | `models/rl_policies/` |

---

## Dataset

| Property | Value |
|---|---|
| **Name** | UbiqLog4UCI (Android Life Logging Dataset) |
| **Source** | UCI Machine Learning Repository |
| **URL** | [UBIQLOG_UCI_LINK] |
| **Licence** | CC BY 4.0 |
| **Size** | 9.7M events, 35 users |
| **Transitions extracted** | 208,695 (after filtering) |
| **Users retained** | 31 (4 removed for < 100 transitions) |

---

## Development Tools

| Tool | Version | Purpose |
|---|---|---|
| Git | ≥ 2.40 | Version control |
| GitHub Actions | -- | CI pipeline (`.github/workflows/`) |
| pyproject.toml | -- | Project metadata and build config |
| Node.js | ≥ 18 | Dashboard dev server |
| npm | ≥ 9 | Dashboard package manager |

---

*Last updated: 2026-06-14. All version pins reflect the minimum tested version; later minor versions should be compatible.*
