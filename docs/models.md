# GraphMind V6 — Model Catalogue

## Models Used (External Open-Weight)

### Gemma 2B — `google/gemma-2b`

| Property | Value |
|---|---|
| **Provider** | Google DeepMind |
| **HuggingFace** | [https://huggingface.co/google/gemma-2b](https://huggingface.co/google/gemma-2b) |
| **Parameters** | 2 billion |
| **Variant** | Instruction-tuned |
| **Licence** | Gemma Terms of Use (open for research and commercial use) |
| **Role in GraphMind** | Post-decision NL explanation generation (Step 6 of 7) |
| **Inference latency** | < 300ms on CPU (4-bit GGUF quantisation) |
| **RAM requirement** | ~1.5 GB (int4 quantisation) |

**Sample output:**
```
"Preloading Spotify because you typically switch from YouTube in the evening."
```

Gemma fires **after** all KPIs are measured. Setting `ENABLE_GEMMA=false` produces byte-for-byte identical benchmark CSVs.

---

## Models Published (Custom, In-House)

### EmbeddingTransformerReranker — 31 Per-User Models

| Property | Value |
|---|---|
| **Architecture** | 2-layer Transformer encoder + linear head |
| **Input** | 34-dim: 32-dim app embedding + time_bucket_norm + day_of_week_norm |
| **Output** | Reranked candidate scores |
| **Training** | Per-user, chronological split, ~30 epochs |
| **File pattern** | `models/saved/v6_reranker_ubiqlog_{user_id}_{gender}.pt` |
| **Quantity** | 31 user-specific models |
| **Committed to GitHub** | ✅ Yes — all 68 files (`.pt` + `_meta.pkl`) |

**Key innovation:** Training one Transformer per user (rather than a shared global model) eliminates cross-user contamination and captures individual behavioural patterns. This is the primary reason V6 achieves 97.92% vs V5's 80.51%.

### PPO Memory Allocation Agent

| Property | Value |
|---|---|
| **Framework** | Stable-Baselines3 PPO |
| **Action space** | `MultiDiscrete([5, 5, 5])` — 125 discrete actions |
| **State dim** | 109 (app OHE + temporal + cache occupancy + hit history) |
| **File** | `models/rl_policies/user_00_ppo.zip` |
| **HuggingFace** | [https://huggingface.co/dheerajsait/GraphMind_PPO](https://huggingface.co/dheerajsait/GraphMind_PPO) |
| **License** | Apache 2.0 |

**Loading the PPO model:**
```python
from stable_baselines3 import PPO
model = PPO.load("models/rl_policies/user_00_ppo.zip")
```

---

## Baseline Models

These models are used as comparison baselines in the 14-policy benchmark:

| Model | File | Purpose |
|---|---|---|
| ARIMA (UbiqLog) | `models/saved/arima_ubiqlog.pkl` | Time-series forecasting baseline |
| LSTM (UbiqLog) | `models/saved/lstm_ubiqlog.pt` | LSTM sequence model baseline |
| Prophet (UbiqLog) | `models/saved/prophet_ubiqlog.pkl` | Meta Prophet baseline |
| ARIMA (Synthetic) | `models/saved/arima_synthetic.pkl` | Synthetic dataset baseline |
| LSTM (Synthetic) | `models/saved/lstm_synthetic.pt` | Synthetic dataset baseline |
| Prophet (Synthetic) | `models/saved/prophet_synthetic.pkl` | Synthetic dataset baseline |

All baseline models are **pre-trained and committed to GitHub**. No download or retraining needed for `--cache` mode.
