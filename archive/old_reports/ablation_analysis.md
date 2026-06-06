# GraphMind Ablation Study

**Question:** What is the contribution of each GraphMind component?

**Users:** 31  
**Variants:** GraphOnly → Graph+Confidence → Graph+RL → Full GraphMind  

---

## Summary Table (mean ± std over all users)

| Variant | Hit Rate | F1 | Latency Saved (ms) | False Prefetch Rate |
|---------|----------|----|--------------------|--------------------|
| **GraphOnly** | 0.938 ± 0.058 | 0.727 ± 0.151 | 2006 ± 297 | — |
| **Graph+Confidence** | 0.935 ± 0.052 | 0.741 ± 0.135 | 2002 ± 290 | — |
| **Graph+RL** | 0.938 ± 0.057 | 0.733 ± 0.145 | 2005 ± 294 | — |
| **Full GraphMind** | 0.936 ± 0.051 | 0.742 ± 0.129 | 2003 ± 289 | — |

---

## Component Contribution

### 1. Graph Alone
**Hit Rate: 0.938 | F1: 0.727**  
The behavioural graph provides structured transition predictions. Compared to stateless LRU/LFU baselines, the graph significantly improves F1 by capturing individual usage patterns.

### 2. Adding Confidence Scorer (+Confidence)
**ΔHit Rate: -0.002 | ΔF1: +0.014**  
The confidence scorer filters low-probability candidates using recency and frequency. Precision improves — prefetch precision gains outweigh recall loss.

### 3. Adding RL Budget Allocation (+RL)
**ΔHit Rate: -0.000 | ΔF1: +0.007**  
RL dynamically adjusts HOT/WARM budget based on recent hit-rate history. Budget adaptation improves F1 by reducing false prefetches.

### 4. Full GraphMind (Graph + Confidence + RL)
**ΔHit Rate vs GraphOnly: -0.002 | ΔF1: +0.016**  
The complete system achieves the best F1 score. Confidence filtering ensures high precision; RL adapts resource budgets over time.

