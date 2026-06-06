# GraphMind Statistical Analysis

**Method:** Paired t-test + Bootstrap 95% CI + Cohen's d  
**n:** 31 paired user observations  
**Significance threshold:** α = 0.05  
**Bootstrap iterations:** 2000  
**Comparisons:** GraphMindRL vs Markov-2, GlobalMarkov2, GraphOnly, Graph+Confidence  

---

## Hit Rate

| Comparison | Δ (mean) | 95% CI Treatment | p-value | Sig? | Cohen's d | Effect |
|-----------|---------|-----------------|---------|------|----------|--------|
| GraphMindRL vs Markov-2 | +0.0061 | [0.9161, 0.9518] | 0.1548 | ❌ | 0.099 | negligible |
| GraphMindRL vs GlobalMarkov2 | +0.0226 | [0.9145, 0.9518] | 0.0001 | ✅ | 0.357 | small |
| GraphMindRL vs GraphOnly | -0.0023 | [0.9157, 0.9519] | 0.3444 | ❌ | -0.040 | negligible |
| GraphMindRL vs Graph+Confidence | +0.0002 | [0.9158, 0.9518] | 0.8052 | ❌ | 0.004 | negligible |

## F1

| Comparison | Δ (mean) | 95% CI Treatment | p-value | Sig? | Cohen's d | Effect |
|-----------|---------|-----------------|---------|------|----------|--------|
| GraphMindRL vs Markov-2 | +0.0128 | [0.6939, 0.7876] | 0.1196 | ❌ | 0.092 | negligible |
| GraphMindRL vs GlobalMarkov2 | +0.0633 | [0.6927, 0.7859] | 0.0001 | ✅ | 0.448 | small |
| GraphMindRL vs GraphOnly | +0.0157 | [0.6948, 0.7868] | 0.1086 | ❌ | 0.110 | negligible |
| GraphMindRL vs Graph+Confidence | +0.0015 | [0.6977, 0.7860] | 0.5911 | ❌ | 0.011 | negligible |

## Latency Saved Ms

| Comparison | Δ (mean) | 95% CI Treatment | p-value | Sig? | Cohen's d | Effect |
|-----------|---------|-----------------|---------|------|----------|--------|
| GraphMindRL vs Markov-2 | +8.6958 | [1898.7360, 2097.0934] | 0.1611 | ❌ | 0.029 | negligible |
| GraphMindRL vs GlobalMarkov2 | +32.7455 | [1900.4250, 2101.9187] | 0.0001 | ✅ | 0.107 | negligible |
| GraphMindRL vs GraphOnly | -3.1990 | [1900.7524, 2103.8793] | 0.3541 | ❌ | -0.011 | negligible |
| GraphMindRL vs Graph+Confidence | +0.3261 | [1899.5800, 2100.0841] | 0.8100 | ❌ | 0.001 | negligible |

---

## Summary

- **3/12** comparisons statistically significant (p < 0.05)

### Key Findings

- **F1** — GraphMindRL vs Markov-2: 0.0128 absolute improvement, p=0.1196, d=0.09 (negligible), sig=no
- **F1** — GraphMindRL vs GlobalMarkov2: 0.0633 absolute improvement, p=0.0001, d=0.45 (small), sig=yes
- **F1** — GraphMindRL vs GraphOnly: 0.0157 absolute improvement, p=0.1086, d=0.11 (negligible), sig=no
- **F1** — GraphMindRL vs Graph+Confidence: 0.0015 absolute improvement, p=0.5911, d=0.01 (negligible), sig=no
