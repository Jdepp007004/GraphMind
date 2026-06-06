# GraphMind V4 — Statistical Analysis

**Methods:** Paired t-test + Wilcoxon signed-rank + Bootstrap 95% CI + Cohen's d  
**n:** 31 paired user observations  
**α:** 0.05  
**Bootstrap iterations:** 2000  

---

## Hit Rate

| Comparison | Δ (mean) | 95% CI (treatment) | t-test p | Wilcoxon p | Sig | Cohen's d | Effect |
|-----------|---------|-------------------|---------|-----------|-----|----------|--------|
| RLAdaptiveEnsemble vs GraphMindRL | -0.0182 | [0.8931, 0.9391] | 0.0002 | 0.0000 | ✅ | -0.297 | small |
| RLAdaptiveEnsemble vs Markov-2 | -0.0121 | [0.8905, 0.9396] | 0.0114 | 0.0004 | ✅ | -0.175 | negligible |
| RLAdaptiveEnsemble vs VariableOrderMarkov | -0.0025 | [0.8914, 0.9394] | 0.1006 | 0.1563 | ❌ | -0.037 | negligible |
| RLAdaptiveEnsemble vs Graph+Confidence | -0.0180 | [0.8913, 0.9393] | 0.0001 | 0.0000 | ✅ | -0.292 | small |
| GraphMindRL vs Markov-2 | +0.0061 | [0.9145, 0.9523] | 0.1548 | 0.5163 | ❌ | 0.099 | negligible |
| GraphMindRL vs GlobalMarkov2 | +0.0226 | [0.9153, 0.9519] | 0.0001 | 0.0000 | ✅ | 0.357 | small |
| VariableOrderMarkov vs Markov-2 | -0.0096 | [0.8930, 0.9416] | 0.0277 | 0.0010 | ✅ | -0.139 | negligible |
| ContextMarkov vs Markov-2 | -0.0126 | [0.8925, 0.9389] | 0.0068 | 0.0002 | ✅ | -0.183 | negligible |
| ClusterMarkov vs GlobalMarkov2 | -0.0123 | [0.8704, 0.9265] | 0.0002 | 0.0003 | ✅ | -0.161 | negligible |

## F1

| Comparison | Δ (mean) | 95% CI (treatment) | t-test p | Wilcoxon p | Sig | Cohen's d | Effect |
|-----------|---------|-------------------|---------|-----------|-----|----------|--------|
| RLAdaptiveEnsemble vs GraphMindRL | -0.1253 | [0.5429, 0.6841] | 0.0002 | 0.0003 | ✅ | -0.741 | medium |
| RLAdaptiveEnsemble vs Markov-2 | -0.1125 | [0.5514, 0.6837] | 0.0001 | 0.0000 | ✅ | -0.641 | medium |
| RLAdaptiveEnsemble vs VariableOrderMarkov | -0.0079 | [0.5500, 0.6856] | 0.0009 | 0.0007 | ✅ | -0.040 | negligible |
| RLAdaptiveEnsemble vs Graph+Confidence | -0.1238 | [0.5458, 0.6871] | 0.0002 | 0.0000 | ✅ | -0.722 | medium |
| GraphMindRL vs Markov-2 | +0.0128 | [0.6963, 0.7900] | 0.1196 | 0.1694 | ❌ | 0.092 | negligible |
| GraphMindRL vs GlobalMarkov2 | +0.0633 | [0.6923, 0.7852] | 0.0001 | 0.0000 | ✅ | 0.448 | small |
| VariableOrderMarkov vs Markov-2 | -0.1046 | [0.5588, 0.6911] | 0.0003 | 0.0000 | ✅ | -0.600 | medium |
| ContextMarkov vs Markov-2 | -0.1200 | [0.5378, 0.6778] | 0.0001 | 0.0000 | ✅ | -0.692 | medium |
| ClusterMarkov vs GlobalMarkov2 | -0.0674 | [0.5388, 0.6798] | 0.0049 | 0.0396 | ✅ | -0.378 | small |

## Latency Saved Ms

| Comparison | Δ (mean) | 95% CI (treatment) | t-test p | Wilcoxon p | Sig | Cohen's d | Effect |
|-----------|---------|-------------------|---------|-----------|-----|----------|--------|
| RLAdaptiveEnsemble vs GraphMindRL | -26.5823 | [1865.0940, 2078.0797] | 0.0001 | 0.0000 | ✅ | -0.088 | negligible |
| RLAdaptiveEnsemble vs Markov-2 | -17.8865 | [1871.3085, 2085.4645] | 0.0106 | 0.0004 | ✅ | -0.057 | negligible |
| RLAdaptiveEnsemble vs VariableOrderMarkov | -3.8668 | [1869.3691, 2080.4791] | 0.0994 | 0.1132 | ❌ | -0.013 | negligible |
| RLAdaptiveEnsemble vs Graph+Confidence | -26.2561 | [1864.1154, 2084.1976] | 0.0001 | 0.0000 | ✅ | -0.087 | negligible |
| GraphMindRL vs Markov-2 | +8.6958 | [1900.9160, 2097.6253] | 0.1611 | 0.5387 | ❌ | 0.029 | negligible |
| GraphMindRL vs GlobalMarkov2 | +32.7455 | [1902.1781, 2105.8623] | 0.0001 | 0.0000 | ✅ | 0.107 | negligible |
| VariableOrderMarkov vs Markov-2 | -14.0197 | [1872.8603, 2088.1245] | 0.0270 | 0.0011 | ✅ | -0.045 | negligible |
| ContextMarkov vs Markov-2 | -18.6094 | [1868.2719, 2079.1913] | 0.0064 | 0.0002 | ✅ | -0.060 | negligible |
| ClusterMarkov vs GlobalMarkov2 | -17.9665 | [1846.9974, 2064.3468] | 0.0002 | 0.0002 | ✅ | -0.056 | negligible |

---

## Summary

**22/27** comparisons statistically significant (p < 0.05)

### Key Findings

- **RLAdaptiveEnsemble vs GraphMindRL** (F1): Δ=-0.1253 (degradation), p=0.0002 (**significant**), d=-0.74 (medium)
- **RLAdaptiveEnsemble vs Graph+Confidence** (F1): Δ=-0.1238 (degradation), p=0.0002 (**significant**), d=-0.72 (medium)
- **ContextMarkov vs Markov-2** (F1): Δ=-0.1200 (degradation), p=0.0001 (**significant**), d=-0.69 (medium)
- **RLAdaptiveEnsemble vs Markov-2** (F1): Δ=-0.1125 (degradation), p=0.0001 (**significant**), d=-0.64 (medium)
- **VariableOrderMarkov vs Markov-2** (F1): Δ=-0.1046 (degradation), p=0.0003 (**significant**), d=-0.60 (medium)
- **GraphMindRL vs GlobalMarkov2** (F1): Δ=+0.0633 (improvement), p=0.0001 (**significant**), d=0.45 (small)
- **ClusterMarkov vs GlobalMarkov2** (F1): Δ=-0.0674 (degradation), p=0.0049 (**significant**), d=-0.38 (small)
- **GraphMindRL vs Markov-2** (F1): Δ=+0.0128 (improvement), p=0.1196 (not significant), d=0.09 (negligible)
- **RLAdaptiveEnsemble vs VariableOrderMarkov** (F1): Δ=-0.0079 (degradation), p=0.0009 (**significant**), d=-0.04 (negligible)
