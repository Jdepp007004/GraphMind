# GraphMind Fast Benchmark Report

**Users:** 31 usable UbiqLog users  
**Policies:** 6  
**Split:** 80% train / 10% val / 10% test (chronological)  
**Latency source:** Measured — Samsung Galaxy A23, app_launch_latency.csv  

---

## Hit Rate (mean ± std, 95% CI)

| Policy | Mean | Median | Std | P95 | 95% CI |
|--------|------|--------|-----|-----|--------|
| **Markov-1** | 0.9380 | 0.9560 | 0.0583 | 0.9867 | [0.9155, 0.9555] |
| **GraphOnly** | 0.9380 | 0.9560 | 0.0583 | 0.9867 | [0.9155, 0.9555] |
| **GraphMindRL** | 0.9357 | 0.9518 | 0.0513 | 0.9813 | [0.9155, 0.9511] |
| **Graph+Confidence** | 0.9355 | 0.9495 | 0.0518 | 0.9809 | [0.9156, 0.9512] |
| **Markov-2** | 0.9297 | 0.9450 | 0.0682 | 0.9865 | [0.9037, 0.9500] |
| **GlobalMarkov2** | 0.9132 | 0.9266 | 0.0713 | 0.9796 | [0.8862, 0.9356] |

## F1 Score (mean ± std)

| Policy | Mean | Median | Std | P95 |
|--------|------|--------|-----|-----|
| **GraphMindRL** | 0.7424 | 0.7641 | 0.1292 | 0.9113 |
| **Graph+Confidence** | 0.7408 | 0.7809 | 0.1354 | 0.9161 |
| **Markov-2** | 0.7295 | 0.7599 | 0.1449 | 0.9024 |
| **Markov-1** | 0.7267 | 0.7565 | 0.1509 | 0.9012 |
| **GraphOnly** | 0.7267 | 0.7565 | 0.1509 | 0.9012 |
| **GlobalMarkov2** | 0.6790 | 0.6955 | 0.1480 | 0.8671 |

## Latency Saved (ms, mean ± std)

| Policy | Mean | Median | Std | P95 |
|--------|------|--------|-----|-----|
| **Markov-1** | 2005.7 | 2036.0 | 297.1 | 2384.4 |
| **GraphOnly** | 2005.7 | 2036.0 | 297.1 | 2384.4 |
| **GraphMindRL** | 2002.5 | 2025.7 | 289.1 | 2385.1 |
| **Graph+Confidence** | 2002.2 | 2030.2 | 289.7 | 2385.1 |
| **Markov-2** | 1993.8 | 2009.6 | 310.9 | 2388.4 |
| **GlobalMarkov2** | 1969.8 | 2009.4 | 313.1 | 2377.7 |
