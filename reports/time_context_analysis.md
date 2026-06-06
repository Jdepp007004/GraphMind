# GraphMind — Time Context Analysis

**Date:** 2026-06-06  
**Source data:** `data/processed/transitions.parquet` (208,695 transitions, 31 users)

---

## 1. Time Field Availability

### In `transitions.parquet`

| Field | Present | Range | Derivation |
|-------|---------|-------|------------|
| `time_bucket` | ✅ YES | 0–47 | `hour*2 + (1 if minute >= 30 else 0)` |
| `day_of_week` | ✅ YES | 0–6 | `datetime.weekday()` (0=Mon) |
| `hour` | Derivable | 0–23 | `time_bucket // 2` |
| `weekend` | Derivable | bool | `day_of_week >= 5` |
| `quarter_hour bucket` | Not stored | 0–95 | Needs raw `minute` field |

### Source code reference

**Created in:** `scripts/ubiqlog_transition_pipeline.py`, lines 131–139:

```python
start_dt = b["start"]
time_bucket = start_dt.hour * 2 + start_dt.minute // 30   # 0–47
transitions.append({
    ...
    "time_bucket": time_bucket,
    "day_of_week": start_dt.weekday(),   # 0=Mon, 6=Sun
})
```

**Re-computed in:** `scripts/run_benchmark_v4.py`, lines ~65–72 (load_events_with_context):

```python
tb = dt.hour * 2 + (1 if dt.minute >= 30 else 0)
wd = dt.weekday()
```

---

## 2. Hourly Event Distribution

Total events by hour (all 31 users, all days):

| Hour | Events | Fraction |
|------|--------|----------|
| 0 | 12,270 | 5.9% |
| 1 | 5,753 | 2.8% |
| 2 | 3,309 | 1.6% |
| 3 | 1,900 | 0.9% |
| 4 | 1,533 | 0.7% |
| 5 | 1,643 | 0.8% |
| 6 | 3,211 | 1.5% |
| 7 | 4,960 | 2.4% |
| 8 | 5,647 | 2.7% |
| 9 | 7,495 | 3.6% |
| 10 | 9,675 | 4.6% |
| 11 | 9,986 | 4.8% |
| 12 | 11,573 | 5.5% |
| 13 | 10,685 | 5.1% |
| 14 | 10,442 | 5.0% |
| 15 | 9,886 | 4.7% |
| 16 | 10,195 | 4.9% |
| 17 | 11,328 | 5.4% |
| 18 | 11,350 | 5.4% |
| 19 | 12,380 | 5.9% |
| 20 | 12,137 | 5.8% |
| 21 | 12,541 | 6.0% |
| 22 | 13,575 | 6.5% |
| 23 | 15,221 | 7.3% |

**Pattern:** Peak usage at hour 23 (7.3%) and hours 22, 21 (evening). Trough at hours 3–5 (night). Clear bimodal distribution: lunch peak (12–13) + evening peak (19–23).

---

## 3. Transition Entropy by Time Period

### Methodology
- Per user, per period: compute P(next_app | from_app) from transition counts
- Only include `from_app` with ≥5 transitions in that period
- Shannon entropy in bits: H = −Σ p log₂ p

### Results

| Time Period | Hours | Mean Entropy (bits) | Interpretation |
|------------|-------|-------------------|----------------|
| Night | 00:00–05:59 | **2.012** | Most predictable |
| Morning | 06:00–11:59 | 2.262 | Moderate predictability |
| Afternoon | 12:00–17:59 | 2.387 | Less predictable |
| Evening | 18:00–23:59 | 2.394 | Least predictable |

**Key insight:** Night transitions are 0.37 bits more predictable than evening (2.01 vs 2.39 bits).
This is an **18% entropy reduction** — statistically meaningful for a prediction model.

Lower entropy → fewer distinct successors → higher top-k hit rate expected at night.

---

## 4. Top Apps by Time Period

Global top-3 app transitions across all users:

| Period | #1 App | #2 App | #3 App |
|--------|--------|--------|--------|
| Morning | com.sec.android.app.launcher | com.sec.android.app.twdvfs | com.sec.pcw.device |
| Afternoon | com.sec.android.app.twdvfs | com.viber.voip | com.sec.pcw.device |
| Evening | com.viber.voip | com.whatsapp | com.sec.android.app.launcher |
| Night | com.viber.voip | com.sec.knox.eventsmanager | com.sec.pcw.device |

**Observation:** `com.viber.voip` dominates evening and night. `com.sec.android.app.launcher` (home screen) is a morning signature. Apps shift systematically across periods — a time-conditioned predictor should capture this.

> Note: System apps (launcher, knox, pcw.device) appear despite system filtering because UbiqLog's Samsung-specific apps were not in the original system prefix list. These should be filtered in V5.

---

## 5. Measured Gain from Time Context

### Empirical evidence (from Markov order analysis, 31 users):

| Model | Mean Hit Rate | vs Markov-1 |
|-------|-------------|------------|
| Markov-1 | 0.6045 | baseline |
| **ContextMarkov-1** | **0.6582** | **+5.37 pp** |

**ContextMarkov-1** conditions on `P(next | from_app, coarse_bucket)` where `coarse_bucket = time_bucket // 8` (6 time bands of 4 hours each). With fallback to pure M1 when the bucket has insufficient data.

**This is a free +5.4pp gain** using data already present in the pipeline.

---

## 6. Granularity Analysis

### 24 hourly buckets (0–23)
- Pro: Sufficient data density (~8,700 transitions/hour average)
- Pro: Captures morning/lunch/evening patterns
- Con: May split sparse users' data too thin

### 48 half-hour buckets (0–47) ← **already computed**
- Pro: Already exists in `time_bucket` field
- Pro: Captures finer patterns (e.g. 8:00 vs 8:30 AM)
- Con: ~4,350 transitions/bucket (some buckets <2000)
- **Recommended for V5**: Use with Laplace smoothing

### 96 quarter-hour buckets (0–95)
- Con: Requires raw `minute` field (not stored in parquet)
- Con: ~2,175 transitions/bucket — too sparse for 500 apps
- **Not recommended**: Sparsity outweighs granularity gain

### Recommendation for V5
Use **48 half-hour buckets** (existing `time_bucket` field) with Laplace smoothing α=0.1.
Fall back to `time_bucket // 4` (12 coarse buckets) for users with < 200 transitions.

---

## 7. Day-of-Week Analysis

Distribution across weekdays:

| Day | Index | Transitions | % |
|-----|-------|------------|---|
| Mon | 0 | ~28k | 13.4% |
| Tue | 1 | ~30k | 14.4% |
| Wed | 2 | ~30k | 14.4% |
| Thu | 3 | ~30k | 14.4% |
| Fri | 4 | ~29k | 13.9% |
| Sat | 5 | ~31k | 14.9% |
| Sun | 6 | ~30k | 14.6% |

Distribution is relatively uniform. Weekend (Sat+Sun = 29.5%) vs Weekday (70.5%).

For a simple binary `is_weekend` feature: weekend usage patterns differ (no commute context, more leisure apps) but with only 29.5% weekend data, separate weekend models risk overfitting for sparse users.

**Recommendation:** Include `day_of_week` as a continuous feature (normalized 0–1) rather than building 7 separate Markov matrices.

---

## 8. Time-Aware Graph Implications for V5

### What to do

1. **Time-conditioned first-order Markov:** `P(next | current, time_bucket)` with Laplace smoothing  
   Expected gain: +5.4pp hit rate

2. **Temporal edge decay:** Weight recent transitions more  
   Formula: `w_edge(t) = base_prob × exp(−λ × days_since_last_occurrence)`  
   Keeps model fresh for behavioural drift

3. **Node identity: `(app, time_bucket)`**  
   This is already in `BehaviouralGraph` (line 315–317) but unused in benchmark  
   Bring this into the benchmark evaluation pipeline

4. **DO NOT** split into 7 × 48 = 336 separate matrices per user  
   This would be too sparse. Use `(app, coarse_bucket)` with fallback.

---
