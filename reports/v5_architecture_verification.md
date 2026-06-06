# V5 Architecture Verification

**Date:** 2026-06-06  
**Method:** Static code analysis + runtime identity tests  
**Source files:** `scripts/run_benchmark_v4.py`, `src/core/graph_engine.py`, `src/prefetch/confidence_prefetch.py`, `src/rl/environment_v2.py`

---

## Q1: Is GraphOnly mathematically identical to Markov-1?

**Answer: YES — verified at runtime across all 31 users.**

### Structural evidence

**`GraphOnlyPolicy.train()` (run_benchmark_v4.py ~L481):**

```python
class GraphOnlyPolicy(Policy):
    def train(self, apps, **kw):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c[apps[i-1]][apps[i]] += 1         # ← app→app count
        for s, d in c.items():
            t = sum(d.values())
            self._g[s] = dict(sorted(
                {k: v/t for k,v in d.items()}.items(),
                key=lambda x: -x[1]
            ))

    def predict(self, cur, prev=None, tb=0, wd=0):
        return list(self._g.get(cur, {}).keys())[:HOT_SIZE]  # top-k by P
```

**`Markov1Policy.train()` (run_benchmark_v4.py ~L431):**

```python
class Markov1Policy(Policy):
    def train(self, apps, **kw):
        c = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(apps)):
            c[apps[i-1]][apps[i]] += 1         # ← identical loop
        for s, d in c.items():
            t = sum(d.values())
            self._m[s] = dict(sorted(
                {k: v/t for k,v in d.items()}.items(),
                key=lambda x: -x[1]
            ))

    def predict(self, cur, prev=None, tb=0, wd=0):
        return list(self._m.get(cur, {}).keys())[:HOT_SIZE]  # identical
```

Both implement `P(next | current_app)`. Both sort by descending probability. Both return top-k. The only difference is the attribute name (`self._g` vs `self._m`).

### Runtime confirmation

```
Empirical per-user F1 difference: 0.000000 (max), 0.000000 (mean)
Conclusion: GraphOnly ≡ Markov-1 on all 31 users
```

**The "graph" label in GraphOnly adds zero predictive value over plain Markov-1.**

---

## Q2: Does GraphMindRL use time_bucket, day_of_week, or previous app?

### Code trace

**`GraphMindRLPolicy.predict()` (run_benchmark_v4.py):**

```python
def predict(self, cur, prev=None, tb=0, wd=0):
    if cur not in self._g: return []
    tot = self._total or 1.0
    cands = {}
    for app, p in self._g[cur].items():
        conf = 0.5*p + 0.3*self._rec.get(app,0) + 0.2*(self._freq.get(app,0)/tot)
        if conf >= self._thresh:
            cands[app] = conf
    return sorted(cands, key=lambda a: -cands[a])[:self._budget]
```

| Parameter | Present in signature? | Used in formula? | Effect |
|-----------|----------------------|-----------------|--------|
| `cur` | ✅ | ✅ | Graph lookup key |
| `prev` | ✅ | ❌ | **Ignored** |
| `tb` | ✅ | ❌ | **Ignored** |
| `wd` | ✅ | ❌ | **Ignored** |

**Finding:** `GraphMindRLPolicy.predict()` ignores `prev`, `tb`, and `wd`. It only uses:
1. `self._g[cur]` — Markov-1 graph transition probabilities
2. `self._rec[app]` — exponentially decaying recency (updated per-step, not time-bucket-aware)
3. `self._freq[app]` — running frequency count (not time-bucket-aware)

**GraphMindRL is a recency/frequency-boosted Markov-1. Time and history are not used.**

### The confidence formula

```
confidence(app) = 0.5 × P(app | cur_app)          ← graph transition probability
                + 0.3 × recency(app)               ← exp-decay score, NOT time-of-day
                + 0.2 × (freq(app) / total_events) ← global frequency, NOT time-conditioned

threshold = 0.05 (adaptive: ±0.03–0.08 based on hit rate)
budget    = HOT_SIZE (adaptive: 3–8 based on hit rate)
```

This is **first-order Markov with recency and frequency boosting**, not a time-aware or history-aware predictor.

---

## Q3: Is BehaviouralGraph (src/core/graph_engine.py) used in benchmark evaluation?

**Answer: NO.**

### Call graph

```
run_benchmark_v4.py::main()
  └─ evaluate_policy(policy, ...)
       └─ policy.predict(cur, prev, tb, wd)
            └─ self._g[cur]   ← dict, built in-process
            └─ NO call to BehaviouralGraph, ConfidencePrefetch, or EventBus
```

The benchmark policies build their own inline Markov dictionaries from the raw `apps` list. They do NOT:
- Import `BehaviouralGraph`
- Import `ConfidencePrefetch`
- Import `EventBus`
- Use `src/core/`, `src/prefetch/`, or `src/rl/environment_v2.py`

### Where BehaviouralGraph IS used

`BehaviouralGraph` is instantiated only in:
- `src/rl/environment_v2.py` (SB3 PPO training, not in benchmark)
- `src/prefetch/daemon.py` (background prefetch daemon, not in benchmark)
- Unit tests

**The BehaviouralGraph architecture and the benchmark evaluation pipeline have diverged.** The benchmark does not exercise any of the production graph code.

---

## Q4: Does GraphMindRL prediction depend only on P(next | current_app)?

**Answer: YES — with recency/frequency adjustments, but no temporal conditioning.**

### Evidence

The complete information used by `GraphMindRL` at prediction time:

```
Inputs available:   cur_app, prev_app, time_bucket, weekday
Inputs USED:        cur_app only (as graph lookup key)
                    recency(any_app)   [running, not time-binned]
                    frequency(any_app) [running, not time-binned]
```

The mathematical form is:

```
score(app) = 0.5 × P(app | cur_app)
           + 0.3 × recency(app)     [∈ [0, 1], exp-decay with each step]
           + 0.2 × freq(app)/total  [∈ [0, 1], cumulative count]

predict = argtop_k(score, k=budget, threshold=thresh)
```

This does NOT use:
- `P(app | prev_app, cur_app)` — no second-order context
- `P(app | cur_app, time_bucket)` — no time conditioning
- `P(app | cur_app, weekday)` — no day-of-week conditioning

---

## Complete Pipeline Call Graph

```
run_benchmark_v4.py
│
├─ load_events_with_context(user_id)
│    ├─ reads UbiqLog .txt files directly
│    ├─ parses: pkg, dt, time_bucket, weekday
│    └─ returns: (apps[], tbs[], wds[])
│
├─ evaluate_policy(policy, train_apps, val_apps, test_apps, ...)
│    ├─ policy.train(train_apps, tbs, wds, ...)
│    │    └─ builds dict[app→dict[app,prob]]  ← pure Markov-1
│    │
│    ├─ Cache warm-up: last 20 train events
│    │
│    └─ for each test event:
│         ├─ policy.predict(cur, prev, tb, wd)
│         │    └─ returns top-k from dict lookup
│         │         (tb, wd IGNORED by most policies)
│         │
│         ├─ Cache.prefetch(predictions)
│         ├─ Cache.lookup(actual_next_app)
│         ├─ MeasuredLatencyModel.saved(app, tier)
│         └─ policy.update(app, hit)
│
└─ aggregate: hit_rate, precision, recall, F1, lat_saved
```

---

## Feature Usage Matrix

| Feature | Markov-1 | GraphOnly | GraphMindRL | Graph+Conf | M2 | CtxMarkov |
|---------|----------|-----------|-------------|------------|-----|-----------|
| `cur_app` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `prev_app` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `time_bucket` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| `weekday` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| `recency` | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| `frequency` | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| `BehaviouralGraph` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**None of the top-performing policies (GraphMindRL, Graph+Conf, M2) use `time_bucket` or `weekday`.**

---

## Summary of Verification Findings

| Claim | Verified | Evidence |
|-------|----------|---------|
| GraphOnly ≡ Markov-1 | ✅ TRUE | Identical code + zero F1 difference across 31 users |
| GraphMindRL uses time_bucket | ❌ FALSE | `tb` param ignored in predict() |
| GraphMindRL uses prev_app | ❌ FALSE | `prev` param ignored in predict() |
| GraphMindRL uses BehaviouralGraph | ❌ FALSE | No import, no call |
| GraphMindRL depends only on P(next\|cur) | ✅ TRUE | Only `self._g[cur]` used as primary signal |
| time_bucket EXISTS in data | ✅ TRUE | Present in parquet and passed to predict() |
| time_bucket USED in evaluation | ❌ FALSE (most policies) | Only ContextMarkov uses it |

---
