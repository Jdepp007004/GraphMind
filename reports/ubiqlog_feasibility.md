# UbiqLog Dataset Feasibility Assessment

**Generated:** 2026-06-06  
**Dataset:** UbiqLog4UCI — 35 users, 10,587,892 total records, 820,603 Application events

---

## Dataset Statistics (Actual)

| Metric                     | Value              |
|----------------------------|--------------------|
| Total event records        | 10,587,892         |
| Total Application events   | 820,603            |
| Total WiFi events          | 8,778,706          |
| Total Location events      | 726,677            |
| Total Bluetooth events     | 117,262            |
| Total Call events          | 99,591             |
| Total SMS events           | 29,144             |
| Total Activity events      | 15,641             |
| Total Picture events       | 267                |
| Users with 0 app events    | 2 (3_M, 4_F)       |
| Usable users               | 33                 |

---

## Per-User Application Event Counts

| User  | App Events | Files | Usable? |
|-------|-----------|-------|---------|
| 10_M  | 1,073     | 9     | ⚠ sparse |
| 11_F  | 27,829    | 61    | ✅       |
| 12_M  | 35,052    | 56    | ✅       |
| 13_F  | 11,938    | 52    | ✅       |
| 14_F  | 7,039     | 70    | ✅       |
| 15_F  | 14,506    | 24    | ✅       |
| 16_F  | 22,455    | 55    | ✅       |
| 17_F  | 16,552    | 57    | ✅       |
| 18_F  | 107,801   | 65    | ✅ rich  |
| 19_F  | 77,360    | 57    | ✅ rich  |
| 1_M   | 9,486     | 30    | ✅       |
| 20_M  | 12,023    | 69    | ✅       |
| 21_F  | 4,561     | 35    | ✅       |
| 22_M  | 47,499    | 49    | ✅       |
| 23_F  | 5,787     | 28    | ✅       |
| 24_F  | 54,251    | 51    | ✅ rich  |
| 25_F  | 455       | 8     | ❌ too sparse |
| 26_F  | 14,706    | 48    | ✅       |
| 27_F  | 19,600    | 55    | ✅       |
| 28_F  | 49,528    | 73    | ✅ rich  |
| 29_F  | 12,786    | 57    | ✅       |
| 2_F   | 14,408    | 27    | ✅       |
| 30_F  | 9,565     | 30    | ✅       |
| 31_F  | 26,015    | 67    | ✅       |
| 32_F  | 12,556    | 42    | ✅       |
| 33_F  | 66,383    | 61    | ✅ rich  |
| 34_F  | 5,366     | 46    | ✅       |
| 35_F  | 42,627    | 111   | ✅ rich  |
| 3_M   | 0         | 53    | ❌ no app events |
| 4_F   | 0         | 38    | ❌ no app events |
| 5_F   | 22,722    | 55    | ✅       |
| 6_M   | 12,502    | 57    | ✅       |
| 7_F   | 41,874    | 49    | ✅       |
| 8_M   | 13,169    | 52    | ✅       |
| 9_M   | 1,129     | 21    | ⚠ sparse |

**Excluded:** 3_M (0 app events), 4_F (0 app events), 25_F (455 events — < 500 threshold)  
**Sparse (include with warning):** 10_M (1,073 events), 9_M (1,129 events)  
**Total usable users: 32**

---

## Transition Reconstruction

**Can app-to-app transitions be reconstructed?**

**YES.** The `Application` event schema provides:
- `ProcessName` — unique app identifier
- `Start` — session start timestamp (parseable)
- `End` — session end timestamp

**Algorithm:**
1. Extract all `Application` events per user across all daily files
2. Sort chronologically by `Start` timestamp
3. Filter zero-duration sessions and known system services
4. Consecutive events form transitions: `event[i].ProcessName → event[i+1].ProcessName`
5. Constraint: only count transitions where gap between `event[i].End` and `event[i+1].Start` < MAX_GAP (e.g., 1 hour)

**Complication:** Overlapping sessions (background services). Resolution: take the foreground app (shortest session duration or most recent start) when multiple apps run simultaneously.

---

## Model Training Feasibility

### Markov-1 (First-Order Markov Chain)

**✅ FEASIBLE**

Requirements: sequence of app transitions  
Available: 820,603 Application events → estimated ~600K–750K clean transitions after filtering  
Minimum per user: ~1,073 events (10_M), well above minimum of ~200 needed  
All 32 usable users support Markov-1 training.

### Markov-2 (Second-Order Markov Chain)

**✅ FEASIBLE**

Requirements: bigram history of transitions (A→B→C)  
Available: sufficient volume for all users with > 5,000 events  
Sparse users (10_M: 1,073, 9_M: 1,129): Markov-2 matrices will be thin but trainable  
**Recommendation:** For users < 2,000 events, fall back to Markov-1 for Markov-2 cells with zero counts

### GraphMind (Behavioural Graph)

**✅ FEASIBLE**

Requirements: app transitions + contextual features  
Available context: Activity (physical), WiFi (location proxy), Call/SMS (social), Bluetooth (proximity)  
Context richness varies by user:
- 2013 Iranian cohort: rich Location, Activity, WiFi, SMS data
- 2014 US cohort (1_M, 2_F): WiFi + Bluetooth + Activity only

**Note:** No `battery` or `time_bucket` fields in raw UbiqLog data — these must be derived:
- `time_bucket` = `Start.hour * 2 + (Start.minute // 30)` → 0–47 bucket
- `battery` — NOT AVAILABLE in UbiqLog; use constant 100.0 or omit

### Confidence Prefetch

**✅ FEASIBLE**

Requirements: transition probabilities + recency/frequency tracking  
Available: all Application events with timestamps  
Time-bucket context: derivable from `Start` timestamps  
The confidence scorer uses `transition_prob`, `recency`, `frequency`, `context` — all computable from the UbiqLog data.

### RL (ResourceAllocationPolicy)

**✅ FEASIBLE with caveats**

Requirements: episodic event stream with cache hit/miss feedback  
Available: 32 users with sufficient event streams for episode generation  
**Caveat:** Battery level not available — RL observation vector must omit or zero-fill battery dimension.  
**Caveat:** GraphMindEnvV2 designed for synthetic events with `battery` field — adapter required.  
**Recommendation:** Set `battery=100.0` as constant for all UbiqLog events during RL training.

---

## What Information is Missing

| Information           | Missing | Impact                              | Workaround                             |
|-----------------------|---------|-------------------------------------|----------------------------------------|
| Battery level         | ❌      | RL observation; MemoryManager       | Use constant 100.0                     |
| Warm start latency    | ✅ FIXED | LatencyModel                       | Now available in dataset               |
| App category taxonomy | Partial | Classification/security model       | Map known packages; unknown → `utility`|
| Screen on/off         | ❌      | Cannot distinguish idle vs active   | Treat all sessions as foreground       |
| Headphones state      | ❌      | Context feature                     | Use constant False                     |
| Calendar events       | ❌      | Context feature                     | Use constant None                      |
| Foreground flag       | ❌      | Overlap resolution                  | Use heuristic: shortest concurrent session |
| Device ID / user demographics | Partial | Only ID+gender in folder name  | Parse from directory name              |

---

## Final Verdict

| Task                | Feasible | Confidence |
|---------------------|----------|------------|
| Markov-1 training   | ✅ Yes   | High       |
| Markov-2 training   | ✅ Yes   | High       |
| GraphMind training  | ✅ Yes   | High       |
| Confidence Prefetch | ✅ Yes   | High       |
| RL training         | ✅ Yes   | Medium (battery missing) |
| Benchmark evaluation| ✅ Yes   | High       |
| Statistical analysis| ✅ Yes   | High (32 users = good n) |

**The dataset is fully usable for the complete GraphMind v2 benchmark pipeline.**
