# GraphMind Repository Cleanup Audit Report

**Date:** 2026-06-06  
**Auditor:** GraphMind V5 Freeze Review  
**Status:** Backend frozen. F1=0.7745 confirmed reproducible.

---

## Legend

- 🔴 **P0 — Must Fix** (blocks submission or causes demo failure)
- 🟡 **P1 — Should Fix** (judge-visible hygiene issues)
- 🟢 **P2 — Nice To Have** (cosmetic)

---

## 1. Repository Structure Audit

### 🔴 P0 — Dead / Unused Files

| File | Issue |
|------|-------|
| `scripts/run_v5_rl_graph.py` | **Moved to archive** ✅ (was failed GraphRL EdgeScorer experiment, F1~0.45) |
| `results/ppo_training_metrics.csv` | 160KB of PPO training curves from abandoned PPO approach — not referenced by dashboard or any kept script |
| `results/training_curves.json` | JSON duplicate of above |
| `results/device_report.json` | Device Analyzer era device report — `{"device": "synthetic"}` — not real |
| `data/cold_graph.db` | 77MB SQLite — synthetic Device Analyzer era graph, never used in UbiqLog pipeline |
| `data/synthetic/` | Synthetic Device Analyzer user data — entire directory obsolete |
| `data/base_graphs/` | Old synthetic graph snapshots — never used in production |

### 🟡 P1 — Duplicate / Redundant Files

| File | Issue |
|------|-------|
| `reports/user_summary.csv` + `reports/user_ranking.csv` + `reports/top_users.csv` | Should live in `data/processed/`, not `reports/`. Duplicated in reports. |
| `results/v5_rl_ablation.csv` | Superseded by `v5_all_experiments.csv` which is more complete |
| `results/v5_graph_study.csv` | Superseded by `v5_all_experiments.csv` |
| `results/v5_temporal_decay.csv` | Superseded by `v5_all_experiments.csv` |
| `results/v5_time_context.csv` | Superseded by `v5_all_experiments.csv` |
| `results/v5_order_analysis.csv` | Superseded by `v5_all_experiments.csv` |
| `GRAPHMIND_BUILD_SPEC.md` | 80KB planning doc — not needed for submission |
| `GRAPHMIND_FULL_AUDIT_REPORT.md` | 76KB old audit — superseded by `reports/graphmind_architecture_audit.md` |
| `samsung_everything_doc.md` | 81KB scratchpad — internal notes, not for judges |

### 🟡 P1 — Orphaned / Temporary Reports

| File | Issue |
|------|-------|
| `reports/temporal_decay_study.md` | Experiment that failed — move to `archive/old_reports/` |
| `reports/context_plus_order_study.md` | Moved to archive ✅ already |
| `reports/performance_review_v4.md` | Superseded by `final_production_report.md` |
| `reports/rl_ablation_v5.md` | Inline data — dashboard should reference `v5_all_experiments.csv` instead |

### 🟢 P2 — Generated Files That Should Not Be in Git

| File | Issue |
|------|-------|
| `results/snapshots/` | Model snapshot binaries — `.gitignore` already excludes pkl but check dir |
| `logs/` | Runtime log files — check if in `.gitignore` |
| `models/rl_policies/` | PPO model weights — already in `.gitignore` ✅ |

---

## 2. Python Code Audit

### 🔴 P0 — Production Safety Issues

#### `src/benchmarks/graphmind_policy_runner.py`
Imports `battery` parameter handling:
```python
# score_candidates() still accepts battery: float = 100.0
# but battery is NOT AVAILABLE in UbiqLog — always passed as 100.0
# Must NOT drive any logic in the policy runner
```
**Risk:** If battery parameter is used for branching in demo, it creates a mismatch.

#### `src/benchmarks/latency_model.py`
Contains old **literature values** in comments that conflict with the real Samsung A23 measurements:
```python
# These should be clearly marked as SUPERSEDED by datasets/app_launch_latency.csv
```

### 🟡 P1 — Dead Classes / Functions

| File | Dead Code |
|------|----------|
| `src/rl/reward.py` | `REWARD_ALPHA..ZETA` weights in settings.py — never used by `environment_v2.py` |
| `src/agents/rl_trainer_agent.py` | `train_ppo()` call — PPO was abandoned. Node exists but function is never triggered in production flow |
| `src/benchmarks/case_study.py` | `get_case_study_user()` — not called by dashboard API |
| `src/benchmarks/provenance.py` | `LatencyProvenance` dataclass — never instantiated anywhere |
| `src/benchmarks/advanced_metrics.py` | `compute_app_level_metrics()` — reads `user_*_simulation_log.json` which are now archived |
| `src/graph_playback/snapshot_manager.py` | `save_snapshot()` / `get_snapshot()` — never called from production path |

### 🟡 P1 — Unused Imports (Key Files)

| File | Unused Imports |
|------|---------------|
| `src/benchmarks/baselines_v2.py` | `from scipy import stats` — not used in baselines themselves |
| `src/rl/environment_v2.py` | `from src.core.memory_manager import MemoryManager` — env manages its own cache |
| `src/agents/orchestrator.py` | `from src.security.sensitivity_model import SensitivityModel` — not wired in graph |

### 🟢 P2 — Experimental Code That Should Stay (src/experiments/)

All three files correctly relocated. `__init__.py` narrative is accurate. No action needed.

---

## 3. Production Safety Audit

### ✅ ALL GREEN — Configuration Verified

| Setting | Expected | Actual (`config/settings.py`) | Status |
|---------|----------|-------------------------------|--------|
| `W_TRANSITION` | 0.50 | `0.50` | ✅ |
| `W_RECENCY` | 0.10 | `0.10` | ✅ |
| `W_FREQUENCY` | 0.40 | `0.40` | ✅ |
| `W_CONTEXT` | 0.00 | `0.00` | ✅ |
| `THRESHOLD` | 0.16 | `0.16` | ✅ |
| Adaptive threshold in `confidence_prefetch.py` | present | Added `_hit_history`, `_ADAPT_STEP=0.005` | ✅ |
| `PREFETCH_TOP_K` (HOT_SIZE) | 5 | `5` | ✅ |
| `WARM_TIER_CAPACITY` | 15 | `150` | ⚠️ |

> ⚠️ **P1 WARNING:** `WARM_TIER_CAPACITY = 150` in settings.py vs `WARM_SIZE = 15` in benchmark scripts.
> The benchmark used WARM=15. The production daemon may over-provision. Update `settings.py` to match.

---

## 4. Git Hygiene Audit

### ✅ Passing

| Check | Status |
|-------|--------|
| `.gitignore` excludes `venv/` | ✅ |
| `.gitignore` excludes `__pycache__/` | ✅ |
| `.gitignore` excludes `archive/` | ✅ (added 2026-06-06) |
| `.gitignore` excludes `__MACOSX/` | ✅ (added 2026-06-06) |
| `.gitignore` excludes `*.pkl` in graphs/ and markov/ | ✅ |
| No `.env` secrets committed | ✅ (`.env.example` is safe) |
| `data/cold_graph.db` in `.gitignore` | ✅ (`data/cold_graph.db`) |

### 🟡 P1 — Issues

| Issue | Detail |
|-------|--------|
| `data/cold_graph.db` **is in `.gitignore` but may still be tracked** | Run `git ls-files data/cold_graph.db` to confirm. 77MB binary should NOT be in git history |
| `logs/` directory not in `.gitignore` | Runtime logs may accumulate |
| `results/ppo_training_metrics.csv` (160KB) | Large CSV committed — not needed for submission |
| `results/training_curves.json` (7.4KB) | PPO artifact — not needed |

### 🟢 P2

| Issue | Detail |
|-------|--------|
| No absolute paths in committed Python files | ✅ All use `os.path.join(PROJECT_ROOT, ...)` pattern |
| No API keys detected | ✅ |
| `.env.example` is clean template | ✅ |

---

## 5. Dashboard Readiness Audit

### 🔴 P0 — Missing (Will Be Built)

| Component | Status | Note |
|-----------|--------|------|
| Clean API layer | ❌ MISSING | No `/api/` routes exist |
| Benchmark result loader | ❌ MISSING | `final_production_results.csv` not connected |
| Graph loader | ❌ MISSING | No Markov-to-JSON conversion |
| Cache simulator data source | ❌ MISSING | No transition stream API |
| Playback data source | ❌ MISSING | No per-user sequence endpoint |
| Dashboard pages (all 7) | ❌ MISSING | Default Next.js template only |

**Current dashboard state:** Default `create-next-app` template. No content. No components.  
**Required action:** Complete rebuild (started in parallel with this audit).

---

## 6. Recommended Fix Order

### P0 — Must Fix Before Submission

1. ✅ Production config matches benchmark (already done)
2. ✅ `archive/` excluded from git (already done)
3. 🔴 Build dashboard (in progress)
4. 🔴 Fix `WARM_TIER_CAPACITY = 150` → `15` in settings.py
5. 🔴 Verify `data/cold_graph.db` is NOT tracked by git (77MB binary)

### P1 — Should Fix

6. Remove `results/ppo_training_metrics.csv` and `results/training_curves.json`
7. Remove `results/device_report.json`
8. Remove `data/cold_graph.db` from git history if tracked
9. Add `logs/` to `.gitignore`

### P2 — Nice To Have

10. Consolidate `v5_rl_ablation.csv` / `v5_graph_study.csv` / etc. into `archive/`
11. Move `reports/user_summary.csv` to `data/processed/`
12. Remove internal planning docs (`GRAPHMIND_BUILD_SPEC.md`, `samsung_everything_doc.md`)
