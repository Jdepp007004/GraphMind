# Changelog

All notable changes to GraphMind are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [V5 -- Production] -- 2025-06-07 (Samsung EnnovateX AX Hackathon 2025)

### 🏆 Official Result
**F1 = 0.7745 · p = 0.0115 · Cohen's d = 0.491 · Cache Hit Rate = 93.1%**

### Added
- **GraphMindRL V5** -- production-grade RL-enhanced Markov prefetch engine
- **Adaptive confidence threshold** -- RL controller adjusting threshold in real time based on rolling 20-step hit rate
- **Two-tier cache architecture** -- HOT (5 apps, 0ms) and WARM (15 apps, ~200ms) tiers
- **Confidence scoring formula** -- fuses Markov transitions (0.5), recency (0.1), and frequency (0.4)
- **7-page Next.js 15 dashboard** -- full interactive analytics suite
  - Executive Overview (`/`)
  - Benchmark Explorer (`/benchmark`)
  - Optimization Journey (`/journey`)
  - Graph Explorer (`/graph`)
  - Cache Simulator (`/simulator`)
  - User Playback (`/playback`)
  - Research Validation (`/research`)
- **Multi-agent orchestration** -- 6 specialized agents (Prefetch, RLTrainer, GraphManager, DriftDetector, Security, Orchestrator)
- **Full documentation suite** -- architecture, benchmarking, reproducibility, models, datasets, user guide

### Changed
- Replaced static threshold (0.15) with RL-adaptive threshold (base 0.16, ±0.005)
- Upgraded confidence weights from (0.5, 0.2, 0.3) to (0.5, 0.1, 0.4) -- Phase 11A weight grid result
- Redesigned dashboard from dark AI aesthetic to clean Figma/Notion-style

### Fixed
- Graph edge weight normalization for users with sparse transition histories
- Unknown app handling to prevent security boundary violations
- Memory leak in graph engine edge cleanup on long sessions

### Removed
- Superseded intermediate results (V1–V4 raw benchmarks archived)
- Stale app_launch_latency.csv dataset (superseded by latency_statistics.csv)

---

## [V4 -- Candidate] -- 2025-05-20

### Added
- `RL_LatencyFocus` policy -- optimized for latency reduction (F1 = 0.7539)
- Full Phase 11 benchmark suite with 8 hypothesis-test-decision cycles
- Statistical significance testing: paired t-test + Cohen's d for all policies
- Device validation module for Samsung Galaxy A-series device profiles
- Advanced benchmark fairness tests

### Changed
- Moved to chronological 80/10/10 train/val/test split (from random split)
- Increased test user count to 31 (from 28, after quality filtering)

---

## [V3 -- Iteration 3] -- 2025-05-05

### Added
- Credibility hardening: benchmark provenance tracking
- Schema validation for all event types
- Graph normalization for transition probability matrices
- Security hardening: retention policies for unknown apps
- Graph scalability and stress testing suite
- Benchmark reproducibility suite
- Device validation tests

---

## [V2 -- Iteration 2] -- 2025-04-20

### Added
- Samsung telemetry ingestion layer (ADB connector, usage stats collector)
- Prediction reasoning engine with explainability module
- Graph evolution playback system
- Security visualization pipeline
- Adaptive drift analytics
- CLI wizard for Samsung device onboarding
- Advanced evaluation suite (leave-one-persona-out cross-validation)
- Dynamic top-k prefetch study

---

## [V1 -- Iteration 1] -- 2025-04-05

### Added
- Core graph engine with weighted directed Markov graph (NetworkX)
- Three-tier memory manager (HOT/WARM/COLD)
- Event bus for agent communication
- PPO reinforcement learning environment (Gymnasium + Stable-Baselines3)
- Data pipeline: UbiqLog CSV loader and transition sequence extractor
- Context encoder for time-of-day and temporal features
- Event simulator for synthetic data generation
- 5 LangGraph agents with orchestration layer
- Prefetch daemon with APScheduler integration
- Full test suite (30 test files, pytest)
- Streamlit live monitoring dashboard (replaced by Next.js in V5)
- Baseline comparisons: Markov-1, Markov-2, LFU, LRU

---

*GraphMind -- Samsung EnnovateX AX Hackathon 2025*
