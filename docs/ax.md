# AX Methodology — Agentic Development Workflow

> **GraphMindRL V5 — Samsung EnnovateX AX Hackathon 2025**
> This document describes the agentic AI-assisted development methodology used to design, implement, evaluate, and validate the GraphMindRL V5 prefetch system.

---

## Table of Contents

1. [Overview](#overview)
2. [Agentic Development Philosophy](#agentic-development-philosophy)
3. [Research Planning Process](#research-planning-process)
4. [The Research Loop](#the-research-loop)
5. [Benchmark Automation](#benchmark-automation)
6. [Decision Gates](#decision-gates)
7. [Reasoning Workflow](#reasoning-workflow)
8. [Statistical Validation Workflow](#statistical-validation-workflow)
9. [Tool Use](#tool-use)
10. [Context Management](#context-management)
11. [Experiment Memory](#experiment-memory)
12. [Architecture Iteration Process](#architecture-iteration-process)
13. [What Worked](#what-worked)
14. [What Failed](#what-failed)
15. [Lessons Learned](#lessons-learned)
16. [Future Work](#future-work)

---

## Overview

GraphMindRL V5 was developed using an agentic development workflow in which an AI assistant operated as a peer engineer — responsible for research planning, experiment design, implementation, benchmarking, statistical validation, and architectural decision-making — under human supervision and final approval.

This document describes the methodology, tools, decision logic, and lessons learned from that process. It is intended both as a technical record and as a demonstration of AX (Agentic eXperience) development practice for the Samsung EnnovateX AX Hackathon.

The key thesis of the project's development methodology is:

> **Every architectural decision must be preceded by a falsifiable hypothesis, implemented as a benchmark, and accepted or rejected based on statistical evidence.**

This distinguishes AX development from ad-hoc "try things and see" engineering. The agent maintained a structured experiment log, enforced decision gates before any change was committed to the production configuration, and explicitly documented failures alongside successes.

---

## Agentic Development Philosophy

### Role of the AI Agent

The AI agent in this project operated at the level of a **senior research engineer**. Specifically:

- It maintained the research state across multiple sessions (experiment log, current best model, baseline performance).
- It proposed, justified, implemented, and evaluated its own experiments.
- It generated all benchmark scripts, statistical analysis code, and documentation.
- It enforced production discipline: the production configuration was declared frozen early and the agent refused to modify it without explicit human approval and new benchmark evidence.

### Human-in-the-Loop

Human oversight was maintained at two critical junctures:

1. **Hypothesis approval**: Before implementing any non-trivial experiment, the agent presented the hypothesis, expected ΔF1, implementation complexity, and risk to the human for a go/no-go decision.
2. **Production freeze**: The human explicitly approved the production freeze before the dashboard phase began, ensuring no further model experimentation would interfere with the submission.

### Principles Observed

| Principle | Implementation |
|---|---|
| Evidence-first | No change to production config without benchmark evidence |
| Explicit failure logging | Rejected hypotheses are documented, not silently discarded |
| Reproducibility by default | Every benchmark script is deterministic; seeds are fixed |
| Statistical conservatism | Significance threshold α = 0.05; effect size (Cohen's d) required |
| Separation of concerns | Model research and dashboard development were strictly separated |

---

## Research Planning Process

The research planning process followed a structured funnel:

### Phase 0 — Architecture Audit

Before any modelling, the agent conducted a full architecture audit to verify the mathematical equivalence of the existing components. The key finding was that the existing `GraphOnly` model was mathematically identical to a first-order Markov chain (Markov-1): both compute P(next | current) and select the top-k candidates by probability. This established the true baseline.

### Phase 1 — Baseline Establishment

A rigorous baseline was established by benchmarking all existing policies on the exact evaluation protocol (80/10/10 chronological split, 31 users, paired t-test against Markov-1). This gave:

- Markov-1 (GraphOnly): F1 = 0.7267
- GraphMindRL Baseline: F1 = 0.7424

The GraphMindRL Baseline became the reference point for all subsequent experiments.

### Phase 2 — Hypothesis Generation

The agent generated a prioritised hypothesis backlog ranked by:

1. Expected ΔF1 (estimated from literature and intuition)
2. Implementation complexity (hours of work)
3. Risk of interference with existing good signals

The initial backlog contained 12 hypotheses. Five were selected for immediate testing based on the above criteria.

### Phase 3 — Experiment Prioritisation

Experiments were sequenced so that:
- Fast-to-implement experiments ran first (enabling rapid learning)
- Risky or expensive experiments ran only if simpler approaches had been exhausted
- Experiments with high expected information value ran before low-value ones

---

## The Research Loop

The core research loop executed for each hypothesis:

```
1. HYPOTHESIS
   State the hypothesis formally:
   "Changing X will improve F1 by approximately Y because Z."

2. IMPLEMENTATION
   Write the benchmark script.
   Do NOT modify production code.
   Create an isolated experiment variant.

3. BENCHMARK
   Run the benchmark script.
   Record: F1, precision, recall, hit rate, latency saved, p-value, Cohen's d.

4. ANALYSIS
   Compare against baseline.
   Apply paired t-test (n=31 users).
   Compute Cohen's d.

5. DECISION
   ACCEPT if: ΔF1 > 0 AND p < 0.05 AND Cohen's d > 0.2
   REJECT if: ΔF1 ≤ 0 OR p ≥ 0.05 OR Cohen's d ≤ 0.2

6. INTEGRATION (if ACCEPTED)
   Merge the change into the production configuration.
   Update the baseline.
   Document the improvement.

7. ARCHIVAL
   Whether accepted or rejected, archive:
   - The benchmark result CSV
   - The analysis report
   - The decision rationale
```

This loop was executed **8 times** across 5 major research phases, consuming approximately 40 hours of compute and development time.

---

## Benchmark Automation

All benchmarks were implemented as standalone Python scripts in `scripts/`. Each script:

- Is **self-contained**: no dependency on interactive state or Jupyter notebooks.
- Is **deterministic**: fixed random seeds, deterministic data splits.
- Produces **structured CSV output** suitable for automated analysis.
- Includes **statistical testing** (paired t-test) inline.

The canonical benchmark entry point is:

```bash
python scripts/run_phase11_e.py
```

This script evaluates 9 policies, computes per-user F1 scores, runs paired t-tests against the baseline, and outputs a final comparison CSV. The script was written to be runnable in under 5 minutes on any modern laptop.

### Benchmark Design Principles

**Chronological splitting**: The 80/10/10 split is always chronological — training data comes from earlier events, test data from later events. Random splitting would leak future information into the training period and inflate all metrics.

**Per-user evaluation**: F1 is computed per user and then averaged. This is more honest than pooling all events because it prevents heavy users from dominating the metric.

**Paired statistical testing**: Because each user appears in both conditions (baseline and experimental), a paired t-test is the correct test. It accounts for between-user variability and tests whether the experimental condition is better *for the same users*.

**Two-metric requirement**: Accepting a hypothesis required both p < 0.05 (statistical significance) and Cohen's d > 0.2 (effect size). This guards against statistically significant but practically negligible improvements.

---

## Decision Gates

Three formal decision gates were enforced during the project:

### Gate 1 — Pre-Experiment Gate

Triggered before any experiment is implemented. The agent must answer:

1. What is the hypothesis?
2. What is the expected ΔF1 and why?
3. What is the implementation complexity?
4. What is the risk to the existing production config?
5. What would constitute a negative result?

If the human is not satisfied with these answers, the experiment is queued but not started.

### Gate 2 — Post-Benchmark Gate

Triggered after every benchmark run. The agent must answer:

1. What was the result?
2. Does it meet the acceptance criteria (p < 0.05, Cohen's d > 0.2, ΔF1 > 0)?
3. Is the result reproducible (second run)?
4. Should this be merged into the production config?

If accepted: update production config, update baseline, archive result.
If rejected: archive result, document failure reason, do not modify production config.

### Gate 3 — Production Freeze Gate

Triggered when the agent determines that the current production config is ready for the dashboard phase. The agent presents:

1. Current best F1 and statistical evidence.
2. Remaining untested hypotheses and their expected value.
3. Recommendation: continue experimenting or freeze.

Human must explicitly approve the freeze. After freeze, **no model, config, or benchmark modification is allowed**.

The freeze was approved at F1 = 0.7745 on 2026-06-06, after the agent determined that remaining hypotheses had insufficient expected value to justify the risk of destabilising a statistically significant result.

---

## Reasoning Workflow

The agent's reasoning process for each non-trivial decision followed a structured chain:

### Decision: Accept or reject a hypothesis

```
OBSERVATION: [What the benchmark showed]
   ↓
COMPARISON: [Versus the current baseline]
   ↓
STATISTICS: [p-value, Cohen's d interpretation]
   ↓
MECHANISM: [Why does this work or fail? Is there a causal explanation?]
   ↓
RISK ASSESSMENT: [Could accepting this harm other users or future experiments?]
   ↓
DECISION: [ACCEPT / REJECT / INVESTIGATE FURTHER]
   ↓
DOCUMENTATION: [Archive result + rationale]
```

### Decision: What to try next

```
CURRENT STATE: [Best F1, remaining hypotheses]
   ↓
EXPECTED VALUE: [ΔF1 × probability of success for each hypothesis]
   ↓
OPPORTUNITY COST: [What is the cost of trying this vs. doing something else?]
   ↓
MARGINAL RETURN: [Is there likely any gain left from further model work?]
   ↓
PRIORITISATION: [Ordered experiment queue]
```

This explicit reasoning chain was maintained throughout the project and is reflected in the decision gate documents in `reports/`.

---

## Statistical Validation Workflow

### 1. Per-User Metric Computation

For each user u and policy π:

```
precision_u(π) = TP_u / (TP_u + FP_u)
recall_u(π)    = TP_u / (TP_u + FN_u)
F1_u(π)        = 2 × precision_u × recall_u / (precision_u + recall_u)
```

Where:
- TP = predicted app that the user actually opened next
- FP = predicted app that the user did not open next
- FN = app the user opened that was not predicted

### 2. Paired t-Test

For each user u, compute `d_u = F1_u(V5) - F1_u(Baseline)`.

```
t = mean(d) / (std(d) / sqrt(n))
p = 2 × P(T > |t|)   [two-tailed]
```

With n = 31 users:
- t = 2.681
- p = 0.0115

### 3. Effect Size

```
Cohen's d = mean(d) / std(d) = 0.491
```

Interpretation: medium-to-large effect. The improvement is not only statistically significant but practically meaningful.

### 4. Reproducibility

The benchmark was run twice on identical hardware and software:

| Run | Date | F1 |
|-----|------|----|
| Run 1 | 2026-06-06 09:39 | 0.7745 |
| Run 2 | 2026-06-06 10:00 | 0.7745 |

Identical results confirm that the benchmark is deterministic and the result is not a statistical artefact.

---

## Tool Use

The agent used the following tools during the development process:

| Tool | Purpose | Frequency |
|------|---------|-----------|
| `view_file` | Reading source code and data files | Very high |
| `write_to_file` | Creating new scripts, configs, reports | High |
| `replace_file_content` | Targeted edits to existing files | High |
| `run_command` | Executing benchmark scripts and shell commands | High |
| `search_web` | Literature search, API documentation | Medium |
| `grep_search` | Finding patterns in code | Medium |
| `list_dir` | Repository structure navigation | Medium |
| `browser_subagent` | Dashboard verification and screenshot capture | Low |
| `generate_image` | Creating visual assets | Low |

Tool use was always purposeful — the agent did not issue redundant commands. Before each tool call, the agent reasoned about whether the tool was necessary and what outcome it expected.

---

## Context Management

The agent managed a long-running research context across multiple sessions. Key context management strategies:

### Knowledge Items (KIs)

Critical experiment results and architectural decisions were captured in structured knowledge items at the end of each session. These persisted across context resets and enabled the agent to pick up where it left off without re-reading the entire repository.

### Experiment Log

The file `results/v5_all_experiments.csv` served as the persistent experiment log. Every benchmark result was appended here, regardless of outcome. This meant the agent could always consult the complete history of what had been tried, preventing redundant experiments.

### Production Config as Ground Truth

The file `config/settings.py` served as the single source of truth for the production configuration. The agent never made changes to this file based on memory alone — every change required a fresh benchmark to justify it.

### Frozen State Documentation

At the production freeze gate, the agent created `reports/pre_dashboard_summary.md` (now archived) documenting the exact state of the repository, the current production config, and all remaining open questions. This document served as the handoff from research phase to dashboard phase.

---

## Experiment Memory

The agent maintained the following state throughout the project:

| State | Storage Location | Update Frequency |
|---|---|---|
| Best F1 | `config/settings.py` (implicit) | Each accepted hypothesis |
| Baseline F1 | `results/final_production_results.csv` | Fixed after freeze |
| Experiment log | `results/v5_all_experiments.csv` | Every benchmark run |
| Decision rationale | `reports/v5_decision_gate.md` | Each gate |
| Architecture state | `reports/v5_architecture_verification.md` | Each structural change |
| Hypothesis backlog | Maintained in agent context | Each session |

When context was reset between sessions, the agent re-read these files to reconstruct the research state before proposing new experiments.

---

## Architecture Iteration Process

The architecture evolved through the following stages:

### Stage 1 — Baseline Architecture (GraphOnly / Markov-1)

**Architecture**: Per-user transition probability table. Select top-k apps by P(next | current).

**Finding**: This is the simplest possible baseline. F1 = 0.7267.

**Decision**: Use as starting point. Establish evaluation harness.

### Stage 2 — Second-Order Markov (Markov-2)

**Hypothesis**: Conditioning on the last *two* apps should capture more context.

**Finding**: F1 = 0.7355. Small improvement (+0.0088) but not statistically significant vs. Markov-1.

**Decision**: Reject. The UbiqLog sessions are too short to reliably learn second-order transitions.

### Stage 3 — Confidence Scoring (Graph+Confidence)

**Hypothesis**: Weighting transition probability with recency and frequency should improve precision.

**Finding**: F1 = 0.7369. Marginal improvement over Markov-1, still below Baseline.

**Decision**: The confidence scoring idea is sound but the weights are wrong.

### Stage 4 — RL Threshold Controller

**Hypothesis**: A self-adapting threshold avoids the need for per-user threshold tuning.

**Finding**: GraphMindRL_Baseline achieves F1 = 0.7424 (+0.0157 vs Markov-1).

**Decision**: Accept. RL threshold controller becomes a permanent component.

### Stage 5 — Time Context Evaluation

**Hypothesis**: Conditioning on time-of-day should capture daily behavioural rhythms.

**Finding**: All four time granularities (6h, 2h, 1h, 30min bands) hurt F1. Coverage was 94–98%, so data sparsity was not the cause. The conditional distributions P(next | app, time_band) were too noisy on 2-month datasets.

**Decision**: Reject time context from confidence scoring. Retain in RL state representation for monitoring.

### Stage 6 — Modified Kneser-Ney Smoothing

**Hypothesis**: KN smoothing should improve transition probability estimates for rarely-seen transitions.

**Finding**: F1 = 0.7421. Not a statistically significant improvement.

**Decision**: Reject. The training data is sufficient; smoothing adds complexity without gain.

### Stage 7 — Phase 11A Weight Grid Search

**Hypothesis**: The default weights (0.5/0.2/0.2/0.1) may not be optimal.

**Finding**: Best weights are (0.5/0.1/0.4/0.0) → F1 = 0.7733. ΔF1 = +0.0309 vs baseline.

**Decision**: Accept. Update production config.

### Stage 8 — Phase 11B+E Combined Optimisation

**Hypothesis**: Combining the optimal weights with the optimal threshold (0.16) should stack.

**Finding**: F1 = 0.7745. Statistically significant (p = 0.0115). Reproducible.

**Decision**: Accept. Freeze production config. Begin dashboard phase.

---

## What Worked

### 1. Empirical discipline over intuition

The single most impactful practice was the strict requirement that every hypothesis be tested against a quantitative benchmark before being accepted. This prevented several plausible-sounding ideas (time-context, Kneser-Ney smoothing) from making it into production where they would have hurt performance.

### 2. Separating confidence signals

The decomposition of the confidence score into three orthogonal signals (transition probability, recency, frequency) proved highly productive. Each signal captures a different aspect of user behaviour that the others miss. The frequency signal in particular was underweighted in the initial architecture.

### 3. Adaptive threshold

The RL-based adaptive threshold was the single most important innovation. It allowed the system to self-calibrate per user, eliminating the need for per-user threshold tuning while simultaneously improving the precision/recall balance.

### 4. Chronological evaluation

Insisting on chronological train/test splits from the beginning ensured that the evaluation protocol matched the real deployment scenario. This prevented optimistic metric inflation that would have led to incorrect conclusions.

### 5. Statistical testing as a default

Requiring p < 0.05 and Cohen's d > 0.2 for every accepted hypothesis eliminated several results that would have been accepted under a simpler "is F1 higher?" decision rule — and correctly so, as those results were not reproducible.

### 6. Production freeze discipline

Explicitly freezing the production config at a well-defined point prevented scope creep and guaranteed that the dashboard phase could proceed without the model changing under it.

---

## What Failed

### 1. Time-context features

The most significant failed hypothesis was that time-of-day features would improve prediction. The intuition was strong — people open different apps at different times — and the coverage was high (94–98%). But the conditional distributions were too noisy on 2-month datasets.

**Lesson**: More features are not always better. Noisy features can actively hurt a well-calibrated model.

### 2. Markov-2 (second-order transitions)

Second-order Markov chains carry more information in theory but require more data to estimate reliably. On UbiqLog sessions that average only a few hundred transitions per user, the second-order transition table is too sparse to be useful.

**Lesson**: Model complexity must be matched to data volume.

### 3. Kneser-Ney smoothing

This failed for a different reason: the first-order Markov tables were already well-estimated with the available data. Smoothing is most useful when many transitions are unseen; here, nearly all common transitions were seen in training.

**Lesson**: Validate the premise of an optimisation before implementing it. Check whether the problem it solves actually exists in your data.

### 4. PPO reinforcement learning

An early attempt to use Proximal Policy Optimisation (PPO) as the prefetch policy was abandoned. PPO requires substantially more training data and time to converge than was available, and the simpler adaptive threshold approach outperformed it decisively.

**Lesson**: Complex RL is not always better than a well-designed heuristic controller.

### 5. Cluster-based Markov

A variant that clustered similar apps and computed transitions at the cluster level failed to improve F1. App categories (maps, social, productivity) are too coarse to capture the fine-grained sequential patterns in UbiqLog.

**Lesson**: Domain abstractions that seem natural do not always align with the patterns in the data.

---

## Lessons Learned

### On agentic development

1. **Explicit state management is critical.** An AI agent operating across multiple sessions needs explicit, persistent records of what has been tried, what worked, and what the current production state is. Relying on agent memory alone is insufficient.

2. **Decision gates prevent premature optimism.** Without formal gates, it is tempting to accept small improvements or commit changes based on a single run. Gates enforce reproducibility and statistical rigour as defaults.

3. **Separation of research and production is essential.** All experiments were implemented as isolated scripts that did not touch the production configuration. This made it safe to try risky ideas without jeopardising the best known result.

4. **Document failures explicitly.** Failed experiments are as valuable as successes. They constrain the hypothesis space and prevent future re-investigation of dead ends.

5. **Human approval at key decision points.** The human's role was not to micromanage individual commands but to approve major transitions: hypothesis prioritisation, production freeze, dashboard launch. This kept the human in the loop without creating unnecessary friction.

### On the problem

1. **Behavioural data is noisy.** Smartphone app sequences contain substantial noise from background processes, notifications, and accidental touches. Robust models must focus on strong signals.

2. **Frequency is underrated.** In most recommendation literature, recency is the dominant temporal signal. In app prefetching, raw frequency proved more predictive than recency on UbiqLog — possibly because people's app usage habits are more stable than their browsing habits.

3. **Short sequences limit model complexity.** With 2 months of data and ~200 transitions per user in the test set, models that require many parameters (Markov-2, neural networks) are at a disadvantage relative to simple, well-calibrated models.

---

## Future Work

The following directions are promising but were not pursued within the hackathon timeline:

### 1. Longitudinal context features

Time-of-day and day-of-week features failed on 2-month datasets. With 12+ months of data, the conditional distributions should stabilise. A future version could implement a data-length gate: only activate context features when the user has at least N months of history.

### 2. Session-aware Markov chains

The current model treats all transitions equally. Modelling within-session transitions separately from between-session transitions (where the user has been away from the phone for a significant time) could capture more nuanced patterns.

### 3. On-device learning

The current model is trained offline and deployed. An on-device learning variant could incrementally update the transition probabilities as the user's behaviour evolves, without requiring a central server.

### 4. Multi-user transfer learning

The current model is purely per-user. A transfer learning approach could initialise each user's model from a population-level prior, which would be especially valuable for new users with sparse history.

### 5. Confidence calibration

The current confidence score is not calibrated to output true probabilities. Platt scaling or isotonic regression could calibrate the scores, which would make the threshold more interpretable and potentially improve the precision/recall trade-off.

### 6. Neural sequence models

With substantially more data (e.g., 6–12 months per user), a lightweight sequence model (e.g., a small LSTM or Transformer) trained per-user might outperform the Markov approach. The AX development workflow would be directly applicable: define the benchmark protocol, implement the model as an isolated script, benchmark against the frozen Markov baseline.

---

*This document describes the agentic development methodology used in GraphMindRL V5. All experimental results referenced here are recorded in `results/v5_all_experiments.csv` and individual phase CSVs. All architectural decisions are documented in `reports/`.*
