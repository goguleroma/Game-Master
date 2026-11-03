# Simulated Games for AI Agent Training: Deficits, Bugs, and the Sim-to-Real Gap — A Case Study of the Game Master Project

**Authors:** Game Master Research Team  
**Date:** November 2026  
**Repository:** https://github.com/Game-Master  
**Status:** Research Paper v1.0

---

## Abstract

This paper presents a comprehensive analysis of simulated game environments as training grounds for autonomous AI agents, using the Game Master (GM) project as a primary case study. Through 38 controlled experiments spanning grid-based navigation, control learning, episodic memory, neural policy training, and strategy search, we identify fundamental deficits that limit the transferability of skills learned in simulation to real games. Key findings include: (1) BFS-based planners are optimal for grid navigation but cannot generalize to partially observable, stochastic, or continuous domains; (2) frontier-based exploration saturates quickly and fails under distribution shift; (3) conservative episodic retrieval improves efficiency but cannot compensate for planner limitations; (4) DQN policies collapse during fine-tuning due to representation mismatches and sparse reward signals; (5) order-dependent learning in persistent memory creates non-reproducible outcomes; (6) no cross-game transfer has been demonstrated. We situate these findings within the broader literature on sim-to-real transfer, procedural generation generalization, and causal confusion in reinforcement learning. We conclude that the current simulated adapter has reached its empirical limit and that Phase 4—real perception (MiniGrid) combined with PathTree/MCTS-Lite planning with belief state—is the necessary architectural evolution.

---

## 1. Introduction

### 1.1 Motivation

The promise of general game-playing agents—systems that can observe, understand, and play arbitrary video games without per-title engineering—has driven research from the Arcade Learning Environment (Bellemare et al., 2013) to ProcGen (Cobbe et al., 2020) and beyond. Yet the *sim-to-real gap* remains a fundamental barrier: agents that master simulated environments frequently fail when deployed on real games due to differences in visual complexity, control latency, stochasticity, and causal structure.

The Game Master (GM) project takes a different approach: rather than training end-to-end on pixels, it builds a hybrid architecture with explicit modules for perception, world modeling, memory, planning, and control learning. The `simulated_game/` adapter serves as a controlled testbed for developing and validating these modules before connecting to real games.

### 1.2 Contributions

This paper makes the following contributions:

1. **Taxonomy of GM's simulated environment** — We characterize the `GridGame` adapter as a partially observable, procedurally dynamic, multi-objective grid world with anonymous controls.
2. **Empirical deficit catalog** — We synthesize 38 experiments documenting specific failure modes: planner saturation, exploration collapse, memory order-dependence, neural policy collapse, and zero transfer.
3. **Literature grounding** — We connect each deficit to established results in sim-to-real transfer, generalization in RL, and causal reasoning.
4. **Architectural diagnosis** — We show why the current architecture (BFS + frontier ranking + conservative retrieval) has reached its empirical limit.
5. **Phase 4 prescription** — We specify the required evolution: real perception + PathTree/MCTS-Lite planning with belief state.

---

## 2. The Game Master Simulated Environment

### 2.1 GridGame Specification

The `GridGame` class (`simulated_game/game.py`) implements a 30×30 grid world with the following properties:

| Property | Specification |
|----------|---------------|
| **Grid size** | 30×30 cells |
| **Vision radius** | Euclidean radius 4 (partial observability) |
| **Agent start** | Fixed at (0, 1) |
| **Target** | Random free cell at Manhattan distance ≥ 4; moves every 2 agent actions |
| **Obstacles** | 1 moving obstacle (relocates every action) + static obstacles from defeated entities |
| **Pursuer** | Enemy at (7, 1) with vision radius 4; activates within range, pursues via BFS |
| **Combat** | 10 HP each; attack deals 5 damage (rebalanced from 3 in Exp 35); defend negates; entity retreats after attacking |
| **Controls** | 6 anonymous inputs: `control_1`–`control_4` = cardinal moves, `control_5` = attack, `control_6` = defend |
| **Observation** | Structured dict with `objects`, `controls`, `hud` (position, health, visible cells, target if visible, obstacles, entities), `goal`, `confidence` |

**Key design decisions:**
- Controls are *anonymous*: the agent receives `control_1`–`control_6` without semantic labels
- Target mobility: moves every 2 turns to a random free cell
- Partial observability: target and obstacles outside vision radius are omitted from observation
- No reward signal provided; agent must infer progress from state changes

### 2.2 Two Evaluation Scenarios

The benchmark (`simulated_game/benchmark.py`) defines two scenarios:

| Scenario | Target | Pursuer | Purpose |
|----------|--------|---------|---------|
| **Static** | Fixed | Disabled | Isolate navigation + control learning |
| **Dynamic** | Moves every 2 turns | Enabled | Full complexity: moving target, pursuer, combat |

All experiments use paired A/B methodology: same seeds, same order, volatile (fresh memory each episode) vs. persistent (shared SQLite database).

---

## 3. Architecture Overview

### 3.1 Hybrid Architecture

GM follows a hybrid architecture (per `ARCHITECTURE.md` and `DECISIONS.md` Decision 2):

```
┌─────────────────────────────────────────────────────────────┐
│                      GameEngine (Core)                       │
│  - Injects: Policy, ActionExecutor, ObservationAdapter      │
│  - Owns: Generic memory (STM/MTM/LTM/GSM/GM)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌────────────────┐ ┌──────────────┐ ┌────────────────┐
│ Perception     │ │ Decision     │ │ Control        │
│ (mock/stub)    │ │ (Planner)    │ │ (Executor)     │
└────────────────┘ └──────────────┘ └────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌────────────────┐ ┌──────────────┐ ┌────────────────┐
│ World Model    │ │ Memory       │ │ Learning       │
│ (Grid)         │ │ (Spatial,    │ │ (Control       │
│                │ │  Episodic,   │ │  Learner,      │
│                │ │  Control)    │ │  Neural Policy)│
└────────────────┘ └──────────────┘ └────────────────┘
```

The `simulated_game/` adapter composes:
- **Planner**: BFS shortest-path + frontier ranking (`planner.py`)
- **ExplorationPolicy**: Orchestrates planner, control learner, memory, PathTree (`exploration_policy.py`)
- **Memory**: Spatial grid, episodic cases, control trials, entity tracking (`memory.py`)
- **ControlLearner** / **WorldModelLearner**: Contextual transition models, entity dynamics (`control_learner.py`)
- **NeuralPolicy**: DQN with intent encoding (`neural_policy.py`)
- **PathTreePlanner**: MCTS-Lite with particle filtering (`path_tree.py`)

---

## 4. Experimental Findings: Deficit Catalog

### 4.1 Navigation and Planning Deficits

#### 4.1.1 BFS Planner Optimality Ceiling (Experiments 5, 19, 38)

**Finding:** BFS is optimal for shortest-path navigation on a grid with known topology. In the static scenario (Exp 19, 32), both volatile and persistent memory achieve 40/40 wins. No planner improvement is possible without changing the problem structure.

**Evidence:**
- Exp 19 static: 40/40 wins both conditions, 100% directional prediction accuracy
- Exp 32 static: 40/40 volatile, 39/40 persistent (1 timeout), mean turns 74.45 vs 69.05
- Exp 38: "BFS planner: optimal for grid with partial observability; no margin for multi-step reasoning"

**Literature connection:** BFS optimality in known graphs is textbook (Cormen et al.). The deficit emerges when the graph is *unknown*, *stochastic*, or *continuous*—precisely the sim-to-real gap.

#### 4.1.2 Partial Observability Causes Planning Failure (Experiment 9)

**Finding:** With vision radius 4, when the target is outside visibility, the planner returns `wait` and the episode stalls. The agent has no exploration policy without spatial memory.

**Evidence:** Exp 9: "if target is hidden, planner chooses `wait` and game stops without progress... GM depends on known coordinates and does not yet possess exploration or useful spatial memory."

**Literature connection:** This is the classic *exploration-exploitation* dilemma in POMDPs (Kaelbling et al., 1998). The agent needs a belief state over target location, not just a point estimate.

#### 4.1.3 Dynamic Target Breaks BFS (Experiments 19, 33–37)

**Finding:** In the dynamic scenario, the target moves every 2 turns and is visible only ~6 turns/episode (Exp 19). BFS chasing the last known position fails when the target moves.

**Evidence:**
- Exp 19 dynamic: volatile 25/40 wins, persistent 22/40 wins — persistence *reduces* wins
- Exp 33: New planner branch (switch to exploration when `goal_is_dynamic` and target hidden) improves coverage 0.311 → 0.731, persistent wins 22 → 25
- Exp 36: Improvement attributed to *frontier ranking*, not dynamic goal branch
- Exp 37: Ablation shows improvement *entirely due to episodic retrieval* (ON: 137/200 wins, OFF: 110/200 wins)

**Deficit:** The planner lacks a *belief state* over target position. It either chases a stale coordinate or switches to undirected frontier exploration. No probabilistic tracking, no multi-hypothesis planning.

### 4.2 Exploration Deficits

#### 4.2.1 Frontier Ranking Saturation (Experiments 21, 36, 38)

**Finding:** Frontier ranking parameters (`visit_penalty`, `information_gain_bonus`, `goal_change_threshold`) have been evolved (Exp 21) and saturated. The best variant (Exp 21) achieved 10/20 holdout wins vs 8/20 baseline but with 6 paired regressions — not promoted.

**Evidence:**
- Exp 21: 14 variants, 5 training seeds, 20 holdout seeds. Best: `visit_penalty=0.074`, `information_gain_bonus=0`, `goal_change_threshold=5`. Net +2 wins but 6 regressions.
- Exp 36: A/B old vs new ranking: coverage 0.302 → 0.731, persistent wins 23 → 25.
- Exp 38: "Frontier ranking: parameters saturated; no margin for multi-step reasoning."

**Literature connection:** Frontier-based exploration (Yamauchi, 1997) is known to be myopic — it maximizes immediate information gain but cannot plan multi-step information acquisition (e.g., "go to a vantage point to see the target"). This is the *active perception* problem (Bajcsy, 1988).

#### 4.2.2 No Genuine Exploration Strategy Learning (Experiments 10, 17–18)

**Finding:** "Exploration" in GM is a deterministic heuristic (visit least-visited frontier). The control learner discovers *which control moves in which direction* but does not *learn to explore*.

**Evidence:**
- Exp 10: "deterministic heuristic exploration, not learning"
- Exp 17–18: Control learner maps planner's desired direction → anonymous control by observing displacement. This is *control calibration*, not exploration strategy.

**Deficit:** The agent cannot *invent* exploration strategies (e.g., systematic sweeping, boundary following, loop detection). It executes a fixed heuristic.

### 4.3 Memory and Learning Deficits

#### 4.3.1 Order-Dependent Persistent Memory (Experiments 14, 16, 17, 31)

**Finding:** Persistent memory across episodes creates order-dependent outcomes. Different seed orders produce different win rates.

**Evidence:**
- Exp 14: Persistent memory reduced control trials (39.95 → 2.58) but *decreased* win rate (62.5% → 55.0%) and *increased* timeouts (10 → 17). "Learning order matters; 40 seeds insufficient for statistical reliability."
- Exp 16: Persistent improved wins (17 → 25) but increased timeouts (4 → 11). Reverse seed order: 27 wins vs 25 — *order matters*.
- Exp 17: Three random orders, persistent wins 4, 3, 4 vs volatile 5 — persistent *worse* in all three.
- Exp 31: 10 orders of same 10 seeds: volatile 60/100 wins (consistent), persistent 58/100 wins (4–7 per order). Paired outcomes changed 26/100 times (12 improvements, 14 regressions).

**Root cause:** The control learner uses a *threshold-based stopping rule* (stop trying a control after N samples). Early experiences permanently bias the policy. No mechanism for *unlearning* or *re-evaluating* with more data.

**Literature connection:** This is *catastrophic interference* / *order effects* in continual learning (McCloskey & Cohen, 1989; French, 1999). The fix requires either rehearsal, elastic weight consolidation (Kirkpatrick et al., 2017), or Bayesian updating — none implemented.

#### 4.3.2 Conservative Episodic Retrieval Cannot Overcome Planner Limits (Experiments 30, 32, 37)

**Finding:** Episodic retrieval (DecisionCase memory) only acts as *tie-break* between equal-cost BFS paths. It improves efficiency (fewer blocks, fewer turns) but not win rate in static scenario; in dynamic scenario, it *is* the improvement (Exp 37).

**Evidence:**
- Exp 30: Static: retrieval guided 221 decisions (volatile) vs 50 (persistent) but wins 6/10 vs 4/10. Dynamic: 213 vs 120 guided, wins 2/10 vs 6/10. "Persistent mode combines control trials and episodic cases; causal effect not isolated."
- Exp 32: Static 40 seeds: persistent 39/40 wins (1 timeout), mean turns 69 vs 74. "Better efficiency, not more wins."
- Exp 37: **Critical result** — Ablation ON vs OFF: Retrieval ON = 137/200 wins (68.5%), coverage 0.58, mean turns 111.87; Retrieval OFF = 110/200 wins (55.0%), coverage 0.287, mean turns 123.2, 5 stalls. **Delta: +27 wins, -11.3 turns, +0.293 coverage.** Replicated across 5 orders.

**Deficit:** Retrieval is *passive* — it only biases among planner-generated options. It cannot *propose* new paths, *correct* planner errors, or *plan* multi-step. The conservative design (similarity ≥ 0.65, support ≥ 2, tie-break only) prevents harmful overgeneralization but also prevents recovery from planner blindness.

**Literature connection:** This mirrors *case-based reasoning* (Aamodt & Plaza, 1994) with conservative retrieval. The limitation is known: CBR excels at interpolation, fails at extrapolation (Richter & Weber, 2013).

#### 4.3.3 Entity Categorization Without Semantic Grounding (Experiments 11, 12)

**Finding:** Entities are tracked by anonymous ID and categorized by movement heuristics (`stationary`, `mobile`, `approaches_agent`, `moves_away_from_agent`). No semantic labels (enemy, NPC, projectile) are learned.

**Evidence:**
- Exp 11: Pursuer included in `obstacles` field without `enemy` label. GM avoids current position but "does not recognize or anticipate pursuit." 8/10 wins, 2 captures.
- Exp 12: Categories inferred from motion only; "not semantic learning; does not infer intent or dangerousness."

**Deficit:** The agent cannot transfer *entity knowledge* across games because categories are purely kinematic. A "pursuer" in Game A and a "homing missile" in Game B share dynamics but have no shared representation.

**Literature connection:** This is the *symbol grounding problem* (Harnad, 1990) and *causal confusion* (Zhang et al., 2020) — the agent learns correlations (movement patterns) without causal models (intent, threat).

### 4.4 Neural Policy Deficits

#### 4.4.1 DQN Collapse During Fine-Tuning (Experiments 22–29)

**Finding:** DQN policies trained from scratch or with behavior cloning warm-start *degrade* during fine-tuning. Best checkpoints are always the behavior cloning (BC) checkpoints *before* DQN fine-tuning.

**Evidence:**
- Exp 22 (smoke test): 300-episode DQN from scratch: 0/20 holdout wins vs 0/20 baseline.
- Exp 24 (full training): 300 episodes, 43,685 updates. Best validation (ep 250): 5/20 vs 10/20 baseline. Holdout: 6/40 wins, 26 losses vs 24/40 baseline.
- Exp 25 (warm-start): 100 BC episodes → 300 DQN. Best validation (ep 150): 13/20 vs 10/20. Holdout: 11/40 wins, 3 losses vs 24/40 baseline.
- Exp 27 (intent BFS): BC validation 10/20 (vs 3/20 old encoder). DQN degraded validation to 2/20. Holdout: 24/40 vs 26/40 baseline.
- Exp 28 (progress reward): BC validation 11/20. DQN degraded to 0/20. Holdout A: 23/40 vs 21/40; Holdout B: 27/40 vs 26/40. *Small aggregate gain but paired regressions in both.*
- Exp 29 (third holdout): All three (baseline, intent, progress) 24/40 wins. Neural: 6 improvements, 6 regressions vs baseline.

**Diagnosis (Exp 26):**
- 18.9% of neural actions produced *no position change* vs 4.2% baseline
- 13.2% actions were attack/defend (`control_5`, `control_6`) in inappropriate contexts
- Encoder (40 features: target offset, adjacent obstacles, visit counts, intent one-hot) lacks BFS path, alternative distances, pursuer direction
- Progress reward only triggers when target *observed at same coordinate* — rare for mobile target
- Imitation fidelity low: BC achieved 3/20 validation (old encoder) → 10/20 (intent encoder) but still not baseline parity

**Literature connection:** This is *distribution shift* in offline/online RL (Fujimoto et al., 2019; Levine et al., 2020). The BC policy visits states the DQN policy doesn't, and vice versa. Without conservative policy improvement (e.g., CQL, AWAC) or explicit regularization to the BC policy, fine-tuning diverges. The *representation mismatch* (no path information in encoder) is a *causal feature deficiency* (Zhang et al., 2020).

#### 4.4.2 No Transfer Demonstrated (All Neural Experiments)

**Finding:** Neural policies are trained and evaluated *only* on the GridGame adapter. No cross-environment evaluation exists.

**Deficit:** Without transfer evaluation, neural policy work is *adapter-specific engineering*, not general capability development.

### 4.5 Strategy Search Deficits

#### 4.5.1 Evolutionary Search Fails Promotion Criteria (Experiments 21, 33)

**Finding:** Strategy search evaluates parameter variants on training seeds and validates on holdout. No variant has met the conservative promotion criterion (≥2 paired win improvements, zero regressions, or ≥5% turn reduction with same outcomes).

**Evidence:**
- Exp 21: Best variant +2 net wins but 6 paired regressions → not promoted.
- Exp 33: Dynamic scenario search — best candidate = baseline (score 0.85223), holdout tied 24/20 wins → not promoted.

**Deficit:** The search space (3 parameters) is too narrow to overcome architectural limits. The promotion criterion is appropriately conservative but reveals that *parameter tuning cannot fix representation deficits*.

---

## 5. Sim-to-Real Gap Analysis

### 5.1 Structural Differences Between GridGame and Real Games

| Dimension | GridGame (Sim) | Real Games | Gap Severity |
|-----------|----------------|------------|--------------|
| **Observation** | Structured dict with exact coordinates | Pixels (RGB), partial, noisy | **Critical** |
| **State** | Fully known grid topology | Hidden, stochastic, continuous | **Critical** |
| **Controls** | 6 anonymous discrete inputs | Keyboard/mouse/gamepad, analog, combinatorial | **Critical** |
| **Dynamics** | Deterministic grid transitions + RNG | Physics engines, animation cancels, frame-perfect | **High** |
| **Rewards** | None (inferred from state) | Sparse, delayed, shaped, multi-objective | **High** |
| **Entities** | 1 pursuer, 1 target, 1 obstacle | Dozens to thousands, diverse behaviors | **High** |
| **Goals** | Single `reach_goal` | Hierarchical, implicit, changing | **High** |
| **Reset** | Deterministic seed | Checkpoints, menus, loading screens | **Medium** |

### 5.2 GM's Simulated Adapter as a "Narrow Simulator"

The `simulated_game/` adapter is a *narrow simulator* in the sense of Tobin et al. (2017) — it simulates only the *abstract decision loop* (observation → decision → action → verification) but not the *perception* or *control* loops. This is by design (Decision 4: minimal operational loop first), but it means:

1. **No visual generalization** — The agent never learns to parse pixels.
2. **No control generalization** — Anonymous controls map 1:1 to cardinal directions; real games have context-sensitive controls (e.g., "jump" only on ground).
3. **No temporal generalization** — Fixed turn-based steps; real games run at 60+ FPS with variable frame timing.
4. **No distributional robustness** — Single environment, fixed rules; real games have patches, mods, variants.

### 5.3 Sim-to-Real Transfer Literature vs. GM Status

| Literature Finding | GM Status |
|-------------------|-----------|
| Domain randomization enables sim-to-real (Tobin et al., 2017) | Not implemented; single deterministic environment |
| System identification closes dynamics gap (Chebotar et al., 2019) | Control learner learns effects but not dynamics model |
| Progressive nets / adapters for transfer (Rusu et al., 2016) | No multi-environment architecture; core frozen but no adapter retraining demonstrated |
| Causal confusion prevents transfer (Zhang et al., 2020) | Entity categories kinematic only; no causal models |
| Belief state planning for POMDPs (Kaelbling et al., 1998) | BFS on point estimate; no belief state |
| MCTS with learned models (Silver et al., 2018; Schrittwieser et al., 2020) | PathTreePlanner implemented but not connected to real perception |

---

## 6. Generalization Deficits

### 6.1 Procedural Generation Generalization

The GridGame uses procedural generation for:
- Target position (random free cell, distance ≥ 4)
- Obstacle relocation (random free cell each turn)
- Target relocation (random free cell every 2 turns)
- Enemy combat choice (random attack/defend)

**Deficit:** All experiments use *fixed seeds* (0–39, 1000–1039, etc.) and *paired evaluation on identical seeds*. This measures *reproducibility*, not *generalization*. True generalization requires:
- Evaluation on *unseen* seeds from the *same distribution* (done partially in holdout)
- Evaluation on *shifted* distributions (e.g., larger grid, more obstacles, different enemy behavior) — **not done**
- Evaluation on *different games* — **not done**

Exp 21 holdout (seeds 1004815–1004834) is the closest to generalization test, but it's the *same environment* with different RNG streams.

### 6.2 Distribution Shift Robustness

**Finding:** The system is brittle to distribution shifts within the same environment:
- Exp 14, 16, 17, 31: Seed order changes outcomes dramatically (order-dependent learning)
- Exp 19: Dynamic vs static scenario — same agent, different win rates (25 vs 40)
- Exp 35: Combat rebalancing (damage 3→5) changed strategic stakes but no re-evaluation

**No experiments test:** Larger grids, different vision radii, additional entities, continuous movement, stochastic transitions, or partial observability beyond radius 4.

### 6.3 Causal vs. Correlational Learning

The control learner and entity tracker learn *correlations*:
- Control → displacement (correlational, not causal: blocked moves look like "no effect")
- Entity movement → category (kinematic only)
- Context → reward (contextual bandit, not structural causal model)

**Literature:** This is *causal confusion* (Zhang et al., 2020; Pearl, 2009). The agent cannot distinguish:
- "Control_1 moves me up" vs "Control_1 moves me up *when not blocked*"
- "Entity approaches me" vs "Entity moves toward my *previous* position" (pursuit vs coincidence)

Without *interventional* learning (trying controls in varied contexts to disambiguate), the agent builds spurious correlations that fail under distribution shift.

---

## 7. Training Bugs and Implementation Deficits

### 7.1 Order-Dependent Learning (Bug Class: Non-Reproducibility)

**Location:** `ControlLearner.choose()` threshold logic; `GridMemoryStore` persistent SQLite.

**Symptom:** Persistent memory across episodes creates path dependence. Early episodes bias control mappings permanently.

**Experiments affected:** 14, 16, 17, 31

**Fix required:** Bayesian updating, rehearsal buffer, or explicit re-evaluation schedule. Current "stop after N samples" is a hard threshold with no forgetting mechanism.

### 7.2 Neural Policy Collapse During Fine-Tuning (Bug Class: Training Instability)

**Location:** `NeuralPolicy` DQN fine-tuning loop; `ExplorationPolicy` integration.

**Symptom:** Validation performance degrades monotonically during DQN fine-tuning after BC warm-start. Best checkpoint is always pre-fine-tuning.

**Experiments affected:** 24, 25, 27, 28

**Root causes identified (Exp 26):**
1. Encoder lacks BFS path information (only local features)
2. Progress reward sparse for mobile targets
3. Action space includes attack/defend in navigation contexts
4. No conservative regularization (CQL, KL to BC policy)
5. Imitation fidelity insufficient (10/20 validation vs 13/20 baseline)

### 7.3 Retrieval Confounded with Control Learning (Bug Class: Experimental Confound)

**Location:** `GridMemoryStore` combines control trials (SQLite) and episodic cases (deque) in persistent mode.

**Symptom:** Experiments 14, 16, 17, 30, 31, 32 compare volatile vs persistent where persistent includes *both* control trial persistence AND episodic retrieval. Causal attribution impossible.

**Fix applied in Exp 37:** `--disable-episodic-retrieval` flag isolates retrieval effect. Result: retrieval accounts for full improvement.

### 7.4 Entity Tracking Conflates Pursuer and Obstacles (Bug Class: Observation Leakage)

**Location:** `GridGame.observe()` includes enemy in `obstacles` field without semantic distinction.

**Symptom:** Agent treats pursuer as static obstacle. Cannot learn "this entity chases me" vs "this entity is a wall."

**Experiments affected:** 11, 12

**Fix required:** Separate entity observations with stable IDs and observed properties (health, velocity). Current `visible_entities` list exists but planner doesn't use it differentially.

### 7.5 Progress Reward Misalignment (Bug Class: Reward Hacking Risk)

**Location:** `ExplorationPolicy._compute_effects()` and `ControlLearner` reward computation.

**Symptom:** Progress reward only when target observed at same coordinate pre/post action. For mobile target (moves every 2 turns, visible ~6 turns/episode), reward is nearly always zero.

**Experiments affected:** 26, 28

**Fix attempted (Exp 28):** Intent-based progress reward using BFS step *before* action. Improved holdout slightly but didn't prevent DQN collapse.

---

## 8. Episodic Memory Limitations

### 8.1 Current Implementation

`GridMemoryStore` maintains:
- **DecisionCase buffer**: 2000 cases max, retrieval window 256 most recent
- **Similarity metric**: Weighted combination of 8 context features (goal visibility, target direction, adjacent obstacles, entity distance, health, frontier count)
- **Retrieval policy**: Minimum similarity 0.65, minimum support 2 cases per direction, tie-break only on equal-cost paths
- **Persistence**: SQLite per game_id, pruned to 2000 cases

### 8.2 Fundamental Limitations

| Limitation | Evidence | Consequence |
|------------|----------|-------------|
| **Passive tie-break only** | Exp 30, 32, 37 | Cannot correct planner; only biases among planner's options |
| **Coordinate-free context** | `build_decision_context()` | Loses metric information; "target at (5,3)" → "target_direction=[1,0]" |
| **Fixed similarity weights** | `_context_similarity()` weights hardcoded | Cannot adapt importance to task phase (e.g., combat vs navigation) |
| **No temporal abstraction** | Cases are single-step (context, direction, reward) | Cannot retrieve multi-step plans or strategies |
| **Per-game isolation** | `game_id` partitioning | No cross-game transfer; each game starts empty |
| **Conservative thresholds** | similarity ≥ 0.65, support ≥ 2 | High precision, low recall; misses useful analogies |

### 8.3 Retrieval as "Local Policy Correction"

The retrieval system functions as a *local policy correction* mechanism: when BFS offers multiple equal-cost paths (e.g., go around obstacle left vs right), retrieval biases toward the direction that historically led to better outcomes in similar contexts.

This is valuable (Exp 37: +27 wins, +0.293 coverage) but fundamentally limited — it cannot:
- Propose a path BFS doesn't see (e.g., "wait for obstacle to move")
- Plan multi-step (e.g., "go to vantage point, then approach target")
- Generalize across context shifts (e.g., new obstacle configuration)

---

## 9. Neural Policy Collapse: Deep Diagnosis

### 9.1 Architecture Mismatch

The `NeuralPolicy` (`neural_policy.py`) uses a 40-feature encoder:
- Target offset (dx, dy) — 2
- Adjacent obstacles (4 booleans) — 4
- Visit counts (4 directions) — 4
- Entity distance bucket — 1
- Health bucket — 1
- Frontier bucket — 1
- **Intent BFS** (4 one-hot + 1 frontier flag) — 5 (added Exp 27)
- **Total**: 18 base + 5 intent = 23? (Code shows 40 — likely includes more HUD features)

**Missing critical features:**
- Full BFS path (not just next step)
- Alternative path distances
- Pursuer position and predicted trajectory
- Target mobility belief (is it moving? where?)
- Control effect predictions (from ControlLearner)
- Uncertainty estimates

### 9.2 Reward Function Pathology

The reward combines:
- Terminal: +1 win, -1 loss
- Progress: `target_distance_delta` (only when target visible at same coordinate)
- Curiosity: State novelty (hash-based)
- Combat: Implicit via health changes

**Problems:**
1. Progress reward near-zero for mobile target (Exp 26)
2. Curiosity rewards *any* new state, not *goal-relevant* states
3. No shaping for "approach target when hidden" or "avoid pursuer"
4. Attack/defend controls have no dedicated reward signal

### 9.3 Training Dynamics

| Phase | Episodes | Validation | Holdout | Outcome |
|-------|----------|------------|---------|---------|
| Scratch DQN (Exp 24) | 300 | 5/20 (ep 250) | 6/40, 26 losses | Collapse |
| BC + DQN (Exp 25) | 100 BC + 300 DQN | 13/20 (ep 150) | 11/40, 3 losses | BC best, DQN degrades |
| Intent + BC (Exp 27) | 100 BC | 10/20 | 24/40 vs 26/40 | BC best, DQN degrades |
| Progress reward (Exp 28) | 100 BC + 300 DQN | 11/20 → 0/20 | 23/40, 27/40 | BC best, DQN degrades |

**Consistent pattern:** BC checkpoint is best. DQN fine-tuning *always* degrades validation performance. This indicates **distribution shift between BC policy and DQN policy** compounded by **insufficient representation** and **misaligned reward**.

### 9.4 Why This Matters for Sim-to-Real

If a neural policy cannot surpass a hand-coded BFS planner *in the same simulator it was trained in*, it has no hope of transferring to a real game where:
- Observation is high-dimensional pixels
- Dynamics are unknown and stochastic
- Reward is sparse or absent
- Action space is larger and continuous

The neural policy work, while technically sophisticated, has not demonstrated *general learning capability* — only *adapter-specific approximation* of a known-optimal planner.

---

## 10. Synthesis: Why the Simulated Adapter Has Reached Its Limit

### 10.1 Empirical Limit Evidence (Experiment 38)

> **Experiment 38 Conclusion:** "Nessun ulteriore miglioramento misurabile possibile nell'architettura attuale."
> - BFS Planner: Optimal for grid with partial observability; no margin for multi-step reasoning.
> - Frontier Ranking: Parameters saturated (visit_penalty, information_gain_bonus).
> - Retrieval: Conservative (tie-break only on equal cost); cannot compensate BFS limits.
> - Dynamic Goal Handling: Already optimal (switch to exploration when dynamic and hidden).
> - Neural Policy (DQN): Collapse in fine-tuning, regressions in all holdouts.
> - Strategy Search: 6 paired regressions, no promotion.

### 10.2 Architectural Ceiling Analysis

The current architecture solves **grid navigation with partial observability and anonymous controls**. It does this by:
1. BFS on known/remembered grid topology → optimal path to visible target/frontier
2. Frontier ranking → heuristic exploration when target hidden
3. Control learner → map planner direction → anonymous control
4. Episodic retrieval → bias tie-breaks toward historically successful directions
5. Entity tracking → kinematic categorization

**What it cannot do:**
- **Belief state planning**: Maintain distribution over target position, update with observations
- **Multi-step reasoning**: Plan "go to vantage point → observe target → approach"
- **Uncertainty-aware control**: "Try control_1 to see if it moves me up" vs "I know control_1 moves me up"
- **Causal model learning**: Distinguish "control_1 moves me up" from "control_1 moves me up when not blocked"
- **Transfer**: Apply any learned component to a different environment

### 10.3 The Phase 4 Imperative

The ROADMAP and DECISIONS (Decision 14) mandate Phase 4:
1. **Real Perception**: Capture/preprocess/parse for MiniGrid (first real environment)
2. **PathTree Planning (MCTS-Lite)**: Replace flat BFS with tree search maintaining belief state, handling uncertainty, integrating retrieval as prior, supporting warm-start (tree reuse)
3. **Transfer Validation**: Deploy on second environment with core frozen + adapter retrained

This is not incremental — it's an **architectural phase transition** from *reactive planning on known topology* to *deliberative planning under uncertainty*.

---

## 11. Related Work and Literature Positioning

### 11.1 Simulated Game Environments

| Environment | Key Characteristic | GM Relation |
|-------------|-------------------|-------------|
| ALE (Bellemare et al., 2013) | 57 Atari games, pixel input | GM targets broader than Atari; no pixel perception yet |
| ProcGen (Cobbe et al., 2020) | Procedurally generated levels, pixel input | GM's GridGame is procedural but structured, not pixel |
| MiniGrid (Chevalier-Boisvert et al., 2019) | Grid worlds, partial observability, missions | **Phase 4 target**; GM will connect via adapter |
| NetHack (Küttler et al., 2020) | Complex, stochastic, text+graphics | Beyond current scope |
| VGDL (Schaul, 2013) | General video game description language | GM uses custom GridGame, not VGDL |

### 11.2 Sim-to-Real Transfer

- **Domain Randomization** (Tobin et al., 2017): Vary simulator parameters to cover real distribution. *GM: Not implemented.*
- **System Identification** (Chebotar et al., 2019): Fit simulator parameters to real data. *GM: ControlLearner learns effects but not parametric dynamics.*
- **Sim-to-Real via Adapters** (Rusu et al., 2016; Parisotto et al., 2016): Progressive nets, policy distillation. *GM: Architecture supports adapter swap but no transfer demonstrated.*
- **Causal Sim-to-Real** (Zhang et al., 2020; Schölkopf et al., 2021): Learn causal models invariant across domains. *GM: Entity categories kinematic only; no causal discovery.*

### 11.3 Generalization in RL

- **Procedural Generalization** (Cobbe et al., 2020; Justesen et al., 2018): Train on distribution, test on held-out levels. *GM: Fixed seeds, paired evaluation; no distribution shift tests.*
- **Meta-RL / Fast Adaptation** (Duan et al., 2016; Wang et al., 2016): Learn to adapt quickly. *GM: Persistent memory across episodes but order-dependent, no meta-learning.*
- **Curriculum Learning** (Narvekar et al., 2020): Sequence tasks by difficulty. *GM: Static → Dynamic is a curriculum but not systematic.*

### 11.4 Planning Under Uncertainty

- **POMDP Solvers** (Kaelbling et al., 1998; Silver & Veness, 2010): Belief state planning. *GM: No belief state; point estimates only.*
- **MCTS with Learned Models** (Silver et al., 2018; Schrittwieser et al., 2020): AlphaZero, MuZero. *GM: PathTreePlanner (MCTS-Lite) implemented but not yet integrated with real perception.*
- **Active Perception** (Bajcsy, 1988; Chen et al., 2019): Plan to reduce uncertainty. *GM: Frontier exploration is myopic information gain; no multi-step info planning.*

### 11.5 Episodic Memory in RL

- **Differentiable Neural Computers** (Graves et al., 2016): External memory with attention. *GM: Conservative CBR, not differentiable.*
- **Episodic RL** (Blundell et al., 2016; Pritzel et al., 2017): Nearest-neighbor value estimation. *GM: Similar but tie-break only, not value function.*
- **Memory-Augmented RL** (Oh et al., 2016; Santoro et al., 2018): Structured memory. *GM: Multi-tier (STM/MTM/LTM/GSM/GM) but consolidation not activated.*

---

## 12. Recommendations and Phase 4 Specification

### 12.1 Immediate Actions (Pre-Phase 4)

1. **Fix order-dependent learning**: Replace threshold-based control trial stopping with Bayesian updating or scheduled re-evaluation.
2. **Isolate retrieval ablation permanently**: Add `--disable-episodic-retrieval` to benchmark as standard flag.
3. **Separate entity observations**: Distinguish pursuers, obstacles, NPCs in observation schema.
4. **Document all hyperparameters**: Current experiments vary seeds, orders, turn limits, damage values — create a canonical benchmark config.

### 12.2 Phase 4: Real Perception + PathTree Planning

#### 12.2.1 Perception Module (`perception/`)

| Component | Specification |
|-----------|---------------|
| **Capture** | Screen capture (Windows: DXGI/Duplication API; Linux: X11/Wayland) at 10–30 FPS |
| **Preprocess** | Resize, normalize, color space conversion, frame stacking |
| **Parser** | MiniGrid parser: detect grid, agent, goal, walls, objects from rendered RGB |
| **Adapter Protocol** | `ObservationAdapter` with `observe(frame) -> dict` matching core `GameState` expectations |

#### 12.2.2 PathTree Planner (`simulated_game/agent/path_tree.py` → `core/planning/path_tree.py`)

| Feature | Current (Grid) | Phase 4 (General) |
|---------|----------------|-------------------|
| **State** | Known grid topology | Belief state (particle filter over world states) |
| **Search** | MCTS with 25 particles, depth 7 | MCTS-Lite with progressive widening, depth adaptive |
| **Uncertainty** | Entity predictions only | Full belief: target position, entity dynamics, control effects |
| **Retrieval** | Tie-break prior | Prior over actions + value initialization |
| **Warm-start** | None | Tree reuse across steps (persistent search tree) |
| **Integration** | Adapter-specific | Core module, adapter-agnostic |

#### 12.2.3 Transfer Validation Protocol

1. **Environment 1**: MiniGrid (e.g., `MiniGrid-Empty-8x8`, `MiniGrid-DoorKey`, `MiniGrid-Dynamic-Obstacles`)
2. **Core frozen**: `GameEngine`, `Memory`, `PathTreePlanner`, `ControlLearner` unchanged
3. **Adapter retrained**: `MiniGridObservationAdapter`, `MiniGridActionExecutor`, reward parser
4. **Metrics**: Sample efficiency (episodes to threshold), asymptotic performance, catastrophic forgetting check
5. **Success criterion**: ≥50% of Environment 1 performance on Environment 2 (e.g., MiniGrid → ProcGen or custom) with <2x samples

### 12.3 Deprecations (Per Decision 14)

- Neural policy (DQN) — not a priority; does not block Phase 4
- Strategy search — not a priority; parameter space too narrow
- GridGame-specific heuristics (frontier ranking, dynamic goal branch) — remain in adapter only

---

## 13. Conclusion

The Game Master project has conducted a rigorous, experiment-driven exploration of a simulated game adapter (`GridGame`) as a testbed for autonomous agent components. Through 38 documented experiments, we have:

1. **Built a functional minimal loop**: Observation → BFS planning → control calibration → action → verification → memory (Experiments 1–10)
2. **Identified and fixed specific deficits**: Target memory loss (Exp 19), dynamic goal handling (Exp 33), frontier ranking (Exp 36), retrieval attribution (Exp 37)
3. **Reached the empirical limit** of the BFS + frontier + retrieval architecture (Exp 38)
4. **Failed to demonstrate** neural policy superiority, cross-game transfer, or robust generalization

The deficits cataloged here — planner optimality ceiling, exploration myopia, order-dependent memory, conservative retrieval, neural collapse, causal confusion — are not unique to GM. They are **manifestations of fundamental challenges** in sim-to-real transfer, generalization under distribution shift, and learning causal models from observational data.

The path forward is clear: **Phase 4 must introduce real perception and belief-state planning**. The simulated adapter has served its purpose as a controlled testbed for module development. Its limits are now the project's compass — pointing toward the architectural evolution required for a truly general game-playing agent.

---

## Appendix A: Experiment Index

| Exp | Date | Focus | Key Result |
|-----|------|-------|------------|
| 1–4 | 10/27 | Core loop, mock data | Reproducible loop, 7.35μs/frame |
| 5–8 | 10/27 | Grid game basics | BFS works, moving target/obstacle |
| 9 | 10/27 | Partial observability | Planner stalls when target hidden |
| 10 | 10/27 | Spatial memory + exploration | 10/10 seeds reach hidden target |
| 11 | 10/27 | Pursuer entity | 8/10 wins, no pursuit anticipation |
| 12 | 10/27 | Entity categorization | Kinematic only, no semantics |
| 13 | 10/27 | Combat trial-and-error | 6/10 wins, contextual learning only |
| 14 | 10/27 | Persistent memory | Fewer trials but more timeouts, order-dependent |
| 15 | 10/27 | UCB control selection | 27/40 wins both conditions, persistence no change |
| 16 | 10/27 | General control learning | Persistent 25/40 vs volatile 17/40, order-dependent |
| 17 | 10/27 | 6 anonymous controls | Persistent worse across 3 orders |
| 18 | 10/27 | Directional control learning | Fewer control trials, persistent wins 21–22 vs 25 |
| 19 | 10/28 | Target memory, static/dynamic | Static 40/40 both; dynamic 25 vs 22, target loss fixed |
| 20 | 10/28 | Core/adapter separation | Clean architectural boundary verified |
| 21 | 10/28 | Strategy search | Best +2 wins but 6 regressions, not promoted |
| 22 | 10/28 | DQN smoke test | Pipeline works, 0/20 holdout wins |
| 23 | 10/28 | Persistent DQN training | Checkpoint resume works, no efficacy |
| 24 | 10/28 | Full DQN from scratch | 6/40 wins, 26 losses vs 24/40 baseline |
| 25 | 10/28 | BC warm-start + DQN | 11/40 wins, 3 losses vs 24/40 baseline |
| 26 | 10/28 | Neural collapse diagnosis | 18.9% no-move, 13.2% attack/defend, sparse reward |
| 27 | 10/28 | Intent BFS in encoder | BC 10/20 val, DQN degrades to 2/20; holdout 24/40 |
| 28 | 10/28 | Progress reward from intent | BC 11/20 val, DQN degrades to 0/20; holdout +2/+1 wins |
| 29 | 10/29 | Third holdout | All 24/40, 6 improvements / 6 regressions |
| 30 | 10/29 | Episodic retrieval (small) | Mixed, confounded with control persistence |
| 31 | 10/29 | Retrieval order sensitivity | Persistent 58/100 (4–7/order) vs volatile 60/100 |
| 32 | 10/29 | Static benchmark extended | 40/40 vs 39/40, fewer blocks/turns persistent |
| 33 | 10/30 | Stop conditions, dynamic search | Dynamic search no promo; dynamic goal branch helps |
| 34 | 10/31 | Dynamic validation (5 orders) | Persistent 137/200 wins (68.5%), KEEP |
| 35 | 10/31 | Combat rebalance | Damage 3→5, strategic stakes increased |
| 36 | 10/31 | Attribution to frontier ranking | Improvement from ranking, not dynamic branch |
| 37 | 10/31 | Ablation retrieval ON/OFF | **Retrieval = full improvement**: +27 wins, +0.293 coverage |
| 38 | 10/31 | Phase 3 closure, Phase 4 plan | Empirical limit reached; Phase 4 mandated |

---

## Appendix B: Key Metrics Summary (Dynamic Scenario, Persistent Memory)

| Metric | Retrieval ON (Exp 37) | Retrieval OFF (Exp 37) | Delta |
|--------|----------------------|------------------------|-------|
| Wins (200 eps) | 137 (68.5%) | 110 (55.0%) | **+27** |
| Timeouts | 62 | 89 (+5 stalls) | -27 |
| Mean turns (wins) | 111.87 | 123.2 | **-11.3** |
| Coverage | 0.58 | 0.287 | **+0.293** |
| Blocked actions/seq | 3 | 118 | -115 |
| Paired outcome changes | 19/seq | — | Consistent across 5 orders |

---

## Appendix C: Glossary

- **BFS**: Breadth-First Search (shortest path on unweighted graph)
- **BC**: Behavior Cloning (supervised imitation learning)
- **CBR**: Case-Based Reasoning
- **DQN**: Deep Q-Network
- **MCTS**: Monte Carlo Tree Search
- **POMDP**: Partially Observable Markov Decision Process
- **Sim-to-Real**: Transfer from simulation to real world
- **UCB**: Upper Confidence Bound (exploration bonus)
- **GM**: Game Master (project) / General Memory (memory tier)

---

## References

[1] Bellemare, M. G., et al. (2013). The Arcade Learning Environment. *JAIR*.
[2] Cobbe, K., et al. (2020). Leveraging Procedural Generation to Benchmark Reinforcement Learning. *ICML*.
[3] Chevalier-Boisvert, M., et al. (2019). MiniGrid: Minimal Gridworld Environment. *GitHub*.
[4] Tobin, J., et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. *IROS*.
[5] Chebotar, Y., et al. (2019). Closing the Sim-to-Real Loop. *RSS*.
[6] Rusu, A. A., et al. (2016). Progressive Neural Networks. *arXiv:1606.04671*.
[7] Zhang, A., et al. (2020). Causal Confusion in Imitation Learning. *NeurIPS*.
[8] Kaelbling, L. P., et al. (1998). Planning and Acting in Partially Observable Stochastic Domains. *AI Magazine*.
[9] Silver, D., et al. (2018). A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go. *Science*.
[10] Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *Nature*.
[11] Yamauchi, B. (1997). A Frontier-Based Approach for Autonomous Exploration. *CIRA*.
[12] Bajcsy, R. (1988). Active Perception. *Proceedings of the IEEE*.
[13] McCloskey, M., & Cohen, N. J. (1989). Catastrophic Interference in Connectionist Networks. *Psychology of Learning and Motivation*.
[14] Kirkpatrick, J., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. *PNAS*.
[15] Fujimoto, S., et al. (2019). Off-Policy Deep Reinforcement Learning without Exploration. *ICML*.
[16] Levine, S., et al. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives. *arXiv:2005.01643*.
[17] Aamodt, A., & Plaza, E. (1994). Case-Based Reasoning: Foundational Issues. *KI*.
[18] Richter, M. M., & Weber, R. (2013). Case-Based Reasoning: A Textbook. *Springer*.
[19] Harnad, S. (1990). The Symbol Grounding Problem. *Physica D*.
[20] Pearl, J. (2009). Causality. *Cambridge University Press*.
[21] Blundell, C., et al. (2016). Model-Free Episodic Control. *arXiv:1606.04460*.
[22] Pritzel, A., et al. (2017). Neural Episodic Control. *ICML*.
[23] Oh, J., et al. (2016). Control of Memory, Active Perception, and Action in Minecraft. *ICML*.
[24] Santoro, A., et al. (2018). Relational Recurrent Neural Networks. *NeurIPS*.
[25] Duan, Y., et al. (2016). RL2: Fast Reinforcement Learning via Slow Reinforcement Learning. *arXiv:1611.02779*.
[26] Wang, J. X., et al. (2016). Learning to Reinforcement Learn. *arXiv:1611.05763*.
[27] Narvekar, S., et al. (2020). Curriculum Learning for Reinforcement Learning Domains. *JAIR*.
[28] Justesen, N., et al. (2018). Illuminating Generalization in Deep Reinforcement Learning. *arXiv:1802.07235*.
[29] Cormen, T. H., et al. (2009). Introduction to Algorithms. *MIT Press*.
[30] Schölkopf, B., et al. (2021). Toward Causal Representation Learning. *Proceedings of the IEEE*.

---

*End of Research Paper*