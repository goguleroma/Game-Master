"""
World Model Learner - Game-agnostic transition learning for agent and entities.

This module provides:
- WorldModelLearner: Learns state, action → next_state for ALL entities + agent
- ControlLearner: Backward-compatible API wrapper for existing code
"""

import json
import math
import random
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

Position = Tuple[int, int]


@dataclass
class AgentTransitionModel:
    """Per (context, control): distribution of observed effects."""
    context: Tuple[int, ...]
    control: str
    samples: List[Dict[str, float]] = field(default_factory=list)
    mean_effects: Dict[str, float] = field(default_factory=dict)
    cov_effects: Dict[str, float] = field(default_factory=dict)  # variance for uncertainty
    reward_samples: List[float] = field(default_factory=list)
    mean_reward: float = 0.0
    last_updated: int = 0

    def add_sample(self, effects: Dict[str, float], reward: float, turn: int) -> None:
        self.samples.append(effects)
        self.reward_samples.append(reward)
        self.last_updated = turn
        self._recompute_stats()

    def _recompute_stats(self) -> None:
        if not self.samples:
            return
        # Mean effects
        all_keys: Set[str] = set()
        for s in self.samples:
            all_keys.update(s.keys())
        
        n = len(self.samples)
        self.mean_effects = {}
        self.cov_effects = {}
        for key in all_keys:
            values = [s.get(key, 0.0) for s in self.samples]
            mean = sum(values) / n
            self.mean_effects[key] = mean
            # Variance (diagonal covariance)
            if n > 1:
                var = sum((v - mean) ** 2 for v in values) / (n - 1)
                self.cov_effects[key] = var
            else:
                self.cov_effects[key] = 0.0
        
        # Mean reward
        if self.reward_samples:
            self.mean_reward = sum(self.reward_samples) / len(self.reward_samples)
        else:
            self.mean_reward = 0.0


@dataclass
class EntityDynamicsModel:
    """Predictive model for a single entity from observations (no semantic labels)."""
    entity_id: str
    positions: List[Tuple[Position, Position, int]] = field(default_factory=list)  # (entity_pos, agent_pos, turn)
    velocities: List[Tuple[float, float]] = field(default_factory=list)  # (vx, vy) per step
    category: str = "unclassified"  # "stationary", "mobile", "approaches_agent", "moves_away_from_agent", "erratic"
    category_confidence: float = 0.0
    approach_streak: int = 0
    retreat_streak: int = 0
    last_seen_turn: int = -1
    
    # For erratic detection
    velocity_changes: int = 0
    last_velocity: Optional[Tuple[float, float]] = None

    def predict_next_positions(self, horizon: int, agent_pos: Position) -> List[Position]:
        """Predict entity positions for next `horizon` steps given agent position."""
        if not self.positions:
            return []
        
        current_pos = self.positions[-1][0]
        
        if self.category == "stationary":
            return [current_pos] * horizon
        
        elif self.category == "approaches_agent":
            # Predict movement toward agent
            preds = []
            pos = current_pos
            for _ in range(horizon):
                dx = agent_pos[0] - pos[0]
                dy = agent_pos[1] - pos[1]
                # Move one step toward agent (greedy)
                step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
                step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
                # Prefer larger component
                if abs(dx) >= abs(dy):
                    pos = (pos[0] + step_x, pos[1])
                else:
                    pos = (pos[0], pos[1] + step_y)
                preds.append(pos)
            return preds
        
        elif self.category == "moves_away_from_agent":
            # Predict movement away from agent
            preds = []
            pos = current_pos
            for _ in range(horizon):
                dx = agent_pos[0] - pos[0]
                dy = agent_pos[1] - pos[1]
                step_x = -1 if dx > 0 else (1 if dx < 0 else 0)
                step_y = -1 if dy > 0 else (1 if dy < 0 else 0)
                if abs(dx) >= abs(dy):
                    pos = (pos[0] + step_x, pos[1])
                else:
                    pos = (pos[0], pos[1] + step_y)
                preds.append(pos)
            return preds
        
        elif self.category in ("mobile", "erratic"):
            # Extend average velocity
            if not self.velocities:
                return [current_pos] * horizon
            avg_vx = sum(v[0] for v in self.velocities[-5:]) / min(5, len(self.velocities))
            avg_vy = sum(v[1] for v in self.velocities[-5:]) / min(5, len(self.velocities))
            preds = []
            pos = current_pos
            for step in range(1, horizon + 1):
                nx = round(pos[0] + avg_vx * step)
                ny = round(pos[1] + avg_vy * step)
                preds.append((nx, ny))
            return preds
        
        else:
            # Unknown/unclassified - assume stationary
            return [current_pos] * horizon

    def update(self, new_pos: Position, agent_pos: Position, turn: int) -> None:
        """Update model with new observation."""
        # Store position history with agent position at that time
        self.positions.append((new_pos, agent_pos, turn))
        if len(self.positions) > 50:  # Keep last 50 positions
            self.positions.pop(0)
        
        # Compute velocity and update approach/retreat streaks
        if len(self.positions) >= 2:
            prev_pos, prev_agent_pos, prev_turn = self.positions[-2]
            dt = turn - prev_turn
            if dt > 0:
                vx = (new_pos[0] - prev_pos[0]) / dt
                vy = (new_pos[1] - prev_pos[1]) / dt
                self.velocities.append((vx, vy))
                if len(self.velocities) > 20:
                    self.velocities.pop(0)
                
                # Detect erratic behavior (velocity direction changes)
                if self.last_velocity is not None:
                    prev_vx, prev_vy = self.last_velocity
                    # Check if velocity reversed or changed significantly
                    dot = prev_vx * vx + prev_vy * vy
                    prev_speed = math.hypot(prev_vx, prev_vy)
                    curr_speed = math.hypot(vx, vy)
                    if prev_speed > 0.1 and curr_speed > 0.1 and dot < -0.5 * prev_speed * curr_speed:
                        self.velocity_changes += 1
                self.last_velocity = (vx, vy)
                
                # Update approach/retreat streaks using agent position at each observation time
                # This correctly handles the case where both agent and entity move
                if dt == 1:
                    # Distance from entity to agent at previous observation
                    prev_dist = abs(prev_pos[0] - prev_agent_pos[0]) + abs(prev_pos[1] - prev_agent_pos[1])
                    # Distance from entity to agent at current observation
                    curr_dist = abs(new_pos[0] - agent_pos[0]) + abs(new_pos[1] - agent_pos[1])
                    
                    # Also check if entity moved toward agent's previous position
                    # (entity moves toward where agent WAS, not where agent IS now)
                    entity_to_prev_agent = abs(prev_pos[0] - prev_agent_pos[0]) + abs(prev_pos[1] - prev_agent_pos[1])
                    new_entity_to_prev_agent = abs(new_pos[0] - prev_agent_pos[0]) + abs(new_pos[1] - prev_agent_pos[1])
                    entity_moved_toward_prev_agent = new_entity_to_prev_agent < entity_to_prev_agent
                    
                    # Check if entity's velocity points toward agent's PREVIOUS position
                    # (entity moves toward where agent WAS at the time of its move)
                    vx = new_pos[0] - prev_pos[0]
                    vy = new_pos[1] - prev_pos[1]
                    to_prev_agent_x = prev_agent_pos[0] - prev_pos[0]
                    to_prev_agent_y = prev_agent_pos[1] - prev_pos[1]
                    velocity_toward_prev_agent = (vx * to_prev_agent_x + vy * to_prev_agent_y) > 0
                    
                    # Check if entity maintains minimal distance (1) while moving toward agent
                    # This handles pursuers that stay at distance 1 and move in parallel with agent
                    maintaining_minimal_distance = (curr_dist <= 1) and (vx != 0 or vy != 0) and velocity_toward_prev_agent
                    
                    if curr_dist < prev_dist or entity_moved_toward_prev_agent or velocity_toward_prev_agent or maintaining_minimal_distance:
                        self.approach_streak += 1
                        self.retreat_streak = 0
                    elif curr_dist > prev_dist:
                        self.retreat_streak += 1
                        self.approach_streak = 0
                    else:
                        # Distance unchanged - don't reset streak if entity moved toward agent
                        if not entity_moved_toward_prev_agent and not velocity_toward_prev_agent and not maintaining_minimal_distance:
                            self.approach_streak = 0
                            self.retreat_streak = 0
                elif dt > 1:
                    # Gap in observations - reset streaks as we can't reliably track
                    self.approach_streak = 0
                    self.retreat_streak = 0
        
        self.last_seen_turn = turn
        
        # Re-infer category
        self.category, self.category_confidence = self._infer_category()

    def _infer_category(self) -> Tuple[str, float]:
        """Infer entity category from observations (no semantic labels)."""
        # Need minimum observations
        if len(self.positions) < 3:
            return "unclassified", 0.0
        
        # Check approach/retreat streaks first (strongest signals)
        # Lowered threshold to 2 for faster pursuer detection
        if self.approach_streak >= 2:
            conf = min(1.0, self.approach_streak / (self.approach_streak + 1))
            return "approaches_agent", conf
        
        if self.retreat_streak >= 2:
            conf = min(1.0, self.retreat_streak / (self.retreat_streak + 1))
            return "moves_away_from_agent", conf
        
        # Check for consistent approach toward agent (even without streak)
        # If most recent velocities point toward agent, it's a pursuer
        # Use agent position from last observation
        if len(self.velocities) >= 3 and self.positions:
            current_pos = self.positions[-1][0]
            last_agent_pos = self.positions[-1][1]  # agent position at last observation
            toward_agent_count = 0
            for vx, vy in self.velocities[-3:]:
                # Check if velocity points roughly toward agent
                to_agent_x = last_agent_pos[0] - current_pos[0]
                to_agent_y = last_agent_pos[1] - current_pos[1]
                # Velocity points toward agent if dot product > 0
                if vx * to_agent_x + vy * to_agent_y > 0:
                    toward_agent_count += 1
            
            if toward_agent_count >= 2:
                return "approaches_agent", 0.7
        
        # Check for erratic (frequent velocity changes) - but only if not consistently approaching
        if len(self.velocities) >= 5 and self.velocity_changes >= 3:
            return "erratic", min(1.0, self.velocity_changes / 5.0)
        
        # Check average speed
        if len(self.velocities) >= 5:
            avg_speed = sum(math.hypot(v[0], v[1]) for v in self.velocities[-5:]) / 5
            if avg_speed > 0.1:
                return "mobile", min(1.0, avg_speed)
        
        # Default: stationary
        return "stationary", 1.0


@dataclass
class ContextTransitionModel:
    """Transition model for a context: action → next_context_distribution."""
    context: Tuple[int, ...]
    action_models: Dict[str, AgentTransitionModel] = field(default_factory=dict)  # control -> model
    entity_interactions: Dict[str, Dict[str, float]] = field(default_factory=dict)  # entity_id -> effect_on_agent
    visit_count: int = 0

    def get_or_create_action_model(self, control: str) -> AgentTransitionModel:
        if control not in self.action_models:
            self.action_models[control] = AgentTransitionModel(context=self.context, control=control)
        return self.action_models[control]


class WorldModelLearner:
    """
    Game-agnostic world model learner.
    
    Learns:
    - Agent: context, control → agent_displacement + HUD effects
    - Entities: entity_id → predicted positions(horizon) + velocity + category
    - Environment: context → transition probabilities
    
    All from numerical observations only, no game-specific semantics.
    """
    
    def __init__(
        self,
        max_contexts: int = 10000,
        max_entities: int = 100,
        prediction_horizon: int = 5,
        min_samples_for_prediction: int = 3,
        exploration_bonus: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.max_contexts = max_contexts
        self.max_entities = max_entities
        self.prediction_horizon = prediction_horizon
        self.min_samples_for_prediction = min_samples_for_prediction
        self.exploration_bonus = exploration_bonus
        self._rng = random.Random(seed)
        
        # Core models
        self.context_models: Dict[Tuple[int, ...], ContextTransitionModel] = {}
        self.entity_models: Dict[str, EntityDynamicsModel] = {}
        
        # Backward compatibility: direction evidence per control (global, not per-context)
        self._direction_evidence: Dict[str, _DirectionEvidence] = {}
        
        # Pending transition for async observation
        self._pending: Optional[Tuple[
            Tuple[int, ...],  # context
            str,              # control
            Dict[str, float], # before snapshot
            Tuple[Optional[float], float],  # before utility
            Optional[Position], # before position
        ]] = None
        
        self.latest_inference: Optional[Dict[str, Any]] = None
        self._memory_loaded = False

    # === AGENT METHODS (compatible with ControlLearner API) ===

    def select_control_for_direction(
        self,
        available_controls: List[str],
        desired_direction: Tuple[int, int],
        current_context: Tuple[int, ...],
        agent_position: Position,
    ) -> Optional[str]:
        """
        Select control that maximizes alignment with desired_direction.
        
        Uses learned mean_effects + exploration bonus for under-tried controls.
        Considers entity risk (avoid controls that lead toward pursuers).
        """
        if not available_controls:
            return None
        
        context_model = self.context_models.get(current_context)
        if context_model is None:
            # No context model - fall back to global direction evidence
            return self._choose_toward_direction_global(available_controls, desired_direction)
        
        # Score each control
        best_control = None
        best_score = float('-inf')
        
        # Get entity risk for this position
        entity_risk = self._compute_entity_risk_at(
            agent_position, desired_direction, current_context, available_controls
        )
        
        for control in available_controls:
            action_model = context_model.action_models.get(control)
            
            if action_model is None or len(action_model.samples) == 0:
                # Completely unexplored control: exploration bonus
                score = self.exploration_bonus * self._rng.random()
            else:
                # Use learned displacement (even with few samples)
                dx = action_model.mean_effects.get("__agent_dx", 0.0)
                dy = action_model.mean_effects.get("__agent_dy", 0.0)
                
                # Alignment with desired direction
                alignment = dx * desired_direction[0] + dy * desired_direction[1]
                
                # Exploration bonus (UCB-style) - decreases with more samples
                n_samples = len(action_model.samples)
                total_samples = sum(len(m.samples) for m in context_model.action_models.values())
                if total_samples > 0 and n_samples > 0:
                    exploration = self.exploration_bonus * math.sqrt(math.log(total_samples) / n_samples)
                else:
                    exploration = self.exploration_bonus
                
                # Confidence weight: lower confidence with fewer samples
                confidence = min(1.0, n_samples / max(1, self.min_samples_for_prediction))
                
                # Risk penalty
                risk_penalty = entity_risk.get(control, 0.0)
                
                # Score: weighted alignment + exploration - risk
                score = confidence * alignment + exploration - risk_penalty
            
            if score > best_score:
                best_score = score
                best_control = control
        
        return best_control

    def _choose_toward_direction_global(
        self,
        controls: List[str],
        desired_direction: Tuple[int, int],
    ) -> Optional[str]:
        """Fallback using global direction evidence (backward compat)."""
        learned: Dict[str, Tuple[float, float]] = {}
        unexplored: List[str] = []
        
        for control in controls:
            evidence = self._direction_evidence.get(control)
            if evidence is None or not evidence.origins:
                unexplored.append(control)
                continue
            if evidence.moving_samples:
                learned[control] = (
                    evidence.dx_total / evidence.moving_samples,
                    evidence.dy_total / evidence.moving_samples,
                )
                continue
            if len(evidence.origins) >= 4:
                learned[control] = (0.0, 0.0)
        
        aligned = [
            (direction[0] * desired_direction[0] + direction[1] * desired_direction[1], control)
            for control, direction in learned.items()
            if direction[0] * desired_direction[0] + direction[1] * desired_direction[1] > 0
        ]
        
        if aligned:
            return max(aligned, key=lambda item: (item[0], item[1]))[1]
        if unexplored:
            return self._rng.choice(unexplored)
        if learned:
            return max(
                controls,
                key=lambda c: (
                    learned.get(c, (0.0, 0.0))[0] * desired_direction[0]
                    + learned.get(c, (0.0, 0.0))[1] * desired_direction[1],
                    abs(learned.get(c, (0.0, 0.0))[0]) + abs(learned.get(c, (0.0, 0.0))[1]),
                    c,
                ),
            )
        return None

    def _compute_entity_risk_at(
        self,
        agent_pos: Position,
        desired_direction: Tuple[int, int],
        context: Tuple[int, ...],
        available_controls: List[str],
    ) -> Dict[str, float]:
        """Compute risk penalty for each control based on entity predictions."""
        risk = {}
        
        # Get context model for predicted displacements
        context_model = self.context_models.get(context)
        
        for entity_id, model in self.entity_models.items():
            if model.category in ("approaches_agent", "mobile") and model.category_confidence > 0.5:
                # Predict where entity will be for next few steps
                preds = model.predict_next_positions(3, agent_pos)
                
                for control in available_controls:
                    # Get predicted displacement for this control
                    pred_disp = (0.0, 0.0)
                    if context_model:
                        action_model = context_model.action_models.get(control)
                        if action_model and action_model.samples:
                            pred_disp = (
                                action_model.mean_effects.get("__agent_dx", 0.0),
                                action_model.mean_effects.get("__agent_dy", 0.0),
                            )
                    
                    # Fallback to global direction evidence
                    if pred_disp == (0.0, 0.0):
                        evidence = self._direction_evidence.get(control)
                        if evidence and evidence.moving_samples > 0:
                            pred_disp = (
                                evidence.dx_total / evidence.moving_samples,
                                evidence.dy_total / evidence.moving_samples,
                            )
                    
                    # Predict agent position after taking this control
                    next_agent_pos = (
                        agent_pos[0] + int(round(pred_disp[0])),
                        agent_pos[1] + int(round(pred_disp[1])),
                    )
                    
                    # Check if entity will be near that position
                    for step, pred_pos in enumerate(preds):
                        dist = abs(pred_pos[0] - next_agent_pos[0]) + abs(pred_pos[1] - next_agent_pos[1])
                        if dist <= 1:
                            penalty = 0.5 * model.category_confidence / (step + 1)
                            risk[control] = risk.get(control, 0.0) + penalty
                            break  # Only penalize once per control per entity
        
        return risk

    def predicted_displacement(self, control: str, context: Tuple[int, ...]) -> Tuple[float, float]:
        """Return mean (dx, dy) learned for control in context."""
        context_model = self.context_models.get(context)
        if context_model is None:
            # Fallback to global evidence
            evidence = self._direction_evidence.get(control)
            if evidence is None or evidence.moving_samples == 0:
                return (0.0, 0.0)
            return (
                evidence.dx_total / evidence.moving_samples,
                evidence.dy_total / evidence.moving_samples,
            )
        
        action_model = context_model.action_models.get(control)
        if action_model is None or not action_model.samples:
            return (0.0, 0.0)
        
        return (
            action_model.mean_effects.get("__agent_dx", 0.0),
            action_model.mean_effects.get("__agent_dy", 0.0),
        )

    def known_controls(self, context: Optional[Tuple[int, ...]] = None) -> List[str]:
        """Controls with at least min_samples in this context (or any context if None)."""
        if context is not None:
            context_model = self.context_models.get(context)
            if context_model is None:
                return []
            return [
                control for control, model in context_model.action_models.items()
                if len(model.samples) >= self.min_samples_for_prediction
            ]
        
        # Global: controls with any samples
        known = []
        for control, evidence in self._direction_evidence.items():
            if evidence.moving_samples >= self.min_samples_for_prediction:
                known.append(control)
        return known

    def record_agent_transition(
        self,
        context: Tuple[int, ...],
        control: str,
        before_snapshot: Dict[str, float],
        after_snapshot: Dict[str, float],
        reward: float,
        turn: int,
        effects: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record agent transition, update all models."""
        # Use pre-computed effects if provided, otherwise compute from snapshots
        if effects is None:
            effects = {
                key: after_snapshot.get(key, 0.0) - before_snapshot.get(key, 0.0)
                for key in before_snapshot.keys() | after_snapshot.keys()
                if after_snapshot.get(key, 0.0) != before_snapshot.get(key, 0.0)
            }
        
        # Ensure context model exists
        if context not in self.context_models:
            if len(self.context_models) >= self.max_contexts:
                # Evict oldest (simple LRU by last_updated)
                oldest = min(self.context_models.items(), key=lambda kv: kv[1].visit_count)
                del self.context_models[oldest[0]]
            self.context_models[context] = ContextTransitionModel(context=context)
        
        context_model = self.context_models[context]
        context_model.visit_count += 1
        
        # Get or create action model
        action_model = context_model.get_or_create_action_model(control)
        action_model.add_sample(effects, reward, turn)
        
        # Update global direction evidence (backward compat)
        self._add_direction_evidence(control, effects)
        
        # Update legacy trials dict for backward compatibility
        self._update_legacy_trials(context, control, effects)
        
        # Update latest inference
        self._update_latest_inference(context, control, action_model)

    def _update_legacy_trials(
        self,
        context: Tuple[int, ...],
        control: str,
        effects: Dict[str, float],
    ) -> None:
        """Update legacy trials dict for backward compatibility (used by tests)."""
        # This is a no-op in base WorldModelLearner; overridden in ControlLearner
        pass

    def _update_latest_inference(
        self,
        context: Tuple[int, ...],
        control: str,
        action_model: AgentTransitionModel,
    ) -> None:
        self.latest_inference = {
            "control": control,
            "context": context,
            "samples": len(action_model.samples),
            "mean_reward": action_model.mean_reward,
            "mean_effects": {
                k: v for k, v in action_model.mean_effects.items()
                if v != 0 and k not in {"__agent_x", "__agent_y"}
            },
        }

    # === ENTITY METHODS (NEW) ===

    def observe_entity(
        self,
        entity_id: str,
        position: Position,
        agent_position: Position,
        turn: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntityDynamicsModel:
        """Update/create dynamic model for entity. Returns updated model."""
        if entity_id not in self.entity_models:
            if len(self.entity_models) >= self.max_entities:
                # Evict oldest
                oldest = min(self.entity_models.items(), key=lambda kv: kv[1].last_seen_turn)
                del self.entity_models[oldest[0]]
            self.entity_models[entity_id] = EntityDynamicsModel(entity_id=entity_id)
        
        model = self.entity_models[entity_id]
        model.update(position, agent_position, turn)
        return model

    def predict_entity_positions(
        self,
        entity_id: str,
        horizon: int,
        agent_position: Position,
    ) -> List[Position]:
        """Predict future positions for entity."""
        model = self.entity_models.get(entity_id)
        if model is None:
            return []
        return model.predict_next_positions(horizon, agent_position)

    def get_entity_category(self, entity_id: str) -> Tuple[str, float]:
        """Return (category, confidence) for entity."""
        model = self.entity_models.get(entity_id)
        if model is None:
            return "unknown", 0.0
        return model.category, model.category_confidence

    def get_all_entity_predictions(
        self,
        horizon: int,
        agent_position: Position,
    ) -> Dict[str, List[Position]]:
        """Predictions for all known entities - for MCTS BeliefState."""
        return {
            eid: model.predict_next_positions(horizon, agent_position)
            for eid, model in self.entity_models.items()
        }

    # === CONTEXT METHODS ===

    def get_context_transition_model(self, context: Tuple[int, ...]) -> Optional[ContextTransitionModel]:
        return self.context_models.get(context)

    @property
    def trials(self) -> dict[tuple[int, ...], dict[str, list[dict[str, float]]]]:
        """Backward-compatible property returning trials in old format."""
        result: dict[tuple[int, ...], dict[str, list[dict[str, float]]]] = {}
        for context, ctm in self.context_models.items():
            for control, atm in ctm.action_models.items():
                if atm.samples:
                    result.setdefault(context, {}).setdefault(control, []).extend(atm.samples)
        return result

    # === PERSISTENCE ===

    def save_all(self, db_path: Path, game_id: str) -> None:
        """Persist agent_transitions, entity_models, context_models to SQLite."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            # Create tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_entity_models (
                    game_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    category_confidence REAL NOT NULL,
                    velocity_x REAL,
                    velocity_y REAL,
                    approach_streak INTEGER,
                    retreat_streak INTEGER,
                    last_seen_turn INTEGER,
                    last_agent_x INTEGER,
                    last_agent_y INTEGER,
                    updated_at INTEGER,
                    PRIMARY KEY (game_id, entity_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_entity_positions (
                    game_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    agent_x INTEGER,
                    agent_y INTEGER,
                    PRIMARY KEY (game_id, entity_id, turn)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_context_transitions (
                    game_id TEXT NOT NULL,
                    context TEXT NOT NULL,
                    control TEXT NOT NULL,
                    mean_agent_dx REAL,
                    mean_agent_dy REAL,
                    mean_reward REAL,
                    sample_count INTEGER,
                    PRIMARY KEY (game_id, context, control)
                )
            """)
            
            # Save entity models
            now = int(time.time())
            for entity_id, model in self.entity_models.items():
                avg_vx = 0.0
                avg_vy = 0.0
                if model.velocities:
                    avg_vx = sum(v[0] for v in model.velocities) / len(model.velocities)
                    avg_vy = sum(v[1] for v in model.velocities) / len(model.velocities)
                
                last_agent_x = model.last_agent_pos[0] if model.last_agent_pos else None
                last_agent_y = model.last_agent_pos[1] if model.last_agent_pos else None
                
                conn.execute("""
                    INSERT OR REPLACE INTO game_entity_models
                    (game_id, entity_id, category, category_confidence, velocity_x, velocity_y,
                     approach_streak, retreat_streak, last_seen_turn, last_agent_x, last_agent_y, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_id, entity_id, model.category, model.category_confidence,
                    avg_vx, avg_vy, model.approach_streak, model.retreat_streak,
                    model.last_seen_turn, last_agent_x, last_agent_y, now
                ))
                
                # Save position history (last 20 positions)
                for pos, turn in model.positions[-20:]:
                    conn.execute("""
                        INSERT OR REPLACE INTO game_entity_positions
                        (game_id, entity_id, turn, x, y, agent_x, agent_y)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (game_id, entity_id, turn, pos[0], pos[1], last_agent_x, last_agent_y))
            
            # Save context transitions (summary only)
            for context, ctm in self.context_models.items():
                for control, atm in ctm.action_models.items():
                    if atm.samples:
                        conn.execute("""
                            INSERT OR REPLACE INTO game_context_transitions
                            (game_id, context, control, mean_agent_dx, mean_agent_dy, mean_reward, sample_count)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            game_id, json.dumps(context), control,
                            atm.mean_effects.get("__agent_dx", 0.0),
                            atm.mean_effects.get("__agent_dy", 0.0),
                            atm.mean_reward,
                            len(atm.samples)
                        ))
            
            conn.commit()

    def load_all(self, db_path: Path, game_id: str) -> None:
        """Load all models from SQLite."""
        if not db_path.exists():
            return
        
        with closing(sqlite3.connect(db_path)) as conn:
            # Load entity models
            rows = conn.execute("""
                SELECT entity_id, category, category_confidence, velocity_x, velocity_y,
                       approach_streak, retreat_streak, last_seen_turn, last_agent_x, last_agent_y
                FROM game_entity_models
                WHERE game_id = ?
            """, (game_id,)).fetchall()
            
            for row in rows:
                (entity_id, category, conf, vx, vy, approach_streak, retreat_streak,
                 last_seen_turn, last_agent_x, last_agent_y) = row
                
                model = EntityDynamicsModel(entity_id=entity_id)
                model.category = category
                model.category_confidence = conf
                model.approach_streak = approach_streak
                model.retreat_streak = retreat_streak
                model.last_seen_turn = last_seen_turn
                if last_agent_x is not None and last_agent_y is not None:
                    model.last_agent_pos = (last_agent_x, last_agent_y)
                if vx is not None and vy is not None:
                    model.velocities.append((vx, vy))
                    model.last_velocity = (vx, vy)
                
                # Load position history
                pos_rows = conn.execute("""
                    SELECT turn, x, y FROM game_entity_positions
                    WHERE game_id = ? AND entity_id = ?
                    ORDER BY turn
                """, (game_id, entity_id)).fetchall()
                
                for turn, x, y in pos_rows:
                    model.positions.append(((x, y), turn))
                
                self.entity_models[entity_id] = model
            
            # Load context transitions (summary only, rebuild action models)
            rows = conn.execute("""
                SELECT context, control, mean_agent_dx, mean_agent_dy, mean_reward, sample_count
                FROM game_context_transitions
                WHERE game_id = ?
            """, (game_id,)).fetchall()
            
            for context_json, control, mean_dx, mean_dy, mean_reward, sample_count in rows:
                context = tuple(json.loads(context_json))
                if context not in self.context_models:
                    self.context_models[context] = ContextTransitionModel(context=context)
                ctm = self.context_models[context]
                atm = ctm.get_or_create_action_model(control)
                atm.mean_effects["__agent_dx"] = mean_dx
                atm.mean_effects["__agent_dy"] = mean_dy
                atm.mean_reward = mean_reward
                # Note: we don't restore individual samples, just summary stats
                # Create dummy samples with required keys for _DirectionEvidence.observe()
                # Use a dummy position (0, 0) and the learned mean displacement
                dummy_sample = {
                    "__agent_dx": mean_dx,
                    "__agent_dy": mean_dy,
                    "__agent_x": 0.0,
                    "__agent_y": 0.0,
                }
                atm.samples = [dummy_sample] * sample_count
        
        # Rebuild direction evidence from loaded context models
        self._rebuild_direction_evidence()

    def _rebuild_direction_evidence(self) -> None:
        """Rebuild global direction evidence from context models."""
        self._direction_evidence.clear()
        for ctm in self.context_models.values():
            for control, atm in ctm.action_models.items():
                for sample in atm.samples:
                    self._add_direction_evidence(control, sample)

    # === DIAGNOSTICS ===

    def get_diagnostics(self) -> Dict[str, Any]:
        """Stats for debugging."""
        category_dist: Dict[str, int] = {}
        for model in self.entity_models.values():
            category_dist[model.category] = category_dist.get(model.category, 0) + 1
        
        total_samples = sum(
            len(atm.samples)
            for ctm in self.context_models.values()
            for atm in ctm.action_models.values()
        )
        
        return {
            "num_contexts": len(self.context_models),
            "num_entities": len(self.entity_models),
            "total_transition_samples": total_samples,
            "avg_samples_per_context": (
                total_samples / len(self.context_models) if self.context_models else 0
            ),
            "category_distribution": category_dist,
            "entity_details": {
                eid: {
                    "category": m.category,
                    "confidence": m.category_confidence,
                    "positions": len(m.positions),
                    "velocities": len(m.velocities),
                    "approach_streak": m.approach_streak,
                    "retreat_streak": m.retreat_streak,
                }
                for eid, m in self.entity_models.items()
            },
        }

    # === INTERNAL HELPERS ===

    def _add_direction_evidence(self, control: str, sample: Dict[str, float]) -> None:
        self._direction_evidence.setdefault(control, _DirectionEvidence()).observe(sample)

    @staticmethod
    def _numeric_snapshot(observation: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric snapshot from observation."""
        snapshot: Dict[str, float] = {}

        def visit(value: Any, path: str) -> None:
            if isinstance(value, bool):
                snapshot[path] = float(value)
                snapshot[f"{path}#present"] = 1.0
            elif isinstance(value, (int, float)):
                snapshot[path] = float(value)
                snapshot[f"{path}#present"] = 1.0
            elif isinstance(value, str):
                snapshot[f"{path}={value}"] = 1.0
            elif value is None:
                snapshot[f"{path}#null"] = 1.0
            elif isinstance(value, dict):
                snapshot[f"{path}#keys={','.join(sorted(str(k) for k in value.keys()))}"] = 1.0
                for key, child in value.items():
                    visit(child, f"{path}.{key}" if path else str(key))
            elif isinstance(value, list):
                snapshot[f"{path}#length"] = float(len(value))
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")

        visit(observation, "")
        return snapshot

    @staticmethod
    def _context_from_state(state: Any) -> Tuple[int, ...]:
        """Extract context from state (same as old ControlLearner)."""
        position = getattr(state, "current_position", None)
        if position is None:
            return ()
        nearby_entities = [
            (
                abs(entity.position[0] - position[0]) + abs(entity.position[1] - position[1]),
                sum(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in entity.metadata.values()
                ),
            )
            for entity in getattr(state, "entities", [])
            if entity.entity_id is not None and entity.position is not None
        ]
        if not nearby_entities:
            return ()
        nearest_distance = min(distance for distance, _ in nearby_entities)
        observable_features = sum(
            feature_count
            for distance, feature_count in nearby_entities
            if distance == nearest_distance
        )
        return (nearest_distance, observable_features)

    @staticmethod
    def _position(state: Any) -> Optional[Position]:
        pos = getattr(state, "current_position", None)
        if isinstance(pos, (list, tuple)) and len(pos) == 2 and all(isinstance(p, int) for p in pos):
            return (pos[0], pos[1])
        return None

    @staticmethod
    def _utility(state: Any, memory: Any) -> Tuple[Optional[float], float]:
        position = getattr(state, "current_position", None)
        target = getattr(state, "target_position", None)
        distance = None
        if (isinstance(position, (list, tuple)) and len(position) == 2 and
            isinstance(target, (list, tuple)) and len(target) == 2 and
            all(isinstance(p, int) for p in position) and all(isinstance(t, int) for t in target)):
            distance = abs(position[0] - target[0]) + abs(position[1] - target[1])
        goal_progress: Optional[float] = float(distance) if distance is not None else None
        
        grid_size = getattr(state, "grid_size", None)
        if (isinstance(grid_size, tuple) and len(grid_size) == 2 and
            all(isinstance(v, int) and v > 0 for v in grid_size)):
            area = max(grid_size[0] * grid_size[1], 1)
            if goal_progress is not None:
                goal_progress /= max(sum(grid_size), 1)
        else:
            area = 1
        
        spatial = getattr(memory, "spatial", None)
        known_cells = getattr(spatial, "grid", {})
        coverage = len(known_cells) / area if isinstance(known_cells, dict) else 0.0
        return goal_progress, coverage

    # === BACKWARD COMPATIBILITY: ControlLearner API ===

    def choose(
        self,
        state: Any,
        observation: Dict[str, Any],
        available_controls: Any,
        memory: Any = None,
        desired_direction: Optional[Tuple[int, int]] = None,
    ) -> Optional[str]:
        """Backward-compatible choose method."""
        if not self._memory_loaded and memory is not None:
            load_trials = getattr(memory, "load_control_trials", None)
            if callable(load_trials):
                trials = load_trials()
                # Convert old trials format to new context models
                self._load_trials_legacy(trials)
            load_outcomes = getattr(memory, "load_control_outcomes", None)
            if callable(load_outcomes):
                outcomes = load_outcomes()
                self._load_outcomes_legacy(outcomes)
            self._memory_loaded = True
        
        self._observe_transition(observation, state, memory)
        
        if not isinstance(available_controls, list):
            return None
        
        controls = [c for c in available_controls if isinstance(c, str) and c]
        if not controls:
            return None
        
        context = self._context_from_state(state)
        agent_pos = self._position(state)
        
        # Infer desired_direction from target_position if not provided
        if desired_direction is None and agent_pos is not None:
            target = getattr(state, "target_position", None)
            if isinstance(target, (list, tuple)) and len(target) == 2 and all(isinstance(t, int) for t in target):
                dx = target[0] - agent_pos[0]
                dy = target[1] - agent_pos[1]
                if dx != 0 or dy != 0:
                    desired_direction = (
                        1 if dx > 0 else (-1 if dx < 0 else 0),
                        1 if dy > 0 else (-1 if dy < 0 else 0),
                    )
        
        if desired_direction is not None and agent_pos is not None:
            control = self.select_control_for_direction(
                controls, desired_direction, context, agent_pos
            )
            if control is not None:
                self._pending = (
                    context, control,
                    self._numeric_snapshot(observation),
                    self._utility(state, memory),
                    agent_pos,
                )
                return control
        
        # Exploration fallback
        context_model = self.context_models.get(context)
        if context_model and all(
            len(context_model.action_models.get(c, AgentTransitionModel(context, c)).samples) >= 3
            for c in controls
        ):
            return None
        
        context_outcomes = {}
        if context_model:
            context_outcomes = {c: m.reward_samples for c, m in context_model.action_models.items()}
        
        counts = {c: len(context_outcomes.get(c, [])) for c in controls}
        total = sum(counts.values())
        scores = {
            c: ((sum(context_outcomes.get(c, [])) / counts[c]) if counts[c] else 0.0)
            + 0.35 * math.sqrt(math.log(total + 2) / (counts[c] + 1))
            for c in controls
        }
        best_score = max(scores.values())
        candidates = [c for c, s in scores.items() if s == best_score]
        control = self._rng.choice(candidates)
        
        self._pending = (
            context, control,
            self._numeric_snapshot(observation),
            self._utility(state, memory),
            agent_pos,
        )
        return control

    def _observe_transition(
        self,
        observation: Dict[str, Any],
        state: Any,
        memory: Any,
    ) -> None:
        if self._pending is None:
            return
        context, control, before, before_utility, before_position = self._pending
        after = self._numeric_snapshot(observation)
        after_position = self._position(state)
        
        if before_position is not None and after_position is not None:
            effects = {
                key: after.get(key, 0.0) - before.get(key, 0.0)
                for key in before.keys() | after.keys()
                if after.get(key, 0.0) != before.get(key, 0.0)
            }
            effects["__agent_dx"] = float(after_position[0] - before_position[0])
            effects["__agent_dy"] = float(after_position[1] - before_position[1])
            effects["__agent_x"] = float(before_position[0])
            effects["__agent_y"] = float(before_position[1])
        else:
            effects = {
                key: after.get(key, 0.0) - before.get(key, 0.0)
                for key in before.keys() | after.keys()
                if after.get(key, 0.0) != before.get(key, 0.0)
            }
        
        after_utility = self._utility(state, memory)
        if before_utility[0] is not None and after_utility[0] is not None:
            reward = before_utility[0] - after_utility[0]
        else:
            reward = after_utility[1] - before_utility[1]
        
        # Record using new API
        turn = len(getattr(memory, "experiences", [])) if memory else 0
        self.record_agent_transition(context, control, before, after, reward, turn, effects)
        
        # Persist via memory if available
        save_trial = getattr(memory, "save_control_trial", None)
        if callable(save_trial):
            save_trial(context, control, effects)
        
        self._pending = None

    def observe_action_result(
        self,
        details: Dict[str, Any],
        memory: Any,
    ) -> None:
        if self._pending is None:
            return
        status = details.get("status")
        if status not in {"goal_reached", "won", "caught", "lost"}:
            return
        context, control, _, _, before_position = self._pending
        reward = 1.0 if status in {"goal_reached", "won"} else -1.0
        position = details.get("position")
        after_position = self._position_value(position)
        if before_position is not None and after_position is not None:
            effects = {
                "__agent_dx": float(after_position[0] - before_position[0]),
                "__agent_dy": float(after_position[1] - before_position[1]),
                "__agent_x": float(before_position[0]),
                "__agent_y": float(before_position[1]),
            }
            turn = len(getattr(memory, "experiences", [])) if memory else 0
            self.record_agent_transition(context, control, {}, effects, reward, turn)
            save_trial = getattr(memory, "save_control_trial", None)
            if callable(save_trial):
                save_trial(context, control, effects)
        self._pending = None

    def record_trial(
        self,
        context: Tuple[int, ...],
        control: str,
        effects: Dict[str, float],
    ) -> None:
        """Backward compat: record trial directly."""
        # Default reward 0 for manual trials
        turn = 0
        self.record_agent_transition(context, control, {}, effects, 0.0, turn)

    def load_trials(
        self,
        trials: Dict[Tuple[int, ...], Dict[str, List[Dict[str, float]]]],
    ) -> None:
        """Backward compat: load trials from memory."""
        self._load_trials_legacy(trials)

    def _load_trials_legacy(
        self,
        trials: Dict[Tuple[int, ...], Dict[str, List[Dict[str, float]]]],
    ) -> None:
        for context, context_trials in trials.items():
            if context not in self.context_models:
                self.context_models[context] = ContextTransitionModel(context=context)
            ctm = self.context_models[context]
            for control, samples in context_trials.items():
                atm = ctm.get_or_create_action_model(control)
                for sample in samples:
                    atm.samples.append(sample)
                atm._recompute_stats()
        self._rebuild_direction_evidence()

    def _load_outcomes_legacy(
        self,
        outcomes: Dict[Tuple[int, ...], Dict[str, List[float]]],
    ) -> None:
        # Outcomes are now stored within action_models as reward_samples
        for context, context_outcomes in outcomes.items():
            if context in self.context_models:
                ctm = self.context_models[context]
                for control, rewards in context_outcomes.items():
                    atm = ctm.get_or_create_action_model(control)
                    atm.reward_samples = rewards
                    atm.mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    @staticmethod
    def _position_value(position: Any) -> Optional[Position]:
        if isinstance(position, (list, tuple)) and len(position) == 2 and all(isinstance(p, int) for p in position):
            return (position[0], position[1])
        return None


class _DirectionEvidence:
    """Backward-compatible direction evidence tracking."""
    def __init__(self) -> None:
        self.origins: list[tuple[int, int]] = []
        self.dx_total = 0.0
        self.dy_total = 0.0
        self.moving_samples = 0

    def observe(self, effects: Dict[str, float]) -> None:
        dx = effects.get("__agent_dx", 0.0)
        dy = effects.get("__agent_dy", 0.0)
        if dx != 0 or dy != 0:
            self.moving_samples += 1
            self.dx_total += dx
            self.dy_total += dy


# Alias for backward compatibility
ControlLearner = WorldModelLearner