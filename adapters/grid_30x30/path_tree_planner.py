"""
MCTS-Lite Path Planner with Belief State for Grid 30x30.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .world_model import WorldState
from .memory import GridMemoryStore
from .control_learner import ControlLearner, WorldModelLearner
from .planner import ActionCandidate


Position = Tuple[int, int]


class CellBelief(Enum):
    """Belief about a cell's traversability."""
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2


@dataclass
class EntityModel:
    """
    Predictive model for a tracked entity.
    
    Stores enough information to predict future positions over a horizon.
    """
    entity_id: str
    category: str  # "stationary", "mobile", "approaches_agent", "moves_away_from_agent"
    position: Position
    velocity: Tuple[float, float] = (0.0, 0.0)
    behavior_confidence: float = 0.0
    last_seen_turn: int = 0
    
    def predict_next_positions(self, horizon: int) -> List[Position]:
        """
        Predict entity positions for the next `horizon` steps.
        
        Simple constant-velocity prediction; subclasses can override
        for more sophisticated dynamics (e.g., pursuit, patrol).
        """
        positions = []
        px, py = self.position
        vx, vy = self.velocity
        for step in range(1, horizon + 1):
            nx = round(px + vx * step)
            ny = round(py + vy * step)
            positions.append((nx, ny))
        return positions
    
    def update(self, new_position: Position, turn: int) -> None:
        """Update model with new observation."""
        dt = turn - self.last_seen_turn
        if dt > 0:
            self.velocity = (
                (new_position[0] - self.position[0]) / dt,
                (new_position[1] - self.position[1]) / dt,
            )
        self.position = new_position
        self.last_seen_turn = turn


@dataclass
class BeliefState:
    """
    Probabilistic belief state for planning under partial observability.
    
    Components:
    - goal_particles: Particle filter over possible goal positions (pos, weight)
    - cell_beliefs: Map from position to CellBelief (FREE, OBSTACLE, UNKNOWN)
    - entity_models: Predictive models for tracked entities
    - entropy: Global uncertainty measure (higher = more uncertain)
    - turn: Current game turn for temporal reasoning
    """
    goal_particles: List[Tuple[Position, float]] = field(default_factory=list)
    cell_beliefs: Dict[Position, CellBelief] = field(default_factory=dict)
    entity_models: Dict[str, EntityModel] = field(default_factory=dict)
    entropy: float = 1.0
    turn: int = 0
    
    def update(self, observation: Dict[str, Any], memory: GridMemoryStore) -> "BeliefState":
        """
        Bayesian update of belief state given new observation and memory.
        
        Returns a new BeliefState (immutable update pattern).
        """
        new_belief = BeliefState(
            goal_particles=[],
            cell_beliefs=dict(self.cell_beliefs),
            entity_models=dict(self.entity_models),
            entropy=0.0,
            turn=self.turn + 1,
        )
        
        # Extract observation data
        hud = observation.get("hud", {}) if isinstance(observation, dict) else {}
        grid = hud.get("grid", {}) if isinstance(hud, dict) else {}
        visible_cells = grid.get("visible_cells", [])
        if not isinstance(visible_cells, list):
            visible_cells = []
        
        target_pos = grid.get("target")
        if isinstance(target_pos, (list, tuple)) and len(target_pos) == 2:
            target_pos = (int(target_pos[0]), int(target_pos[1]))
        else:
            target_pos = None
        
        obstacle_positions = grid.get("obstacles", [])
        if not isinstance(obstacle_positions, list):
            obstacle_positions = []
        
        observed_entities = grid.get("entities", [])
        if not isinstance(observed_entities, list):
            observed_entities = []
        
        agent_pos = memory.working.current_observation
        if isinstance(agent_pos, dict):
            agent_pos = agent_pos.get("position")
        if isinstance(agent_pos, (list, tuple)) and len(agent_pos) == 2:
            agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        else:
            agent_pos = memory.spatial.grid and next(iter(memory.spatial.grid), None)
        
        # 1. Update goal particles
        new_belief.goal_particles = self._update_goal_particles(
            target_pos, visible_cells, memory, new_belief.turn
        )
        
        # 2. Update cell beliefs from observation
        self._update_cell_beliefs(new_belief, visible_cells, obstacle_positions, target_pos)
        
        # 3. Update entity models
        self._update_entity_models(new_belief, observed_entities, agent_pos, new_belief.turn)
        
        # 4. Recompute entropy
        new_belief.entropy = self._compute_entropy(new_belief)
        
        return new_belief
    
    def _update_goal_particles(
        self,
        target_pos: Optional[Position],
        visible_cells: List[Position],
        memory: GridMemoryStore,
        turn: int,
    ) -> List[Tuple[Position, float]]:
        """Update particle filter over goal positions."""
        visible_set = set(visible_cells)
        known_goals = memory.spatial.known_goal_positions
        goal_is_dynamic = memory.spatial.goal_is_dynamic
        
        if target_pos is not None:
            # Goal observed: collapse particles to observed position with high weight
            return [(target_pos, 1.0)]
        
        # Goal not observed: propagate existing particles
        if not self.goal_particles:
            # Initialize from memory if no particles exist
            if known_goals:
                if goal_is_dynamic:
                    # Spread particles around known goals
                    particles = []
                    for goal in known_goals:
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                pos = (goal[0] + dx, goal[1] + dy)
                                if pos not in visible_set:
                                    particles.append((pos, 1.0 / 9.0))
                    return particles[:30]  # Cap at num_particles
                else:
                    # Static goal: uniform over known goals
                    weight = 1.0 / len(known_goals)
                    return [(g, weight) for g in known_goals]
            else:
                # No goal knowledge: uniform over frontier
                frontier = memory.spatial.frontier
                if frontier:
                    weight = 1.0 / len(frontier)
                    return [(f, weight) for f in frontier]
                return []
        
        # Propagate particles according to dynamics
        new_particles = []
        for pos, weight in self.goal_particles:
            if goal_is_dynamic:
                # Goal moves every 2 turns: random walk
                if turn % 2 == 0:
                    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                        new_pos = (pos[0] + dx, pos[1] + dy)
                        if new_pos not in visible_set:
                            new_particles.append((new_pos, weight / 5.0))
                else:
                    new_particles.append((pos, weight))
            else:
                # Static goal: keep particles, but eliminate those in visible cells
                if pos not in visible_set:
                    new_particles.append((pos, weight))
        
        # Normalize weights
        total_weight = sum(w for _, w in new_particles)
        if total_weight > 0:
            new_particles = [(p, w / total_weight) for p, w in new_particles]
        
        # Resample if too few particles
        if len(new_particles) < 5 and known_goals:
            for goal in known_goals:
                if goal not in visible_set:
                    new_particles.append((goal, 1.0 / len(known_goals)))
        
        return new_particles[:30]  # Cap
    
    def _update_cell_beliefs(
        self,
        new_belief: BeliefState,
        visible_cells: List[Position],
        obstacle_positions: List[Position],
        target_pos: Optional[Position],
    ) -> None:
        """Update cell beliefs from observation."""
        visible_set = set(visible_cells)
        obstacle_set = set(obstacle_positions)
        
        # Mark visible cells as FREE (unless obstacle)
        for pos in visible_cells:
            if pos in obstacle_set:
                new_belief.cell_beliefs[pos] = CellBelief.OBSTACLE
            else:
                new_belief.cell_beliefs[pos] = CellBelief.FREE
        
        # Mark obstacles
        for pos in obstacle_positions:
            new_belief.cell_beliefs[pos] = CellBelief.OBSTACLE
        
        # Goal cell is FREE when visible
        if target_pos is not None:
            new_belief.cell_beliefs[target_pos] = CellBelief.FREE
        
        # Cells no longer visible but previously known: keep their belief
        # (handled by copying self.cell_beliefs at start)
    
    def _update_entity_models(
        self,
        new_belief: BeliefState,
        observed_entities: List[Dict[str, Any]],
        agent_pos: Optional[Position],
        turn: int,
    ) -> None:
        """Update predictive models for tracked entities."""
        observed_ids = set()
        
        # Use WorldModelLearner if available for better entity predictions
        if self.world_model_learner is not None and agent_pos is not None:
            for ent in observed_entities:
                if not isinstance(ent, dict):
                    continue
                entity_id = ent.get("id")
                position = ent.get("position")
                if not entity_id or not isinstance(entity_id, str):
                    continue
                if not isinstance(position, (list, tuple)) or len(position) != 2:
                    continue
                pos = (int(position[0]), int(position[1]))
                observed_ids.add(entity_id)
                
                # Update/create model in WorldModelLearner
                wm_model = self.world_model_learner.observe_entity(
                    entity_id, pos, agent_pos, turn
                )
                
                # Copy to belief state's local EntityModel for simulation
                new_belief.entity_models[entity_id] = EntityModel(
                    entity_id=entity_id,
                    category=wm_model.category,
                    position=wm_model.position,
                    velocity=wm_model.velocities[-1] if wm_model.velocities else (0.0, 0.0),
                    behavior_confidence=wm_model.category_confidence,
                    last_seen_turn=turn,
                )
        else:
            # Fallback: local entity models
            for ent in observed_entities:
                if not isinstance(ent, dict):
                    continue
                entity_id = ent.get("id")
                position = ent.get("position")
                if not entity_id or not isinstance(entity_id, str):
                    continue
                if not isinstance(position, (list, tuple)) or len(position) != 2:
                    continue
                pos = (int(position[0]), int(position[1]))
                observed_ids.add(entity_id)
                
                if entity_id in new_belief.entity_models:
                    model = new_belief.entity_models[entity_id]
                    model.update(pos, turn)
                    # Update category from memory if available
                    mem_entity = new_belief.entity_models.get(entity_id)
                    if mem_entity:
                        model.category = mem_entity.category
                        model.behavior_confidence = mem_entity.behavior_confidence
                else:
                    # New entity - create model
                    category = "unclassified"
                    confidence = 0.0
                    mem_ent = new_belief.entity_models.get(entity_id)
                    if mem_ent:
                        category = mem_ent.category
                        confidence = mem_ent.behavior_confidence
                    new_belief.entity_models[entity_id] = EntityModel(
                        entity_id=entity_id,
                        category=category,
                        position=pos,
                        behavior_confidence=confidence,
                        last_seen_turn=turn,
                    )
        
        # Remove entities not seen for too long (optional, keep for now)
        # Could add max_age parameter
    
    def _compute_entropy(self, belief: BeliefState) -> float:
        """Compute global uncertainty entropy."""
        if not belief.goal_particles:
            return 1.0
        
        # Entropy of goal distribution
        entropy = 0.0
        for _, weight in belief.goal_particles:
            if weight > 0:
                entropy -= weight * math.log(weight)
        
        # Normalize by max entropy (log N)
        max_entropy = math.log(len(belief.goal_particles)) if belief.goal_particles else 1.0
        goal_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Add cell belief uncertainty
        unknown_cells = sum(1 for v in belief.cell_beliefs.values() if v == CellBelief.UNKNOWN)
        total_cells = max(len(belief.cell_beliefs), 1)
        cell_entropy = unknown_cells / total_cells
        
        # Weighted combination
        return 0.7 * goal_entropy + 0.3 * cell_entropy
    
    def sample_goal(self, rng: Optional[random.Random] = None) -> Optional[Position]:
        """Sample a goal position from the particle distribution."""
        if not self.goal_particles:
            return None
        
        # Weighted random sample
        _rng = rng or random
        rand = _rng.random()
        cumulative = 0.0
        for pos, weight in self.goal_particles:
            cumulative += weight
            if rand <= cumulative:
                return pos
        return self.goal_particles[-1][0]


@dataclass
class PathTreeNode:
    """
    Node in the MCTS path planning tree.
    
    Each node represents a belief state reached by taking an action.
    Uses UCB1 for selection with Dirichlet prior from episodic retrieval.
    """
    belief: BeliefState
    action: Optional[ActionCandidate] = None
    action_direction: Optional[Position] = None  # Direction taken to reach this node
    children: Dict[Position, "PathTreeNode"] = field(default_factory=dict)
    parent: Optional["PathTreeNode"] = None
    visit_count: int = 0
    value_sum: float = 0.0
    uncertainty: float = 1.0
    retrieval_prior: Dict[Position, float] = field(default_factory=dict)
    depth: int = 0
    
    @property
    def value(self) -> float:
        """Mean value of this node."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
    
    def ucb_score(self, parent_visits: int, c: float = 1.414) -> float:
        """
        UCB1 score with retrieval prior bonus.
        
        Unvisited nodes get infinite score to ensure exploration.
        """
        if self.visit_count == 0:
            # Unvisited nodes: prioritize those with retrieval prior
            prior = self.retrieval_prior.get(self.action_direction, 0.0)
            return float('inf') if prior > 0 else float('inf')
        
        if parent_visits <= 0:
            # Root node not visited yet, use exploration only
            return self.value + c * math.sqrt(1.0 / self.visit_count)
        
        exploitation = self.value
        exploration = c * math.sqrt(math.log(parent_visits) / self.visit_count)
        prior_bonus = self.retrieval_prior.get(self.action_direction, 0.0) * 0.1
        return exploitation + exploration + prior_bonus
    
    def is_fully_expanded(self, valid_directions: List[Position]) -> bool:
        """Check if all valid directions have been expanded."""
        return all(d in self.children for d in valid_directions)
    
    def best_child(self, c: float = 1.414, valid_directions: Optional[List[Position]] = None) -> Optional["PathTreeNode"]:
        """Select best child by UCB score, considering only valid directions."""
        if not self.children:
            return None
        candidates = self.children.values()
        if valid_directions is not None:
            candidates = [c for c in candidates if c.action_direction in valid_directions]
        if not candidates:
            return None
        return max(candidates, key=lambda child: child.ucb_score(self.visit_count, c))
    
    def most_visited_child(self, valid_directions: Optional[List[Position]] = None) -> Optional["PathTreeNode"]:
        """Select child with highest visit count (robust policy), considering only valid directions.
        
        Ties are broken by average value (higher is better).
        """
        if not self.children:
            return None
        candidates = self.children.values()
        if valid_directions is not None:
            candidates = [c for c in candidates if c.action_direction in valid_directions]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda child: (
                child.visit_count,
                child.value_sum / child.visit_count if child.visit_count > 0 else float('-inf')
            )
        )


class GridPathTreePlanner:
    """
    MCTS-Lite planner with Belief State for partially observable grid worlds.
    
    Features:
    - Particle filter belief over goal positions
    - Cell belief tracking (FREE/OBSTACLE/UNKNOWN)
    - Entity predictive models
    - Episodic retrieval priors for guided exploration
    - Tree reuse (warm-start) across turns
    - Deterministic given seed
    
    Usage:
        planner = GridPathTreePlanner(max_depth=8, num_simulations=50, seed=42)
        action = planner.plan(world_state, memory)
    """
    
    # Cardinal directions
    DIRECTIONS: Tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
    
    def __init__(
        self,
        max_depth: int = 8,
        num_simulations: int = 50,
        exploration_constant: float = 1.414,
        num_particles: int = 30,
        control_learner: Optional[ControlLearner] = None,
        world_model_learner: Optional[WorldModelLearner] = None,
        memory: Optional[GridMemoryStore] = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize the path tree planner.
        
        Args:
            max_depth: Maximum tree depth for simulations
            num_simulations: Number of MCTS iterations per plan call
            exploration_constant: UCB exploration constant (c)
            num_particles: Number of particles for goal belief
            control_learner: Optional ControlLearner for transition prediction
            world_model_learner: Optional WorldModelLearner for entity predictions
            memory: Optional GridMemoryStore for retrieval priors
            seed: Random seed for deterministic behavior
        """
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.num_particles = num_particles
        self.control_learner = control_learner
        self.world_model_learner = world_model_learner
        self.memory = memory
        self._rng = random.Random(seed)
        self._root: Optional[PathTreeNode] = None
        self._last_plan_turn = -1
    
    def plan(self, state: WorldState, memory: GridMemoryStore) -> ActionCandidate:
        """
        Run MCTS planning and return the best action.
        
        Args:
            state: Current world state
            memory: Grid memory store for retrieval and beliefs
            
        Returns:
            Best ActionCandidate found by MCTS
        """
        # 1. Build root BeliefState from state + memory
        root_belief = self._build_root_belief(state, memory)
        
        # 2. Build/reuse root PathTreeNode (warm-start)
        if self._root is not None and self._last_plan_turn == state.confidence:  # proxy for turn
            self._root = self.reuse_tree(self._root, root_belief)
        else:
            self._root = PathTreeNode(belief=root_belief, depth=0)
        
        self._last_plan_turn = int(state.confidence) if isinstance(state.confidence, (int, float)) else 0
        
        # Apply retrieval priors from memory
        if memory is not None:
            context = self._build_retrieval_context(state, memory)
            preferences = memory.retrieve_direction_preferences(context)
            self._root.retrieval_prior = preferences
        
        # 3. Run MCTS for num_simulations
        for _ in range(self.num_simulations):
            # Selection
            leaf = self._select(self._root)
            
            # Expansion
            if leaf.visit_count > 0 and leaf.depth < self.max_depth:
                children = self._expand(leaf, memory)
                if children:
                    leaf = self._rng.choice(children)
            
            # Simulation
            reward = self._simulate(leaf, memory)
            
            # Backpropagation
            self._backpropagate(leaf, reward)
        
        # 4. Choose best child by visit count (robust policy)
        valid_dirs = self._get_valid_directions(self._root)
        best_child = self._root.most_visited_child(valid_dirs)
        if best_child is None or best_child.action is None:
            # Fallback: wait action
            return ActionCandidate(name="wait", params={"duration": 0.1}, utility=0.1, confidence=0.1)
        
        # 5. Record guided decision if retrieval influenced the choice
        # Use root's retrieval_prior (built from current state context) 
        # since children build their own from different belief contexts
        if (self.memory is not None 
            and best_child.action_direction is not None
            and best_child.action.name != "wait"):
            
            retrieval_prior = self._root.retrieval_prior.get(best_child.action_direction, 0.0)
            if retrieval_prior > 0.05:
                self.memory.record_guided_decision()
        
        # 6. Cache tree for next turn
        # (root already cached in self._root)
        
        return best_child.action
    
    def _build_retrieval_context(self, state: WorldState, memory: GridMemoryStore) -> Dict[str, Any]:
        """Build context for episodic retrieval."""
        # Reuse the planner's context builder
        from .planner import GridPlanner
        # We need a temporary planner instance or static method
        # Since build_decision_context is an instance method, create one
        temp_planner = GridPlanner()
        # Use the memory's method directly
        return memory.build_decision_context(state)
    
    def _select(self, node: PathTreeNode) -> PathTreeNode:
        """
        Selection phase: traverse tree using UCB until leaf or unexpanded node.
        """
        current = node
        while current.children and current.is_fully_expanded(self._get_valid_directions(current)):
            valid_dirs = self._get_valid_directions(current)
            current = current.best_child(self.exploration_constant, valid_dirs)
            if current is None:
                break
        return current
    
    def _get_valid_directions(self, node: PathTreeNode) -> List[Position]:
        """Get valid movement directions from current belief state."""
        # Use the agent position from the belief state
        pos = self._estimate_agent_position(node.belief, self.memory)
        if pos is None:
            # Fallback: use first goal particle or memory
            pos = node.belief.goal_particles[0][0] if node.belief.goal_particles else (0, 0)
            if self.memory and self.memory.spatial.grid:
                for p, cell in self.memory.spatial.grid.items():
                    if cell.visited:
                        pos = p
                        break
        
        valid = []
        for dx, dy in self.DIRECTIONS:
            new_pos = (pos[0] + dx, pos[1] + dy)
            # Check if not known obstacle
            belief = node.belief.cell_beliefs.get(new_pos, CellBelief.UNKNOWN)
            if belief != CellBelief.OBSTACLE:
                valid.append((dx, dy))
        
        # Add control actions if control_learner available
        if self.control_learner:
            # Could add mapped controls as "directions"
            pass
        
        return valid if valid else list(self.DIRECTIONS)

    def _is_direction_valid(self, node: PathTreeNode, direction: Position) -> bool:
        """Check if a direction is valid (not into known obstacle) from current node."""
        # Estimate agent position from node belief
        agent_pos = self._estimate_agent_position(node.belief, self.memory)
        if agent_pos is None:
            return True  # Can't verify, assume valid
        
        new_pos = (agent_pos[0] + direction[0], agent_pos[1] + direction[1])
        belief = node.belief.cell_beliefs.get(new_pos, CellBelief.UNKNOWN)
        return belief != CellBelief.OBSTACLE
    
    def _expand(self, node: PathTreeNode, memory: GridMemoryStore) -> List[PathTreeNode]:
        """
        Expansion phase: generate children for valid directions and known controls.
        
        Uses control_learner.predicted_displacement() for controls with known effects
        instead of assuming unit cardinal movement.
        """
        valid_dirs = self._get_valid_directions(node)
        children = []
        
        # Get retrieval priors for this context
        retrieval_priors = {}
        if memory is not None and node.belief.turn >= 0:
            context = self._build_retrieval_context_from_belief(node.belief, memory)
            retrieval_priors = memory.retrieve_direction_preferences(context)
        
        # Expand cardinal directions (baseline)
        for direction in valid_dirs:
            if direction in node.children:
                continue
            
            # Create child belief by simulating action
            child_belief = self._simulate_transition(node.belief, direction, memory)
            
            # Create action candidate
            action_name = self._direction_to_action(direction)
            action = ActionCandidate(
                name=action_name,
                params={"direction": direction},
                utility=0.0,
                confidence=0.5,
            )
            
            child = PathTreeNode(
                belief=child_belief,
                action=action,
                action_direction=direction,
                parent=node,
                depth=node.depth + 1,
                retrieval_prior=retrieval_priors,
                uncertainty=child_belief.entropy,
            )
            node.children[direction] = child
            children.append(child)
        
        # Expand known controls from control_learner
        if self.control_learner is not None:
            # Build context from belief state for per-context predictions
            context_key = ()
            if memory is not None:
                retrieval_context = self._build_retrieval_context_from_belief(node.belief, memory)
                context_key = self._context_to_key(retrieval_context)
            for control_name in self.control_learner.known_controls(context_key):
                pred = self.control_learner.predicted_displacement(control_name, context_key)
                if pred is None or pred == (0.0, 0.0):
                    continue  # control with no known effect
                
                # Round predicted displacement to cardinal direction
                direction = (int(round(pred[0])), int(round(pred[1])))
                if direction not in self.DIRECTIONS:
                    continue  # only support cardinal directions for now
                
                if direction in node.children:
                    continue  # already expanded this direction
                
                # Check if direction is valid (not into known obstacle)
                if not self._is_direction_valid(node, direction):
                    continue
                
                # Create child belief with predicted transition
                child_belief = self._simulate_transition(node.belief, direction, memory)
                
                # Create action candidate for the control
                action = ActionCandidate(
                    name=control_name,
                    params={"direction": direction, "control": control_name},
                    utility=0.0,
                    confidence=0.7,  # higher confidence for learned controls
                )
                
                child = PathTreeNode(
                    belief=child_belief,
                    action=action,
                    action_direction=direction,
                    parent=node,
                    depth=node.depth + 1,
                    retrieval_prior=retrieval_priors,
                    uncertainty=child_belief.entropy,
                )
                node.children[direction] = child
                children.append(child)
        
        return children
    
    def _simulate_transition(
        self,
        belief: BeliefState,
        direction: Position,
        memory: GridMemoryStore,
    ) -> BeliefState:
        """
        Simulate belief state transition after taking an action.
        
        Uses WorldModelLearner for entity predictions if available.
        """
        # Estimate agent position for WorldModelLearner predictions
        agent_pos = self._estimate_agent_position(belief, memory)
        
        new_belief = BeliefState(
            goal_particles=list(belief.goal_particles),
            cell_beliefs=dict(belief.cell_beliefs),
            entity_models=dict(belief.entity_models),
            entropy=belief.entropy,
            turn=belief.turn + 1,
        )
        
        # Move agent position (simplified - we track goal particles relative to agent)
        dx, dy = direction
        new_particles = []
        for (gx, gy), weight in new_belief.goal_particles:
            # Agent moves (dx, dy), so goal appears to move (-dx, -dy) relative to agent
            new_particles.append(((gx - dx, gy - dy), weight))
        new_belief.goal_particles = new_particles
        
        # Update entity models using WorldModelLearner if available
        if self.world_model_learner is not None and agent_pos is not None:
            # Get predictions for all entities at next step
            entity_preds = self.world_model_learner.get_all_entity_predictions(1, agent_pos)
            for entity_id, preds in entity_preds.items():
                if entity_id in new_belief.entity_models and preds:
                    new_belief.entity_models[entity_id].position = preds[0]
        else:
            # Fallback: local entity models with constant velocity
            for model in new_belief.entity_models.values():
                preds = model.predict_next_positions(1)
                if preds:
                    model.position = preds[0]
        
        # Recompute entropy
        new_belief.entropy = self._compute_entropy(new_belief)
        
        return new_belief
    
    def _simulate(self, node: PathTreeNode, memory: GridMemoryStore) -> float:
        """
        Simulation/rollout phase: greedy rollout to goal with retrieval hints.
        
        Uses control_learner.select_control_for_direction() to choose controls
        that realize the desired direction, with predicted displacement.
        
        Returns discounted cumulative reward.
        
        Optimized: limits rollout steps, caches context/retrieval prefs.
        """
        belief = node.belief
        total_reward = 0.0
        discount = 1.0
        gamma = 0.95
        
        # Sample a goal from belief
        goal = belief.sample_goal(self._rng)
        if goal is None:
            # No goal belief - explore
            return self._rollout_explore(belief, memory, gamma)
        
        # Get current agent position (approximate from belief)
        agent_pos = self._estimate_agent_position(belief, memory)
        if agent_pos is None:
            return 0.0
        
        # Limit rollout steps for performance (max 8 steps regardless of tree depth)
        max_rollout_steps = min(8, self.max_depth - node.depth)
        
        # Pre-build context for retrieval (cached per simulation)
        retrieval_context = None
        retrieval_prefs = {}
        if memory is not None:
            retrieval_context = self._build_retrieval_context_from_belief(belief, memory)
            retrieval_prefs = memory.retrieve_direction_preferences(retrieval_context)
        
        # Pre-compute context key for control learner
        context_key = ()
        if retrieval_context is not None:
            context_key = self._context_to_key(retrieval_context)
        
        # Greedy rollout toward goal
        for step in range(max_rollout_steps):
            if agent_pos == goal:
                total_reward += discount * 10.0  # Goal reward
                break
            
            # Find best direction toward goal
            best_dir = self._greedy_direction(agent_pos, goal, belief, memory)
            if best_dir is None:
                break
            
            # Use control_learner to select control for this direction
            chosen_control = None
            pred = best_dir  # Default to cardinal direction
            
            if self.control_learner is not None:
                # Get available controls for this context
                available_controls = self.control_learner.known_controls(context_key)
                
                # If no controls known in this context, fall back to all controls
                if not available_controls:
                    available_controls = self.control_learner.known_controls()
                
                if available_controls:
                    chosen_control = self.control_learner.select_control_for_direction(
                        available_controls, best_dir, context_key, agent_pos
                    )
                    if chosen_control is not None:
                        pred_disp = self.control_learner.predicted_displacement(chosen_control, context_key)
                        if pred_disp is not None:
                            pred = (int(round(pred_disp[0])), int(round(pred_disp[1])))
            
            # Check for obstacles using predicted displacement
            next_pos = (agent_pos[0] + pred[0], agent_pos[1] + pred[1])
            cell_belief = belief.cell_beliefs.get(next_pos, CellBelief.UNKNOWN)
            
            # Check if next_pos is a frontier cell (high exploration value)
            is_frontier = memory and next_pos in memory.spatial.frontier
            
            if cell_belief == CellBelief.OBSTACLE:
                total_reward += discount * -1.0  # Hit obstacle
                break
            elif cell_belief == CellBelief.FREE:
                if is_frontier:
                    total_reward += discount * 0.5  # Frontier bonus during goal-seeking
                else:
                    total_reward += discount * 0.1  # Step reward
            else:
                total_reward += discount * 0.3  # Unknown cell (info gain) - INCREASED from 0.05
            
            # Penalize moving into cells where entities (especially pursuers) are predicted to be
            # Simplified: only check next position (not adjacent), reduced horizon
            if self.world_model_learner is not None:
                for entity_id, model in belief.entity_models.items():
                    entity_preds = model.predict_next_positions(2)  # Reduced from 3
                    for step_idx, pred_pos in enumerate(entity_preds):
                        if pred_pos == next_pos:
                            category, conf = self.world_model_learner.get_entity_category(entity_id)
                            penalty_factor = 1.0 / (step_idx + 1)
                            if category == "approaches_agent" and conf > 0.5:
                                total_reward += discount * -8.0 * penalty_factor
                            elif category in ("mobile", "approaches_agent") and conf > 0.3:
                                total_reward += discount * -3.0 * penalty_factor
                            break
            
            # Apply retrieval hint bonus (use pre-computed prefs)
            if best_dir in retrieval_prefs:
                total_reward += discount * retrieval_prefs[best_dir] * 0.5
            
            # Transition using predicted displacement
            agent_pos = next_pos
            belief = self._simulate_transition(belief, pred, memory)
            discount *= gamma
        
        return total_reward
    
    def _rollout_explore(
        self,
        belief: BeliefState,
        memory: GridMemoryStore,
        gamma: float,
    ) -> float:
        """Rollout when no goal is known: explore frontier."""
        total_reward = 0.0
        discount = 1.0
        
        agent_pos = self._estimate_agent_position(belief, memory)
        if agent_pos is None or not memory:
            return 0.0
        
        frontier = memory.spatial.frontier
        if not frontier:
            return 0.0
        
        # Find nearest frontier
        target = min(frontier, key=lambda f: abs(f[0] - agent_pos[0]) + abs(f[1] - agent_pos[1]))
        
        for _ in range(self.max_depth):
            if agent_pos == target:
                total_reward += discount * 5.0
                break
            
            best_dir = self._greedy_direction(agent_pos, target, belief, memory)
            if best_dir is None:
                break
            
            next_pos = (agent_pos[0] + best_dir[0], agent_pos[1] + best_dir[1])
            cell_belief = belief.cell_beliefs.get(next_pos, CellBelief.UNKNOWN)
            
            if cell_belief == CellBelief.OBSTACLE:
                total_reward += discount * -1.0
                break
            elif cell_belief == CellBelief.UNKNOWN:
                total_reward += discount * 0.5  # High info gain for unknown
            else:
                total_reward += discount * 0.1
            
            agent_pos = next_pos
            belief = self._simulate_transition(belief, best_dir, memory)
            discount *= gamma
        
        return total_reward
    
    def _greedy_direction(
        self,
        agent_pos: Position,
        goal: Position,
        belief: BeliefState,
        memory: GridMemoryStore,
    ) -> Optional[Position]:
        """Find best cardinal direction toward goal avoiding known obstacles."""
        best_dir = None
        best_dist = float('inf')
        
        for dx, dy in self.DIRECTIONS:
            next_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
            
            # Skip known obstacles
            if belief.cell_beliefs.get(next_pos) == CellBelief.OBSTACLE:
                continue
            
            # Check bounds if known
            if memory and memory.spatial.grid:
                # Loose bounds check
                pass
            
            dist = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
            if dist < best_dist:
                best_dist = dist
                best_dir = (dx, dy)
        
        return best_dir
    
    def _estimate_agent_position(
        self,
        belief: BeliefState,
        memory: GridMemoryStore,
    ) -> Optional[Position]:
        """Estimate current agent position from belief/memory."""
        # Try to get from memory's working memory or spatial grid
        if memory and memory.working.current_observation:
            obs = memory.working.current_observation
            if isinstance(obs, dict):
                # Check top-level position (legacy)
                pos = obs.get("position")
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    return (int(pos[0]), int(pos[1]))
                # Check nested position in hud.grid (actual GridGame format)
                hud = obs.get("hud", {})
                if isinstance(hud, dict):
                    grid = hud.get("grid", {})
                    if isinstance(grid, dict):
                        pos = grid.get("position")
                        if isinstance(pos, (list, tuple)) and len(pos) == 2:
                            return (int(pos[0]), int(pos[1]))
        
        # Fallback: use last visited cell in memory
        if memory and memory.spatial.grid:
            for pos, cell in memory.spatial.grid.items():
                if cell.visited:
                    return pos
        
        # Last resort: use first goal particle as reference (inverted)
        if belief.goal_particles:
            gx, gy = belief.goal_particles[0][0]
            return (gx, gy)  # Rough estimate
        
        return (0, 0)
    
    def _backpropagate(self, node: PathTreeNode, reward: float) -> None:
        """
        Backpropagation phase: update statistics up to root.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += reward
            # Update uncertainty as running variance approximation
            if current.visit_count > 1:
                mean = current.value_sum / current.visit_count
                # Simple uncertainty: value variance proxy
                current.uncertainty = current.belief.entropy / max(current.visit_count, 1)
            current = current.parent
    
    def _build_root_belief(self, state: WorldState, memory: GridMemoryStore) -> BeliefState:
        """
        Build initial BeliefState from WorldState and GridMemoryStore.
        """
        belief = BeliefState(
            goal_particles=[],
            cell_beliefs={},
            entity_models={},
            entropy=1.0,
            turn=0,
        )
        
        # 1. Initialize goal particles from memory and current observation
        known_goals = memory.spatial.known_goal_positions
        goal_is_dynamic = memory.spatial.goal_is_dynamic
        
        # If target is visible in current observation, use it directly
        if state.target_position is not None:
            belief.goal_particles = [(state.target_position, 1.0)]
        elif known_goals:
            if goal_is_dynamic:
                # Spread particles around known goals
                particles = []
                for goal in known_goals:
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            pos = (goal[0] + dx, goal[1] + dy)
                            particles.append((pos, 1.0 / (9 * len(known_goals))))
                belief.goal_particles = particles[:self.num_particles]
            else:
                # Static goal: uniform over known goals
                weight = 1.0 / len(known_goals)
                belief.goal_particles = [(g, weight) for g in known_goals]
        else:
            # No known goals: uniform over frontier
            frontier = memory.spatial.frontier
            if frontier:
                weight = 1.0 / len(frontier)
                belief.goal_particles = [(f, weight) for f in frontier]
        
        # 2. Cell beliefs from memory grid
        for pos, cell in memory.spatial.grid.items():
            if cell.obstacle is True:
                belief.cell_beliefs[pos] = CellBelief.OBSTACLE
            elif cell.seen:
                belief.cell_beliefs[pos] = CellBelief.FREE
            else:
                belief.cell_beliefs[pos] = CellBelief.UNKNOWN
        
        # Add current observation cells
        for pos in state.visible_cells:
            if pos not in belief.cell_beliefs:
                belief.cell_beliefs[pos] = CellBelief.FREE
        for pos in state.obstacle_positions:
            belief.cell_beliefs[pos] = CellBelief.OBSTACLE
        if state.target_position:
            belief.cell_beliefs[state.target_position] = CellBelief.FREE
        
        # 3. Entity models from memory entities
        for entity_id, ent_mem in memory.entities.items():
            category = ent_mem.category
            confidence = ent_mem.behavior_confidence
            if ent_mem.category == "mobile" and ent_mem.approach_streak >= 3:
                category = "approaches_agent"
            elif ent_mem.category == "mobile" and ent_mem.retreat_streak >= 3:
                category = "moves_away_from_agent"
            
            belief.entity_models[entity_id] = EntityModel(
                entity_id=entity_id,
                category=category,
                position=ent_mem.position,
                behavior_confidence=confidence,
                last_seen_turn=ent_mem.last_seen_turn,
            )
        
        # Also add entities from current state
        for entity in state.entities:
            if entity.entity_id and entity.position:
                if entity.entity_id not in belief.entity_models:
                    belief.entity_models[entity.entity_id] = EntityModel(
                        entity_id=entity.entity_id,
                        category=entity.kind,
                        position=entity.position,
                        behavior_confidence=entity.confidence,
                        last_seen_turn=0,
                    )
        
        # 4. Compute initial entropy
        belief.entropy = self._compute_entropy(belief)
        
        return belief
    
    def _build_retrieval_context_from_belief(
        self,
        belief: BeliefState,
        memory: GridMemoryStore,
    ) -> Dict[str, Any]:
        """Build retrieval context from belief state."""
        agent_pos = self._estimate_agent_position(belief, memory)
        goal_pos = belief.sample_goal(self._rng)
        
        target_direction = None
        if agent_pos and goal_pos:
            dx = goal_pos[0] - agent_pos[0]
            dy = goal_pos[1] - agent_pos[1]
            target_direction = [
                0 if dx == 0 else (1 if dx > 0 else -1),
                0 if dy == 0 else (1 if dy > 0 else -1),
            ]
        
        # Adjacent obstacles from belief
        adjacent_obstacles = []
        if agent_pos:
            for dx, dy in self.DIRECTIONS:
                cell = (agent_pos[0] + dx, agent_pos[1] + dy)
                belief_val = belief.cell_beliefs.get(cell)
                if belief_val == CellBelief.OBSTACLE:
                    adjacent_obstacles.append(True)
                elif belief_val == CellBelief.FREE:
                    adjacent_obstacles.append(False)
                else:
                    adjacent_obstacles.append(None)
        else:
            adjacent_obstacles = [None, None, None, None]
        
        # Entity distance
        nearest_entity_dist = None
        for model in belief.entity_models.values():
            if agent_pos:
                dist = abs(model.position[0] - agent_pos[0]) + abs(model.position[1] - agent_pos[1])
                if nearest_entity_dist is None or dist < nearest_entity_dist:
                    nearest_entity_dist = dist
        
        entity_bucket = None
        if nearest_entity_dist is not None:
            if nearest_entity_dist <= 1:
                entity_bucket = 0
            elif nearest_entity_dist <= 2:
                entity_bucket = 1
            elif nearest_entity_dist <= 4:
                entity_bucket = 2
            else:
                entity_bucket = 3
        
        # Health bucket (from memory HUD if available)
        health_bucket = None
        if memory and memory.working.current_observation:
            hud = memory.working.current_observation.get("hud", {}) if isinstance(memory.working.current_observation, dict) else {}
            health = hud.get("health")
            if isinstance(health, (int, float)):
                if health <= 3:
                    health_bucket = 0
                elif health <= 6:
                    health_bucket = 1
                else:
                    health_bucket = 2
        
        frontier_count = len(memory.spatial.frontier) if memory else 0
        
        return {
            "goal_visible": goal_pos is not None and agent_pos == goal_pos,
            "goal_known": goal_pos is not None,
            "target_direction": target_direction,
            "goal_dynamic": memory.spatial.goal_is_dynamic if memory else False,
            "adjacent_obstacles": adjacent_obstacles,
            "entity_distance_bucket": entity_bucket,
            "health_bucket": health_bucket,
            "frontier_bucket": min(frontier_count, 2),
        }
    
    def _context_to_key(self, context: Dict[str, Any]) -> Tuple[int, ...]:
        """Convert decision context dict to tuple key for control_learner."""
        required_keys = [
            "goal_visible",
            "goal_known",
            "target_direction",
            "goal_dynamic",
            "adjacent_obstacles",
            "entity_distance_bucket",
            "health_bucket",
            "frontier_bucket",
        ]
        key_parts: list[int] = []
        for k in required_keys:
            v = context.get(k)
            if v is None:
                key_parts.append(-1)
            elif isinstance(v, bool):
                key_parts.append(1 if v else 0)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if item is None:
                        key_parts.append(-1)
                    else:
                        key_parts.append(int(item))
            elif isinstance(v, int):
                key_parts.append(v)
            else:
                key_parts.append(0)
        return tuple(key_parts)
    
    def _direction_to_action(self, direction: Position) -> str:
        """Convert direction tuple to action name."""
        mapping = {
            (0, -1): "move_up",
            (1, 0): "move_right",
            (0, 1): "move_down",
            (-1, 0): "move_left",
        }
        return mapping.get(direction, "wait")
    
    def _compute_entropy(self, belief: BeliefState) -> float:
        """Compute entropy (helper for static access)."""
        if not belief.goal_particles:
            return 1.0
        
        entropy = 0.0
        for _, weight in belief.goal_particles:
            if weight > 0:
                entropy -= weight * math.log(weight)
        
        max_entropy = math.log(len(belief.goal_particles)) if belief.goal_particles else 1.0
        goal_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        unknown_cells = sum(1 for v in belief.cell_beliefs.values() if v == CellBelief.UNKNOWN)
        total_cells = max(len(belief.cell_beliefs), 1)
        cell_entropy = unknown_cells / total_cells
        
        return 0.7 * goal_entropy + 0.3 * cell_entropy
    
    def reuse_tree(self, old_root: PathTreeNode, new_belief: BeliefState) -> PathTreeNode:
        """Reuse tree from previous turn with updated root belief."""
        # Create new root with updated belief, keep children
        new_root = PathTreeNode(
            belief=new_belief,
            depth=0,
            retrieval_prior=old_root.retrieval_prior,
        )
        new_root.children = old_root.children
        for child in new_root.children.values():
            child.parent = new_root
        return new_root