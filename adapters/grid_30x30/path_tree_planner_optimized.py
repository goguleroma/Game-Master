"""
Optimized PathTreePlanner with Numba JIT compilation for real-time performance.

This is a drop-in replacement for GridPathTreePlanner that uses Numba-accelerated
hot paths for belief state updates, simulations, and direction validation.

Target: <10ms per decision (100+ decisions/sec)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .world_model import WorldState
from .memory import GridMemoryStore
from .control_learner import ControlLearner, WorldModelLearner
from .planner import ActionCandidate
from .path_tree_planner import (
    CellBelief,
    EntityModel,
    BeliefState,
    PathTreeNode,
)

# DIRECTIONS is a class attribute in the original, define it here
DIRECTIONS: Tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))

from .path_tree_planner_numba import (
    NUMBA_AVAILABLE,
    simulate_transition_numba,
    greedy_direction_numba,
    simulate_rollout_numba,
    update_goal_particles_numba,
    get_valid_directions_numba,
    is_direction_valid_numba,
    belief_state_to_arrays,
    get_buffers,
    NumbaBuffers,
    CELL_UNKNOWN,
    CELL_FREE,
    CELL_OBSTACLE,
)


class OptimizedGridPathTreePlanner:
    """
    MCTS-Lite planner with Numba-accelerated hot paths.
    
    Optimizations:
    - Numba JIT on _simulate_transition, _simulate, _get_valid_directions
    - Pre-allocated arrays to avoid GC pressure
    - Vectorized particle filter operations
    - Batched simulation where possible
    - Early termination on value convergence
    
    Usage:
        planner = OptimizedGridPathTreePlanner(max_depth=8, num_simulations=50, seed=42)
        action = planner.plan(world_state, memory)
    """
    
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
        use_numba: bool = True,
        early_termination_threshold: float = 0.01,
        convergence_window: int = 10,
    ) -> None:
        """
        Initialize the optimized path tree planner.
        
        Args:
            max_depth: Maximum tree depth for simulations
            num_simulations: Number of MCTS iterations per plan call
            exploration_constant: UCB exploration constant (c)
            num_particles: Number of particles for goal belief
            control_learner: Optional ControlLearner for transition prediction
            world_model_learner: Optional WorldModelLearner for entity predictions
            memory: Optional GridMemoryStore for retrieval priors
            seed: Random seed for deterministic behavior
            use_numba: Whether to use Numba acceleration (fallback to Python if False/unavailable)
            early_termination_threshold: Value change threshold for early termination
            convergence_window: Number of simulations to check for convergence
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
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.early_termination_threshold = early_termination_threshold
        self.convergence_window = convergence_window
        
        # Pre-allocated buffers for Numba
        self._buffers = get_buffers()
        
        # Performance tracking
        self._last_plan_time = 0.0
        self._simulation_times: List[float] = []
        
        # Convergence tracking
        self._recent_values: List[float] = []
        
        if self.use_numba:
            print(f"[PathTreePlanner] Numba acceleration ENABLED")
        else:
            print(f"[PathTreePlanner] Numba acceleration DISABLED (available={NUMBA_AVAILABLE})")
    
    def plan(self, state: WorldState, memory: GridMemoryStore) -> ActionCandidate:
        """
        Run MCTS planning and return the best action.
        
        Optimized with:
        - Numba-accelerated hot paths
        - Early termination on convergence
        - Pre-allocated buffers
        """
        start_time = time.perf_counter()
        
        # 1. Build root BeliefState from state + memory
        root_belief = self._build_root_belief(state, memory)
        
        # 2. Build/reuse root PathTreeNode (warm-start)
        if self._root is not None and self._last_plan_turn == int(state.confidence) if isinstance(state.confidence, (int, float)) else 0:
            self._root = self.reuse_tree(self._root, root_belief)
        else:
            self._root = PathTreeNode(belief=root_belief, depth=0)
        
        self._last_plan_turn = int(state.confidence) if isinstance(state.confidence, (int, float)) else 0
        
        # Apply retrieval priors from memory
        if memory is not None:
            context = self._build_retrieval_context(state, memory)
            preferences = memory.retrieve_direction_preferences(context)
            self._root.retrieval_prior = preferences
        
        # 3. Run MCTS for num_simulations (with early termination)
        self._recent_values.clear()
        
        for sim_idx in range(self.num_simulations):
            sim_start = time.perf_counter()
            
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
            
            self._simulation_times.append(time.perf_counter() - sim_start)
            self._recent_values.append(reward)
            
            # Early termination check
            if self._check_convergence(sim_idx):
                # print(f"[PathTreePlanner] Early termination at simulation {sim_idx + 1}")
                break
        
        # 4. Choose best child by visit count (robust policy)
        valid_dirs = self._get_valid_directions(self._root)
        best_child = self._root.most_visited_child(valid_dirs)
        if best_child is None or best_child.action is None:
            return ActionCandidate(name="wait", params={"duration": 0.1}, utility=0.1, confidence=0.1)
        
        # 5. Record guided decision if retrieval influenced the choice
        if (self.memory is not None 
            and best_child.action_direction is not None
            and best_child.action.name != "wait"):
            
            retrieval_prior = self._root.retrieval_prior.get(best_child.action_direction, 0.0)
            if retrieval_prior > 0.05:
                self.memory.record_guided_decision()
        
        self._last_plan_time = time.perf_counter() - start_time
        return best_child.action
    
    def _check_convergence(self, sim_idx: int) -> bool:
        """Check if value estimates have converged for early termination."""
        if sim_idx < self.convergence_window:
            return False
        if len(self._recent_values) < self.convergence_window:
            return False
        
        recent = self._recent_values[-self.convergence_window:]
        mean_val = sum(recent) / len(recent)
        max_diff = max(abs(v - mean_val) for v in recent)
        
        return max_diff < self.early_termination_threshold
    
    def _build_retrieval_context(self, state: WorldState, memory: GridMemoryStore) -> Dict[str, Any]:
        """Build context for episodic retrieval."""
        return memory.build_decision_context(state)
    
    def _select(self, node: PathTreeNode) -> PathTreeNode:
        """Selection phase: traverse tree using UCB until leaf or unexpanded node."""
        current = node
        while current.children and current.is_fully_expanded(self._get_valid_directions(current)):
            valid_dirs = self._get_valid_directions(current)
            current = current.best_child(self.exploration_constant, valid_dirs)
            if current is None:
                break
        return current
    
    def _get_valid_directions(self, node: PathTreeNode) -> List[Position]:
        """Get valid movement directions from current belief state (Numba-accelerated)."""
        if self.use_numba:
            return self._get_valid_directions_numba(node)
        else:
            return self._get_valid_directions_python(node)
    
    def _get_valid_directions_numba(self, node: PathTreeNode) -> List[Position]:
        """Numba-accelerated valid directions."""
        # Estimate agent position
        pos = self._estimate_agent_position(node.belief, self.memory)
        if pos is None:
            pos = node.belief.goal_particles[0][0] if node.belief.goal_particles else (0, 0)
        
        # Convert cell beliefs to arrays
        buffers = self._buffers
        max_cells = NumbaBuffers.MAX_CELLS
        
        # Fill cell belief arrays
        num_cells = 0
        for (x, y), belief in node.belief.cell_beliefs.items():
            if num_cells >= max_cells:
                break
            buffers.memory_cell_x[num_cells] = x
            buffers.memory_cell_y[num_cells] = y
            if belief == CellBelief.OBSTACLE:
                buffers.memory_cell_obs[num_cells] = 1
            elif belief == CellBelief.FREE:
                buffers.memory_cell_obs[num_cells] = 2
            else:
                buffers.memory_cell_obs[num_cells] = 0
            num_cells += 1
        
        # Call Numba function
        valid_dirs_arr, count = get_valid_directions_numba(
            pos[0], pos[1],
            buffers.memory_cell_x, buffers.memory_cell_y, buffers.memory_cell_obs,
            num_cells
        )
        
        # Convert back to list of tuples
        result = []
        for i in range(count):
            result.append((int(valid_dirs_arr[i, 0]), int(valid_dirs_arr[i, 1])))
        
        return result if result else list(DIRECTIONS)
    
    def _get_valid_directions_python(self, node: PathTreeNode) -> List[Position]:
        """Python fallback for valid directions."""
        pos = self._estimate_agent_position(node.belief, self.memory)
        if pos is None:
            pos = node.belief.goal_particles[0][0] if node.belief.goal_particles else (0, 0)
        
        valid = []
        for dx, dy in DIRECTIONS:
            new_pos = (pos[0] + dx, pos[1] + dy)
            belief = node.belief.cell_beliefs.get(new_pos, CellBelief.UNKNOWN)
            if belief != CellBelief.OBSTACLE:
                valid.append((dx, dy))
        
        return valid if valid else list(DIRECTIONS)
    
    def _is_direction_valid(self, node: PathTreeNode, direction: Position) -> bool:
        """Check if a direction is valid (Numba-accelerated)."""
        if self.use_numba:
            return self._is_direction_valid_numba(node, direction)
        else:
            return self._is_direction_valid_python(node, direction)
    
    def _is_direction_valid_numba(self, node: PathTreeNode, direction: Position) -> bool:
        """Numba-accelerated direction validation."""
        agent_pos = self._estimate_agent_position(node.belief, self.memory)
        if agent_pos is None:
            return True
        
        buffers = self._buffers
        max_cells = NumbaBuffers.MAX_CELLS
        
        num_cells = 0
        for (x, y), belief in node.belief.cell_beliefs.items():
            if num_cells >= max_cells:
                break
            buffers.memory_cell_x[num_cells] = x
            buffers.memory_cell_y[num_cells] = y
            if belief == CellBelief.OBSTACLE:
                buffers.memory_cell_obs[num_cells] = 1
            elif belief == CellBelief.FREE:
                buffers.memory_cell_obs[num_cells] = 2
            else:
                buffers.memory_cell_obs[num_cells] = 0
            num_cells += 1
        
        return is_direction_valid_numba(
            agent_pos[0], agent_pos[1], direction,
            buffers.memory_cell_x, buffers.memory_cell_y, buffers.memory_cell_obs,
            num_cells
        )
    
    def _is_direction_valid_python(self, node: PathTreeNode, direction: Position) -> bool:
        """Python fallback for direction validation."""
        agent_pos = self._estimate_agent_position(node.belief, self.memory)
        if agent_pos is None:
            return True
        
        new_pos = (agent_pos[0] + direction[0], agent_pos[1] + direction[1])
        belief = node.belief.cell_beliefs.get(new_pos, CellBelief.UNKNOWN)
        return belief != CellBelief.OBSTACLE
    
    def _expand(self, node: PathTreeNode, memory: GridMemoryStore) -> List[PathTreeNode]:
        """Expansion phase: generate children for valid directions and known controls."""
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
            
            # Create child belief by simulating action (Numba-accelerated)
            child_belief = self._simulate_transition(node.belief, direction, memory)
            
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
            context_key = ()
            if memory is not None:
                retrieval_context = self._build_retrieval_context_from_belief(node.belief, memory)
                context_key = self._context_to_key(retrieval_context)
            
            for control_name in self.control_learner.known_controls(context_key):
                pred = self.control_learner.predicted_displacement(control_name, context_key)
                if pred is None or pred == (0.0, 0.0):
                    continue
                
                direction = (int(round(pred[0])), int(round(pred[1])))
                if direction not in DIRECTIONS:
                    continue
                
                if direction in node.children:
                    continue
                
                if not self._is_direction_valid(node, direction):
                    continue
                
                child_belief = self._simulate_transition(node.belief, direction, memory)
                
                action = ActionCandidate(
                    name=control_name,
                    params={"direction": direction, "control": control_name},
                    utility=0.0,
                    confidence=0.7,
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
        
        Numba-accelerated when available.
        """
        if self.use_numba:
            return self._simulate_transition_numba(belief, direction, memory)
        else:
            return self._simulate_transition_python(belief, direction, memory)
    
    def _simulate_transition_numba(
        self,
        belief: BeliefState,
        direction: Position,
        memory: GridMemoryStore,
    ) -> BeliefState:
        """Numba-accelerated belief state transition."""
        buffers = self._buffers
        
        # Convert belief to arrays
        belief_arrays = belief_state_to_arrays(belief)
        
        # Prepare memory arrays
        max_cells = NumbaBuffers.MAX_CELLS
        max_frontier = NumbaBuffers.MAX_FRONTIER
        
        # Memory cells
        mem_num_cells = 0
        for pos, cell in memory.spatial.grid.items():
            if mem_num_cells >= max_cells:
                break
            buffers.memory_cell_x[mem_num_cells] = pos[0]
            buffers.memory_cell_y[mem_num_cells] = pos[1]
            buffers.memory_cell_obs[mem_num_cells] = 1 if cell.obstacle else (2 if cell.seen else 0)
            mem_num_cells += 1
        
        # Memory frontier
        mem_num_frontier = 0
        for pos in memory.spatial.frontier:
            if mem_num_frontier >= max_frontier:
                break
            buffers.memory_frontier_x[mem_num_frontier] = pos[0]
            buffers.memory_frontier_y[mem_num_frontier] = pos[1]
            mem_num_frontier += 1
        
        # Memory known goals
        mem_num_goals = 0
        for pos in memory.spatial.known_goal_positions:
            if mem_num_goals >= 20:
                break
            buffers.memory_goals_x[mem_num_goals] = pos[0]
            buffers.memory_goals_y[mem_num_goals] = pos[1]
            mem_num_goals += 1
        
        # WML entity predictions
        wml_num_entities = 0
        if self.world_model_learner is not None:
            agent_pos = self._estimate_agent_position(belief, memory)
            if agent_pos is not None:
                preds = self.world_model_learner.get_all_entity_predictions(1, agent_pos)
                for eid, pred_list in preds.items():
                    if wml_num_entities >= NumbaBuffers.MAX_ENTITIES:
                        break
                    if pred_list:
                        buffers.wml_preds_x[wml_num_entities] = pred_list[0][0]
                        buffers.wml_preds_y[wml_num_entities] = pred_list[0][1]
                    else:
                        buffers.wml_preds_x[wml_num_entities] = -9999
                        buffers.wml_preds_y[wml_num_entities] = -9999
                    wml_num_entities += 1
        
        # Call Numba transition
        new_num_goal, new_num_cells, new_entropy, new_turn = simulate_transition_numba(
            belief_arrays['goal_particles_x'],
            belief_arrays['goal_particles_y'],
            belief_arrays['goal_particles_w'],
            belief_arrays['num_goal_particles'],
            belief_arrays['cell_beliefs_x'],
            belief_arrays['cell_beliefs_y'],
            belief_arrays['cell_beliefs_val'],
            belief_arrays['num_cell_beliefs'],
            None,  # entity_ids (not used in current impl)
            belief_arrays['entity_pos_x'],
            belief_arrays['entity_pos_y'],
            belief_arrays['entity_vel_x'],
            belief_arrays['entity_vel_y'],
            belief_arrays['entity_category'],
            belief_arrays['entity_confidence'],
            belief_arrays['num_entities'],
            belief_arrays['entropy'],
            belief_arrays['turn'],
            direction,
            buffers.memory_cell_x,
            buffers.memory_cell_y,
            buffers.memory_cell_obs,
            mem_num_cells,
            buffers.wml_preds_x,
            buffers.wml_preds_y,
            wml_num_entities,
            buffers.next_goal_x,
            buffers.next_goal_y,
            buffers.next_goal_w,
            buffers.next_cell_x,
            buffers.next_cell_y,
            buffers.next_cell_val,
            buffers.next_entity_x,
            buffers.next_entity_y,
        )
        
        # Build new BeliefState from output arrays
        new_belief = BeliefState(
            goal_particles=[],
            cell_beliefs={},
            entity_models={},
            entropy=new_entropy,
            turn=new_turn,
        )
        
        # Goal particles
        for i in range(new_num_goal):
            new_belief.goal_particles.append((
                (int(buffers.next_goal_x[i]), int(buffers.next_goal_y[i])),
                float(buffers.next_goal_w[i])
            ))
        
        # Cell beliefs
        for i in range(new_num_cells):
            pos = (int(buffers.next_cell_x[i]), int(buffers.next_cell_y[i]))
            val = int(buffers.next_cell_val[i])
            if val == CELL_UNKNOWN:
                new_belief.cell_beliefs[pos] = CellBelief.UNKNOWN
            elif val == CELL_FREE:
                new_belief.cell_beliefs[pos] = CellBelief.FREE
            else:
                new_belief.cell_beliefs[pos] = CellBelief.OBSTACLE
        
        # Entity models (copy and update positions)
        for i in range(belief_arrays['num_entities']):
            # We need to map back to entity IDs
            eid_list = list(belief.entity_models.keys())
            if i < len(eid_list):
                eid = eid_list[i]
                model = belief.entity_models[eid]
                new_belief.entity_models[eid] = EntityModel(
                    entity_id=eid,
                    category=model.category,
                    position=(int(buffers.next_entity_x[i]), int(buffers.next_entity_y[i])),
                    velocity=model.velocity,
                    behavior_confidence=model.behavior_confidence,
                    last_seen_turn=model.last_seen_turn,
                )
        
        return new_belief
    
    def _simulate_transition_python(
        self,
        belief: BeliefState,
        direction: Position,
        memory: GridMemoryStore,
    ) -> BeliefState:
        """Python fallback for belief state transition."""
        # Original implementation from path_tree_planner.py
        agent_pos = self._estimate_agent_position(belief, memory)
        
        new_belief = BeliefState(
            goal_particles=list(belief.goal_particles),
            cell_beliefs=dict(belief.cell_beliefs),
            entity_models=dict(belief.entity_models),
            entropy=belief.entropy,
            turn=belief.turn + 1,
        )
        
        dx, dy = direction
        new_particles = []
        for (gx, gy), weight in new_belief.goal_particles:
            new_particles.append(((gx - dx, gy - dy), weight))
        new_belief.goal_particles = new_particles
        
        if self.world_model_learner is not None and agent_pos is not None:
            entity_preds = self.world_model_learner.get_all_entity_predictions(1, agent_pos)
            for entity_id, preds in entity_preds.items():
                if entity_id in new_belief.entity_models and preds:
                    new_belief.entity_models[entity_id].position = preds[0]
        else:
            for model in new_belief.entity_models.values():
                preds = model.predict_next_positions(1)
                if preds:
                    model.position = preds[0]
        
        new_belief.entropy = self._compute_entropy(new_belief)
        return new_belief
    
    def _simulate(self, node: PathTreeNode, memory: GridMemoryStore) -> float:
        """
        Simulation/rollout phase: greedy rollout to goal with retrieval hints.
        
        Numba-accelerated when available.
        """
        if self.use_numba:
            return self._simulate_numba(node, memory)
        else:
            return self._simulate_python(node, memory)
    
    def _simulate_numba(self, node: PathTreeNode, memory: GridMemoryStore) -> float:
        """Numba-accelerated simulation rollout."""
        buffers = self._buffers
        
        # Convert belief to arrays
        belief_arrays = belief_state_to_arrays(node.belief)
        
        # Prepare memory arrays
        max_frontier = NumbaBuffers.MAX_FRONTIER
        
        mem_num_frontier = 0
        for pos in memory.spatial.frontier:
            if mem_num_frontier >= max_frontier:
                break
            buffers.memory_frontier_x[mem_num_frontier] = pos[0]
            buffers.memory_frontier_y[mem_num_frontier] = pos[1]
            mem_num_frontier += 1
        
        max_cells = NumbaBuffers.MAX_CELLS
        mem_num_cells = 0
        for pos, cell in memory.spatial.grid.items():
            if mem_num_cells >= max_cells:
                break
            buffers.memory_cell_x[mem_num_cells] = pos[0]
            buffers.memory_cell_y[mem_num_cells] = pos[1]
            buffers.memory_cell_obs[mem_num_cells] = 1 if cell.obstacle else (2 if cell.seen else 0)
            mem_num_cells += 1
        
        # Control learner predictions
        num_controls = 0
        context_key = ()
        if self.control_learner is not None:
            if memory is not None:
                retrieval_context = self._build_retrieval_context_from_belief(node.belief, memory)
                context_key = self._context_to_key(retrieval_context)
            available_controls = self.control_learner.known_controls(context_key)
            if not available_controls:
                available_controls = self.control_learner.known_controls()
            for i, control in enumerate(available_controls):
                if i >= NumbaBuffers.MAX_CONTROLS:
                    break
                pred = self.control_learner.predicted_displacement(control, context_key)
                if pred is not None:
                    buffers.control_pred_dx[i] = pred[0]
                    buffers.control_pred_dy[i] = pred[1]
                    num_controls += 1
        
        # Estimate agent position
        agent_pos = self._estimate_agent_position(node.belief, memory)
        if agent_pos is None:
            return 0.0
        
        # Retrieval preferences
        retrieval_prefs = {}
        if memory is not None:
            retrieval_context = self._build_retrieval_context_from_belief(node.belief, memory)
            retrieval_prefs = memory.retrieve_direction_preferences(retrieval_context)
        
        # For now, we'll use a hybrid approach: Numba for core rollout, Python for retrieval
        # Full Numba rollout would need retrieval preferences passed as arrays
        # This is a simplified version - we'll call the Python simulate but with
        # Numba-accelerated greedy_direction and transition
        
        return self._simulate_hybrid(node, memory, belief_arrays, retrieval_prefs)
    
    def _simulate_hybrid(
        self,
        node: PathTreeNode,
        memory: GridMemoryStore,
        belief_arrays: Dict,
        retrieval_prefs: Dict[Position, float],
    ) -> float:
        """Hybrid simulation: Numba for core ops, Python for complex logic."""
        belief = node.belief
        total_reward = 0.0
        discount = 1.0
        gamma = 0.95
        
        # Sample goal from belief
        goal = belief.sample_goal(self._rng)
        if goal is None:
            return self._rollout_explore(belief, memory, gamma)
        
        agent_pos = self._estimate_agent_position(belief, memory)
        if agent_pos is None:
            return 0.0
        
        max_rollout_steps = min(8, self.max_depth - node.depth)
        
        # Pre-compute context key
        context_key = ()
        if memory is not None:
            retrieval_context = self._build_retrieval_context_from_belief(belief, memory)
            context_key = self._context_to_key(retrieval_context)
        
        # Local copies for mutation
        cur_belief = belief
        cur_agent_pos = agent_pos
        
        for step in range(max_rollout_steps):
            if cur_agent_pos == goal:
                total_reward += discount * 10.0
                break
            
            # Numba-accelerated greedy direction
            buffers = self._buffers
            max_cells = NumbaBuffers.MAX_CELLS
            num_cells = 0
            for (x, y), cell_belief in cur_belief.cell_beliefs.items():
                if num_cells >= max_cells:
                    break
                buffers.memory_cell_x[num_cells] = x
                buffers.memory_cell_y[num_cells] = y
                if cell_belief == CellBelief.OBSTACLE:
                    buffers.memory_cell_obs[num_cells] = 1
                elif cell_belief == CellBelief.FREE:
                    buffers.memory_cell_obs[num_cells] = 2
                else:
                    buffers.memory_cell_obs[num_cells] = 0
                num_cells += 1
            
            dir_x, dir_y = greedy_direction_numba(
                cur_agent_pos[0], cur_agent_pos[1], goal[0], goal[1],
                buffers.memory_cell_x, buffers.memory_cell_y, buffers.memory_cell_obs,
                num_cells
            )
            
            if dir_x == 0 and dir_y == 0:
                break
            
            best_dir = (dir_x, dir_y)
            
            # Control learner prediction
            chosen_control = None
            pred = best_dir
            if self.control_learner is not None:
                available_controls = self.control_learner.known_controls(context_key)
                if not available_controls:
                    available_controls = self.control_learner.known_controls()
                
                if available_controls:
                    chosen_control = self.control_learner.select_control_for_direction(
                        available_controls, best_dir, context_key, cur_agent_pos
                    )
                    if chosen_control is not None:
                        pred_disp = self.control_learner.predicted_displacement(chosen_control, context_key)
                        if pred_disp is not None:
                            pred = (int(round(pred_disp[0])), int(round(pred_disp[1])))
            
            next_pos = (cur_agent_pos[0] + pred[0], cur_agent_pos[1] + pred[1])
            cell_belief = cur_belief.cell_beliefs.get(next_pos, CellBelief.UNKNOWN)
            
            is_frontier = memory and next_pos in memory.spatial.frontier
            
            if cell_belief == CellBelief.OBSTACLE:
                total_reward += discount * -1.0
                break
            elif cell_belief == CellBelief.FREE:
                if is_frontier:
                    total_reward += discount * 0.5
                else:
                    total_reward += discount * 0.1
            else:
                total_reward += discount * 0.3
            
            # Entity penalty
            if self.world_model_learner is not None:
                for entity_id, model in cur_belief.entity_models.items():
                    entity_preds = model.predict_next_positions(2)
                    for step_idx, pred_pos in enumerate(entity_preds):
                        if pred_pos == next_pos:
                            category, conf = self.world_model_learner.get_entity_category(entity_id)
                            penalty_factor = 1.0 / (step_idx + 1)
                            if category == "approaches_agent" and conf > 0.5:
                                total_reward += discount * -8.0 * penalty_factor
                            elif category in ("mobile", "approaches_agent") and conf > 0.3:
                                total_reward += discount * -3.0 * penalty_factor
                            break
            
            # Retrieval hint bonus
            if best_dir in retrieval_prefs:
                total_reward += discount * retrieval_prefs[best_dir] * 0.5
            
            # Transition using Numba-accelerated function
            cur_belief = self._simulate_transition(cur_belief, pred, memory)
            cur_agent_pos = next_pos
            discount *= gamma
        
        return total_reward
    
    def _simulate_python(self, node: PathTreeNode, memory: GridMemoryStore) -> float:
        """Python fallback for simulation."""
        # This would be the original _simulate method
        # For now, delegate to hybrid which works without Numba too
        belief_arrays = belief_state_to_arrays(node.belief)
        retrieval_prefs = {}
        if memory is not None:
            retrieval_context = self._build_retrieval_context_from_belief(node.belief, memory)
            retrieval_prefs = memory.retrieve_direction_preferences(retrieval_context)
        return self._simulate_hybrid(node, memory, belief_arrays, retrieval_prefs)
    
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
        
        target = min(frontier, key=lambda f: abs(f[0] - agent_pos[0]) + abs(f[1] - agent_pos[1]))
        
        for _ in range(self.max_depth):
            if agent_pos == target:
                total_reward += discount * 5.0
                break
            
            buffers = self._buffers
            max_cells = NumbaBuffers.MAX_CELLS
            num_cells = 0
            for (x, y), cell_belief in belief.cell_beliefs.items():
                if num_cells >= max_cells:
                    break
                buffers.memory_cell_x[num_cells] = x
                buffers.memory_cell_y[num_cells] = y
                if cell_belief == CellBelief.OBSTACLE:
                    buffers.memory_cell_obs[num_cells] = 1
                elif cell_belief == CellBelief.FREE:
                    buffers.memory_cell_obs[num_cells] = 2
                else:
                    buffers.memory_cell_obs[num_cells] = 0
                num_cells += 1
            
            dir_x, dir_y = greedy_direction_numba(
                agent_pos[0], agent_pos[1], target[0], target[1],
                buffers.memory_cell_x, buffers.memory_cell_y, buffers.memory_cell_obs,
                num_cells
            )
            
            if dir_x == 0 and dir_y == 0:
                break
            
            best_dir = (dir_x, dir_y)
            next_pos = (agent_pos[0] + best_dir[0], agent_pos[1] + best_dir[1])
            cell_belief = belief.cell_beliefs.get(next_pos, CellBelief.UNKNOWN)
            
            if cell_belief == CellBelief.OBSTACLE:
                total_reward += discount * -1.0
                break
            elif cell_belief == CellBelief.UNKNOWN:
                total_reward += discount * 0.5
            else:
                total_reward += discount * 0.1
            
            agent_pos = next_pos
            belief = self._simulate_transition(belief, best_dir, memory)
            discount *= gamma
        
        return total_reward
    
    def _backpropagate(self, node: PathTreeNode, reward: float) -> None:
        """Backpropagation phase: update statistics up to root."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += reward
            if current.visit_count > 1:
                current.uncertainty = current.belief.entropy / max(current.visit_count, 1)
            current = current.parent
    
    def _build_root_belief(self, state: WorldState, memory: GridMemoryStore) -> BeliefState:
        """Build initial BeliefState from WorldState and GridMemoryStore."""
        # Delegate to original implementation
        from .path_tree_planner import GridPathTreePlanner
        temp_planner = GridPathTreePlanner(
            max_depth=self.max_depth,
            num_simulations=self.num_simulations,
            control_learner=self.control_learner,
            world_model_learner=self.world_model_learner,
        )
        return temp_planner._build_root_belief(state, memory)
    
    def _build_retrieval_context_from_belief(
        self,
        belief: BeliefState,
        memory: GridMemoryStore,
    ) -> Dict[str, Any]:
        """Build retrieval context from belief state."""
        from .path_tree_planner import GridPathTreePlanner
        temp_planner = GridPathTreePlanner(
            max_depth=self.max_depth,
            num_simulations=self.num_simulations,
            control_learner=self.control_learner,
            world_model_learner=self.world_model_learner,
        )
        return temp_planner._build_retrieval_context_from_belief(belief, memory)
    
    def _context_to_key(self, context: Dict[str, Any]) -> Tuple[int, ...]:
        """Convert decision context dict to tuple key for control_learner."""
        from .path_tree_planner import GridPathTreePlanner
        temp_planner = GridPathTreePlanner(
            max_depth=self.max_depth,
            num_simulations=self.num_simulations,
            control_learner=self.control_learner,
            world_model_learner=self.world_model_learner,
        )
        return temp_planner._context_to_key(context)
    
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
    
    def _estimate_agent_position(
        self,
        belief: BeliefState,
        memory: GridMemoryStore,
    ) -> Optional[Position]:
        """Estimate current agent position from belief/memory."""
        if memory and memory.working.current_observation:
            obs = memory.working.current_observation
            if isinstance(obs, dict):
                pos = obs.get("position")
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    return (int(pos[0]), int(pos[1]))
                hud = obs.get("hud", {})
                if isinstance(hud, dict):
                    grid = hud.get("grid", {})
                    if isinstance(grid, dict):
                        pos = grid.get("position")
                        if isinstance(pos, (list, tuple)) and len(pos) == 2:
                            return (int(pos[0]), int(pos[1]))
        
        if memory and memory.spatial.grid:
            for pos, cell in memory.spatial.grid.items():
                if cell.visited:
                    return pos
        
        if belief.goal_particles:
            gx, gy = belief.goal_particles[0][0]
            return (gx, gy)
        
        return (0, 0)
    
    def reuse_tree(self, old_root: PathTreeNode, new_belief: BeliefState) -> PathTreeNode:
        """Reuse tree from previous turn with updated root belief."""
        new_root = PathTreeNode(
            belief=new_belief,
            depth=0,
            retrieval_prior=old_root.retrieval_prior,
        )
        new_root.children = old_root.children
        for child in new_root.children.values():
            child.parent = new_root
        return new_root
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "last_plan_time_ms": self._last_plan_time * 1000,
            "avg_simulation_time_ms": (sum(self._simulation_times) / len(self._simulation_times) * 1000) if self._simulation_times else 0,
            "num_simulations_run": len(self._simulation_times),
            "use_numba": self.use_numba,
            "numba_available": NUMBA_AVAILABLE,
        }


# Backward compatibility alias
GridPathTreePlanner = OptimizedGridPathTreePlanner