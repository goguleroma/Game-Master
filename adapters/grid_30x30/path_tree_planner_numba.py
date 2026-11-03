"""
Numba-optimized hot paths for PathTreePlanner.

This module contains JIT-compiled functions for the performance-critical
parts of the MCTS planner:
- Belief state transitions (particle filter update)
- Simulation rollouts
- Direction validation
- Particle filtering
"""

import math
from typing import List, Tuple, Optional, Dict
import numpy as np

# Try to import numba, provide fallback if not available
try:
    from numba import njit, types, typed
    from numba.typed import Dict as NumbaDict, List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback: define no-op decorators
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper if args else wrapper(args[0]) if args else wrapper
    NUMBA_AVAILABLE = False

# Type aliases
Position = Tuple[int, int]
Direction = Tuple[int, int]

# Constants for cell beliefs
CELL_UNKNOWN = 0
CELL_FREE = 1
CELL_OBSTACLE = 2

# Cardinal directions - use tuples for Numba compatibility
DIRECTIONS_LIST = ((0, -1), (1, 0), (0, 1), (-1, 0))

# Also provide as numpy array for vectorized ops
DIRECTIONS_NP = np.array([
    [0, -1],   # up
    [1, 0],    # right
    [0, 1],    # down
    [-1, 0],   # left
], dtype=np.int32)


@njit(cache=True)
def estimate_agent_position_from_belief(
    goal_particles_x: np.ndarray,
    goal_particles_y: np.ndarray,
    goal_particles_w: np.ndarray,
    num_particles: int,
) -> Position:
    """
    Estimate agent position from goal particles (inverse of particle positions).
    
    The particles represent goal positions relative to the agent.
    If goal is at (gx, gy) relative to agent, and agent is at (ax, ay),
    then absolute goal is (ax + gx, ay + gy).
    
    We use the mean particle position as a reference point.
    """
    if num_particles == 0:
        return (0, 0)
    
    # Weighted mean of particles
    total_w = 0.0
    mean_x = 0.0
    mean_y = 0.0
    for i in range(num_particles):
        w = goal_particles_w[i]
        if w > 0:
            mean_x += goal_particles_x[i] * w
            mean_y += goal_particles_y[i] * w
            total_w += w
    
    if total_w > 0:
        mean_x /= total_w
        mean_y /= total_w
    
    # Return rounded position as integer
    return (int(round(mean_x)), int(round(mean_y)))


@njit(cache=True)
def get_valid_directions_numba(
    agent_x: int,
    agent_y: int,
    cell_beliefs_x: np.ndarray,
    cell_beliefs_y: np.ndarray,
    cell_beliefs_val: np.ndarray,
    num_cells: int,
) -> Tuple[np.ndarray, int]:
    """
    Get valid movement directions from current position.
    
    Returns array of valid directions and count.
    """
    valid_dirs = np.zeros((4, 2), dtype=np.int32)
    count = 0
    
    for d in range(4):
        dx, dy = DIRECTIONS_LIST[d]
        new_x = agent_x + dx
        new_y = agent_y + dy
        
        # Check if cell is known obstacle
        is_obstacle = False
        for i in range(num_cells):
            if cell_beliefs_x[i] == new_x and cell_beliefs_y[i] == new_y:
                if cell_beliefs_val[i] == CELL_OBSTACLE:
                    is_obstacle = True
                break
        
        if not is_obstacle:
            valid_dirs[count, 0] = dx
            valid_dirs[count, 1] = dy
            count += 1
    
    return valid_dirs, count


@njit(cache=True)
def is_direction_valid_numba(
    agent_x: int,
    agent_y: int,
    direction: Direction,
    cell_beliefs_x: np.ndarray,
    cell_beliefs_y: np.ndarray,
    cell_beliefs_val: np.ndarray,
    num_cells: int,
) -> bool:
    """Check if a direction is valid (not into known obstacle)."""
    new_x = agent_x + direction[0]
    new_y = agent_y + direction[1]
    
    for i in range(num_cells):
        if cell_beliefs_x[i] == new_x and cell_beliefs_y[i] == new_y:
            if cell_beliefs_val[i] == CELL_OBSTACLE:
                return False
            break
    return True


@njit(cache=True)
def simulate_transition_numba(
    # Input belief state
    goal_particles_x: np.ndarray,
    goal_particles_y: np.ndarray,
    goal_particles_w: np.ndarray,
    num_goal_particles: int,
    # Cell beliefs
    cell_beliefs_x: np.ndarray,
    cell_beliefs_y: np.ndarray,
    cell_beliefs_val: np.ndarray,
    num_cell_beliefs: int,
    # Entity models
    entity_ids: np.ndarray,  # String IDs stored as int32 hashes
    entity_pos_x: np.ndarray,
    entity_pos_y: np.ndarray,
    entity_vel_x: np.ndarray,
    entity_vel_y: np.ndarray,
    entity_category: np.ndarray,  # 0=unknown, 1=stationary, 2=mobile, 3=approaches, 4=moves_away, 5=erratic
    entity_confidence: np.ndarray,
    num_entities: int,
    entropy: float,
    turn: int,
    # Action
    direction: Direction,
    # Memory (simplified - just grid info)
    memory_cell_x: np.ndarray,
    memory_cell_y: np.ndarray,
    memory_cell_obstacle: np.ndarray,
    memory_num_cells: int,
    # World model learner predictions (pre-computed)
    wml_entity_preds_x: np.ndarray,
    wml_entity_preds_y: np.ndarray,
    wml_num_entities: int,
    # Output arrays (pre-allocated)
    out_goal_particles_x: np.ndarray,
    out_goal_particles_y: np.ndarray,
    out_goal_particles_w: np.ndarray,
    out_cell_beliefs_x: np.ndarray,
    out_cell_beliefs_y: np.ndarray,
    out_cell_beliefs_val: np.ndarray,
    out_entity_pos_x: np.ndarray,
    out_entity_pos_y: np.ndarray,
) -> Tuple[int, int, float, int]:
    """
    Simulate belief state transition after taking an action.
    
    Returns: (num_goal_particles, num_cell_beliefs, new_entropy, new_turn)
    """
    dx, dy = direction
    
    # 1. Update goal particles: agent moves (dx, dy), so goal appears to move (-dx, -dy)
    for i in range(num_goal_particles):
        out_goal_particles_x[i] = goal_particles_x[i] - dx
        out_goal_particles_y[i] = goal_particles_y[i] - dy
        out_goal_particles_w[i] = goal_particles_w[i]
    
    new_num_goal_particles = num_goal_particles
    
    # 2. Copy cell beliefs
    for i in range(num_cell_beliefs):
        out_cell_beliefs_x[i] = cell_beliefs_x[i]
        out_cell_beliefs_y[i] = cell_beliefs_y[i]
        out_cell_beliefs_val[i] = cell_beliefs_val[i]
    new_num_cell_beliefs = num_cell_beliefs
    
    # 3. Update entity positions using WML predictions or constant velocity
    for i in range(num_entities):
        if i < wml_num_entities and wml_entity_preds_x[i] != -9999:
            # Use WML prediction
            out_entity_pos_x[i] = wml_entity_preds_x[i]
            out_entity_pos_y[i] = wml_entity_preds_y[i]
        else:
            # Constant velocity fallback
            out_entity_pos_x[i] = entity_pos_x[i] + int(round(entity_vel_x[i]))
            out_entity_pos_y[i] = entity_pos_y[i] + int(round(entity_vel_y[i]))
    
    # 4. Recompute entropy (simplified)
    # Goal entropy
    goal_entropy = 0.0
    if new_num_goal_particles > 0:
        for i in range(new_num_goal_particles):
            w = out_goal_particles_w[i]
            if w > 0:
                goal_entropy -= w * math.log(w)
        max_entropy = math.log(new_num_goal_particles)
        if max_entropy > 0:
            goal_entropy /= max_entropy
    
    # Cell entropy
    unknown_count = 0
    for i in range(new_num_cell_beliefs):
        if out_cell_beliefs_val[i] == CELL_UNKNOWN:
            unknown_count += 1
    cell_entropy = unknown_count / max(new_num_cell_beliefs, 1)
    
    new_entropy = 0.7 * goal_entropy + 0.3 * cell_entropy
    new_turn = turn + 1
    
    return new_num_goal_particles, new_num_cell_beliefs, new_entropy, new_turn


@njit(cache=True)
def greedy_direction_numba(
    agent_x: int,
    agent_y: int,
    goal_x: int,
    goal_y: int,
    cell_beliefs_x: np.ndarray,
    cell_beliefs_y: np.ndarray,
    cell_beliefs_val: np.ndarray,
    num_cells: int,
) -> Tuple[int, int]:
    """Find best cardinal direction toward goal avoiding known obstacles."""
    best_dir_x = 0
    best_dir_y = 0
    best_dist = 2147483647  # INT_MAX
    
    for d in range(4):
        dx, dy = DIRECTIONS_LIST[d]
        next_x = agent_x + dx
        next_y = agent_y + dy
        
        # Skip known obstacles
        is_obstacle = False
        for i in range(num_cells):
            if cell_beliefs_x[i] == next_x and cell_beliefs_y[i] == next_y:
                if cell_beliefs_val[i] == CELL_OBSTACLE:
                    is_obstacle = True
                break
        if is_obstacle:
            continue
        
        dist = abs(next_x - goal_x) + abs(next_y - goal_y)
        if dist < best_dist:
            best_dist = dist
            best_dir_x = dx
            best_dir_y = dy
    
    if best_dist == 2147483647:
        return (0, 0)  # No valid direction
    return (best_dir_x, best_dir_y)


@njit(cache=True)
def simulate_rollout_numba(
    # Initial state
    agent_x: int,
    agent_y: int,
    goal_particles_x: np.ndarray,
    goal_particles_y: np.ndarray,
    goal_particles_w: np.ndarray,
    num_goal_particles: int,
    cell_beliefs_x: np.ndarray,
    cell_beliefs_y: np.ndarray,
    cell_beliefs_val: np.ndarray,
    num_cell_beliefs: int,
    entity_pos_x: np.ndarray,
    entity_pos_y: np.ndarray,
    entity_vel_x: np.ndarray,
    entity_vel_y: np.ndarray,
    entity_category: np.ndarray,
    entity_confidence: np.ndarray,
    num_entities: int,
    entropy: float,
    turn: int,
    # Memory
    memory_frontier_x: np.ndarray,
    memory_frontier_y: np.ndarray,
    memory_num_frontier: int,
    memory_cell_x: np.ndarray,
    memory_cell_y: np.ndarray,
    memory_cell_obstacle: np.ndarray,
    memory_num_cells: int,
    # Control learner predictions (pre-computed)
    control_pred_dx: np.ndarray,
    control_pred_dy: np.ndarray,
    num_controls: int,
    # Parameters
    max_steps: int,
    gamma: float,
    # RNG state
    rng_seed: int,
) -> float:
    """
    Run a greedy rollout simulation toward the goal.
    
    Returns discounted cumulative reward.
    """
    # Simple LCG RNG for Numba
    rng_state = rng_seed
    
    def rng_random() -> float:
        nonlocal rng_state
        rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
        return rng_state / 4294967296.0
    
    # Sample goal from particles
    goal_x = 0
    goal_y = 0
    has_goal = False
    
    if num_goal_particles > 0:
        rand = rng_random()
        cumulative = 0.0
        for i in range(num_goal_particles):
            cumulative += goal_particles_w[i]
            if rand <= cumulative:
                goal_x = int(goal_particles_x[i])
                goal_y = int(goal_particles_y[i])
                has_goal = True
                break
        if not has_goal and num_goal_particles > 0:
            goal_x = int(goal_particles_x[num_goal_particles - 1])
            goal_y = int(goal_particles_y[num_goal_particles - 1])
            has_goal = True
    
    if not has_goal:
        # No goal - explore toward frontier
        if memory_num_frontier > 0:
            best_dist = 2147483647
            for i in range(memory_num_frontier):
                dist = abs(memory_frontier_x[i] - agent_x) + abs(memory_frontier_y[i] - agent_y)
                if dist < best_dist:
                    best_dist = dist
                    goal_x = memory_frontier_x[i]
                    goal_y = memory_frontier_y[i]
            has_goal = True
        else:
            return 0.0
    
    total_reward = 0.0
    discount = 1.0
    
    # Local copies for mutation during rollout
    cur_agent_x = agent_x
    cur_agent_y = agent_y
    cur_num_goal = num_goal_particles
    cur_num_cells = num_cell_beliefs
    cur_entropy = entropy
    cur_turn = turn
    
    # Pre-allocated arrays for transition (reused each step)
    next_goal_x = np.zeros(30, dtype=np.int32)
    next_goal_y = np.zeros(30, dtype=np.int32)
    next_goal_w = np.zeros(30, dtype=np.float32)
    next_cell_x = np.zeros(200, dtype=np.int32)
    next_cell_y = np.zeros(200, dtype=np.int32)
    next_cell_val = np.zeros(200, dtype=np.int32)
    next_entity_x = np.zeros(20, dtype=np.int32)
    next_entity_y = np.zeros(20, dtype=np.int32)
    
    for step in range(max_steps):
        if cur_agent_x == goal_x and cur_agent_y == goal_y:
            total_reward += discount * 10.0
            break
        
        # Find best direction toward goal
        dir_x, dir_y = greedy_direction_numba(
            cur_agent_x, cur_agent_y, goal_x, goal_y,
            cell_beliefs_x, cell_beliefs_y, cell_beliefs_val, cur_num_cells
        )
        
        if dir_x == 0 and dir_y == 0:
            break
        
        # Use control prediction if available
        pred_dx = dir_x
        pred_dy = dir_y
        if num_controls > 0:
            # Use first available control prediction
            pred_dx = int(round(control_pred_dx[0]))
            pred_dy = int(round(control_pred_dy[0]))
        
        next_x = cur_agent_x + pred_dx
        next_y = cur_agent_y + pred_dy
        
        # Check cell belief
        cell_belief = CELL_UNKNOWN
        is_frontier = False
        for i in range(cur_num_cells):
            if cell_beliefs_x[i] == next_x and cell_beliefs_y[i] == next_y:
                cell_belief = cell_beliefs_val[i]
                break
        
        # Check if frontier
        for i in range(memory_num_frontier):
            if memory_frontier_x[i] == next_x and memory_frontier_y[i] == next_y:
                is_frontier = True
                break
        
        if cell_belief == CELL_OBSTACLE:
            total_reward += discount * -1.0
            break
        elif cell_belief == CELL_FREE:
            if is_frontier:
                total_reward += discount * 0.5
            else:
                total_reward += discount * 0.1
        else:
            total_reward += discount * 0.3
        
        # Entity penalty check
        for e in range(num_entities):
            if entity_category[e] in (3, 2) and entity_confidence[e] > 0.3:  # approaches or mobile
                # Predict 2 steps
                for step_idx in range(2):
                    pred_ex = entity_pos_x[e] + int(round(entity_vel_x[e] * (step_idx + 1)))
                    pred_ey = entity_pos_y[e] + int(round(entity_vel_y[e] * (step_idx + 1)))
                    if pred_ex == next_x and pred_ey == next_y:
                        penalty_factor = 1.0 / (step_idx + 1)
                        if entity_category[e] == 3 and entity_confidence[e] > 0.5:  # approaches_agent
                            total_reward += discount * -8.0 * penalty_factor
                        else:
                            total_reward += discount * -3.0 * penalty_factor
                        break
        
        # Transition
        cur_num_goal, cur_num_cells, cur_entropy, cur_turn = simulate_transition_numba(
            goal_particles_x, goal_particles_y, goal_particles_w, cur_num_goal,
            cell_beliefs_x, cell_beliefs_y, cell_beliefs_val, cur_num_cells,
            entity_ids=None,  # Simplified
            entity_pos_x=entity_pos_x, entity_pos_y=entity_pos_y,
            entity_vel_x=entity_vel_x, entity_vel_y=entity_vel_y,
            entity_category=entity_category, entity_confidence=entity_confidence,
            num_entities=num_entities,
            entropy=cur_entropy, turn=cur_turn,
            direction=(pred_dx, pred_dy),
            memory_cell_x=memory_cell_x, memory_cell_y=memory_cell_y,
            memory_cell_obstacle=memory_cell_obstacle, memory_num_cells=memory_num_cells,
            wml_entity_preds_x=np.zeros(20, dtype=np.int32),  # Not used in rollout
            wml_entity_preds_y=np.zeros(20, dtype=np.int32),
            wml_num_entities=0,
            out_goal_particles_x=next_goal_x,
            out_goal_particles_y=next_goal_y,
            out_goal_particles_w=next_goal_w,
            out_cell_beliefs_x=next_cell_x,
            out_cell_beliefs_y=next_cell_y,
            out_cell_beliefs_val=next_cell_val,
            out_entity_pos_x=next_entity_x,
            out_entity_pos_y=next_entity_y,
        )
        
        # Copy back for next iteration
        for i in range(cur_num_goal):
            goal_particles_x[i] = next_goal_x[i]
            goal_particles_y[i] = next_goal_y[i]
            goal_particles_w[i] = next_goal_w[i]
        for i in range(cur_num_cells):
            cell_beliefs_x[i] = next_cell_x[i]
            cell_beliefs_y[i] = next_cell_y[i]
            cell_beliefs_val[i] = next_cell_val[i]
        for i in range(num_entities):
            entity_pos_x[i] = next_entity_x[i]
            entity_pos_y[i] = next_entity_y[i]
        
        cur_agent_x = next_x
        cur_agent_y = next_y
        discount *= gamma
    
    return total_reward


@njit(cache=True)
def update_goal_particles_numba(
    goal_particles_x: np.ndarray,
    goal_particles_y: np.ndarray,
    goal_particles_w: np.ndarray,
    num_particles: int,
    target_x: int,
    target_y: int,
    target_observed: bool,
    visible_cells_x: np.ndarray,
    visible_cells_y: np.ndarray,
    num_visible: int,
    known_goals_x: np.ndarray,
    known_goals_y: np.ndarray,
    num_known_goals: int,
    goal_is_dynamic: bool,
    turn: int,
    out_particles_x: np.ndarray,
    out_particles_y: np.ndarray,
    out_particles_w: np.ndarray,
) -> int:
    """
    Update particle filter for goal positions.
    
    Returns new number of particles.
    """
    if target_observed:
        # Goal observed: collapse to observed position
        out_particles_x[0] = target_x
        out_particles_y[0] = target_y
        out_particles_w[0] = 1.0
        return 1
    
    # Build visible set for fast lookup
    # (simplified - just check against arrays)
    
    if num_particles == 0:
        # Initialize from known goals
        if num_known_goals > 0:
            if goal_is_dynamic:
                # Spread around known goals
                idx = 0
                for g in range(num_known_goals):
                    gx = known_goals_x[g]
                    gy = known_goals_y[g]
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if idx >= 30:
                                break
                            # Check if not in visible cells
                            in_visible = False
                            for v in range(num_visible):
                                if visible_cells_x[v] == gx + dx and visible_cells_y[v] == gy + dy:
                                    in_visible = True
                                    break
                            if not in_visible:
                                out_particles_x[idx] = gx + dx
                                out_particles_y[idx] = gy + dy
                                out_particles_w[idx] = 1.0 / 9.0
                                idx += 1
                        if idx >= 30:
                            break
                    if idx >= 30:
                        break
                return min(idx, 30)
            else:
                # Static: uniform over known goals
                w = 1.0 / num_known_goals
                for g in range(min(num_known_goals, 30)):
                    out_particles_x[g] = known_goals_x[g]
                    out_particles_y[g] = known_goals_y[g]
                    out_particles_w[g] = w
                return min(num_known_goals, 30)
        return 0
    
    # Propagate existing particles
    new_count = 0
    for i in range(num_particles):
        px = goal_particles_x[i]
        py = goal_particles_y[i]
        pw = goal_particles_w[i]
        
        if goal_is_dynamic:
            if turn % 2 == 0:
                # Random walk
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    if new_count >= 30:
                        break
                    nx = px + dx
                    ny = py + dy
                    # Check visible
                    in_visible = False
                    for v in range(num_visible):
                        if visible_cells_x[v] == nx and visible_cells_y[v] == ny:
                            in_visible = True
                            break
                    if not in_visible:
                        out_particles_x[new_count] = nx
                        out_particles_y[new_count] = ny
                        out_particles_w[new_count] = pw / 5.0
                        new_count += 1
            else:
                # Stay
                if new_count < 30:
                    out_particles_x[new_count] = px
                    out_particles_y[new_count] = py
                    out_particles_w[new_count] = pw
                    new_count += 1
        else:
            # Static: keep if not visible
            in_visible = False
            for v in range(num_visible):
                if visible_cells_x[v] == px and visible_cells_y[v] == py:
                    in_visible = True
                    break
            if not in_visible and new_count < 30:
                out_particles_x[new_count] = px
                out_particles_y[new_count] = py
                out_particles_w[new_count] = pw
                new_count += 1
    
    # Normalize
    total_w = 0.0
    for i in range(new_count):
        total_w += out_particles_w[i]
    if total_w > 0:
        for i in range(new_count):
            out_particles_w[i] /= total_w
    
    # Resample if too few
    if new_count < 5 and num_known_goals > 0:
        for g in range(num_known_goals):
            gx = known_goals_x[g]
            gy = known_goals_y[g]
            in_visible = False
            for v in range(num_visible):
                if visible_cells_x[v] == gx and visible_cells_y[v] == gy:
                    in_visible = True
                    break
            if not in_visible and new_count < 30:
                out_particles_x[new_count] = gx
                out_particles_y[new_count] = gy
                out_particles_w[new_count] = 1.0 / num_known_goals
                new_count += 1
        
        # Re-normalize
        total_w = 0.0
        for i in range(new_count):
            total_w += out_particles_w[i]
        if total_w > 0:
            for i in range(new_count):
                out_particles_w[i] /= total_w
    
    return min(new_count, 30)


# Helper functions to convert between Python objects and Numba-compatible arrays

def belief_state_to_arrays(belief: 'BeliefState', max_particles: int = 30, max_cells: int = 200, max_entities: int = 20) -> Dict:
    """Convert BeliefState to Numba-compatible arrays."""
    # Goal particles
    gp_x = np.zeros(max_particles, dtype=np.int32)
    gp_y = np.zeros(max_particles, dtype=np.int32)
    gp_w = np.zeros(max_particles, dtype=np.float32)
    num_gp = len(belief.goal_particles)
    for i, (pos, w) in enumerate(belief.goal_particles[:max_particles]):
        gp_x[i] = pos[0]
        gp_y[i] = pos[1]
        gp_w[i] = w
    
    # Cell beliefs
    cb_x = np.zeros(max_cells, dtype=np.int32)
    cb_y = np.zeros(max_cells, dtype=np.int32)
    cb_val = np.zeros(max_cells, dtype=np.int32)
    num_cb = len(belief.cell_beliefs)
    for i, (pos, belief_val) in enumerate(belief.cell_beliefs.items()):
        if i >= max_cells:
            break
        cb_x[i] = pos[0]
        cb_y[i] = pos[1]
        if belief_val == CellBelief.UNKNOWN:
            cb_val[i] = CELL_UNKNOWN
        elif belief_val == CellBelief.FREE:
            cb_val[i] = CELL_FREE
        else:
            cb_val[i] = CELL_OBSTACLE
    
    # Entity models
    e_pos_x = np.zeros(max_entities, dtype=np.int32)
    e_pos_y = np.zeros(max_entities, dtype=np.int32)
    e_vel_x = np.zeros(max_entities, dtype=np.float32)
    e_vel_y = np.zeros(max_entities, dtype=np.float32)
    e_cat = np.zeros(max_entities, dtype=np.int32)
    e_conf = np.zeros(max_entities, dtype=np.float32)
    num_e = len(belief.entity_models)
    cat_map = {
        "unclassified": 0,
        "stationary": 1,
        "mobile": 2,
        "approaches_agent": 3,
        "moves_away_from_agent": 4,
        "erratic": 5,
    }
    for i, (eid, model) in enumerate(belief.entity_models.items()):
        if i >= max_entities:
            break
        e_pos_x[i] = model.position[0]
        e_pos_y[i] = model.position[1]
        e_vel_x[i] = model.velocity[0]
        e_vel_y[i] = model.velocity[1]
        e_cat[i] = cat_map.get(model.category, 0)
        e_conf[i] = model.behavior_confidence
    
    return {
        'goal_particles_x': gp_x,
        'goal_particles_y': gp_y,
        'goal_particles_w': gp_w,
        'num_goal_particles': num_gp,
        'cell_beliefs_x': cb_x,
        'cell_beliefs_y': cb_y,
        'cell_beliefs_val': cb_val,
        'num_cell_beliefs': num_cb,
        'entity_pos_x': e_pos_x,
        'entity_pos_y': e_pos_y,
        'entity_vel_x': e_vel_x,
        'entity_vel_y': e_vel_y,
        'entity_category': e_cat,
        'entity_confidence': e_conf,
        'num_entities': num_e,
        'entropy': belief.entropy,
        'turn': belief.turn,
    }


# Import at module level for type hints
from .path_tree_planner import BeliefState, CellBelief

# Local DIRECTIONS to avoid circular import
DIRECTIONS: Tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


# Pre-allocated buffers for performance
class NumbaBuffers:
    """Pre-allocated arrays to avoid allocation in hot paths."""
    
    MAX_PARTICLES = 30
    MAX_CELLS = 200
    MAX_ENTITIES = 20
    MAX_FRONTIER = 100
    MAX_CONTROLS = 10
    
    def __init__(self):
        # Transition buffers
        self.next_goal_x = np.zeros(self.MAX_PARTICLES, dtype=np.int32)
        self.next_goal_y = np.zeros(self.MAX_PARTICLES, dtype=np.int32)
        self.next_goal_w = np.zeros(self.MAX_PARTICLES, dtype=np.float32)
        self.next_cell_x = np.zeros(self.MAX_CELLS, dtype=np.int32)
        self.next_cell_y = np.zeros(self.MAX_CELLS, dtype=np.int32)
        self.next_cell_val = np.zeros(self.MAX_CELLS, dtype=np.int32)
        self.next_entity_x = np.zeros(self.MAX_ENTITIES, dtype=np.int32)
        self.next_entity_y = np.zeros(self.MAX_ENTITIES, dtype=np.int32)
        
        # Particle update buffers
        self.new_particles_x = np.zeros(self.MAX_PARTICLES, dtype=np.int32)
        self.new_particles_y = np.zeros(self.MAX_PARTICLES, dtype=np.int32)
        self.new_particles_w = np.zeros(self.MAX_PARTICLES, dtype=np.float32)
        
        # WML predictions
        self.wml_preds_x = np.zeros(self.MAX_ENTITIES, dtype=np.int32)
        self.wml_preds_y = np.zeros(self.MAX_ENTITIES, dtype=np.int32)
        
        # Control predictions
        self.control_pred_dx = np.zeros(self.MAX_CONTROLS, dtype=np.float32)
        self.control_pred_dy = np.zeros(self.MAX_CONTROLS, dtype=np.float32)
        
        # Memory arrays
        self.memory_cell_x = np.zeros(self.MAX_CELLS, dtype=np.int32)
        self.memory_cell_y = np.zeros(self.MAX_CELLS, dtype=np.int32)
        self.memory_cell_obs = np.zeros(self.MAX_CELLS, dtype=np.int32)
        self.memory_frontier_x = np.zeros(self.MAX_FRONTIER, dtype=np.int32)
        self.memory_frontier_y = np.zeros(self.MAX_FRONTIER, dtype=np.int32)
        self.memory_goals_x = np.zeros(20, dtype=np.int32)
        self.memory_goals_y = np.zeros(20, dtype=np.int32)
        
        # Visible cells
        self.visible_x = np.zeros(100, dtype=np.int32)
        self.visible_y = np.zeros(100, dtype=np.int32)


# Global buffer instance
_BUFFERS = NumbaBuffers()


def get_buffers() -> NumbaBuffers:
    """Get the global pre-allocated buffers."""
    return _BUFFERS