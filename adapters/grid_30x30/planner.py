"""
BFS-based planner for Grid 30x30 game.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

from .world_model import WorldState, bfs_shortest_path


@dataclass
class ActionCandidate:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    utility: float = 0.0
    confidence: float = 0.0
    memory_guided: bool = False


class GridPlanner:
    """BFS + frontier ranking planner for grid worlds."""
    
    def __init__(
        self,
        strategy: "GridStrategyParameters" | None = None,
    ) -> None:
        from .config import GridStrategyParameters
        self.strategy = strategy or GridStrategyParameters()
        self.last_goal: Optional[Tuple[int, int]] = None
        self._decision_turn = 0
        self._blocked_transitions: dict[
            tuple[Tuple[int, int], Tuple[int, int]], int
        ] = {}

    def choose_action(
        self,
        state: Any,
        memory: Any = None,
        direction_preferences: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Optional[ActionCandidate]:
        self._decision_turn += 1
        self._blocked_transitions = {
            transition: expiry
            for transition, expiry in self._blocked_transitions.items()
            if expiry >= self._decision_turn
        }
        if state is None:
            return None

        if isinstance(state, dict):
            frontier = state.get("frontier_cells", [])
            if not frontier and memory is not None and hasattr(memory, "spatial"):
                frontier = memory.spatial.frontier
            if frontier:
                return ActionCandidate(
                    name="explore_frontier",
                    params={"frontier": frontier[0]},
                    utility=0.8,
                    confidence=0.7,
                )

        if isinstance(state, WorldState) and state.grid_size is not None:
            if direction_preferences is None and memory is not None:
                build_context = getattr(memory, "build_decision_context", None)
                retrieve = getattr(memory, "retrieve_direction_preferences", None)
                if callable(build_context) and callable(retrieve):
                    direction_preferences = retrieve(build_context(state))
            candidate = self._choose_world_action(
                state,
                memory,
                set(self._blocked_transitions),
                self.strategy,
                direction_preferences or {},
            )
            if candidate.memory_guided and memory is not None:
                record_guidance = getattr(memory, "record_guided_decision", None)
                if callable(record_guidance):
                    record_guidance()
            return candidate

        if hasattr(state, "goal") and state.goal is not None:
            if state.goal == "move_left":
                return ActionCandidate(name="move_left", params={"steps": 1}, utility=1.0, confidence=0.9)
            if state.goal == "start":
                return ActionCandidate(name="click", params={"x": 500, "y": 400}, utility=1.0, confidence=0.9)

        if hasattr(state, "objects"):
            if "start_button" in state.objects:
                return ActionCandidate(name="click", params={"x": 500, "y": 400}, utility=1.9, confidence=0.9)
            if "unknown" in state.objects:
                return ActionCandidate(name="wait", params={"duration": 0.1}, utility=0.2, confidence=0.2)

        if memory is not None and hasattr(memory, "spatial"):
            frontier = getattr(memory.spatial, "frontier", [])
            if frontier:
                return ActionCandidate(name="explore_frontier", params={"frontier": frontier[0]}, utility=0.8, confidence=0.7)

        return ActionCandidate(name="wait", params={"duration": 0.1}, utility=0.1, confidence=0.1)

    def record_blocked_transition(
        self,
        start: Tuple[int, int],
        direction: Tuple[int, int],
    ) -> None:
        destination = (start[0] + direction[0], start[1] + direction[1])
        self._blocked_transitions[(start, destination)] = self._decision_turn + 2

    @staticmethod
    def _choose_world_action(
        state: WorldState,
        memory: Any,
        blocked_transitions: set[
            tuple[Tuple[int, int], Tuple[int, int]]
        ],
        strategy: "GridStrategyParameters",
        direction_preferences: Dict[Tuple[int, int], float],
    ) -> ActionCandidate:
        start = state.current_position
        if start is None:
            return ActionCandidate(name="wait", params={"duration": 0.1})

        known_cells = {
            position
            for position, cell in memory.spatial.grid.items()
            if cell.seen and cell.obstacle is not True
        }
        known_cells.add(start)
        blocked = set(state.obstacle_positions)
        bounds = state.grid_size

        if state.target_position is not None:
            targets = [state.target_position]
        elif memory.spatial.goal_is_dynamic:
            # The inferred goal moved before it was re-observed: pursuing
            # remembered positions would chase a stale goal.
            targets = []
        elif memory.spatial.goal_change_count < strategy.goal_change_threshold:
            targets = list(memory.spatial.known_goal_positions)
        else:
            targets = []
        for target in targets:
            path, memory_guided = GridPlanner._path_with_memory_preference(
                start,
                target,
                blocked,
                bounds=bounds,
                allowed=known_cells,
                blocked_transitions=blocked_transitions,
                direction_preferences=direction_preferences,
            )
            if len(path) > 1:
                return ActionCandidate(
                    name=GridPlanner._action_for_step(path[0], path[1]),
                    utility=1.0,
                    confidence=state.confidence,
                    memory_guided=memory_guided,
                )

        frontiers = memory.spatial.frontier
        candidates: list[tuple[float, int, int, int, list[Tuple[int, int]], Tuple[int, int]]] = []
        for frontier in frontiers:
            path = bfs_shortest_path(
                start,
                frontier,
                blocked,
                bounds=bounds,
                allowed=known_cells,
                blocked_transitions=blocked_transitions,
            )
            if len(path) < 2:
                continue
            cell = memory.spatial.grid[frontier]
            x, y = frontier
            information_gain = sum(
                (x + dx, y + dy) not in memory.spatial.grid
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
                if 0 <= x + dx < bounds[0] and 0 <= y + dy < bounds[1]
            )
            score = (
                len(path) - 1
                + cell.visit_count
                * bounds[0]
                * bounds[1]
                * strategy.visit_penalty
                - information_gain * strategy.information_gain_bonus
            )
            candidates.append((
                score,
                frontier[1],
                frontier[0],
                -information_gain,
                path,
                frontier,
            ))

        if candidates:
            baseline = min(
                candidates,
                key=lambda item: (item[0], item[1], item[2], item[3], item[4]),
            )
            tied = [candidate for candidate in candidates if candidate[0] == baseline[0]]
            ranked_tied = []
            for candidate in tied:
                path, path_guided = GridPlanner._path_with_memory_preference(
                    start,
                    candidate[5],
                    blocked,
                    bounds=bounds,
                    allowed=known_cells,
                    blocked_transitions=blocked_transitions,
                    direction_preferences=direction_preferences,
                    baseline_path=candidate[4],
                )
                direction = (path[1][0] - start[0], path[1][1] - start[1])
                ranked_tied.append((
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    candidate[3],
                    path,
                    path_guided,
                    direction_preferences.get(direction, 0.0),
                ))
            selected = min(
                ranked_tied,
                key=lambda item: (
                    item[0],
                    -item[6],
                    item[1],
                    item[2],
                    item[3],
                    item[4],
                ),
            )
            _, _, _, _, path, path_guided, _ = selected
            selected_direction = (
                path[1][0] - start[0],
                path[1][1] - start[1],
            )
            baseline_direction = (
                baseline[4][1][0] - start[0],
                baseline[4][1][1] - start[1],
            )
            baseline_preference = direction_preferences.get(baseline_direction, 0.0)
            memory_guided = path_guided or (
                selected_direction != baseline_direction
                and selected[0] == baseline[0]
                and selected[6] > baseline_preference + 0.05
            )
            return ActionCandidate(
                name=GridPlanner._action_for_step(path[0], path[1]),
                utility=0.8,
                confidence=0.6,
                memory_guided=memory_guided,
            )

        return ActionCandidate(name="wait", params={"duration": 0.1}, utility=0.1, confidence=0.1)

    @staticmethod
    def _path_with_memory_preference(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: set[Tuple[int, int]],
        bounds: Optional[Tuple[int, int]],
        allowed: Optional[set[Tuple[int, int]]],
        blocked_transitions: set[
            tuple[Tuple[int, int], Tuple[int, int]]
        ],
        direction_preferences: Dict[Tuple[int, int], float],
        baseline_path: Optional[list[Tuple[int, int]]] = None,
    ) -> tuple[list[Tuple[int, int]], bool]:
        baseline = baseline_path or bfs_shortest_path(
                start,
                goal,
                blocked,
                bounds=bounds,
                allowed=allowed,
                blocked_transitions=blocked_transitions,
            )
        if len(baseline) < 2 or not direction_preferences:
            return baseline, False

        baseline_direction = (
            baseline[1][0] - start[0],
            baseline[1][1] - start[1],
        )
        selected_path = baseline
        selected_direction = baseline_direction
        selected_score = direction_preferences.get(baseline_direction, 0.0)
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if direction not in direction_preferences:
                continue
            neighbor = (start[0] + direction[0], start[1] + direction[1])
            if neighbor in blocked or (start, neighbor) in blocked_transitions:
                continue
            if bounds is not None and not (
                0 <= neighbor[0] < bounds[0] and 0 <= neighbor[1] < bounds[1]
            ):
                continue
            if allowed is not None and neighbor not in allowed:
                continue
            remaining = bfs_shortest_path(
                neighbor,
                goal,
                blocked,
                bounds=bounds,
                allowed=allowed,
                blocked_transitions=blocked_transitions,
            )
            if not remaining:
                continue
            candidate_path = [start, *remaining]
            if len(candidate_path) != len(baseline):
                continue
            score = direction_preferences[direction]
            if score > selected_score + 0.05:
                selected_path = candidate_path
                selected_direction = direction
                selected_score = score
        return selected_path, selected_direction != baseline_direction

    @staticmethod
    def _action_for_step(
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> str:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return {
            (0, -1): "move_up",
            (1, 0): "move_right",
            (0, 1): "move_down",
            (-1, 0): "move_left",
        }[(dx, dy)]