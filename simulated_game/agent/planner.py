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


@dataclass(frozen=True)
class GridStrategyParameters:
    visit_penalty: float = 1.0
    information_gain_bonus: float = 0.25
    goal_change_threshold: float = 1.0

    def __post_init__(self) -> None:
        for name, value in self.to_mapping().items():
            low, high = (1, 5) if name == "goal_change_threshold" else (0, 2)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(
                    f"{name} must be a finite value between {low} and {high}."
                )
            if name == "goal_change_threshold" and not float(value).is_integer():
                raise ValueError(f"{name} must be a whole number.")

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> "GridStrategyParameters":
        return cls(
            visit_penalty=float(values["visit_penalty"]),
            information_gain_bonus=float(values["information_gain_bonus"]),
            goal_change_threshold=float(values["goal_change_threshold"]),
        )

    def to_mapping(self) -> dict[str, float]:
        return {
            "visit_penalty": self.visit_penalty,
            "information_gain_bonus": self.information_gain_bonus,
            "goal_change_threshold": self.goal_change_threshold,
        }


class Planner:
    def __init__(
        self,
        strategy: GridStrategyParameters | None = None,
    ) -> None:
        self.strategy = strategy or GridStrategyParameters()
        self.last_goal: Optional[Tuple[int, int]] = None
        self._decision_turn = 0
        self._blocked_transitions: dict[
            tuple[Tuple[int, int], Tuple[int, int]], int
        ] = {}

    def choose_action(self, state: Any, memory: Any = None) -> Optional[ActionCandidate]:
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
            return self._choose_world_action(
                state,
                memory,
                set(self._blocked_transitions),
                self.strategy,
            )

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
        strategy: GridStrategyParameters,
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

        targets = (
            [state.target_position]
            if state.target_position is not None
            else (
                []
                if memory.spatial.goal_change_count
                >= strategy.goal_change_threshold
                else list(memory.spatial.known_goal_positions)
            )
        )
        for target in targets:
            path = bfs_shortest_path(
                start,
                target,
                blocked,
                bounds=bounds,
                allowed=known_cells,
                blocked_transitions=blocked_transitions,
            )
            if len(path) > 1:
                return ActionCandidate(
                    name=Planner._action_for_step(path[0], path[1]),
                    utility=1.0,
                    confidence=state.confidence,
                )

        frontiers = memory.spatial.frontier
        candidates: list[tuple[float, int, int, int, list[Tuple[int, int]]]] = []
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
            candidates.append((score, frontier[1], frontier[0], -information_gain, path))

        if candidates:
            _, _, _, _, path = min(candidates)
            return ActionCandidate(
                name=Planner._action_for_step(path[0], path[1]),
                utility=0.8,
                confidence=0.6,
            )

        return ActionCandidate(name="wait", params={"duration": 0.1}, utility=0.1, confidence=0.1)

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
