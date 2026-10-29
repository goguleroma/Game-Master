from collections import Counter
import math
from typing import Any

from core.types import Action, ActionResult, GameState
from learning.dqn import DQNAgent
from .memory import GridMemoryStore
from .planner import Planner
from .world_model import WorldModel


class NeuralGridPolicy:
    """Grid adapter for a learned Q-policy; the network sees observations, not game internals."""

    ACTION_COUNT = 6
    FEATURE_COUNT = 40
    PLANNED_DIRECTION_START = 31

    def __init__(
        self,
        seed: int = 0,
        training: bool = False,
        epsilon_start: float = 1.0,
    ) -> None:
        self.agent = DQNAgent(
            self.FEATURE_COUNT,
            self.ACTION_COUNT,
            seed=seed,
            epsilon_start=epsilon_start,
        )
        self.training = training
        self.world_model = WorldModel()
        self._pending: tuple[
            tuple[float, ...],
            int,
            dict[str, Any],
            dict[str, Any],
        ] | None = None
        self._novel_states: Counter[tuple[int, ...]] = Counter()
        self.losses: list[float] = []

    def begin_episode(self) -> None:
        self._pending = None

    def decide(self, observation: GameState, memory: GridMemoryStore) -> Action | None:
        normalized = self._normalize(observation)
        controls = normalized.get("controls", [])
        if not isinstance(controls, list) or not controls:
            return None
        if len(controls) > self.ACTION_COUNT:
            raise ValueError("Neural grid policy received more controls than its action head supports.")
        if any(not isinstance(control, str) or not control for control in controls):
            raise ValueError("Available controls must be non-empty strings.")

        features = self._encode(normalized, memory)
        if self._pending is not None and self.training:
            prior_features, action_index, before, result = self._pending
            reward = self._reward(before, normalized, features, result)
            mask = tuple(index < len(controls) for index in range(self.ACTION_COUNT))
            self.agent.remember(
                prior_features,
                action_index,
                reward,
                features,
                mask,
                done=False,
            )
            loss = self.agent.train_step()
            if loss is not None:
                self.losses.append(loss)
        self._pending = None

        mask = tuple(index < len(controls) for index in range(self.ACTION_COUNT))
        action_index = self.agent.select_action(
            features,
            mask,
            explore=self.training,
        )
        self._pending = (
            features,
            action_index,
            normalized,
            {},
        )
        return Action(name=controls[action_index])

    def observe_result(
        self,
        result: ActionResult,
        memory: GridMemoryStore,
    ) -> None:
        if self._pending is None or not self.training:
            return
        status = result.details.get("status")
        if status in {"goal_reached", "won", "caught", "lost"}:
            features, action_index, before, _ = self._pending
            reward = self._reward(
                before,
                before,
                features,
                result.details,
                terminal=status,
            )
            self.agent.remember(
                features,
                action_index,
                reward,
                tuple(0.0 for _ in range(self.FEATURE_COUNT)),
                tuple(False for _ in range(self.ACTION_COUNT)),
                done=True,
            )
            loss = self.agent.train_step()
            if loss is not None:
                self.losses.append(loss)
            self._pending = None
            return
        self._pending = (
            self._pending[0],
            self._pending[1],
            self._pending[2],
            dict(result.details),
        )

    def finish_episode(
        self,
        observation: dict[str, Any],
        memory: GridMemoryStore,
    ) -> None:
        if not self.training or self._pending is None:
            self._pending = None
            return
        features, action_index, before, result = self._pending
        next_features = self._encode(observation, memory)
        reward = self._reward(before, observation, next_features, result)
        self.agent.remember(
            features,
            action_index,
            reward,
            next_features,
            tuple(False for _ in range(self.ACTION_COUNT)),
            done=True,
        )
        loss = self.agent.train_step()
        if loss is not None:
            self.losses.append(loss)
        self._pending = None

    def _encode(
        self,
        observation: dict[str, Any],
        memory: GridMemoryStore,
        update_memory: bool = True,
    ) -> tuple[float, ...]:
        state = self.world_model.update(observation)
        grid = observation.get("hud", {}).get("grid", {})
        if update_memory:
            current_turn = len(memory.experiences)
            visible = set(state.visible_cells)
            for cell in visible:
                memory.mark_cell(
                    cell,
                    seen=True,
                    obstacle=False,
                    value="",
                    turn=current_turn,
                )
            memory.observe_goal(state.target_position, list(visible))
            for obstacle in state.obstacle_positions:
                memory.mark_cell(
                    obstacle,
                    seen=True,
                    obstacle=True,
                    value="obstacle",
                    turn=current_turn,
                )
            if state.target_position is not None:
                memory.mark_cell(
                    state.target_position,
                    seen=True,
                    obstacle=False,
                    value="target",
                    turn=current_turn,
                )
            if state.current_position is not None:
                memory.mark_cell(
                    state.current_position,
                    seen=True,
                    visited=True,
                    obstacle=False,
                    value="agent",
                    turn=current_turn,
                )
            for entity in state.entities:
                if entity.entity_id is not None and entity.position is not None:
                    memory.observe_entity(
                        entity.entity_id,
                        entity.position,
                        state.current_position,
                        current_turn,
                    )
            if state.grid_size is not None:
                memory.update_frontier(*state.grid_size)
        width = grid.get("width", 1)
        height = grid.get("height", 1)
        position = state.current_position or (0, 0)
        visible_target = state.target_position
        remembered = (
            memory.spatial.known_goal_positions[-1]
            if memory.spatial.known_goal_positions
            else None
        )
        entities = grid.get("entities", [])
        visible_enemy = next(
            (
                entity
                for entity in entities
                if isinstance(entity, dict)
                and entity.get("id") == "entity-2"
                and isinstance(entity.get("position"), list)
            ),
            None,
        )
        obstacles = {
            tuple(point)
            for point in grid.get("obstacles", [])
            if isinstance(point, list) and len(point) == 2
        }
        x, y = position
        target_offset = (
            (
                (visible_target[0] - x) / max(width, 1),
                (visible_target[1] - y) / max(height, 1),
            )
            if visible_target is not None
            else (0.0, 0.0)
        )
        remembered_offset = (
            (
                (remembered[0] - x) / max(width, 1),
                (remembered[1] - y) / max(height, 1),
            )
            if remembered is not None
            else (0.0, 0.0)
        )
        enemy_position = (
            tuple(visible_enemy["position"])
            if visible_enemy is not None
            else None
        )
        enemy_offset = (
            (
                (enemy_position[0] - x) / max(width, 1),
                (enemy_position[1] - y) / max(height, 1),
            )
            if enemy_position is not None
            else (0.0, 0.0)
        )
        frontier = memory.spatial.frontier
        nearest_frontier = (
            min(frontier, key=lambda cell: abs(cell[0] - x) + abs(cell[1] - y))
            if frontier
            else None
        )
        frontier_offset = (
            (
                (nearest_frontier[0] - x) / max(width, 1),
                (nearest_frontier[1] - y) / max(height, 1),
            )
            if nearest_frontier is not None
            else (0.0, 0.0)
        )
        action_index = self._previous_action_index(observation.get("controls", []), memory)
        planned_direction, intent_is_frontier = self._planning_hint(state, memory)
        direction_features = [
            float(planned_direction == direction)
            for direction in ((0, -1), (1, 0), (0, 1), (-1, 0))
        ]
        adjacent = ((0, -1), (1, 0), (0, 1), (-1, 0))
        values = [
            x / max(width - 1, 1),
            y / max(height - 1, 1),
            float(observation.get("hud", {}).get("health", 0)) / 10.0,
            float(visible_target is not None),
            *target_offset,
            float(remembered is not None),
            *remembered_offset,
            float(memory.spatial.goal_is_dynamic),
            float(visible_enemy is not None),
            *enemy_offset,
            float(visible_enemy.get("health", 0)) / 10.0
            if visible_enemy is not None
            else 0.0,
            *[
                float((x + dx, y + dy) in obstacles)
                for dx, dy in adjacent
            ],
            *[
                min(
                    memory.spatial.grid.get((x + dx, y + dy)).visit_count / 10.0,
                    1.0,
                )
                if memory.spatial.grid.get((x + dx, y + dy)) is not None
                else 0.0
                for dx, dy in adjacent
            ],
            float(nearest_frontier is not None),
            *frontier_offset,
            *[
                float(action_index == index)
                for index in range(self.ACTION_COUNT)
            ],
            *direction_features,
            intent_is_frontier,
            *self._previous_result_features(memory),
        ]
        if len(values) != self.FEATURE_COUNT:
            raise RuntimeError(
                f"Neural observation encoder produced {len(values)} features; "
                f"expected {self.FEATURE_COUNT}."
            )
        features = tuple(float(value) for value in values)
        key = tuple(round(value * 20) for value in features)
        self._novel_states[key] += 1
        return features

    @staticmethod
    def _planning_hint(
        state: Any,
        memory: GridMemoryStore,
    ) -> tuple[tuple[int, int] | None, float]:
        if state.grid_size is None:
            return None, 0.0
        plan = Planner().choose_action(state, memory)
        if plan is None:
            return None, 0.0
        direction = {
            "move_up": (0, -1),
            "move_right": (1, 0),
            "move_down": (0, 1),
            "move_left": (-1, 0),
        }.get(plan.name)
        intent_is_frontier = float(
            direction is not None and plan.utility < 1.0
        )
        return direction, intent_is_frontier

    @staticmethod
    def _normalize(observation: GameState) -> dict[str, Any]:
        return {
            "objects": observation.objects,
            "hud": observation.hud,
            "goal": observation.goal,
            "confidence": observation.confidence,
            "controls": observation.controls,
        }

    @staticmethod
    def _previous_action_index(controls: object, memory: GridMemoryStore) -> int:
        if not isinstance(controls, list):
            return -1
        try:
            return controls.index(memory.working.last_action)
        except ValueError:
            return -1

    @staticmethod
    def _previous_result_features(memory: GridMemoryStore) -> list[float]:
        result = memory.working.last_result
        return [
            float(result in {"moved", "goal_reached"}),
            float(isinstance(result, str) and result.startswith("blocked_")),
            float(result == "completed"),
            float(result in {"caught", "lost"}),
        ]

    def _reward(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        features: tuple[float, ...],
        result: dict[str, Any],
        terminal: str | None = None,
    ) -> float:
        reward = -0.02
        key = tuple(round(value * 20) for value in features)
        reward += 0.25 / math.sqrt(max(self._novel_states[key], 1))
        previous_grid = before.get("hud", {}).get("grid", {})
        current_grid = after.get("hud", {}).get("grid", {})
        previous_visible = {
            tuple(cell)
            for cell in previous_grid.get("visible_cells", [])
            if isinstance(cell, list) and len(cell) == 2
        }
        current_visible = {
            tuple(cell)
            for cell in current_grid.get("visible_cells", [])
            if isinstance(cell, list) and len(cell) == 2
        }
        reward += 0.5 * len(current_visible - previous_visible) / 50.0

        previous_target = previous_grid.get("target")
        current_target = current_grid.get("target")
        previous_position = previous_grid.get("position")
        current_position = current_grid.get("position")
        if (
            isinstance(previous_target, list)
            and previous_target == current_target
            and isinstance(previous_position, list)
            and isinstance(current_position, list)
        ):
            before_distance = abs(previous_position[0] - previous_target[0]) + abs(
                previous_position[1] - previous_target[1]
            )
            after_distance = abs(current_position[0] - current_target[0]) + abs(
                current_position[1] - current_target[1]
            )
            reward += max(-1, min(1, before_distance - after_distance)) * 0.2

        if (
            isinstance(previous_position, list)
            and len(previous_position) == 2
            and isinstance(current_position, list)
            and len(current_position) == 2
        ):
            planned_direction_features = features[
                self.PLANNED_DIRECTION_START : self.PLANNED_DIRECTION_START + 4
            ]
            if any(planned_direction_features):
                planned_index = planned_direction_features.index(1.0)
                planned_direction = (
                    (0, -1),
                    (1, 0),
                    (0, 1),
                    (-1, 0),
                )[planned_index]
                displacement = (
                    current_position[0] - previous_position[0],
                    current_position[1] - previous_position[1],
                )
                if displacement == planned_direction:
                    reward += 0.2
                elif displacement == (0, 0):
                    reward -= 0.15 if str(result.get("status", "")).startswith(
                        "blocked_"
                    ) else 0.05
                else:
                    reward -= 0.1
        outcome = terminal or result.get("status")
        if outcome in {"goal_reached", "won"}:
            reward += 10.0
        elif outcome in {"caught", "lost"}:
            reward -= 5.0
        return reward
