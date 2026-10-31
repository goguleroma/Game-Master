import random
from collections import deque

from core.types import Action, ActionResult


class GridGame:
    _CONTROL_BINDINGS = {
        "control_1": ("move", (0, -1)),
        "control_2": ("move", (1, 0)),
        "control_3": ("move", (0, 1)),
        "control_4": ("move", (-1, 0)),
        "control_5": ("attack", None),
        "control_6": ("defend", None),
    }

    def __init__(
        self,
        seed: int | None = None,
        moving_target: bool = True,
        pursuer_enabled: bool = True,
    ) -> None:
        self.width = 30
        self.height = 30
        self.vision_radius = 4
        self.enemy_vision_radius = 4
        self.position = (0, 1)
        self.obstacles = {(2, 1)}
        self.static_obstacles: set[tuple[int, int]] = set()
        self.enemy_position: tuple[int, int] | None = (7, 1) if pursuer_enabled else None
        self.player_health = 10
        self.enemy_health = 10
        self.enemy_alerted = False
        self.caught = False
        self._move_count = 0
        self._moving_target = moving_target
        self._rng = random.Random(seed)
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) != self.position
            and (x, y) not in self.obstacles
            and (x, y) != self.enemy_position
            and abs(x - self.position[0]) + abs(y - self.position[1]) >= 4
        ]
        self.target = self._rng.choice(candidates)

    @property
    def won(self) -> bool:
        return self.position == self.target

    def observe(self) -> dict[str, object]:
        target_visible = self.is_visible(self.target)
        obstacle_positions = self.obstacles | self.static_obstacles | (
            {self.enemy_position} if self.enemy_position is not None else set()
        )
        visible_obstacles = sorted(
            obstacle
            for obstacle in obstacle_positions
            if self.is_visible(obstacle)
        )
        visible_entities = [
            {"id": "entity-1", "position": list(obstacle)}
            for obstacle in sorted(self.obstacles)
            if self.is_visible(obstacle)
        ]
        if self.enemy_position is not None and self.is_visible(self.enemy_position):
            visible_entities.append({
                "id": "entity-2",
                "position": list(self.enemy_position),
                "health": self.enemy_health,
            })
        objects = ["player"]
        if target_visible:
            objects.append("goal")
        if visible_obstacles:
            objects.append("obstacle")

        return {
            "objects": objects,
            "controls": list(self._CONTROL_BINDINGS),
            "hud": {
                "status": (
                    "won" if self.won else "lost" if self.caught else "playing"
                ),
                "health": self.player_health,
                "grid": {
                    "width": self.width,
                    "height": self.height,
                    "position": list(self.position),
                    "vision_radius": self.vision_radius,
                    "visible_cells": [
                        [x, y]
                        for y in range(self.height)
                        for x in range(self.width)
                        if self.is_visible((x, y))
                    ],
                    "target": list(self.target) if target_visible else None,
                    "obstacles": [list(point) for point in visible_obstacles],
                    "entities": visible_entities,
                },
            },
            "goal": "reach_goal",
            "confidence": 1.0,
        }

    def is_visible(self, cell: tuple[int, int]) -> bool:
        dx = cell[0] - self.position[0]
        dy = cell[1] - self.position[1]
        return dx * dx + dy * dy <= self.vision_radius * self.vision_radius

    def execute(self, action: Action) -> ActionResult:
        if self.won or self.caught:
            return ActionResult(
                success=False,
                details={"status": "game_over", "position": list(self.position)},
            )

        if action.name in self._CONTROL_BINDINGS:
            kind, value = self._CONTROL_BINDINGS[action.name]
            if kind == "move":
                if not isinstance(value, tuple):
                    raise RuntimeError("Movement control is missing its direction.")
                return self._execute_move(value)
            return self._execute_combat(kind)
        if action.name == "wait":
            return ActionResult(
                success=False,
                details={
                    "status": "waiting_for_visible_target",
                    "position": list(self.position),
                },
            )

        directions = {
            "move_up": (0, -1),
            "move_right": (1, 0),
            "move_down": (0, 1),
            "move_left": (-1, 0),
        }
        direction = directions.get(action.name)
        if direction is None:
            return ActionResult(
                success=False,
                details={"status": "unsupported_action", "action": action.name},
            )

        return self._execute_move(direction)

    def _execute_move(self, direction: tuple[int, int]) -> ActionResult:
        previous_position = self.position
        next_position = (
            self.position[0] + direction[0],
            self.position[1] + direction[1],
        )

        if not (0 <= next_position[0] < self.width and 0 <= next_position[1] < self.height):
            status = "blocked_by_boundary"
        elif (
            next_position in self.obstacles
            or next_position in self.static_obstacles
            or next_position == self.enemy_position
        ):
            status = "blocked_by_obstacle"
        else:
            self.position = next_position
            status = "goal_reached" if self.won else "moved"

        self._move_count += 1
        if self.won:
            return ActionResult(
                success=self.position != previous_position,
                details={
                    "status": status,
                    "position": list(self.position),
                    "move_count": self._move_count,
                },
            )

        self._move_obstacle()
        if self._moving_target and self._move_count % 2 == 0:
            self._move_target()
        self._move_enemy()
        return ActionResult(
            success=self.position != previous_position,
            details={
                "status": "caught" if self.caught else status,
                "position": list(self.position),
                "move_count": self._move_count,
            },
        )

    def _execute_combat(self, player_action: str) -> ActionResult:
        if self.enemy_position is None or self._distance(self.position, self.enemy_position) != 1:
            return ActionResult(
                success=True,
                details={
                    "status": "completed",
                },
            )

        enemy_action = self._enemy_combat_choice()

        if player_action == "attack" and enemy_action != "defend":
            self.enemy_health -= 5
        if enemy_action == "attack" and player_action != "defend":
            self.player_health -= 5

        enemy_attacked = enemy_action == "attack"
        if self.player_health <= 0:
            self.player_health = 0
            self.caught = True
        if self.enemy_health <= 0:
            self.enemy_health = 0
            self.static_obstacles.add(self.enemy_position)
            self.enemy_position = None
            self.enemy_alerted = False
        elif not self.caught and enemy_attacked:
            self._retreat_enemy()

        return ActionResult(
            success=True,
            details={
                "status": "caught" if self.caught else "completed",
            },
        )

    def _enemy_combat_choice(self) -> str:
        return self._rng.choice(("attack", "defend"))

    def _retreat_enemy(self) -> None:
        if self.enemy_position is None:
            return
        x, y = self.enemy_position
        candidates = [
            (x + dx, y + dy)
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
            if 0 <= x + dx < self.width
            and 0 <= y + dy < self.height
            and (x + dx, y + dy) not in self.obstacles
            and (x + dx, y + dy) not in self.static_obstacles
            and (x + dx, y + dy) != self.position
            and (x + dx, y + dy) != self.target
        ]
        if candidates:
            farthest = max(self._distance(candidate, self.position) for candidate in candidates)
            self.enemy_position = self._rng.choice(
                [candidate for candidate in candidates if self._distance(candidate, self.position) == farthest]
            )

    def _move_enemy(self) -> None:
        if self.enemy_position is None:
            return
        dx = self.position[0] - self.enemy_position[0]
        dy = self.position[1] - self.enemy_position[1]
        if dx * dx + dy * dy <= self.enemy_vision_radius * self.enemy_vision_radius:
            self.enemy_alerted = True
        if not self.enemy_alerted:
            return

        blocked = self.obstacles | self.static_obstacles | {self.target}
        path = self._find_path(self.enemy_position, self.position, blocked)
        if len(path) > 2:
            self.enemy_position = path[1]

    @staticmethod
    def _distance(first: tuple[int, int], second: tuple[int, int]) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        queue = deque([start])
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                next_position = (current[0] + dx, current[1] + dy)
                if (
                    not (0 <= next_position[0] < self.width and 0 <= next_position[1] < self.height)
                    or next_position in blocked
                    or next_position in came_from
                ):
                    continue
                came_from[next_position] = current
                queue.append(next_position)

        if goal not in came_from:
            return []
        path = []
        current: tuple[int, int] | None = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        return list(reversed(path))

    def _move_obstacle(self) -> None:
        current_obstacle = next(iter(self.obstacles))
        available_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y)
            not in {
                self.position,
                self.target,
                current_obstacle,
                self.enemy_position,
                *self.static_obstacles,
            }
        ]
        if not available_positions:
            raise RuntimeError("No free position available for the moving obstacle.")
        self.obstacles = {self._rng.choice(available_positions)}

    def _move_target(self) -> None:
        current_target = self.target
        available_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y)
            not in {
                self.position,
                *self.obstacles,
                self.enemy_position,
                current_target,
                *self.static_obstacles,
            }
        ]
        if not available_positions:
            raise RuntimeError("No free position available for the moving target.")
        self.target = self._rng.choice(available_positions)


class GridGameInputAdapter:
    def __init__(self, game: GridGame) -> None:
        self.game = game

    def execute(self, action: Action) -> ActionResult:
        return self.game.execute(action)
