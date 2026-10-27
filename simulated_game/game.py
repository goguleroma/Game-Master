import random

from control.input_adapter import InputAdapter
from core.types import Action, ActionResult


class GridGame:
    def __init__(self, seed: int | None = None) -> None:
        self.width = 10
        self.height = 10
        self.vision_radius = 4
        self.position = (0, 1)
        self.obstacles = {(2, 1)}
        self._move_count = 0
        self._rng = random.Random(seed)
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) != self.position
            and (x, y) not in self.obstacles
            and abs(x - self.position[0]) + abs(y - self.position[1]) >= 4
        ]
        self.target = self._rng.choice(candidates)

    @property
    def won(self) -> bool:
        return self.position == self.target

    def observe(self) -> dict[str, object]:
        target_visible = self.is_visible(self.target)
        visible_obstacles = sorted(
            obstacle for obstacle in self.obstacles if self.is_visible(obstacle)
        )
        objects = ["player"]
        if target_visible:
            objects.append("goal")
        if visible_obstacles:
            objects.append("obstacle")

        return {
            "objects": objects,
            "hud": {
                "status": "won" if self.won else "playing",
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

        previous_position = self.position
        next_position = (
            self.position[0] + direction[0],
            self.position[1] + direction[1],
        )

        if not (0 <= next_position[0] < self.width and 0 <= next_position[1] < self.height):
            status = "blocked_by_boundary"
        elif next_position in self.obstacles:
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
                    "obstacle": list(next(iter(self.obstacles))),
                    "target": list(self.target),
                    "move_count": self._move_count,
                },
            )

        self._move_obstacle()
        if self._move_count % 2 == 0:
            self._move_target()
        return ActionResult(
            success=self.position != previous_position,
            details={
                "status": status,
                "position": list(self.position),
                "obstacle": list(next(iter(self.obstacles))),
                "target": list(self.target),
                "move_count": self._move_count,
            },
        )

    def _move_obstacle(self) -> None:
        current_obstacle = next(iter(self.obstacles))
        available_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in {self.position, self.target, current_obstacle}
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
            if (x, y) not in {self.position, *self.obstacles, current_target}
        ]
        if not available_positions:
            raise RuntimeError("No free position available for the moving target.")
        self.target = self._rng.choice(available_positions)


class GridGameInputAdapter(InputAdapter):
    def __init__(self, game: GridGame) -> None:
        self.game = game

    def execute(self, action: Action) -> ActionResult:
        return self.game.execute(action)
