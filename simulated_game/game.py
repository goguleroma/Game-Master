import random
from collections import deque
from typing import TypedDict

from core.types import Action, ActionResult


class Enemy(TypedDict):
    position: tuple[int, int]
    health: int
    alerted: bool


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
        width: int = 30,
        height: int = 30,
        vision_radius: int = 4,
        enemy_vision_radius: int = 4,
        player_max_health: int = 20,
        enemy_max_health: int = 8,
        attack_damage: int = 3,
        num_enemies: int = 5,
        num_static_walls: int = 30,
        target_move_interval: int = 2,
        obstacle_density: float = 0.0,
    ) -> None:
        self.width = width
        self.height = height
        self.vision_radius = vision_radius
        self.enemy_vision_radius = enemy_vision_radius
        self.player_max_health = player_max_health
        self.enemy_max_health = enemy_max_health
        self.attack_damage = attack_damage
        self.num_enemies = num_enemies if pursuer_enabled else 0
        self.num_static_walls = num_static_walls if pursuer_enabled else 0
        self.target_move_interval = target_move_interval
        self.obstacle_density = obstacle_density

        self.position = (0, 1)
        self.obstacles = {(2, 1)}  # The single moving obstacle
        self.static_obstacles: set[tuple[int, int]] = set()
        self.enemies: list[Enemy] = []
        self.player_health = self.player_max_health
        self.caught = False
        self._move_count = 0
        self._moving_target = moving_target
        self._rng = random.Random(seed)

        # Generate static walls first (before placing enemies/target)
        if self.num_static_walls > 0:
            self._generate_static_walls()

        # Initialize enemies
        if self.num_enemies > 0:
            self._initialize_enemies()

        # Place target (ensuring connectivity)
        self.target = self._place_target()

    def _generate_static_walls(self) -> None:
        """Generate static walls ensuring connectivity from player to most of the grid."""
        # We'll generate walls randomly and verify connectivity
        max_attempts = 50
        for _ in range(max_attempts):
            self.static_obstacles.clear()
            # Generate candidate wall positions (excluding player start area)
            candidates = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if (x, y) != self.position
                and (x, y) not in self.obstacles
                # Keep a 5x5 area around player start clear
                and not (0 <= x <= 4 and 0 <= y <= 4)
            ]
            if len(candidates) < self.num_static_walls:
                continue
            walls = self._rng.sample(candidates, self.num_static_walls)
            self.static_obstacles.update(walls)

            # Verify there's a path from player to multiple distant points
            blocked = self.obstacles | self.static_obstacles
            test_targets = [(15, 15), (20, 20), (25, 25), (10, 10), (5, 5), (25, 5), (5, 25)]
            reachable_count = 0
            for test_target in test_targets:
                if test_target not in blocked:
                    path = self._find_path(self.position, test_target, blocked)
                    if path:
                        reachable_count += 1
            
            # Require at least 4 out of 7 test targets to be reachable
            if reachable_count >= 4:
                return  # Connectivity verified

        # Fallback: if we couldn't verify, just use fewer walls
        if len(self.static_obstacles) > 15:
            # Keep only 15 walls to ensure playability
            self.static_obstacles = set(list(self.static_obstacles)[:15])

    def _initialize_enemies(self) -> None:
        """Initialize enemies at random non-overlapping positions."""
        occupied = self.obstacles | self.static_obstacles | {self.position}
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
            # Keep enemies at least distance 4 from player initially
            and (abs(x - self.position[0]) + abs(y - self.position[1])) >= 4
        ]
        if len(candidates) < self.num_enemies:
            # Fallback: reduce enemy count if not enough space
            enemy_positions = self._rng.sample(candidates, len(candidates))
        else:
            enemy_positions = self._rng.sample(candidates, self.num_enemies)

        for pos in enemy_positions:
            self.enemies.append({
                "position": pos,
                "health": self.enemy_max_health,
                "alerted": False,
            })

    def _place_target(self) -> tuple[int, int]:
        """Place target ensuring it's reachable from player position."""
        blocked = self.obstacles | self.static_obstacles | {e["position"] for e in self.enemies}
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) != self.position
            and (x, y) not in blocked
            and abs(x - self.position[0]) + abs(y - self.position[1]) >= 4
        ]
        # Filter to only reachable positions
        reachable = []
        for cand in candidates:
            path = self._find_path(self.position, cand, blocked)
            if path:
                reachable.append(cand)

        if not reachable:
            # Fallback: just pick any valid candidate
            return self._rng.choice(candidates) if candidates else (29, 29)

        return self._rng.choice(reachable)

    @property
    def won(self) -> bool:
        return self.position == self.target

    @property
    def lost(self) -> bool:
        return self.caught

    @property
    def is_game_over(self) -> bool:
        return self.won or self.lost

    # Backward-compatible properties for tests (first enemy)
    @property
    def enemy_position(self) -> tuple[int, int] | None:
        return self.enemies[0]["position"] if self.enemies else None

    @property
    def enemy_health(self) -> int:
        return self.enemies[0]["health"] if self.enemies else 0

    @property
    def enemy_alerted(self) -> bool:
        return self.enemies[0]["alerted"] if self.enemies else False

    @enemy_position.setter
    def enemy_position(self, value: tuple[int, int] | None) -> None:
        if value is None:
            self.enemies.clear()
        elif self.enemies:
            self.enemies[0]["position"] = value
        else:
            self.enemies.append({"position": value, "health": self.enemy_max_health, "alerted": False})

    @enemy_health.setter
    def enemy_health(self, value: int) -> None:
        if self.enemies:
            self.enemies[0]["health"] = value

    @enemy_alerted.setter
    def enemy_alerted(self, value: bool) -> None:
        if self.enemies:
            self.enemies[0]["alerted"] = value

    def observe(self) -> dict[str, object]:
        target_visible = self.is_visible(self.target)
        # Obstacles are only static/moving obstacles, NOT enemies
        obstacle_positions = self.obstacles | self.static_obstacles
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
        # Add visible enemies as entities with unique IDs
        for idx, enemy in enumerate(self.enemies):
            if self.is_visible(enemy["position"]):
                visible_entities.append({
                    "id": f"entity-{idx + 2}",  # entity-2, entity-3, etc.
                    "position": list(enemy["position"]),
                    "health": enemy["health"],
                })
        objects = ["player"]
        if target_visible:
            objects.append("goal")
        if visible_obstacles:
            objects.append("obstacle")

        # Confidence: 1.0 if target visible, 0.5 otherwise
        confidence = 1.0 if target_visible else 0.5

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
            "confidence": confidence,
        }

    def is_visible(self, cell: tuple[int, int]) -> bool:
        dx = abs(cell[0] - self.position[0])
        dy = abs(cell[1] - self.position[1])
        return dx + dy <= self.vision_radius

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
            # wait is a true no-op that doesn't count as a move
            return ActionResult(
                success=True,
                details={
                    "status": "wait",
                    "position": list(self.position),
                    "move_count": self._move_count,
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

        enemy_positions = {e["position"] for e in self.enemies}
        if not (0 <= next_position[0] < self.width and 0 <= next_position[1] < self.height):
            status = "blocked_by_boundary"
        elif (
            next_position in self.obstacles
            or next_position in self.static_obstacles
            or next_position in enemy_positions
        ):
            status = "blocked_by_obstacle"
        else:
            self.position = next_position
            status = "goal_reached" if self.won else "moved"

        # Only increment move_count on successful moves (position actually changed)
        moved = self.position != previous_position
        if moved:
            self._move_count += 1

        if self.won:
            return ActionResult(
                success=moved,
                details={
                    "status": status,
                    "position": list(self.position),
                    "move_count": self._move_count,
                },
            )

        # Enemies get a turn on every player action (even blocked moves)
        # This allows enemies to attack autonomously when adjacent
        self._move_obstacle()
        if self._moving_target and self._move_count % self.target_move_interval == 0:
            self._move_target()
        self._move_enemies()
        return ActionResult(
            success=moved,
            details={
                "status": "caught" if self.caught else status,
                "position": list(self.position),
                "move_count": self._move_count,
            },
        )

    def _execute_combat(self, player_action: str) -> ActionResult:
        # Find adjacent enemy (if any)
        adjacent_enemy = self._get_adjacent_enemy()
        if adjacent_enemy is None:
            return ActionResult(
                success=True,
                details={"status": "completed"},
            )

        enemy_action = self._enemy_combat_choice()

        if player_action == "attack" and enemy_action != "defend":
<<<<<<< Updated upstream
            self.enemy_health -= 3
        if enemy_action == "attack" and player_action != "defend":
            self.player_health -= 3
=======
            adjacent_enemy["health"] -= self.attack_damage
        if enemy_action == "attack" and player_action != "defend":
            self.player_health -= self.attack_damage
>>>>>>> Stashed changes

        enemy_attacked = enemy_action == "attack"
        if self.player_health <= 0:
            self.player_health = 0
            self.caught = True
        if adjacent_enemy["health"] <= 0:
            adjacent_enemy["health"] = 0
            self.static_obstacles.add(adjacent_enemy["position"])
            self.enemies.remove(adjacent_enemy)
        elif not self.caught and enemy_attacked:
            self._retreat_enemy(adjacent_enemy)

        return ActionResult(
            success=True,
            details={
                "status": "caught" if self.caught else "completed",
            },
        )

    def _get_adjacent_enemy(self) -> Enemy | None:
        """Return the first enemy adjacent to the player, or None."""
        for enemy in self.enemies:
            if self._distance(self.position, enemy["position"]) == 1:
                return enemy
        return None

    def _enemy_combat_choice(self) -> str:
        return self._rng.choice(("attack", "defend"))

    def _retreat_enemy(self, enemy: Enemy) -> None:
        x, y = enemy["position"]
        other_enemy_positions = {e["position"] for e in self.enemies if e is not enemy}
        candidates = [
            (x + dx, y + dy)
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
            if 0 <= x + dx < self.width
            and 0 <= y + dy < self.height
            and (x + dx, y + dy) not in self.obstacles
            and (x + dx, y + dy) not in self.static_obstacles
            and (x + dx, y + dy) not in other_enemy_positions
            and (x + dx, y + dy) != self.position
            and (x + dx, y + dy) != self.target
        ]
        if candidates:
            farthest = max(self._distance(candidate, self.position) for candidate in candidates)
            enemy["position"] = self._rng.choice(
                [candidate for candidate in candidates if self._distance(candidate, self.position) == farthest]
            )

    def _move_enemies(self) -> None:
        """Move all alerted enemies toward the player."""
        enemy_positions = {e["position"] for e in self.enemies}
        for enemy in self.enemies:
            # Check if enemy should become alerted (using Manhattan distance)
            dx = abs(self.position[0] - enemy["position"][0])
            dy = abs(self.position[1] - enemy["position"][1])
            if dx + dy <= self.enemy_vision_radius:
                enemy["alerted"] = True
            if not enemy["alerted"]:
                continue

            # Move enemy toward player using BFS
            blocked = self.obstacles | self.static_obstacles | {self.target} | (enemy_positions - {enemy["position"]})
            path = self._find_path(enemy["position"], self.position, blocked)
            if len(path) > 1:  # path[0] is current position, path[1] is next step
                # Check if next step would move onto player (catch)
                next_pos = path[1]
                if next_pos == self.position:
                    self.player_health -= self.attack_damage
                    if self.player_health <= 0:
                        self.player_health = 0
                        self.caught = True
                    # Enemy stays in place after attacking
                else:
                    enemy["position"] = next_pos
                    # Update the set for next enemy's pathfinding
                    enemy_positions = {e["position"] for e in self.enemies}

        # After all enemies move, check if any enemy is now on player (catch)
        if not self.caught:
            for enemy in self.enemies:
                if enemy["position"] == self.position:
                    self.player_health -= self.attack_damage
                    if self.player_health <= 0:
                        self.player_health = 0
                        self.caught = True
                    break

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
        enemy_positions = {e["position"] for e in self.enemies}
        available_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y)
            not in {
                self.position,
                self.target,
                current_obstacle,
                *enemy_positions,
                *self.static_obstacles,
            }
        ]
        if not available_positions:
            # Fallback: keep obstacle in current position
            return
        self.obstacles = {self._rng.choice(available_positions)}

    def _move_target(self) -> None:
        current_target = self.target
        enemy_positions = {e["position"] for e in self.enemies}
        available_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y)
            not in {
                self.position,
                *self.obstacles,
                *enemy_positions,
                current_target,
                *self.static_obstacles,
            }
        ]
        if not available_positions:
            # Fallback: keep target in current position
            return
        self.target = self._rng.choice(available_positions)


class GridGameInputAdapter:
    def __init__(self, game: GridGame) -> None:
        self.game = game

    def execute(self, action: Action) -> ActionResult:
        return self.game.execute(action)
