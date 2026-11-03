"""
Grid 30x30 game environment and input adapter.

This is now a thin wrapper around simulated_game.game.GridGame to avoid duplication.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from core.types import Action, ActionResult
from simulated_game.game import GridGame as SimulatedGridGame


class GridGame:
    """30x30 grid game with moving target and pursuer.
    
    This class wraps simulated_game.game.GridGame with the classic adapter defaults:
    - Single enemy (num_enemies=1)
    - No static walls (num_static_walls=0)
    - Player health: 20, Enemy health: 8, Attack damage: 3
    - Target moves every 2 steps
    - Vision radius: 4
    """

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
        num_enemies: int = 1,
        num_static_walls: int = 0,
        target_move_interval: int = 2,
        obstacle_density: float = 0.0,
    ) -> None:
        # Delegate to the canonical implementation
        self._game = SimulatedGridGame(
            seed=seed,
            moving_target=moving_target,
            pursuer_enabled=pursuer_enabled,
            width=width,
            height=height,
            vision_radius=vision_radius,
            enemy_vision_radius=enemy_vision_radius,
            player_max_health=player_max_health,
            enemy_max_health=enemy_max_health,
            attack_damage=attack_damage,
            num_enemies=num_enemies,
            num_static_walls=num_static_walls,
            target_move_interval=target_move_interval,
            obstacle_density=obstacle_density,
        )

    # Delegate properties for backward compatibility
    @property
    def width(self) -> int:
        return self._game.width

    @property
    def height(self) -> int:
        return self._game.height

    @property
    def vision_radius(self) -> int:
        return self._game.vision_radius

    @vision_radius.setter
    def vision_radius(self, value: int) -> None:
        self._game.vision_radius = value

    @property
    def enemy_vision_radius(self) -> int:
        return self._game.enemy_vision_radius

    @enemy_vision_radius.setter
    def enemy_vision_radius(self, value: int) -> None:
        self._game.enemy_vision_radius = value

    @property
    def player_max_health(self) -> int:
        return self._game.player_max_health

    @player_max_health.setter
    def player_max_health(self, value: int) -> None:
        self._game.player_max_health = value

    @property
    def enemy_max_health(self) -> int:
        return self._game.enemy_max_health

    @enemy_max_health.setter
    def enemy_max_health(self, value: int) -> None:
        self._game.enemy_max_health = value

    @property
    def attack_damage(self) -> int:
        return self._game.attack_damage

    @attack_damage.setter
    def attack_damage(self, value: int) -> None:
        self._game.attack_damage = value

    @property
    def num_enemies(self) -> int:
        return self._game.num_enemies

    @num_enemies.setter
    def num_enemies(self, value: int) -> None:
        self._game.num_enemies = value

    @property
    def num_static_walls(self) -> int:
        return self._game.num_static_walls

    @num_static_walls.setter
    def num_static_walls(self, value: int) -> None:
        self._game.num_static_walls = value

    @property
    def position(self) -> Tuple[int, int]:
        return self._game.position

    @position.setter
    def position(self, value: Tuple[int, int]) -> None:
        self._game.position = value

    @property
    def obstacles(self) -> set[Tuple[int, int]]:
        return self._game.obstacles

    @obstacles.setter
    def obstacles(self, value: set[Tuple[int, int]]) -> None:
        self._game.obstacles = value

    @property
    def static_obstacles(self) -> set[Tuple[int, int]]:
        return self._game.static_obstacles

    @static_obstacles.setter
    def static_obstacles(self, value: set[Tuple[int, int]]) -> None:
        self._game.static_obstacles = value

    @property
    def enemy_position(self) -> Optional[Tuple[int, int]]:
        return self._game.enemy_position

    @enemy_position.setter
    def enemy_position(self, value: Optional[Tuple[int, int]]) -> None:
        self._game.enemy_position = value

    @property
    def enemy_health(self) -> int:
        return self._game.enemy_health

    @enemy_health.setter
    def enemy_health(self, value: int) -> None:
        self._game.enemy_health = value

    @property
    def enemy_alerted(self) -> bool:
        return self._game.enemy_alerted

    @enemy_alerted.setter
    def enemy_alerted(self, value: bool) -> None:
        self._game.enemy_alerted = value

    @property
    def player_health(self) -> int:
        return self._game.player_health

    @player_health.setter
    def player_health(self, value: int) -> None:
        self._game.player_health = value

    @property
    def caught(self) -> bool:
        return self._game.caught

    @caught.setter
    def caught(self, value: bool) -> None:
        self._game.caught = value

    @property
    def target(self) -> Tuple[int, int]:
        return self._game.target

    @target.setter
    def target(self, value: Tuple[int, int]) -> None:
        self._game.target = value

    @property
    def won(self) -> bool:
        return self._game.won

    @won.setter
    def won(self, value: bool) -> None:
        if value:
            # Set position to target to make won=True
            self._game.position = self._game.target
        # Can't make won=False by setting position, as that depends on target

    @property
    def lost(self) -> bool:
        return self._game.lost

    @lost.setter
    def lost(self, value: bool) -> None:
        self._game.lost = value

    @property
    def is_game_over(self) -> bool:
        return self._game.is_game_over

    @property
    def _move_count(self) -> int:
        return self._game._move_count

    @_move_count.setter
    def _move_count(self, value: int) -> None:
        self._game._move_count = value

    @property
    def _moving_target(self) -> bool:
        return self._game._moving_target

    @property
    def _rng(self):
        return self._game._rng

    # Delegate methods
    def observe(self) -> Dict[str, object]:
        return self._game.observe()

    def is_visible(self, cell: Tuple[int, int]) -> bool:
        return self._game.is_visible(cell)

    def execute(self, action: Action) -> ActionResult:
        # Override to use wrapper's _execute_combat for test patching
        if self._game.won or self._game.caught:
            return ActionResult(
                success=False,
                details={"status": "game_over", "position": list(self._game.position)},
            )

        if action.name in self._CONTROL_BINDINGS:
            kind, value = self._CONTROL_BINDINGS[action.name]
            if kind == "move":
                if not isinstance(value, tuple):
                    raise RuntimeError("Movement control is missing its direction.")
                return self._execute_move(value)
            return self._execute_combat(kind)
        if action.name == "wait":
            # Enemies still get a turn on wait
            self._game._move_obstacle()
            if self._game._moving_target and self._game._move_count % self._game.target_move_interval == 0:
                self._game._move_target()
            self._game._move_enemies()
            return ActionResult(
                success=True,
                details={
                    "status": "wait",
                    "position": list(self._game.position),
                    "move_count": self._game._move_count,
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

    def _distance(self, first: Tuple[int, int], second: Tuple[int, int]) -> int:
        return self._game._distance(first, second)

    def _find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        return self._game._find_path(start, goal, blocked)

    def _move_obstacle(self) -> None:
        self._game._move_obstacle()

    def _move_target(self) -> None:
        self._game._move_target()

    def _move_enemy(self) -> None:
        # For backward compatibility, this moves the first enemy
        # The canonical implementation moves all enemies in _move_enemies
        self._game._move_enemies()

    # Expose internal methods for test patching
    # This method can be patched by tests; the inner game's method is not used for combat
    def _enemy_combat_choice(self) -> str:
        return self._rng.choice(("attack", "defend"))

    def _retreat_enemy(self, enemy: dict) -> None:
        self._game._retreat_enemy(enemy)

    def _get_adjacent_enemy(self):
        return self._game._get_adjacent_enemy()

    # Override _execute_combat to use the wrapper's _enemy_combat_choice (for test patching)
    def _execute_combat(self, player_action: str) -> ActionResult:
        # Find the first enemy adjacent to player
        adjacent_enemy = None
        for enemy in self._game.enemies:
            if self._distance(self._game.position, enemy["position"]) == 1:
                adjacent_enemy = enemy
                break
        
        if adjacent_enemy is None:
            return ActionResult(
                success=True,
                details={"status": "completed"},
            )

        # Use the wrapper's _enemy_combat_choice which can be patched
        enemy_action = self._enemy_combat_choice()

        if player_action == "attack" and enemy_action != "defend":
            adjacent_enemy["health"] -= self._game.attack_damage
        if enemy_action == "attack" and player_action != "defend":
            self._game.player_health -= self._game.attack_damage

        enemy_attacked = enemy_action == "attack"
        if self._game.player_health <= 0:
            self._game.player_health = 0
            self._game.caught = True
        if adjacent_enemy["health"] <= 0:
            adjacent_enemy["health"] = 0
            self._game.static_obstacles.add(adjacent_enemy["position"])
            self._game.enemies.remove(adjacent_enemy)
        elif not self._game.caught and enemy_attacked:
            self._retreat_enemy(adjacent_enemy)

        # Sync enemy_health for backward compatibility (first enemy's health)
        self._game.enemy_health = self._game.enemies[0]["health"] if self._game.enemies else 0

        return ActionResult(
            success=True,
            details={
                "status": "caught" if self._game.caught else "completed",
            },
        )

    def _execute_move(self, direction: Tuple[int, int]) -> ActionResult:
        return self._game._execute_move(direction)

    # Expose enemies list for backward compatibility
    @property
    def enemies(self) -> list[dict]:
        return self._game.enemies


class GridGameInputAdapter:
    """ActionExecutor implementation for GridGame."""

    def __init__(self, game: GridGame) -> None:
        self.game = game

    def execute(self, action: Action) -> ActionResult:
        return self.game.execute(action)