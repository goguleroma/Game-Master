from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class WorldEntity:
    kind: str
    position: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    entity_id: Optional[str] = None


@dataclass
class WorldState:
    entities: List[WorldEntity] = field(default_factory=list)
    visible_cells: List[Tuple[int, int]] = field(default_factory=list)
    unknown_cells: List[Tuple[int, int]] = field(default_factory=list)
    frontier_cells: List[Tuple[int, int]] = field(default_factory=list)
    current_position: Optional[Tuple[int, int]] = None
    target_position: Optional[Tuple[int, int]] = None
    obstacle_positions: List[Tuple[int, int]] = field(default_factory=list)
    grid_size: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    goal: Optional[str] = None


class WorldModel:
    def __init__(self) -> None:
        self.state = WorldState()

    def update(
        self,
        observation: Optional[Dict[str, Any]],
        position: Optional[Tuple[int, int]] = None,
    ) -> WorldState:
        if observation is None:
            return WorldState(current_position=position, confidence=0.0)

        hud = observation.get("hud", {})
        grid = observation.get("grid", hud.get("grid", {}))
        visible_cells = self._positions(
            grid.get("visible_cells", observation.get("visible_cells", []))
        )
        unknown_cells = self._positions(observation.get("unknown_cells", []))
        frontier_cells = self._positions(observation.get("frontier_cells", []))
        current_position = self._position(grid.get("position")) or position
        target_position = self._position(grid.get("target"))
        obstacle_positions = self._positions(grid.get("obstacles", []))
        width = grid.get("width")
        height = grid.get("height")
        grid_size = (
            (width, height)
            if isinstance(width, int) and isinstance(height, int)
            else None
        )
        entities = []

        observed_entities = grid.get("entities", observation.get("entities", []))
        if isinstance(observed_entities, list):
            for item in observed_entities:
                if not isinstance(item, dict):
                    continue
                entity_position = self._position(item.get("position"))
                entity_id = item.get("id")
                if entity_position is None or not isinstance(entity_id, str):
                    continue
                entities.append(
                    WorldEntity(
                        kind=item.get("kind", "unknown"),
                        position=entity_position,
                        confidence=item.get("confidence", 0.0),
                        metadata={
                            **item.get("metadata", {}),
                            **({"health": item["health"]} if "health" in item else {}),
                        },
                        entity_id=entity_id,
                    )
                )
        if current_position is not None:
            entities.append(WorldEntity(kind="agent", position=current_position, confidence=1.0))
        if target_position is not None:
            entities.append(WorldEntity(kind="target", position=target_position, confidence=1.0))
        entities.extend(
            WorldEntity(kind="obstacle", position=obstacle, confidence=1.0)
            for obstacle in obstacle_positions
        )

        self.state = WorldState(
            entities=entities,
            visible_cells=visible_cells,
            unknown_cells=unknown_cells,
            frontier_cells=frontier_cells,
            current_position=current_position,
            target_position=target_position,
            obstacle_positions=obstacle_positions,
            grid_size=grid_size,
            confidence=observation.get("confidence", 0.0),
            goal=observation.get("goal"),
        )
        return self.state

    @staticmethod
    def _position(value: Any) -> Optional[Tuple[int, int]]:
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(isinstance(part, int) for part in value)
        ):
            return (value[0], value[1])
        return None

    @classmethod
    def _positions(cls, values: Any) -> List[Tuple[int, int]]:
        if not isinstance(values, list):
            return []
        positions = [cls._position(value) for value in values]
        return [position for position in positions if position is not None]


def bfs_shortest_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: set[Tuple[int, int]],
    bounds: Optional[Tuple[int, int]] = None,
    allowed: Optional[set[Tuple[int, int]]] = None,
    blocked_transitions: Optional[set[tuple[Tuple[int, int], Tuple[int, int]]]] = None,
) -> List[Tuple[int, int]]:
    queue = deque([start])
    came_from: Dict[Tuple[int, int], Tuple[int, int] | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0] + dx, current[1] + dy)
            if bounds is not None and not (
                0 <= nxt[0] < bounds[0] and 0 <= nxt[1] < bounds[1]
            ):
                continue
            if allowed is not None and nxt not in allowed:
                continue
            if nxt in blocked:
                continue
            if blocked_transitions is not None and (current, nxt) in blocked_transitions:
                continue
            if nxt in came_from:
                continue
            came_from[nxt] = current
            queue.append(nxt)

    if goal not in came_from:
        return []

    path: List[Tuple[int, int]] = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path
