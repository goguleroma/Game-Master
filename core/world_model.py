from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class WorldEntity:
    kind: str
    position: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    entities: List[WorldEntity] = field(default_factory=list)
    visible_cells: List[Tuple[int, int]] = field(default_factory=list)
    unknown_cells: List[Tuple[int, int]] = field(default_factory=list)
    frontier_cells: List[Tuple[int, int]] = field(default_factory=list)
    current_position: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    goal: Optional[str] = None


class WorldModel:
    def __init__(self) -> None:
        self.state = WorldState()

    def update(self, observation: Optional[Dict[str, Any]], position: Optional[Tuple[int, int]]) -> WorldState:
        if observation is None:
            return WorldState(current_position=position, confidence=0.0)

        visible_cells = observation.get("visible_cells", [])
        unknown_cells = observation.get("unknown_cells", [])
        frontier_cells = observation.get("frontier_cells", [])
        entities = []

        for item in observation.get("entities", []):
            entities.append(
                WorldEntity(
                    kind=item.get("kind", "unknown"),
                    position=item.get("position"),
                    confidence=item.get("confidence", 0.0),
                    metadata=item.get("metadata", {}),
                )
            )

        self.state = WorldState(
            entities=entities,
            visible_cells=visible_cells,
            unknown_cells=unknown_cells,
            frontier_cells=frontier_cells,
            current_position=position,
            confidence=observation.get("confidence", 0.0),
            goal=observation.get("goal"),
        )
        return self.state


def bfs_shortest_path(start: Tuple[int, int], goal: Tuple[int, int], blocked: set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    queue = deque([start])
    came_from: Dict[Tuple[int, int], Tuple[int, int] | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in blocked:
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
