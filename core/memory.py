from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Position = Tuple[int, int]


@dataclass
class CellMemory:
    seen: bool = False
    visited: bool = False
    obstacle: Optional[bool] = None
    value: Optional[str] = None
    last_seen_turn: int = -1


@dataclass
class WorkingMemory:
    current_observation: Optional[Dict[str, Any]] = None
    last_action: Optional[str] = None
    last_result: Optional[str] = None
    last_position: Optional[Position] = None
    visible_goal: Optional[Position] = None


@dataclass
class SpatialMemory:
    grid: Dict[Position, CellMemory] = field(default_factory=dict)
    frontier: List[Position] = field(default_factory=list)
    known_goal_positions: List[Position] = field(default_factory=list)


@dataclass
class KnowledgeItem:
    key: str
    value: Any
    confidence: float
    evidence_count: int
    context: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    def __init__(self) -> None:
        self.working = WorkingMemory()
        self.spatial = SpatialMemory()
        self.knowledge: List[KnowledgeItem] = []
        self.experiences: List[Dict[str, Any]] = []

    def update_working_memory(
        self,
        observation: Optional[Dict[str, Any]],
        action: Optional[str],
        result: Optional[str],
        position: Optional[Position],
    ) -> None:
        self.working.current_observation = observation
        self.working.last_action = action
        self.working.last_result = result
        self.working.last_position = position

    def mark_cell(
        self,
        pos: Position,
        seen: bool = True,
        visited: bool = False,
        obstacle: Optional[bool] = None,
        value: Optional[str] = None,
        turn: int = 0,
    ) -> None:
        cell = self.spatial.grid.setdefault(pos, CellMemory())
        cell.seen = seen
        cell.visited = visited or cell.visited
        if obstacle is not None:
            cell.obstacle = obstacle
        if value is not None:
            cell.value = value
        cell.last_seen_turn = turn

    def update_frontier(self) -> None:
        frontier: List[Position] = []
        for pos, cell in self.spatial.grid.items():
            if cell.seen and not cell.visited:
                frontier.append(pos)
        self.spatial.frontier = frontier

    def add_knowledge(
        self,
        key: str,
        value: Any,
        confidence: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.knowledge.append(
            KnowledgeItem(
                key=key,
                value=value,
                confidence=confidence,
                evidence_count=1,
                context=context or {},
            )
        )

    def record_experience(self, event: Dict[str, Any]) -> None:
        self.experiences.append(event)

    def record(self, entry: Dict[str, Any]) -> None:
        self.record_experience(entry)
