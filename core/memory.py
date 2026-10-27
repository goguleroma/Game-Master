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
    visit_count: int = 0


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


@dataclass
class EntityMemory:
    position: Position
    category: str = "unclassified"
    confidence: float = 0.0
    behavior_hypothesis: Optional[str] = None
    behavior_confidence: float = 0.0
    observation_count: int = 1
    movement_evidence: int = 0
    approach_evidence: int = 0
    retreat_evidence: int = 0
    stationary_evidence: int = 0
    approach_streak: int = 0
    retreat_streak: int = 0
    last_seen_turn: int = -1
    last_agent_position: Optional[Position] = None


class MemoryStore:
    def __init__(self) -> None:
        self.working = WorkingMemory()
        self.spatial = SpatialMemory()
        self.knowledge: List[KnowledgeItem] = []
        self.experiences: List[Dict[str, Any]] = []
        self.entities: Dict[str, EntityMemory] = {}

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self.experiences

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
        if visited:
            cell.visit_count += 1

    def update_frontier(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        frontier: List[Position] = []
        for pos, cell in self.spatial.grid.items():
            if not cell.seen or cell.obstacle is True:
                continue
            if width is None or height is None:
                is_frontier = not cell.visited
            else:
                x, y = pos
                is_frontier = any(
                    0 <= x + dx < width
                    and 0 <= y + dy < height
                    and (x + dx, y + dy) not in self.spatial.grid
                    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
                )
            if is_frontier:
                frontier.append(pos)
        if width is not None and height is not None and not any(
            (x, y) not in self.spatial.grid
            for y in range(height)
            for x in range(width)
        ):
            frontier = [
                pos
                for pos, cell in self.spatial.grid.items()
                if cell.seen and cell.obstacle is not True
            ]
        self.spatial.frontier = sorted(frontier)

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

    def observe_entity(
        self,
        entity_id: str,
        position: Position,
        agent_position: Optional[Position],
        turn: int,
    ) -> EntityMemory:
        entity = self.entities.get(entity_id)
        if entity is None:
            entity = EntityMemory(
                position=position,
                last_seen_turn=turn,
                last_agent_position=agent_position,
            )
            self.entities[entity_id] = entity
            return entity

        previous_position = entity.position
        previous_agent_position = entity.last_agent_position
        entity.observation_count += 1
        entity.last_seen_turn = turn
        entity.behavior_hypothesis = None
        entity.behavior_confidence = 0.0

        if position == previous_position:
            entity.stationary_evidence += 1
            entity.approach_streak = 0
            entity.retreat_streak = 0
            if entity.stationary_evidence >= 3:
                entity.category = "stationary"
                entity.confidence = entity.stationary_evidence / max(
                    entity.observation_count - 1, 1
                )
        else:
            entity.movement_evidence += 1
            entity.category = "mobile"
            entity.confidence = entity.movement_evidence / max(
                entity.observation_count - 1, 1
            )
            if previous_agent_position is not None:
                previous_distance = self._manhattan_distance(
                    previous_position, previous_agent_position
                )
                current_distance = self._manhattan_distance(
                    position, previous_agent_position
                )
                if current_distance < previous_distance:
                    entity.approach_evidence += 1
                    entity.approach_streak += 1
                    entity.retreat_streak = 0
                elif current_distance > previous_distance:
                    entity.retreat_evidence += 1
                    entity.retreat_streak += 1
                    entity.approach_streak = 0
                else:
                    entity.approach_streak = 0
                    entity.retreat_streak = 0

            if entity.approach_streak >= 3:
                entity.behavior_hypothesis = "approaches_agent"
                entity.behavior_confidence = entity.approach_streak / (
                    entity.approach_streak + 2
                )
            elif entity.retreat_streak >= 3:
                entity.behavior_hypothesis = "moves_away_from_agent"
                entity.behavior_confidence = entity.retreat_streak / (
                    entity.retreat_streak + 2
                )

        entity.position = position
        entity.last_agent_position = agent_position
        return entity

    @staticmethod
    def _manhattan_distance(first: Position, second: Position) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def record_experience(self, event: Dict[str, Any]) -> None:
        self.experiences.append(event)

    def record(self, entry: Dict[str, Any]) -> None:
        self.record_experience(entry)
