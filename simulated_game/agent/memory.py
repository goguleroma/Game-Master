from dataclasses import dataclass, field
from collections import deque
import json
import math
import sqlite3
import time
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Position = Tuple[int, int]
DECISION_CASE_LIMIT = 2000
DECISION_RETRIEVAL_WINDOW = 256
_DIRECTIONS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


class ShortTermMemory:
    def __init__(
        self,
        max_frames: int = 60,
        max_age_seconds: float = 12.0,
    ) -> None:
        self.max_frames = max_frames
        self.max_age_seconds = max_age_seconds
        self.frames: deque[tuple[float, object]] = deque(maxlen=max_frames)

    def add(self, frame: object, timestamp: Optional[float] = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        self.frames.append((now, frame))
        while self.frames and now - self.frames[0][0] > self.max_age_seconds:
            self.frames.popleft()


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
    current_observation: Optional[object] = None
    last_action: Optional[str] = None
    last_result: Optional[str] = None


@dataclass
class SpatialMemory:
    grid: Dict[Position, CellMemory] = field(default_factory=dict)
    frontier: List[Position] = field(default_factory=list)
    known_goal_positions: List[Position] = field(default_factory=list)
    last_observed_goal_position: Optional[Position] = None
    goal_change_count: int = 0
    goal_is_dynamic: bool = False


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


@dataclass(frozen=True)
class DecisionCase:
    context: Dict[str, Any]
    direction: Position
    reward: float
    outcome: str


class GridMemoryStore:
    def __init__(
        self,
        game_id: Optional[str] = None,
        database_path: Optional[str | Path] = None,
    ) -> None:
        self.short_term = ShortTermMemory()
        self.working = WorkingMemory()
        self.spatial = SpatialMemory()
        self.knowledge: List[KnowledgeItem] = []
        self.general_knowledge: List[KnowledgeItem] = []
        self.experiences: List[Dict[str, Any]] = []
        self.entities: Dict[str, EntityMemory] = {}
        self.decision_cases: deque[DecisionCase] = deque(
            maxlen=DECISION_CASE_LIMIT
        )
        self._decision_cases_loaded = False
        self._decision_case_writes = 0
        self.retrieval_metrics = {
            "queries": 0,
            "queries_with_matches": 0,
            "matched_cases": 0,
            "guided_decisions": 0,
        }
        self.last_retrieval: Dict[str, Any] = {
            "matched_cases": 0,
            "preferences": {},
        }
        self.game_id = game_id
        self.database_path = Path(database_path) if database_path is not None else None
        if self.database_path is not None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS game_control_trials (
                        game_id TEXT NOT NULL,
                        context TEXT NOT NULL,
                        control TEXT NOT NULL,
                        effects TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS game_control_outcomes (
                        game_id TEXT NOT NULL,
                        context TEXT NOT NULL,
                        control TEXT NOT NULL,
                        reward REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS game_decision_cases (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        context TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        reward REAL NOT NULL,
                        outcome TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS game_decision_cases_by_game
                    ON game_decision_cases (game_id, id)
                    """
                )
                connection.commit()

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self.experiences

    def observe_frame(self, frame: object) -> None:
        self.short_term.add(frame)

    def load_control_trials(
        self,
    ) -> dict[tuple[int, ...], dict[str, list[dict[str, float]]]]:
        if self.database_path is None or self.game_id is None:
            return {}
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT context, control, effects
                FROM game_control_trials
                WHERE game_id = ?
                """,
                (self.game_id,),
            ).fetchall()
        trials: dict[tuple[int, ...], dict[str, list[dict[str, float]]]] = {}
        for context_json, control, effects_json in rows:
            context = tuple(json.loads(context_json))
            effects = json.loads(effects_json)
            trials.setdefault(context, {}).setdefault(control, []).append(effects)
        return trials

    def save_control_trial(
        self,
        context: tuple[int, ...],
        control: str,
        effects: dict[str, float],
    ) -> None:
        if self.database_path is None or self.game_id is None:
            return
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO game_control_trials (game_id, context, control, effects)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self.game_id,
                    json.dumps(context),
                    control,
                    json.dumps(effects, sort_keys=True),
                ),
            )
            connection.commit()

    def load_control_outcomes(
        self,
    ) -> dict[tuple[int, ...], dict[str, list[float]]]:
        if self.database_path is None or self.game_id is None:
            return {}
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT context, control, reward
                FROM game_control_outcomes
                WHERE game_id = ?
                """,
                (self.game_id,),
            ).fetchall()
        outcomes: dict[tuple[int, ...], dict[str, list[float]]] = {}
        for context_json, control, reward in rows:
            context = tuple(json.loads(context_json))
            outcomes.setdefault(context, {}).setdefault(control, []).append(reward)
        return outcomes

    def save_control_outcome(
        self,
        context: tuple[int, ...],
        control: str,
        reward: float,
    ) -> None:
        if self.database_path is None or self.game_id is None:
            return
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO game_control_outcomes (game_id, context, control, reward)
                VALUES (?, ?, ?, ?)
                """,
                (self.game_id, json.dumps(context), control, reward),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        if self.database_path is None:
            raise RuntimeError("Persistent memory has no database path")
        return sqlite3.connect(self.database_path)

    def build_decision_context(
        self,
        state: object,
        observation: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a coordinate-free context for retrieving prior grid decisions."""
        if observation is None:
            current = self.working.current_observation
            observation = current if isinstance(current, dict) else {}
        hud = observation.get("hud", {})
        if not isinstance(hud, dict):
            hud = {}
        grid = hud.get("grid", {})
        if not isinstance(grid, dict):
            grid = {}

        position = self._position(getattr(state, "current_position", None))
        visible_target = self._position(getattr(state, "target_position", None))
        remembered_target = (
            self.spatial.known_goal_positions[-1]
            if self.spatial.known_goal_positions
            else None
        )
        target = visible_target or remembered_target
        target_direction = None
        if position is not None and target is not None:
            dx = target[0] - position[0]
            dy = target[1] - position[1]
            target_direction = [
                0 if dx == 0 else (1 if dx > 0 else -1),
                0 if dy == 0 else (1 if dy > 0 else -1),
            ]

        known_obstacles = set(
            self._positions(getattr(state, "obstacle_positions", []))
        )
        adjacent_obstacles: list[Optional[bool]] = []
        for dx, dy in _DIRECTIONS:
            cell_position = (
                (position[0] + dx, position[1] + dy)
                if position is not None
                else None
            )
            if cell_position is None:
                adjacent_obstacles.append(None)
                continue
            if cell_position in known_obstacles:
                adjacent_obstacles.append(True)
                continue
            cell = self.spatial.grid.get(cell_position)
            adjacent_obstacles.append(
                cell.obstacle if cell is not None else None
            )

        nearby_entities = [
            self._manhattan_distance(position, entity.position)
            for entity in getattr(state, "entities", [])
            if position is not None
            and getattr(entity, "entity_id", None) is not None
            and self._position(getattr(entity, "position", None)) is not None
        ]
        nearest_entity_distance = (
            min(nearby_entities) if nearby_entities else None
        )
        entity_distance_bucket = (
            None
            if nearest_entity_distance is None
            else 0
            if nearest_entity_distance <= 1
            else 1
            if nearest_entity_distance <= 2
            else 2
            if nearest_entity_distance <= 4
            else 3
        )

        health = hud.get("health")
        health_bucket = (
            None
            if not isinstance(health, (int, float))
            else 0
            if health <= 3
            else 1
            if health <= 6
            else 2
        )
        frontier_count = len(self.spatial.frontier)
        return {
            "goal_visible": visible_target is not None,
            "goal_known": target is not None,
            "target_direction": target_direction,
            "goal_dynamic": self.spatial.goal_is_dynamic,
            "adjacent_obstacles": adjacent_obstacles,
            "entity_distance_bucket": entity_distance_bucket,
            "health_bucket": health_bucket,
            "frontier_bucket": min(frontier_count, 2),
        }

    def record_decision_outcome(
        self,
        context: Dict[str, Any],
        planned_direction: Position,
        before_position: Position,
        target_position: Optional[Position],
        details: Dict[str, Any],
    ) -> None:
        """Store a bounded, scored case after observing the action result."""
        status = details.get("status")
        if not isinstance(status, str):
            return
        after_position = self._position(details.get("position"))
        outcome = status
        if status in {"goal_reached", "won"}:
            reward = 1.0
        elif status in {"caught", "lost"}:
            reward = -1.0
        elif status.startswith("blocked_"):
            reward = -0.75
        elif after_position is None:
            reward = 0.0
        else:
            actual_direction = (
                after_position[0] - before_position[0],
                after_position[1] - before_position[1],
            )
            if actual_direction == (0, 0):
                reward = -0.1
            elif target_position is not None:
                before_distance = self._manhattan_distance(
                    before_position, target_position
                )
                after_distance = self._manhattan_distance(
                    after_position, target_position
                )
                reward = max(-0.5, min(0.5, (before_distance - after_distance) * 0.5))
            else:
                alignment = (
                    actual_direction[0] * planned_direction[0]
                    + actual_direction[1] * planned_direction[1]
                )
                reward = 0.4 if alignment > 0 else -0.4 if alignment < 0 else 0.0

        self.record_decision_case(context, planned_direction, reward, outcome)

    def record_decision_case(
        self,
        context: Dict[str, Any],
        direction: Position,
        reward: float,
        outcome: str,
    ) -> None:
        if direction not in _DIRECTIONS:
            raise ValueError("Decision cases require a cardinal planned direction")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
            or not -1.0 <= reward <= 1.0
        ):
            raise ValueError("Decision case reward must be between -1 and 1")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError("Decision case outcome must be a non-empty string")
        if not isinstance(context, dict):
            raise ValueError("Decision case context must be a mapping")
        if not self._valid_decision_context(context):
            raise ValueError("Decision case context does not match the grid schema")

        context_json = json.dumps(context, sort_keys=True, allow_nan=False)
        normalized_context = json.loads(context_json)
        self._ensure_decision_cases_loaded()
        case = DecisionCase(
            context=normalized_context,
            direction=direction,
            reward=float(reward),
            outcome=outcome,
        )
        self.decision_cases.append(case)
        if self.database_path is not None and self.game_id is not None:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO game_decision_cases
                        (game_id, schema_version, context, direction, reward, outcome)
                    VALUES (?, 1, ?, ?, ?, ?)
                    """,
                    (
                        self.game_id,
                        context_json,
                        json.dumps(direction),
                        float(reward),
                        outcome,
                    ),
                )
                self._decision_case_writes += 1
                if self._decision_case_writes % 250 == 0:
                    connection.execute(
                        """
                        DELETE FROM game_decision_cases
                        WHERE game_id = ? AND id NOT IN (
                            SELECT id FROM game_decision_cases
                            WHERE game_id = ? ORDER BY id DESC LIMIT ?
                        )
                        """,
                        (self.game_id, self.game_id, DECISION_CASE_LIMIT),
                    )
                connection.commit()

    def retrieve_direction_preferences(
        self,
        context: Dict[str, Any],
        minimum_similarity: float = 0.65,
        minimum_support: int = 2,
    ) -> Dict[Position, float]:
        """Retrieve similar past cases and return confidence-weighted rewards."""
        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between 0 and 1")
        if minimum_support < 1:
            raise ValueError("minimum_support must be positive")
        if not isinstance(context, dict) or not self._valid_decision_context(context):
            raise ValueError("Retrieval context does not match the grid schema")
        self._ensure_decision_cases_loaded()
        self.retrieval_metrics["queries"] += 1

        matches = [
            (case, similarity)
            for case in islice(reversed(self.decision_cases), DECISION_RETRIEVAL_WINDOW)
            if (similarity := self._context_similarity(context, case.context))
            >= minimum_similarity
        ]
        if matches:
            self.retrieval_metrics["queries_with_matches"] += 1
            self.retrieval_metrics["matched_cases"] += len(matches)

        grouped: Dict[Position, List[Tuple[DecisionCase, float]]] = {}
        for case, similarity in matches:
            grouped.setdefault(case.direction, []).append((case, similarity))

        preferences: Dict[Position, float] = {}
        details: Dict[str, Dict[str, float | int]] = {}
        for direction, cases in grouped.items():
            cases = sorted(cases, key=lambda item: item[1], reverse=True)[:25]
            if len(cases) < minimum_support:
                continue
            total_weight = sum(similarity for _, similarity in cases)
            if total_weight <= 0:
                continue
            mean_reward = sum(
                case.reward * similarity for case, similarity in cases
            ) / total_weight
            mean_similarity = total_weight / len(cases)
            confidence = min(1.0, len(cases) / 4.0) * mean_similarity
            score = mean_reward * confidence
            preferences[direction] = score
            details[f"{direction[0]},{direction[1]}"] = {
                "support": len(cases),
                "mean_reward": round(mean_reward, 4),
                "mean_similarity": round(mean_similarity, 4),
                "confidence": round(confidence, 4),
                "score": round(score, 4),
            }

        self.last_retrieval = {
            "matched_cases": len(matches),
            "preferences": details,
        }
        return preferences

    def record_guided_decision(self) -> None:
        self.retrieval_metrics["guided_decisions"] += 1

    def retrieval_summary(self) -> Dict[str, int]:
        return {
            **self.retrieval_metrics,
            "available_cases": len(self.decision_cases),
        }

    def _ensure_decision_cases_loaded(self) -> None:
        if self._decision_cases_loaded:
            return
        self._decision_cases_loaded = True
        if self.database_path is None or self.game_id is None:
            return
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT context, direction, reward, outcome
                FROM game_decision_cases
                WHERE game_id = ? AND schema_version = 1
                ORDER BY id DESC LIMIT ?
                """,
                (self.game_id, DECISION_CASE_LIMIT),
            ).fetchall()
            row_count = connection.execute(
                "SELECT COUNT(*) FROM game_decision_cases WHERE game_id = ?",
                (self.game_id,),
            ).fetchone()[0]
        self._decision_case_writes = int(row_count)
        for context_json, direction_json, reward, outcome in reversed(rows):
            try:
                context = json.loads(context_json)
                direction = tuple(json.loads(direction_json))
                if (
                    isinstance(context, dict)
                    and self._valid_decision_context(context)
                    and direction in _DIRECTIONS
                    and isinstance(reward, (int, float))
                    and -1.0 <= reward <= 1.0
                    and isinstance(outcome, str)
                ):
                    self.decision_cases.append(
                        DecisionCase(context, direction, float(reward), outcome)
                    )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

    @classmethod
    def _context_similarity(
        cls,
        first: Dict[str, Any],
        second: Dict[str, Any],
    ) -> float:
        weights = {
            "goal_visible": 0.08,
            "goal_known": 0.06,
            "target_direction": 0.25,
            "goal_dynamic": 0.10,
            "adjacent_obstacles": 0.25,
            "entity_distance_bucket": 0.14,
            "health_bucket": 0.07,
            "frontier_bucket": 0.05,
        }
        similarities = {
            "goal_visible": float(first.get("goal_visible") == second.get("goal_visible")),
            "goal_known": float(first.get("goal_known") == second.get("goal_known")),
            "target_direction": cls._direction_similarity(
                first.get("target_direction"), second.get("target_direction")
            ),
            "goal_dynamic": float(first.get("goal_dynamic") == second.get("goal_dynamic")),
            "adjacent_obstacles": cls._obstacle_similarity(
                first.get("adjacent_obstacles"), second.get("adjacent_obstacles")
            ),
            "entity_distance_bucket": cls._bucket_similarity(
                first.get("entity_distance_bucket"),
                second.get("entity_distance_bucket"),
                maximum_distance=4,
            ),
            "health_bucket": cls._bucket_similarity(
                first.get("health_bucket"), second.get("health_bucket"),
                maximum_distance=2,
            ),
            "frontier_bucket": cls._bucket_similarity(
                first.get("frontier_bucket"), second.get("frontier_bucket"),
                maximum_distance=2,
            ),
        }
        return sum(weights[key] * similarities[key] for key in weights)

    @staticmethod
    def _valid_decision_context(context: Dict[str, Any]) -> bool:
        required = {
            "goal_visible",
            "goal_known",
            "target_direction",
            "goal_dynamic",
            "adjacent_obstacles",
            "entity_distance_bucket",
            "health_bucket",
            "frontier_bucket",
        }
        if set(context) != required:
            return False
        if any(
            not isinstance(context[key], bool)
            for key in ("goal_visible", "goal_known", "goal_dynamic")
        ):
            return False
        direction = context["target_direction"]
        if direction is not None and not (
            isinstance(direction, (list, tuple))
            and len(direction) == 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value in {-1, 0, 1}
                for value in direction
            )
        ):
            return False
        obstacles = context["adjacent_obstacles"]
        if not (
            isinstance(obstacles, (list, tuple))
            and len(obstacles) == 4
            and all(value is None or isinstance(value, bool) for value in obstacles)
        ):
            return False
        if not GridMemoryStore._valid_bucket(
            context["entity_distance_bucket"], {0, 1, 2, 3}, allow_none=True
        ):
            return False
        if not GridMemoryStore._valid_bucket(
            context["health_bucket"], {0, 1, 2}, allow_none=True
        ):
            return False
        return GridMemoryStore._valid_bucket(
            context["frontier_bucket"], {0, 1, 2}, allow_none=False
        )

    @staticmethod
    def _valid_bucket(value: Any, allowed: set[int], allow_none: bool) -> bool:
        if value is None:
            return allow_none
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value in allowed
        )

    @staticmethod
    def _direction_similarity(first: Any, second: Any) -> float:
        if first is None or second is None:
            return float(first is None and second is None)
        if not (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) == len(second) == 2
        ):
            return 0.0
        first_vector = tuple(int(value) for value in first)
        second_vector = tuple(int(value) for value in second)
        first_norm = math.sqrt(sum(value * value for value in first_vector))
        second_norm = math.sqrt(sum(value * value for value in second_vector))
        if first_norm == 0 or second_norm == 0:
            return float(first_norm == second_norm)
        cosine = sum(a * b for a, b in zip(first_vector, second_vector)) / (
            first_norm * second_norm
        )
        return (cosine + 1.0) / 2.0

    @staticmethod
    def _obstacle_similarity(first: Any, second: Any) -> float:
        if not (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) == len(second) == 4
        ):
            return 0.0
        scores = [
            1.0
            if a == b
            else 0.5
            if a is None or b is None
            else 0.0
            for a, b in zip(first, second)
        ]
        return sum(scores) / len(scores)

    @staticmethod
    def _bucket_similarity(first: Any, second: Any, maximum_distance: int) -> float:
        if first is None or second is None:
            return float(first is None and second is None)
        if not (
            isinstance(first, int)
            and isinstance(second, int)
            and not isinstance(first, bool)
            and not isinstance(second, bool)
        ):
            return 0.0
        return max(0.0, 1.0 - abs(first - second) / maximum_distance)

    @staticmethod
    def _position(value: object) -> Optional[Position]:
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(isinstance(part, int) for part in value)
        ):
            return value[0], value[1]
        return None

    @classmethod
    def _positions(cls, values: object) -> List[Position]:
        if not isinstance(values, list):
            return []
        positions = [cls._position(value) for value in values]
        return [position for position in positions if position is not None]

    def update_working_memory(
        self,
        observation: Optional[object],
        action: Optional[str],
        result: Optional[str],
    ) -> None:
        self.working.current_observation = observation
        self.working.last_action = action
        self.working.last_result = result

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

    def observe_goal(
        self,
        position: Optional[Position],
        visible_cells: List[Position],
    ) -> None:
        visible = set(visible_cells)
        previous_goals = self.spatial.known_goal_positions
        if position is not None:
            if (
                self.spatial.last_observed_goal_position is not None
                and position != self.spatial.last_observed_goal_position
            ):
                self.spatial.goal_change_count += 1
                self.spatial.goal_is_dynamic = True
            self.spatial.last_observed_goal_position = position
            for previous in previous_goals:
                if previous != position:
                    cell = self.spatial.grid.get(previous)
                    if cell is not None and cell.value == "target":
                        cell.value = ""
            self.spatial.known_goal_positions = [position]
            return

        self.spatial.known_goal_positions = [
            goal for goal in previous_goals if goal not in visible
        ]
        for goal in visible.intersection(previous_goals):
            cell = self.spatial.grid.get(goal)
            if cell is not None and cell.value == "target":
                cell.value = ""

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
