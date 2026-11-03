"""
GridMemoryStore for Grid 30x30 game.
Spatial + episodic memory with top-down attention.
"""

from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
import math
import sqlite3
import time
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio


Position = Tuple[int, int]
DECISION_CASE_LIMIT = 2000
DECISION_RETRIEVAL_WINDOW = 256
_DIRECTIONS: Tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass
class AttentionRequest:
    """Top-down attention request for focused observation."""
    roi: Tuple[Position, Position]  # Region of interest (top-left, bottom-right)
    priority: float  # 0.0 to 1.0
    reason: str  # Why this region is important
    timestamp: float = field(default_factory=time.monotonic)
    source: str = "planner"  # 'planner', 'consolidator', 'policy', etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionManager:
    """Manages top-down attention requests for focused perception.

    This allows higher-level components (planner, consolidator, policy) to
    request focused observation of specific game regions.
    """
    max_requests: int = 10
    default_priority: float = 0.5

    def __init__(self):
        self.requests: List[AttentionRequest] = []
        self.history: List[AttentionRequest] = []

    def request_observation(
        self,
        roi: Tuple[Position, Position],
        priority: float = 0.5,
        reason: str = "",
        source: str = "planner",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AttentionRequest:
        """Request focused observation of a region.

        Args:
            roi: Region of interest as ((x1, y1), (x2, y2))
            priority: Priority 0.0-1.0 (higher = more important)
            reason: Why this region needs attention
            source: Requesting component
            metadata: Additional context

        Returns:
            The created AttentionRequest
        """
        request = AttentionRequest(
            roi=roi,
            priority=min(1.0, max(0.0, priority)),
            reason=reason,
            source=source,
            metadata=metadata or {},
        )
        self.requests.append(request)
        self.history.append(request)

        # Keep only max_requests
        if len(self.requests) > self.max_requests:
            self.requests = sorted(self.requests, key=lambda r: r.priority, reverse=True)[:self.max_requests]

        return request

    def get_top_requests(self, n: int = 3) -> List[AttentionRequest]:
        """Get top N attention requests by priority."""
        return sorted(self.requests, key=lambda r: r.priority, reverse=True)[:n]

    def clear_requests(self) -> None:
        """Clear current attention requests."""
        self.requests.clear()

    def get_requests_for_region(self, position: Position, radius: int = 5) -> List[AttentionRequest]:
        """Get requests that cover a specific position."""
        matching = []
        for req in self.requests:
            (x1, y1), (x2, y2) = req.roi
            if x1 <= position[0] <= x2 and y1 <= position[1] <= y2:
                matching.append(req)
        return sorted(matching, key=lambda r: r.priority, reverse=True)


class Scope(Enum):
    """Scope of knowledge applicability."""
    EPISODE = "episode"
    SESSION = "session"
    GAME = "game"
    GENERAL = "general"


class SourceMemory(Enum):
    """Source memory tier from which knowledge originated."""
    STM = "stm"       # Short-term memory
    MTM = "mtm"       # Medium-term memory
    LTM = "ltm"       # Long-term memory
    GSM = "gsm"       # General/shared memory
    GM = "gm"         # Game memory


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
    # Provenance tracking
    provenance: Dict[str, Any] = field(default_factory=dict)
    # Scope of applicability
    scope: Scope = Scope.EPISODE
    # Confirmation tracking
    confirmation_count: int = 0
    # Validity tracking
    validity: Dict[str, Any] = field(default_factory=dict)
    # Transferability flag
    transferable: bool = False
    # Consolidation traceability
    consolidated_from: List[str] = field(default_factory=list)


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
        enable_episodic_retrieval: bool = True,
        enable_consolidation: bool = False,
        consolidation_config: Optional["ConsolidationConfig"] = None,
    ) -> None:
        self.enable_episodic_retrieval = enable_episodic_retrieval
        self.enable_consolidation = enable_consolidation
        self._consolidation_config = consolidation_config
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

        # Top-down attention management
        self.attention = AttentionManager()

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
                # GM (General Memory) persistence tables
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS general_knowledge (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        evidence_count INTEGER NOT NULL,
                        context TEXT NOT NULL,
                        provenance TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        confirmation_count INTEGER NOT NULL,
                        validity TEXT NOT NULL,
                        transferable INTEGER NOT NULL,
                        consolidated_from TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS general_knowledge_by_scope
                    ON general_knowledge (scope, confidence DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gm_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_type TEXT NOT NULL,  -- 'control', 'mechanic', 'principle'
                        pattern_key TEXT NOT NULL,
                        pattern_value TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        source_games TEXT NOT NULL,  -- JSON list of game_ids
                        evidence_count INTEGER NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS gm_patterns_by_type
                    ON gm_patterns (pattern_type, confidence DESC)
                    """
                )
                connection.commit()

        # STM event buffer for consolidation
        self._stm_buffer: List[Dict[str, Any]] = []
        self._consolidator: Optional["MemoryConsolidator"] = None
        self._consolidation_task: Optional[asyncio.Task] = None

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self.experiences

    def record(self, entry: dict[str, Any]) -> None:
        """Record an experience entry (implements AgentMemory protocol)."""
        self.experiences.append(entry)

    @property
    def consolidator(self) -> Optional["MemoryConsolidator"]:
        """Lazily initialize and return the MemoryConsolidator."""
        if not self.enable_consolidation:
            return None
        if self._consolidator is None:
            from .memory_consolidator import MemoryConsolidator, ConsolidationConfig
            config = self._consolidation_config or ConsolidationConfig()
            self._consolidator = MemoryConsolidator(self, config)
        return self._consolidator

    def start_consolidation(self, interval_seconds: float = 30.0) -> None:
        """Start background consolidation loop.
        
        Args:
            interval_seconds: Interval between consolidation steps in seconds.
            
        Note:
            Requires a running asyncio event loop. If no loop is running,
            use consolidate_now() for synchronous consolidation.
        """
        if not self.enable_consolidation:
            return
        consolidator = self.consolidator
        if consolidator is None:
            return
        if consolidator._running:
            return
        consolidator.config.consolidation_interval_seconds = interval_seconds
        try:
            loop = asyncio.get_running_loop()
            self._consolidation_task = loop.create_task(consolidator.start())
        except RuntimeError:
            raise RuntimeError(
                "No running event loop. start_consolidation() requires an async context. "
                "Use consolidate_now() for synchronous consolidation or run within an async function."
            )

    def stop_consolidation(self) -> None:
        """Stop background consolidation loop.
        
        Note:
            Requires a running asyncio event loop (the same one used to start consolidation).
        """
        if self._consolidator is None:
            return
        try:
            loop = asyncio.get_running_loop()
            self._consolidation_task = loop.create_task(self._consolidator.stop())
        except RuntimeError:
            raise RuntimeError(
                "No running event loop. stop_consolidation() requires an async context."
            )
        self._consolidation_task = None

    def consolidate_now(self) -> Dict[str, int]:
        """Run one consolidation step synchronously.
        
        Returns:
            Dictionary with counts of promotions at each tier.
        """
        if not self.enable_consolidation or self._consolidator is None:
            return {"stm_to_mtm": 0, "mtm_to_ltm": 0, "ltm_to_gsm": 0, "gsm_to_gm": 0}
        try:
            loop = asyncio.get_running_loop()
            # If loop is running, we can't run synchronously
            # Note: In a running loop, we can't block. Return empty for now.
            return {"stm_to_mtm": 0, "mtm_to_ltm": 0, "ltm_to_gsm": 0, "gsm_to_gm": 0}
        except RuntimeError:
            # No running loop, create one and run
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._consolidator.consolidate_step())
            finally:
                loop.close()

    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics from the consolidator."""
        if self._consolidator is None:
            return {"enabled": False}
        return {"enabled": True, **self._consolidator.get_stats()}

    # ===== Top-Down Attention API =====

    def request_observation(
        self,
        roi: Tuple[Position, Position],
        priority: float = 0.5,
        reason: str = "",
        source: str = "planner",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AttentionRequest:
        """Request focused observation of a game region (top-down attention).

        This allows higher-level components to direct perception toward
        specific areas of interest.

        Args:
            roi: Region of interest as ((x1, y1), (x2, y2))
            priority: Priority 0.0-1.0 (higher = more important)
            reason: Why this region needs attention
            source: Requesting component ('planner', 'consolidator', 'policy', etc.)
            metadata: Additional context

        Returns:
            The created AttentionRequest
        """
        return self.attention.request_observation(
            roi=roi,
            priority=priority,
            reason=reason,
            source=source,
            metadata=metadata,
        )

    def get_attention_requests(self, n: int = 3) -> List[AttentionRequest]:
        """Get top N attention requests by priority.

        Args:
            n: Number of top requests to return

        Returns:
            List of AttentionRequests sorted by priority
        """
        return self.attention.get_top_requests(n)

    def clear_attention_requests(self) -> None:
        """Clear all current attention requests."""
        self.attention.clear_requests()

    def get_attention_for_position(self, position: Position, radius: int = 5) -> List[AttentionRequest]:
        """Get attention requests covering a specific position.

        Args:
            position: Grid position to check
            radius: Radius around position (for expanded ROI matching)

        Returns:
            Matching attention requests sorted by priority
        """
        return self.attention.get_requests_for_region(position, radius)

    def get_attention_stats(self) -> Dict[str, Any]:
        """Get attention manager statistics."""
        return {
            "active_requests": len(self.attention.requests),
            "total_history": len(self.attention.history),
            "top_priorities": [r.priority for r in self.attention.get_top_requests(5)],
        }

    def _feed_stm_buffer(
        self,
        frame: Any,
        reward: float = 0.0,
        action: Optional[str] = None,
        observation: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an event to the STM buffer for consolidation.
        
        This is called by internal methods to feed events to the consolidator.
        """
        if not self.enable_consolidation:
            return
        event = {
            "frame": frame,
            "reward": reward,
            "action": action,
            "observation": observation,
            "metadata": metadata or {},
            "timestamp": time.monotonic(),
        }
        self._stm_buffer.append(event)
        # Also feed to consolidator if available (access property to trigger lazy init)
        consolidator = self.consolidator
        if consolidator is not None:
            consolidator.add_stm_event(
                frame=frame,
                reward=reward,
                action=action,
                observation=observation,
                metadata=metadata,
                timestamp=event["timestamp"],
            )

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

    # ===== GM (General Memory) Persistence =====

    def save_general_knowledge(self) -> int:
        """Save GENERAL scope knowledge items to SQLite for cross-session persistence.

        Returns:
            Number of items saved
        """
        if self.database_path is None:
            return 0

        now = time.monotonic()
        saved = 0
        with closing(self._connect()) as connection:
            for item in self.general_knowledge:
                if item.scope != Scope.GENERAL:
                    continue
                connection.execute(
                    """
                    INSERT OR REPLACE INTO general_knowledge
                    (key, value, confidence, evidence_count, context, provenance,
                     scope, confirmation_count, validity, transferable, consolidated_from,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.key,
                        json.dumps(item.value),
                        item.confidence,
                        item.evidence_count,
                        json.dumps(item.context),
                        json.dumps(item.provenance),
                        item.scope.value,
                        item.confirmation_count,
                        json.dumps(item.validity),
                        int(item.transferable),
                        json.dumps(item.consolidated_from),
                        item.provenance.get("added_at", now),
                        now,
                    ),
                )
                saved += 1
            connection.commit()
        return saved

    def load_general_knowledge(self) -> int:
        """Load GENERAL scope knowledge items from SQLite.

        Returns:
            Number of items loaded
        """
        if self.database_path is None:
            return 0

        loaded = 0
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT key, value, confidence, evidence_count, context, provenance,
                       scope, confirmation_count, validity, transferable, consolidated_from,
                       created_at, updated_at
                FROM general_knowledge
                WHERE scope = ?
                ORDER BY confidence DESC
                """,
                (Scope.GENERAL.value,),
            ).fetchall()

            for row in rows:
                (key, value_json, confidence, evidence_count, context_json,
                 provenance_json, scope_str, confirmation_count, validity_json,
                 transferable, consolidated_from_json, created_at, updated_at) = row

                item = KnowledgeItem(
                    key=key,
                    value=json.loads(value_json),
                    confidence=confidence,
                    evidence_count=evidence_count,
                    context=json.loads(context_json),
                    provenance=json.loads(provenance_json),
                    scope=Scope(scope_str),
                    confirmation_count=confirmation_count,
                    validity=json.loads(validity_json),
                    transferable=bool(transferable),
                    consolidated_from=json.loads(consolidated_from_json),
                )
                self.general_knowledge.append(item)
                loaded += 1
        return loaded

    def save_gm_patterns(self, gm_patterns: Dict[str, Any]) -> int:
        """Save GM (General Memory) patterns to SQLite.

        Args:
            gm_patterns: Dict with pattern_type as key, list of pattern dicts as value

        Returns:
            Number of patterns saved
        """
        if self.database_path is None:
            return 0

        now = time.monotonic()
        saved = 0
        with closing(self._connect()) as connection:
            for pattern_type, patterns in gm_patterns.items():
                for pattern in patterns:
                    pattern_key = pattern.get("key", "")
                    pattern_value = pattern.get("value", {})
                    confidence = pattern.get("confidence", 0.0)
                    source_games = pattern.get("source_games", [])
                    evidence_count = pattern.get("evidence_count", 0)

                    connection.execute(
                        """
                        INSERT OR REPLACE INTO gm_patterns
                        (pattern_type, pattern_key, pattern_value, confidence,
                         source_games, evidence_count, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            pattern_type,
                            pattern_key,
                            json.dumps(pattern_value),
                            confidence,
                            json.dumps(source_games),
                            evidence_count,
                            now,
                            now,
                        ),
                    )
                    saved += 1
            connection.commit()
        return saved

    def load_gm_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load GM patterns from SQLite.

        Returns:
            Dict with pattern_type as key, list of pattern dicts as value
        """
        if self.database_path is None:
            return {}

        patterns = defaultdict(list)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT pattern_type, pattern_key, pattern_value, confidence,
                       source_games, evidence_count, created_at, updated_at
                FROM gm_patterns
                ORDER BY confidence DESC
                """
            ).fetchall()

            for row in rows:
                (pattern_type, pattern_key, pattern_value_json, confidence,
                 source_games_json, evidence_count, created_at, updated_at) = row

                patterns[pattern_type].append({
                    "key": pattern_key,
                    "value": json.loads(pattern_value_json),
                    "confidence": confidence,
                    "source_games": json.loads(source_games_json),
                    "evidence_count": evidence_count,
                    "created_at": created_at,
                    "updated_at": updated_at,
                })
        return dict(patterns)

    def get_gm_persistence_stats(self) -> Dict[str, int]:
        """Get statistics about GM persistence.

        Returns:
            Dict with counts of saved items
        """
        if self.database_path is None:
            return {"general_knowledge": 0, "gm_patterns": 0}

        with closing(self._connect()) as connection:
            gk_count = connection.execute(
                "SELECT COUNT(*) FROM general_knowledge WHERE scope = ?",
                (Scope.GENERAL.value,),
            ).fetchone()[0]

            gp_count = connection.execute(
                "SELECT COUNT(*) FROM gm_patterns"
            ).fetchone()[0]

            return {"general_knowledge": gk_count, "gm_patterns": gp_count}

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

        # Feed to STM buffer for consolidation
        self._feed_stm_buffer(
            frame={"context": context, "planned_direction": planned_direction},
            reward=reward,
            action=details.get("action"),
            observation=details.get("observation"),
            metadata={"outcome": outcome, "before_position": before_position, "target_position": target_position},
        )

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
        if not self.enable_episodic_retrieval:
            # Ablation switch: episodic retrieval is disabled for this run and
            # no prior case may steer the decision.
            return {}
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
        if width is None or height is None:
            return
        frontier: List[Position] = []
        for x in range(width):
            for y in range(height):
                pos = (x, y)
                cell = self.spatial.grid.get(pos)
                if cell is None or not cell.seen:
                    continue
                if cell.obstacle is True:
                    continue
                # Check if any neighbor is unseen
                is_frontier = False
                for dx, dy in _DIRECTIONS:
                    neighbor = (x + dx, y + dy)
                    if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                        continue
                    neighbor_cell = self.spatial.grid.get(neighbor)
                    if neighbor_cell is None or not neighbor_cell.seen:
                        is_frontier = True
                        break
                if is_frontier:
                    frontier.append(pos)
        self.spatial.frontier = frontier

    def observe_entity(
        self,
        entity_id: str,
        position: Position,
        agent_position: Optional[Position],
        turn: int,
    ) -> None:
        ent = self.entities.get(entity_id)
        if ent is None:
            self.entities[entity_id] = EntityMemory(
                position=position,
                last_seen_turn=turn,
                last_agent_position=agent_position,
            )
            return

        # Update approach/retreat streak
        if agent_position is not None and ent.last_agent_position is not None:
            prev_dist = abs(ent.position[0] - ent.last_agent_position[0]) + abs(
                ent.position[1] - ent.last_agent_position[1]
            )
            curr_dist = abs(position[0] - agent_position[0]) + abs(
                position[1] - agent_position[1]
            )
            if curr_dist < prev_dist:
                ent.approach_streak += 1
                ent.retreat_streak = 0
            elif curr_dist > prev_dist:
                ent.retreat_streak += 1
                ent.approach_streak = 0

        # Update category
        ent.position = position
        ent.last_agent_position = agent_position
        ent.last_seen_turn = turn
        ent.observation_count += 1
        if ent.category == "unclassified" and ent.observation_count >= 2:
            ent.category = "mobile"

    def _manhattan_distance(self, a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])