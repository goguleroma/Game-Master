"""Tests for Grid 30x30 adapter components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.types import Action, ActionResult, GameState, Observation
from adapters.grid_30x30.game import GridGame, GridGameInputAdapter
from adapters.grid_30x30.observation_adapter import GridObservationAdapter
from adapters.grid_30x30.planner import GridPlanner, ActionCandidate
from adapters.grid_30x30.world_model import GridWorldModel, WorldState, WorldEntity, bfs_shortest_path
from adapters.grid_30x30.memory import (
    GridMemoryStore,
    ShortTermMemory,
    WorkingMemory,
    SpatialMemory,
    CellMemory,
    KnowledgeItem,
    EntityMemory,
    DecisionCase,
    Scope,
    SourceMemory,
    AttentionRequest,
    AttentionManager,
)
from adapters.grid_30x30.memory_consolidator import (
    GridMemoryConsolidator,
    ConsolidationConfig,
    STMEvent,
)
from adapters.grid_30x30.config import GridStrategyParameters
from simulated_game.game import GridGame as SimulatedGridGame


class TestGridGame:
    """Tests for GridGame wrapper."""

    def test_init_default_params(self) -> None:
        game = GridGame(seed=42)
        assert game.width == 30
        assert game.height == 30
        assert game.vision_radius == 4
        assert game.player_max_health == 20
        assert game.enemy_max_health == 8
        assert game.attack_damage == 3
        assert game.num_enemies == 1  # adapter default: classic 1 enemy
        assert game.num_static_walls == 0  # adapter default: no static walls

    def test_init_custom_params(self) -> None:
        game = GridGame(
            seed=42,
            width=20,
            height=15,
            vision_radius=5,
            player_max_health=30,
            enemy_max_health=10,
            attack_damage=5,
            num_enemies=3,
            num_static_walls=10,
        )
        assert game.width == 20
        assert game.height == 15
        assert game.vision_radius == 5
        assert game.player_max_health == 30
        assert game.enemy_max_health == 10
        assert game.attack_damage == 5
        assert game.num_enemies == 3
        assert game.num_static_walls == 10

    def test_observe_returns_dict(self) -> None:
        game = GridGame(seed=42)
        obs = game.observe()
        assert isinstance(obs, dict)
        assert "objects" in obs
        assert "controls" in obs
        assert "hud" in obs
        assert "goal" in obs
        assert "confidence" in obs

    def test_observe_hud_contains_grid_info(self) -> None:
        game = GridGame(seed=42)
        obs = game.observe()
        hud = obs["hud"]
        assert "grid" in hud
        grid = hud["grid"]
        assert "width" in grid
        assert "height" in grid
        assert "position" in grid
        assert "vision_radius" in grid
        assert "visible_cells" in grid
        assert "target" in grid
        assert "obstacles" in grid
        assert "entities" in grid

    def test_execute_move_action(self) -> None:
        game = GridGame(seed=42)
        game.position = (10, 10)
        start_pos = game.position

        result = game.execute(Action(name="move_right"))

        assert result.success is True
        assert game.position == (11, 10)
        assert result.details["status"] == "moved"

    def test_execute_move_blocked_by_boundary(self) -> None:
        game = GridGame(seed=42)
        game.position = (0, 0)

        result = game.execute(Action(name="move_left"))

        assert result.success is False
        assert result.details["status"] == "blocked_by_boundary"
        assert game.position == (0, 0)

    def test_execute_move_blocked_by_obstacle(self) -> None:
        game = GridGame(seed=42)
        game.position = (5, 5)
        game.obstacles = {(6, 5)}
        game.static_obstacles = set()

        result = game.execute(Action(name="move_right"))

        assert result.success is False
        assert result.details["status"] == "blocked_by_obstacle"
        assert game.position == (5, 5)

    def test_execute_control_actions(self) -> None:
        game = GridGame(seed=42)
        game.position = (10, 10)

        # Test control_1 (move up)
        game.execute(Action(name="control_1"))
        assert game.position == (10, 9)

        game.position = (10, 10)
        # Test control_2 (move right)
        game.execute(Action(name="control_2"))
        assert game.position == (11, 10)

        game.position = (10, 10)
        # Test control_3 (move down)
        game.execute(Action(name="control_3"))
        assert game.position == (10, 11)

        game.position = (10, 10)
        # Test control_4 (move left)
        game.execute(Action(name="control_4"))
        assert game.position == (9, 10)

    def test_execute_wait_action(self) -> None:
        game = GridGame(seed=42)
        game.position = (10, 10)

        result = game.execute(Action(name="wait"))

        assert result.success is True
        assert result.details["status"] == "wait"
        assert game.position == (10, 10)

    def test_execute_unknown_action(self) -> None:
        game = GridGame(seed=42)

        result = game.execute(Action(name="unknown_action"))

        assert result.success is False
        assert result.details["status"] == "unsupported_action"

    def test_game_over_states(self) -> None:
        game = GridGame(seed=42)
        assert game.won is False
        assert game.lost is False
        assert game.is_game_over is False

        game.won = True
        assert game.won is True
        assert game.is_game_over is True

        game = GridGame(seed=42)
        game.caught = True
        assert game.lost is True
        assert game.is_game_over is True

    def test_is_visible(self) -> None:
        game = GridGame(seed=42)
        game.position = (10, 10)
        game.vision_radius = 4

        assert game.is_visible((10, 10)) is True  # Same position
        assert game.is_visible((12, 10)) is True  # Distance 2
        assert game.is_visible((14, 10)) is True  # Distance 4
        assert game.is_visible((15, 10)) is False  # Distance 5

    def test_execute_combat_attack(self) -> None:
        game = GridGame(seed=42)
        game.position = (10, 10)
        game.enemy_position = (11, 10)
        game.enemy_health = 8
        game.player_health = 20

        with patch.object(game, '_enemy_combat_choice', return_value="attack"):
            result = game.execute(Action(name="control_5"))  # attack

        assert game.enemy_health == 5  # 8 - 3 damage
        assert game.player_health == 17  # 20 - 3 damage
        assert result.details["status"] == "completed"

    def test_execute_combat_defend(self) -> None:
        game = GridGame(seed=42)
        game.position = (10, 10)
        game.enemy_position = (11, 10)
        game.enemy_health = 8
        game.player_health = 20

        # Enemy defends -> enemy takes NO damage from player attack
        with patch.object(game, '_enemy_combat_choice', return_value="defend"):
            result = game.execute(Action(name="control_5"))  # attack

        assert game.enemy_health == 8  # No damage because enemy defended
        assert game.player_health == 20  # No damage (player didn't get attacked)
        assert result.details["status"] == "completed"

    def test_enemy_death_becomes_obstacle(self) -> None:
        game = GridGame(seed=42)
        game.position = (10, 10)
        game.enemy_position = (11, 10)
        game.enemy_health = 3  # Will die in one hit

        with patch.object(game, '_enemy_combat_choice', return_value="attack"):
            for _ in range(1):
                game.enemy_position = (11, 10)
                game.enemy_health = 3
                game.execute(Action(name="control_5"))

        assert (11, 10) in game.static_obstacles
        assert len(game.enemies) == 0

    def test_player_death(self) -> None:
        game = GridGame(seed=42)
        game.position = (10, 10)
        game.enemy_position = (11, 10)
        game.player_health = 1

        with patch.object(game, '_enemy_combat_choice', return_value="attack"):
            result = game.execute(Action(name="control_5"))

        assert game.player_health == 0
        assert game.caught is True
        assert result.details["status"] == "caught"


class TestGridGameInputAdapter:
    """Tests for GridGameInputAdapter."""

    def test_execute_delegates_to_game(self) -> None:
        game = GridGame(seed=42)
        game.position = (10, 10)
        adapter = GridGameInputAdapter(game)

        result = adapter.execute(Action(name="move_right"))

        assert result.success is True
        assert game.position == (11, 10)


class TestGridObservationAdapter:
    """Tests for GridObservationAdapter."""

    def test_extract_from_dict(self) -> None:
        adapter = GridObservationAdapter()
        obs_dict = {
            "objects": ["player", "enemy"],
            "hud": {"health": 100},
            "goal": "reach_target",
            "confidence": 0.9,
            "controls": ["move_up", "move_down"],
        }

        state = adapter.extract(obs_dict)

        assert isinstance(state, GameState)
        assert state.objects == ["player", "enemy"]
        assert state.hud == {"health": 100}
        assert state.goal == "reach_target"
        assert state.confidence == 0.9
        assert state.controls == ["move_up", "move_down"]

    def test_extract_from_observation_object(self) -> None:
        adapter = GridObservationAdapter()
        obs = Observation(
            image={
                "objects": ["player"],
                "hud": {"health": 50},
                "goal": "survive",
                "confidence": 0.8,
            },
            timestamp=1.0,
            metadata={},
        )

        state = adapter.extract(obs)

        assert state.objects == ["player"]
        assert state.hud == {"health": 50}
        assert state.goal == "survive"
        assert state.confidence == 0.8

    def test_extract_fallback_for_other_types(self) -> None:
        adapter = GridObservationAdapter()
        obs = Observation(image="not_a_dict", timestamp=1.0, metadata={})

        state = adapter.extract(obs)

        assert isinstance(state, GameState)
        assert state.objects == []
        assert state.hud == {}
        assert state.confidence == 0.0


class TestGridPlanner:
    """Tests for GridPlanner."""

    def test_choose_action_with_world_state(self) -> None:
        planner = GridPlanner()
        memory = Mock()
        memory.spatial = Mock()
        # Set up spatial grid with path from (0,0) to (10,10)
        memory.spatial.grid = {}
        for x in range(11):
            for y in range(11):
                if x == 0 and y == 0:
                    continue  # start
                cell = Mock()
                cell.seen = True
                cell.obstacle = False
                cell.visited = False
                cell.visit_count = 0
                memory.spatial.grid[(x, y)] = cell
        # Also add start cell
        start_cell = Mock()
        start_cell.seen = True
        start_cell.obstacle = False
        start_cell.visited = True
        start_cell.visit_count = 1
        memory.spatial.grid[(0, 0)] = start_cell
        memory.spatial.frontier = [(5, 5)]
        memory.spatial.goal_is_dynamic = False
        memory.spatial.goal_change_count = 0
        memory.spatial.known_goal_positions = [(10, 10)]
        memory.build_decision_context = Mock(return_value={})
        memory.retrieve_direction_preferences = Mock(return_value={})

        state = WorldState(
            current_position=(0, 0),
            target_position=(10, 10),
            obstacle_positions=[],
            grid_size=(30, 30),
            confidence=1.0,
        )

        action = planner.choose_action(state, memory)

        assert action is not None
        assert isinstance(action, ActionCandidate)
        assert action.name in ["move_up", "move_right", "move_down", "move_left"]

    def test_choose_action_explores_frontier_when_no_goal(self) -> None:
        planner = GridPlanner()
        memory = Mock()
        memory.spatial = Mock()
        memory.spatial.frontier = [(5, 5), (6, 6)]
        
        # Set up spatial grid with path from (0,0) to frontiers
        memory.spatial.grid = {}
        # Add start cell
        start_cell = Mock()
        start_cell.seen = True
        start_cell.obstacle = False
        start_cell.visited = True
        start_cell.visit_count = 1
        memory.spatial.grid[(0, 0)] = start_cell
        
        # Add intermediate cells to reach frontier
        for x in range(6):
            for y in range(6):
                if x == 0 and y == 0:
                    continue
                cell = Mock()
                cell.seen = True
                cell.obstacle = False
                cell.visited = False
                cell.visit_count = 0
                memory.spatial.grid[(x, y)] = cell
        
        # Add frontier cells specifically
        for pos in [(5, 5), (6, 6)]:
            cell = Mock()
            cell.seen = True
            cell.obstacle = False
            cell.visited = False
            cell.visit_count = 0
            memory.spatial.grid[pos] = cell

        memory.retrieve_direction_preferences = Mock(return_value={})

        state = WorldState(
            current_position=(0, 0),
            target_position=None,
            obstacle_positions=[],
            grid_size=(30, 30),
            confidence=1.0,
        )

        action = planner.choose_action(state, memory)

        assert action is not None
        # When no goal but frontiers exist, planner should move towards a frontier
        # The action name will be a movement direction like "move_right", "move_down", etc.
        assert action.name in ["move_up", "move_right", "move_down", "move_left"]
        assert action.utility > 0.5  # Should have reasonable utility for exploration

    def test_record_blocked_transition(self) -> None:
        planner = GridPlanner()

        planner.record_blocked_transition((5, 5), (1, 0))

        assert ((5, 5), (6, 5)) in planner._blocked_transitions


class TestGridWorldModel:
    """Tests for GridWorldModel."""

    def test_update_from_observation(self) -> None:
        model = GridWorldModel()
        obs = {
            "hud": {
                "grid": {
                    "width": 30,
                    "height": 30,
                    "position": [10, 10],
                    "target": [20, 20],
                    "visible_cells": [[10, 10], [11, 10], [10, 11]],
                    "obstacles": [[12, 10]],
                    "entities": [
                        {"id": "entity-1", "position": [11, 10], "kind": "obstacle"},
                        {"id": "entity-2", "position": [15, 15], "kind": "enemy", "health": 8},
                    ],
                }
            },
            "goal": "reach_target",
            "confidence": 1.0,
        }

        state = model.update(obs)

        assert state.current_position == (10, 10)
        assert state.target_position == (20, 20)
        assert state.grid_size == (30, 30)
        assert len(state.visible_cells) == 3
        assert state.obstacle_positions == [(12, 10)]
        assert len(state.entities) == 5  # agent + 2 observed entities + target + obstacle
        assert state.goal == "reach_target"
        assert state.confidence == 1.0

    def test_update_with_none_observation(self) -> None:
        model = GridWorldModel()
        state = model.update(None, position=(5, 5))

        assert state.current_position == (5, 5)
        assert state.confidence == 0.0

    def test_entities_include_agent_and_target(self) -> None:
        model = GridWorldModel()
        obs = {
            "hud": {
                "grid": {
                    "width": 30,
                    "height": 30,
                    "position": [10, 10],
                    "target": [20, 20],
                    "visible_cells": [],
                    "obstacles": [],
                    "entities": [],
                }
            },
            "confidence": 1.0,
        }

        state = model.update(obs)

        # Should have agent and target entities
        agent_entities = [e for e in state.entities if e.kind == "agent"]
        target_entities = [e for e in state.entities if e.kind == "target"]
        assert len(agent_entities) == 1
        assert len(target_entities) == 1
        assert agent_entities[0].position == (10, 10)
        assert target_entities[0].position == (20, 20)


class TestBFSShortestPath:
    """Tests for bfs_shortest_path function."""

    def test_simple_path(self) -> None:
        start = (0, 0)
        goal = (2, 0)
        blocked = set()
        bounds = (10, 10)

        path = bfs_shortest_path(start, goal, blocked, bounds=bounds)

        assert len(path) == 3
        assert path[0] == (0, 0)
        assert path[-1] == (2, 0)

    def test_path_avoids_blocked(self) -> None:
        start = (0, 0)
        goal = (2, 0)
        blocked = {(1, 0)}
        bounds = (10, 10)

        path = bfs_shortest_path(start, goal, blocked, bounds=bounds)

        # Should go around
        assert (1, 0) not in path
        assert path[-1] == (2, 0)

    def test_no_path_returns_empty(self) -> None:
        start = (0, 0)
        goal = (2, 0)
        blocked = {(1, 0), (0, 1), (1, 1)}  # Completely surrounded
        bounds = (10, 10)

        path = bfs_shortest_path(start, goal, blocked, bounds=bounds)

        assert path == []

    def test_path_respects_bounds(self) -> None:
        start = (0, 0)
        goal = (5, 5)
        blocked = set()
        bounds = (3, 3)  # Goal outside bounds

        path = bfs_shortest_path(start, goal, blocked, bounds=bounds)

        assert path == []

    def test_path_respects_allowed_cells(self) -> None:
        start = (0, 0)
        goal = (2, 0)
        blocked = set()
        bounds = (10, 10)
        allowed = {(0, 0), (1, 0), (2, 0)}  # Only direct path allowed

        path = bfs_shortest_path(start, goal, blocked, bounds=bounds, allowed=allowed)

        assert path == [(0, 0), (1, 0), (2, 0)]

    def test_blocked_transitions(self) -> None:
        start = (0, 0)
        goal = (2, 0)
        blocked = set()
        bounds = (10, 10)
        blocked_transitions = {((0, 0), (1, 0))}  # Can't go right from start

        path = bfs_shortest_path(start, goal, blocked, bounds=bounds, blocked_transitions=blocked_transitions)

        # Should go up then right then down
        assert (1, 0) not in path


class TestGridMemoryStore:
    """Tests for GridMemoryStore."""

    def test_init(self) -> None:
        memory = GridMemoryStore(game_id="test_game")

        assert memory.game_id == "test_game"
        assert memory.enable_episodic_retrieval is True
        assert memory.enable_consolidation is False
        assert isinstance(memory.short_term, ShortTermMemory)
        assert isinstance(memory.working, WorkingMemory)
        assert isinstance(memory.spatial, SpatialMemory)
        assert isinstance(memory.attention, AttentionManager)

    def test_observe_frame(self) -> None:
        memory = GridMemoryStore()
        frame = {"test": "data"}

        memory.observe_frame(frame)

        assert len(memory.short_term.frames) == 1

    def test_record(self) -> None:
        memory = GridMemoryStore()
        entry = {"action": "move", "reward": 1.0}

        memory.record(entry)

        assert len(memory.records) == 1
        assert memory.records[0] == entry

    def test_update_working_memory(self) -> None:
        memory = GridMemoryStore()

        memory.update_working_memory({"pos": (1, 1)}, "move_right", "moved")

        assert memory.working.current_observation == {"pos": (1, 1)}
        assert memory.working.last_action == "move_right"
        assert memory.working.last_result == "moved"

    def test_mark_cell(self) -> None:
        memory = GridMemoryStore()

        memory.mark_cell((5, 5), seen=True, visited=True, obstacle=False, turn=1)

        cell = memory.spatial.grid[(5, 5)]
        assert cell.seen is True
        assert cell.visited is True
        assert cell.obstacle is False
        assert cell.last_seen_turn == 1
        assert cell.visit_count == 1

    def test_observe_goal(self) -> None:
        memory = GridMemoryStore()

        memory.observe_goal((10, 10), [(10, 10), (11, 10)])

        assert memory.spatial.known_goal_positions == [(10, 10)]
        assert memory.spatial.last_observed_goal_position == (10, 10)

    def test_observe_goal_dynamic(self) -> None:
        memory = GridMemoryStore()

        memory.observe_goal((10, 10), [(10, 10)])
        memory.observe_goal((15, 15), [(15, 15)])  # Goal moved

        assert memory.spatial.goal_change_count == 1
        assert memory.spatial.goal_is_dynamic is True


class TestAttentionManager:
    """Tests for AttentionManager."""

    def test_request_observation(self) -> None:
        manager = AttentionManager()
        roi = ((0, 0), (10, 10))

        request = manager.request_observation(roi, priority=0.8, reason="test", source="planner")

        assert request.roi == roi
        assert request.priority == 0.8
        assert request.reason == "test"
        assert request.source == "planner"
        assert len(manager.requests) == 1

    def test_priority_clamping(self) -> None:
        manager = AttentionManager()

        req1 = manager.request_observation(((0, 0), (10, 10)), priority=1.5)
        req2 = manager.request_observation(((0, 0), (10, 10)), priority=-0.5)

        assert req1.priority == 1.0
        assert req2.priority == 0.0

    def test_get_top_requests(self) -> None:
        manager = AttentionManager()
        manager.request_observation(((0, 0), (10, 10)), priority=0.5)
        manager.request_observation(((5, 5), (15, 15)), priority=0.9)
        manager.request_observation(((10, 10), (20, 20)), priority=0.3)

        top = manager.get_top_requests(2)

        assert len(top) == 2
        assert top[0].priority == 0.9
        assert top[1].priority == 0.5

    def test_clear_requests(self) -> None:
        manager = AttentionManager()
        manager.request_observation(((0, 0), (10, 10)))

        manager.clear_requests()

        assert len(manager.requests) == 0

    def test_get_requests_for_region(self) -> None:
        manager = AttentionManager()
        manager.request_observation(((0, 0), (10, 10)), priority=0.5)
        manager.request_observation(((5, 5), (15, 15)), priority=0.9)

        requests = manager.get_requests_for_region((5, 5))

        assert len(requests) == 2  # Both cover (5, 5)
        assert requests[0].priority == 0.9


class TestGridMemoryConsolidator:
    """Tests for GridMemoryConsolidator."""

    def test_init(self) -> None:
        memory = GridMemoryStore()
        config = ConsolidationConfig()
        consolidator = GridMemoryConsolidator(memory, config)

        assert consolidator.memory == memory
        assert consolidator.config == config
        assert consolidator._running is False

    def test_add_stm_event(self) -> None:
        memory = GridMemoryStore()
        consolidator = GridMemoryConsolidator(memory)

        event = consolidator.add_stm_event(
            frame={"pos": (1, 1)},
            reward=1.0,
            action="move_right",
            observation={"hud": {}},
        )

        assert isinstance(event, STMEvent)
        assert event.reward == 1.0
        assert event.action == "move_right"
        assert len(consolidator._stm_events) == 1

    def test_evaluate_significance_high_reward(self) -> None:
        memory = GridMemoryStore()
        consolidator = GridMemoryConsolidator(memory)
        config = ConsolidationConfig(stm_significance_threshold=0.3)
        consolidator.config = config

        event = STMEvent(
            frame={},
            timestamp=1.0,
            reward=1.0,
            action="attack",
            observation={},
        )

        significance = consolidator.evaluate_significance(event)

        assert significance >= 0.4  # High reward should give high significance

    def test_evaluate_significance_zero_reward(self) -> None:
        memory = GridMemoryStore()
        consolidator = GridMemoryConsolidator(memory)

        event = STMEvent(
            frame={},
            timestamp=1.0,
            reward=0.0,
            action="wait",
            observation={},
        )

        significance = consolidator.evaluate_significance(event)

        assert significance <= 0.3  # Low significance (0.3 for novel but zero reward)

    def test_promote_stm_to_mtm(self) -> None:
        memory = GridMemoryStore()
        consolidator = GridMemoryConsolidator(memory)

        event = STMEvent(
            frame={"pos": (1, 1)},
            timestamp=1.0,
            reward=1.0,
            action="attack",
            observation={},
        )

        hypothesis = consolidator.promote_stm_to_mtm(event, confidence=0.8)

        assert isinstance(hypothesis, KnowledgeItem)
        assert hypothesis.confidence == 0.8
        assert hypothesis.scope == Scope.EPISODE
        assert len(consolidator._mtm_hypotheses) == 1

    def test_confirm_hypothesis(self) -> None:
        memory = GridMemoryStore()
        consolidator = GridMemoryConsolidator(memory)

        event = STMEvent(frame={}, timestamp=1.0, reward=1.0, action="attack", observation={})
        hypothesis = consolidator.promote_stm_to_mtm(event, confidence=0.5)

        result = consolidator.confirm_hypothesis(hypothesis.key, reward=1.0)

        assert result is True
        assert hypothesis.confirmation_count == 1
        assert hypothesis.confidence == 0.6  # 0.5 + 0.1

    def test_confirm_hypothesis_negative_reward(self) -> None:
        memory = GridMemoryStore()
        consolidator = GridMemoryConsolidator(memory)

        event = STMEvent(frame={}, timestamp=1.0, reward=1.0, action="attack", observation={})
        hypothesis = consolidator.promote_stm_to_mtm(event, confidence=0.5)

        result = consolidator.confirm_hypothesis(hypothesis.key, reward=-1.0)

        assert result is True
        assert hypothesis.confidence == 0.35  # 0.5 - 0.15


class TestConsolidationConfig:
    """Tests for ConsolidationConfig."""

    def test_default_values(self) -> None:
        config = ConsolidationConfig()

        assert config.stm_significance_threshold == 0.3
        assert config.mtm_confidence_threshold == 0.6
        assert config.ltm_pattern_min_support == 5
        assert config.gsm_generalization_min_games == 3
        assert config.consolidation_interval_seconds == 1.0


class TestSimulatedGridGame:
    """Tests for the underlying SimulatedGridGame."""

    def test_seed_reproducibility(self) -> None:
        game1 = SimulatedGridGame(seed=42)
        game2 = SimulatedGridGame(seed=42)

        assert game1.target == game2.target
        assert game1.position == game2.position
        assert game1.obstacles == game2.obstacles

    def test_target_placement_reachable(self) -> None:
        game = SimulatedGridGame(seed=42, num_static_walls=10)

        # Target should be reachable from player position
        blocked = game.obstacles | game.static_obstacles | {e["position"] for e in game.enemies}
        path = game._find_path(game.position, game.target, blocked)
        assert len(path) > 0

    def test_static_wall_generation(self) -> None:
        game = SimulatedGridGame(seed=42, num_static_walls=20)

        assert len(game.static_obstacles) <= 20
        # Player start area (5x5) should be clear
        for x in range(5):
            for y in range(5):
                assert (x, y) not in game.static_obstacles

    def test_enemy_initialization(self) -> None:
        game = SimulatedGridGame(seed=42, num_enemies=3)

        assert len(game.enemies) <= 3
        for enemy in game.enemies:
            assert "position" in enemy
            assert "health" in enemy
            assert "alerted" in enemy
            assert enemy["health"] == game.enemy_max_health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])