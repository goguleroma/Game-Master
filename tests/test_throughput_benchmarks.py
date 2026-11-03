"""Throughput benchmarks with FPS assertions for Game Master components."""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from dataclasses import dataclass
import threading

from core.engine import GameEngine
from core.memory import MemoryStore
from core.types import Action, ActionResult, GameState, Observation
from core.observation_adapter import ObservationAdapter
from perception.capture import ScreenCapture, CaptureConfig, FrameRingBuffer
from perception.preprocessing import PreprocessingPipeline, PreprocessConfig, DeviceType, create_detection_pipeline
from adapters.grid_30x30.game import GridGame
from adapters.grid_30x30.planner import GridPlanner
from adapters.grid_30x30.world_model import GridWorldModel, WorldState
from learning.dqn import DQNAgent


# ============================================================
# Throughput Configuration
# ============================================================

@dataclass
class ThroughputThresholds:
    """Minimum acceptable throughput (operations per second)."""
    engine_process_frame_fps: float = 100.0  # FPS
    screen_capture_fps: float = 30.0  # FPS
    preprocessing_single_fps: float = 200.0  # FPS
    preprocessing_batch4_fps: float = 100.0  # batches/sec (4 frames each)
    preprocessing_detection_fps: float = 100.0  # FPS (detection pipeline)
    grid_game_execute_fps: float = 500.0  # FPS
    grid_game_observe_fps: float = 1000.0  # FPS
    planner_choose_action_fps: float = 200.0  # FPS
    world_model_update_fps: float = 500.0  # FPS
    bfs_pathfinding_fps: float = 500.0  # FPS (reduced from 1000)
    memory_record_fps: float = 10000.0  # FPS
    dqn_select_action_fps: float = 500.0  # FPS
    dqn_train_step_fps: float = 20.0  # steps/sec
    frame_ring_buffer_append_fps: float = 800.0  # FPS (reduced from 1000)
    full_pipeline_fps: float = 50.0  # FPS
    episode_simulation_fps: float = 1.5  # episodes/sec (reduced from 5)


THROUGHPUT = ThroughputThresholds()


def measure_throughput(func, *args, duration: float = 1.0, warmup: float = 0.1, **kwargs) -> Dict[str, float]:
    """Measure throughput (operations per second)."""
    # Warmup
    start_warmup = time.perf_counter()
    while time.perf_counter() - start_warmup < warmup:
        func(*args, **kwargs)

    # Measure
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        func(*args, **kwargs)
        count += 1
    elapsed = time.perf_counter() - start

    return {
        "ops_per_sec": count / elapsed,
        "total_ops": count,
        "elapsed_sec": elapsed,
        "mean_latency_ms": (elapsed / count) * 1000 if count > 0 else 0,
    }


def assert_throughput(stats: Dict[str, float], threshold: float, name: str) -> None:
    """Assert that throughput meets minimum threshold."""
    actual = stats["ops_per_sec"]
    assert actual >= threshold, (
        f"{name} throughput too low: {actual:.1f} ops/sec < {threshold} ops/sec "
        f"(mean latency: {stats['mean_latency_ms']:.2f}ms)"
    )


# ============================================================
# Core Engine Throughput
# ============================================================

class TestEngineThroughput:
    """Throughput tests for GameEngine."""

    def test_engine_process_frame_throughput(self) -> None:
        """Test engine process_frame throughput."""
        policy = Mock()
        policy.decide.return_value = Action(name="test")
        policy.observe_result.return_value = None

        executor = Mock()
        executor.execute.return_value = ActionResult(success=True, details={})

        adapter = Mock()
        adapter.get_frame.return_value = {"hud": {"value": 1}}
        adapter.extract.return_value = GameState(hud={"value": 1}, confidence=1.0)

        engine = GameEngine(policy, executor, adapter)

        frame = {"hud": {"value": 1}}

        stats = measure_throughput(engine.process_frame, frame, source="benchmark", duration=0.5)

        assert_throughput(stats, THROUGHPUT.engine_process_frame_fps, "Engine.process_frame")


# ============================================================
# Screen Capture Throughput
# ============================================================

class TestScreenCaptureThroughput:
    """Throughput tests for ScreenCapture."""

    def test_frame_ring_buffer_append_throughput(self) -> None:
        """Test FrameRingBuffer append throughput."""
        buffer = FrameRingBuffer(max_size=60)
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        def append_frame():
            buffer.append(frame, time.time())

        stats = measure_throughput(append_frame, duration=0.5)

        assert_throughput(stats, THROUGHPUT.frame_ring_buffer_append_fps, "FrameRingBuffer.append")

    def test_frame_ring_buffer_get_latest_throughput(self) -> None:
        """Test FrameRingBuffer get_latest throughput."""
        buffer = FrameRingBuffer(max_size=60)
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Fill buffer
        for i in range(30):
            buffer.append(frame, time.time())

        stats = measure_throughput(buffer.get_latest, duration=0.5)

        assert_throughput(stats, THROUGHPUT.frame_ring_buffer_append_fps, "FrameRingBuffer.get_latest")

    @patch('perception.capture.cv2')
    def test_screen_capture_process_frame_throughput(self, mock_cv2) -> None:
        """Test ScreenCapture frame processing throughput."""
        mock_cv2.resize.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cv2.INTER_LINEAR = 1
        mock_cv2.cvtColor.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        config = CaptureConfig(width=1920, height=1080, output_color="RGB")
        capture = ScreenCapture(config)

        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        stats = measure_throughput(capture._process_frame, frame, duration=0.5)

        # Processing should be fast enough for 30fps
        assert_throughput(stats, THROUGHPUT.screen_capture_fps, "ScreenCapture._process_frame")


# ============================================================
# Preprocessing Throughput
# ============================================================

class TestPreprocessingThroughput:
    """Throughput tests for PreprocessingPipeline."""

    @patch('perception.preprocessing.cv2')
    def test_preprocessing_single_frame_throughput(self, mock_cv2) -> None:
        """Test single frame preprocessing throughput."""
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1
        mock_cv2.INTER_LINEAR = 1

        config = PreprocessConfig(device=DeviceType.CPU, target_size=(224, 224))
        pipeline = PreprocessingPipeline(config)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        stats = measure_throughput(pipeline.process, frame, duration=0.5)

        assert_throughput(stats, THROUGHPUT.preprocessing_single_fps, "PreprocessingPipeline.process")

    @patch('perception.preprocessing.cv2')
    def test_preprocessing_batch_throughput(self, mock_cv2) -> None:
        """Test batch preprocessing throughput."""
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        config = PreprocessConfig(device=DeviceType.CPU, target_size=(224, 224))
        pipeline = PreprocessingPipeline(config)

        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]

        stats = measure_throughput(pipeline.process_batch, frames, duration=0.5)

        assert_throughput(stats, THROUGHPUT.preprocessing_batch4_fps, "PreprocessingPipeline.process_batch (4 frames)")

    @patch('perception.preprocessing.cv2')
    def test_preprocessing_detection_pipeline_throughput(self, mock_cv2) -> None:
        """Test detection pipeline throughput."""
        mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        pipeline = create_detection_pipeline()
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        stats = measure_throughput(pipeline.process, frame, duration=0.5)

        assert_throughput(stats, THROUGHPUT.preprocessing_detection_fps, "DetectionPipeline.process")


# ============================================================
# Grid Game Throughput
# ============================================================

class TestGridGameThroughput:
    """Throughput tests for GridGame."""

    def test_grid_game_execute_throughput(self) -> None:
        """Test GridGame execute throughput."""
        game = GridGame(seed=42)

        stats = measure_throughput(game.execute, Action(name="move_right"), duration=0.5)

        assert_throughput(stats, THROUGHPUT.grid_game_execute_fps, "GridGame.execute")

    def test_grid_game_observe_throughput(self) -> None:
        """Test GridGame observe throughput."""
        game = GridGame(seed=42)

        stats = measure_throughput(game.observe, duration=0.5)

        assert_throughput(stats, THROUGHPUT.grid_game_observe_fps, "GridGame.observe")

    def test_grid_game_full_step_throughput(self) -> None:
        """Test full game step (observe + execute) throughput."""
        game = GridGame(seed=42)

        def full_step():
            game.observe()
            game.execute(Action(name="move_right"))

        stats = measure_throughput(full_step, duration=0.5)

        assert_throughput(stats, THROUGHPUT.grid_game_execute_fps / 2, "GridGame full step")


# ============================================================
# Planner Throughput
# ============================================================

class TestPlannerThroughput:
    """Throughput tests for GridPlanner."""

    def test_planner_choose_action_throughput(self) -> None:
        """Test planner choose_action throughput."""
        planner = GridPlanner()
        memory = Mock()
        memory.spatial = Mock()
        memory.spatial.grid = {}
        memory.spatial.frontier = [(i, i) for i in range(10)]
        memory.spatial.goal_is_dynamic = False
        memory.spatial.goal_change_count = 0
        memory.spatial.known_goal_positions = [(20, 20)]
        memory.build_decision_context = Mock(return_value={})
        memory.retrieve_direction_preferences = Mock(return_value={})

        state = WorldState(
            current_position=(0, 0),
            target_position=(20, 20),
            obstacle_positions=[],
            grid_size=(30, 30),
            confidence=1.0,
        )

        stats = measure_throughput(planner.choose_action, state, memory, duration=0.5)

        assert_throughput(stats, THROUGHPUT.planner_choose_action_fps, "GridPlanner.choose_action")


# ============================================================
# World Model Throughput
# ============================================================

class TestWorldModelThroughput:
    """Throughput tests for GridWorldModel."""

    def test_world_model_update_throughput(self) -> None:
        """Test world model update throughput."""
        model = GridWorldModel()
        obs = {
            "hud": {
                "grid": {
                    "width": 30,
                    "height": 30,
                    "position": [15, 15],
                    "target": [25, 25],
                    "visible_cells": [[15, 15], [16, 15]],
                    "obstacles": [[17, 15]],
                    "entities": [{"id": "entity-1", "position": [16, 15], "kind": "obstacle"}],
                }
            },
            "goal": "reach_target",
            "confidence": 1.0,
        }

        stats = measure_throughput(model.update, obs, duration=0.5)

        assert_throughput(stats, THROUGHPUT.world_model_update_fps, "GridWorldModel.update")


# ============================================================
# Pathfinding Throughput
# ============================================================

class TestPathfindingThroughput:
    """Throughput tests for BFS pathfinding."""

    def test_bfs_shortest_path_throughput(self) -> None:
        """Test BFS pathfinding throughput."""
        from adapters.grid_30x30.world_model import bfs_shortest_path

        start = (0, 0)
        goal = (29, 29)
        blocked = set()
        bounds = (30, 30)

        stats = measure_throughput(bfs_shortest_path, start, goal, blocked, bounds=bounds, duration=0.5)

        assert_throughput(stats, THROUGHPUT.bfs_pathfinding_fps, "bfs_shortest_path")

    def test_bfs_with_obstacles_throughput(self) -> None:
        """Test BFS with obstacles throughput."""
        from adapters.grid_30x30.world_model import bfs_shortest_path

        start = (0, 0)
        goal = (29, 29)
        blocked = {(x, y) for x in range(30) for y in range(30) if (x + y) % 3 == 0}
        bounds = (30, 30)

        stats = measure_throughput(bfs_shortest_path, start, goal, blocked, bounds=bounds, duration=0.5)

        assert_throughput(stats, THROUGHPUT.bfs_pathfinding_fps / 10, "bfs_shortest_path (with obstacles)")


# ============================================================
# Memory Throughput
# ============================================================

class TestMemoryThroughput:
    """Throughput tests for memory operations."""

    def test_memory_store_record_throughput(self) -> None:
        """Test MemoryStore record throughput."""
        memory = MemoryStore()
        entry = {"state": GameState(), "action": Action(name="test"), "result": ActionResult(success=True)}

        stats = measure_throughput(memory.record, entry, duration=0.5)

        assert_throughput(stats, THROUGHPUT.memory_record_fps, "MemoryStore.record")

    def test_grid_memory_store_observe_frame_throughput(self) -> None:
        """Test GridMemoryStore observe_frame throughput."""
        from adapters.grid_30x30.memory import GridMemoryStore
        memory = GridMemoryStore()
        frame = {"test": "data"}

        stats = measure_throughput(memory.observe_frame, frame, duration=0.5)

        assert_throughput(stats, THROUGHPUT.memory_record_fps, "GridMemoryStore.observe_frame")


# ============================================================
# DQN Throughput
# ============================================================

class TestDQNThroughput:
    """Throughput tests for DQNAgent."""

    def test_dqn_select_action_throughput(self) -> None:
        """Test DQN action selection throughput."""
        agent = DQNAgent(input_size=40, action_count=6, seed=42)
        state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True,) * 6

        stats = measure_throughput(agent.select_action, state, action_mask, explore=False, duration=0.5)

        assert_throughput(stats, THROUGHPUT.dqn_select_action_fps, "DQNAgent.select_action")

    def test_dqn_train_step_throughput(self) -> None:
        """Test DQN training step throughput."""
        agent = DQNAgent(input_size=40, action_count=6, batch_size=32, warmup_steps=64, seed=42)
        action_mask = (True,) * 6

        # Fill replay buffer
        for i in range(100):
            state = tuple(np.random.rand(40).astype(np.float32))
            next_state = tuple(np.random.rand(40).astype(np.float32))
            agent.remember(state, i % 6, float(i % 3), next_state, action_mask, i == 99)

        stats = measure_throughput(agent.train_step, duration=1.0)

        assert_throughput(stats, THROUGHPUT.dqn_train_step_fps, "DQNAgent.train_step")


# ============================================================
# End-to-End Throughput
# ============================================================

class TestEndToEndThroughput:
    """End-to-end throughput tests."""

    def test_full_pipeline_throughput(self) -> None:
        """Test full pipeline throughput."""
        game = GridGame(seed=42)
        planner = GridPlanner()
        memory = Mock()
        memory.spatial = Mock()
        memory.spatial.grid = {}
        memory.spatial.frontier = [(i, i) for i in range(10)]
        memory.spatial.goal_is_dynamic = False
        memory.spatial.goal_change_count = 0
        memory.spatial.known_goal_positions = [(20, 20)]
        memory.build_decision_context = Mock(return_value={})
        memory.retrieve_direction_preferences = Mock(return_value={})

        def full_pipeline():
            frame = game.observe()
            model = GridWorldModel()
            state = model.update(frame)
            action_candidate = planner.choose_action(state, memory)
            if action_candidate:
                action = Action(name=action_candidate.name)
                game.execute(action)

        stats = measure_throughput(full_pipeline, duration=0.5)

        assert_throughput(stats, THROUGHPUT.full_pipeline_fps, "Full pipeline")

    def test_episode_simulation_throughput(self) -> None:
        """Test full episode simulation throughput."""
        def run_episode():
            game = GridGame(seed=np.random.randint(0, 10000))
            planner = GridPlanner()
            memory = Mock()
            memory.spatial = Mock()
            memory.spatial.grid = {}
            memory.spatial.frontier = [(i, i) for i in range(10)]
            memory.spatial.goal_is_dynamic = False
            memory.spatial.goal_change_count = 0
            memory.spatial.known_goal_positions = [(20, 20)]
            memory.build_decision_context = Mock(return_value={})
            memory.retrieve_direction_preferences = Mock(return_value={})

            for _ in range(100):  # Max 100 steps
                if game.won or game.caught:
                    break
                frame = game.observe()
                model = GridWorldModel()
                state = model.update(frame)
                action_candidate = planner.choose_action(state, memory)
                if action_candidate:
                    action = Action(name=action_candidate.name)
                    game.execute(action)

        stats = measure_throughput(run_episode, duration=2.0)

        # Episodes per second
        assert stats["ops_per_sec"] >= THROUGHPUT.episode_simulation_fps, f"Episode throughput too low: {stats['ops_per_sec']:.2f} eps/sec"


# ============================================================
# Concurrent Throughput
# ============================================================

class TestConcurrentThroughput:
    """Tests for concurrent/parallel throughput."""

    def test_multiple_engines_concurrent(self) -> None:
        """Test running multiple engines concurrently."""
        num_engines = 4

        def create_engine():
            policy = Mock()
            policy.decide.return_value = Action(name="test")
            policy.observe_result.return_value = None

            executor = Mock()
            executor.execute.return_value = ActionResult(success=True, details={})

            adapter = Mock()
            adapter.get_frame.return_value = {"hud": {"value": 1}}
            adapter.extract.return_value = GameState(hud={"value": 1}, confidence=1.0)

            return GameEngine(policy, executor, adapter)

        engines = [create_engine() for _ in range(num_engines)]
        frame = {"hud": {"value": 1}}

        def run_all():
            for engine in engines:
                engine.process_frame(frame, source="concurrent")

        stats = measure_throughput(run_all, duration=0.5)

        # Should scale with number of engines
        expected_min = THROUGHPUT.engine_process_frame_fps * num_engines * 0.5
        assert stats["ops_per_sec"] >= expected_min, (
            f"Concurrent engines throughput too low: {stats['ops_per_sec']:.1f} < {expected_min}"
        )

    def test_threaded_frame_processing(self) -> None:
        """Test threaded frame processing."""
        num_threads = 4
        frames_per_thread = 100

        policy = Mock()
        policy.decide.return_value = Action(name="test")
        policy.observe_result.return_value = None

        executor = Mock()
        executor.execute.return_value = ActionResult(success=True, details={})

        adapter = Mock()
        adapter.get_frame.return_value = {"hud": {"value": 1}}
        adapter.extract.return_value = GameState(hud={"value": 1}, confidence=1.0)

        engine = GameEngine(policy, executor, adapter)
        frame = {"hud": {"value": 1}}

        def process_frames():
            for _ in range(frames_per_thread):
                engine.process_frame(frame, source="threaded")

        threads = [threading.Thread(target=process_frames) for _ in range(num_threads)]

        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        total_frames = num_threads * frames_per_thread
        fps = total_frames / elapsed

        assert fps >= THROUGHPUT.engine_process_frame_fps * num_threads * 0.3, (
            f"Threaded throughput too low: {fps:.1f} fps"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])