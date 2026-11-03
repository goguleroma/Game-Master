"""Latency benchmarks with assertions for Game Master components."""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from dataclasses import dataclass

from core.engine import GameEngine
from core.memory import MemoryStore
from core.types import Action, ActionResult, GameState, Observation
from core.observation_adapter import ObservationAdapter
from perception.capture import ScreenCapture, CaptureConfig, FrameRingBuffer
from perception.preprocessing import PreprocessingPipeline, PreprocessConfig, DeviceType, create_detection_pipeline
from perception.ocr import OCRProcessor, OCRConfig, OCREngine
from perception.detection import DetectionConfig, YOLODetector, ModelType
from adapters.grid_30x30.game import GridGame
from adapters.grid_30x30.planner import GridPlanner
from adapters.grid_30x30.world_model import GridWorldModel, WorldState, bfs_shortest_path
from adapters.grid_30x30.memory import GridMemoryStore
from learning.dqn import DQNAgent


# ============================================================
# Benchmark Configuration
# ============================================================

@dataclass
class LatencyThresholds:
    """Maximum acceptable latencies in milliseconds."""
    engine_process_frame: float = 10.0  # ms
    screen_capture_get_frame: float = 5.0
    frame_ring_buffer_append: float = 2.0
    frame_ring_buffer_diff: float = 50.0
    preprocessing_single: float = 5.0  # CPU
    preprocessing_batch_4: float = 10.0  # CPU batch of 4
    preprocessing_detection_pipeline: float = 10.0  # Detection pipeline
    ocr_recognize: float = 100.0  # Can be slow
    detection_inference: float = 50.0  # ONNX inference
    grid_game_execute: float = 2.0
    grid_planner_choose_action: float = 5.0
    world_model_update: float = 2.0
    bfs_pathfinding: float = 2.0
    memory_store_record: float = 0.5
    dqn_select_action: float = 2.0
    dqn_train_step: float = 50.0


THRESHOLDS = LatencyThresholds()


def benchmark_function(func, *args, iterations: int = 100, warmup: int = 10, **kwargs) -> Dict[str, float]:
    """Run benchmark and return statistics."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
    }


def assert_latency(stats: Dict[str, float], threshold: float, name: str) -> None:
    """Assert that mean latency is below threshold."""
    assert stats["mean"] < threshold, (
        f"{name} latency too high: mean={stats['mean']:.2f}ms > {threshold}ms "
        f"(p95={stats['p95']:.2f}ms, p99={stats['p99']:.2f}ms)"
    )


# ============================================================
# Core Engine Benchmarks
# ============================================================

class TestEngineLatency:
    """Benchmarks for GameEngine."""

    def test_engine_process_frame_latency(self) -> None:
        """Benchmark engine process_frame latency."""
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

        stats = benchmark_function(engine.process_frame, frame, source="benchmark")

        assert_latency(stats, THRESHOLDS.engine_process_frame, "Engine.process_frame")

    def test_engine_process_frame_with_memory_latency(self) -> None:
        """Benchmark engine with memory recording."""
        policy = Mock()
        policy.decide.return_value = Action(name="test")
        policy.observe_result.return_value = None

        executor = Mock()
        executor.execute.return_value = ActionResult(success=True, details={})

        adapter = Mock()
        adapter.get_frame.return_value = {"hud": {"value": 1}}
        adapter.extract.return_value = GameState(hud={"value": 1}, confidence=1.0)

        memory = MemoryStore()
        engine = GameEngine(policy, executor, adapter, memory=memory)

        frame = {"hud": {"value": 1}}

        stats = benchmark_function(engine.process_frame, frame, source="benchmark")

        assert_latency(stats, THRESHOLDS.engine_process_frame * 2, "Engine.process_frame (with memory)")


# ============================================================
# Screen Capture Benchmarks
# ============================================================

class TestScreenCaptureLatency:
    """Benchmarks for ScreenCapture components."""

    def test_frame_ring_buffer_append_latency(self) -> None:
        """Benchmark FrameRingBuffer append."""
        buffer = FrameRingBuffer(max_size=60)
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        def append_frame():
            buffer.append(frame, time.time())

        stats = benchmark_function(append_frame, iterations=1000)
        assert_latency(stats, THRESHOLDS.frame_ring_buffer_append, "FrameRingBuffer.append")

    def test_frame_ring_buffer_diff_latency(self) -> None:
        """Benchmark FrameRingBuffer frame diff."""
        buffer = FrameRingBuffer(max_size=60)
        frame1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        buffer.append(frame1, time.time())
        buffer.append(frame2, time.time())

        def get_diff():
            buffer.get_frame_diff(1)

        stats = benchmark_function(get_diff, iterations=100)
        assert_latency(stats, THRESHOLDS.frame_ring_buffer_diff, "FrameRingBuffer.get_frame_diff")

    @patch('perception.capture.cv2')
    def test_screen_capture_process_frame_latency(self, mock_cv2) -> None:
        """Benchmark ScreenCapture frame processing."""
        mock_cv2.resize.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cv2.INTER_LINEAR = 1
        mock_cv2.cvtColor.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        config = CaptureConfig(width=1920, height=1080, output_color="RGB")
        capture = ScreenCapture(config)

        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        stats = benchmark_function(capture._process_frame, frame, iterations=100)
        assert_latency(stats, 5.0, "ScreenCapture._process_frame")


# ============================================================
# Preprocessing Benchmarks
# ============================================================

class TestPreprocessingLatency:
    """Benchmarks for PreprocessingPipeline."""

    @patch('perception.preprocessing.cv2')
    def test_preprocessing_single_frame_cpu(self, mock_cv2) -> None:
        """Benchmark single frame preprocessing on CPU."""
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1
        mock_cv2.INTER_LINEAR = 1

        config = PreprocessConfig(device=DeviceType.CPU, target_size=(224, 224))
        pipeline = PreprocessingPipeline(config)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        stats = benchmark_function(pipeline.process, frame, iterations=100)
        assert_latency(stats, THRESHOLDS.preprocessing_single, "PreprocessingPipeline.process (CPU)")

    @patch('perception.preprocessing.cv2')
    def test_preprocessing_batch_cpu(self, mock_cv2) -> None:
        """Benchmark batch preprocessing on CPU."""
        mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        config = PreprocessConfig(device=DeviceType.CPU, target_size=(224, 224))
        pipeline = PreprocessingPipeline(config)

        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]

        stats = benchmark_function(pipeline.process_batch, frames, iterations=50)
        assert_latency(stats, THRESHOLDS.preprocessing_batch_4, "PreprocessingPipeline.process_batch (CPU, 4 frames)")

    @patch('perception.preprocessing.cv2')
    def test_preprocessing_detection_pipeline(self, mock_cv2) -> None:
        """Benchmark detection-optimized pipeline."""
        mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 1

        pipeline = create_detection_pipeline()
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        stats = benchmark_function(pipeline.process, frame, iterations=100)
        assert_latency(stats, THRESHOLDS.preprocessing_detection_pipeline, "DetectionPipeline.process")


# ============================================================
# OCR Benchmarks
# ============================================================

class TestOCRLatency:
    """Benchmarks for OCR processing."""

    @patch('perception.ocr.PaddleOCREngine')
    def test_ocr_recognize_latency(self, mock_engine_class) -> None:
        """Benchmark OCR recognize."""
        mock_engine = Mock()
        mock_engine.initialize.return_value = True
        mock_engine.recognize.return_value = []
        mock_engine_class.return_value = mock_engine

        config = OCRConfig(engine=OCREngine.PADDLEOCR)
        processor = OCRProcessor(config)
        processor.initialize()

        frame = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)

        stats = benchmark_function(processor.recognize, frame, iterations=10)
        assert_latency(stats, THRESHOLDS.ocr_recognize, "OCRProcessor.recognize")

    @patch('perception.ocr.PaddleOCREngine')
    def test_ocr_extract_hud_latency(self, mock_engine_class) -> None:
        """Benchmark OCR HUD extraction."""
        mock_engine = Mock()
        mock_engine.initialize.return_value = True
        mock_engine.recognize.return_value = []
        mock_engine_class.return_value = mock_engine

        config = OCRConfig(engine=OCREngine.PADDLEOCR)
        processor = OCRProcessor(config)
        processor.initialize()

        frame = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)

        stats = benchmark_function(processor.extract_hud, frame, iterations=10)
        assert_latency(stats, THRESHOLDS.ocr_recognize * 1.5, "OCRProcessor.extract_hud")


# ============================================================
# Detection Benchmarks
# ============================================================

class TestDetectionLatency:
    """Benchmarks for object detection."""

    @patch('perception.detection.ort')
    @patch('perception.detection.Path.exists')
    def test_detection_inference_latency(self, mock_exists, mock_ort) -> None:
        """Benchmark detection inference."""
        mock_exists.return_value = True
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 3, 640, 640])]
        mock_session.get_outputs.return_value = [Mock(name="output")]
        mock_session.get_modelmeta.return_value = Mock(custom_metadata_map={})
        mock_session.run.return_value = [np.random.rand(1, 84, 8400).astype(np.float32)]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = Mock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 1

        config = DetectionConfig(model_path="model.onnx", confidence_threshold=0.25)
        detector = YOLODetector(config)
        detector.initialize()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        stats = benchmark_function(detector.detect, frame, iterations=20)
        assert_latency(stats, THRESHOLDS.detection_inference, "YOLODetector.detect")


# ============================================================
# Grid Game Benchmarks
# ============================================================

class TestGridGameLatency:
    """Benchmarks for GridGame."""

    def test_grid_game_execute_latency(self) -> None:
        """Benchmark GridGame execute."""
        game = GridGame(seed=42)

        stats = benchmark_function(game.execute, Action(name="move_right"), iterations=1000)
        assert_latency(stats, THRESHOLDS.grid_game_execute, "GridGame.execute")

    def test_grid_game_observe_latency(self) -> None:
        """Benchmark GridGame observe."""
        game = GridGame(seed=42)

        stats = benchmark_function(game.observe, iterations=1000)
        assert_latency(stats, THRESHOLDS.grid_game_execute, "GridGame.observe")


# ============================================================
# Planner Benchmarks
# ============================================================

class TestPlannerLatency:
    """Benchmarks for GridPlanner."""

    def test_planner_choose_action_latency(self) -> None:
        """Benchmark planner choose_action."""
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

        stats = benchmark_function(planner.choose_action, state, memory, iterations=100)
        assert_latency(stats, THRESHOLDS.grid_planner_choose_action, "GridPlanner.choose_action")


# ============================================================
# World Model Benchmarks
# ============================================================

class TestWorldModelLatency:
    """Benchmarks for GridWorldModel."""

    def test_world_model_update_latency(self) -> None:
        """Benchmark world model update."""
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

        stats = benchmark_function(model.update, obs, iterations=1000)
        assert_latency(stats, THRESHOLDS.world_model_update, "GridWorldModel.update")


# ============================================================
# Pathfinding Benchmarks
# ============================================================

class TestPathfindingLatency:
    """Benchmarks for BFS pathfinding."""

    def test_bfs_shortest_path_latency(self) -> None:
        """Benchmark BFS pathfinding."""
        start = (0, 0)
        goal = (29, 29)
        blocked = set()
        bounds = (30, 30)

        stats = benchmark_function(bfs_shortest_path, start, goal, blocked, bounds=bounds, iterations=100)
        assert_latency(stats, THRESHOLDS.bfs_pathfinding, "bfs_shortest_path")

    def test_bfs_with_obstacles_latency(self) -> None:
        """Benchmark BFS with many obstacles."""
        start = (0, 0)
        goal = (29, 29)
        blocked = {(x, y) for x in range(30) for y in range(30) if (x + y) % 3 == 0}
        bounds = (30, 30)

        stats = benchmark_function(bfs_shortest_path, start, goal, blocked, bounds=bounds, iterations=50)
        assert_latency(stats, THRESHOLDS.bfs_pathfinding * 5, "bfs_shortest_path (with obstacles)")


# ============================================================
# Memory Benchmarks
# ============================================================

class TestMemoryLatency:
    """Benchmarks for memory operations."""

    def test_memory_store_record_latency(self) -> None:
        """Benchmark MemoryStore record."""
        memory = MemoryStore()
        entry = {"state": GameState(), "action": Action(name="test"), "result": ActionResult(success=True)}

        stats = benchmark_function(memory.record, entry, iterations=10000)
        assert_latency(stats, THRESHOLDS.memory_store_record, "MemoryStore.record")

    def test_grid_memory_store_observe_frame_latency(self) -> None:
        """Benchmark GridMemoryStore observe_frame."""
        memory = GridMemoryStore()
        frame = {"test": "data"}

        stats = benchmark_function(memory.observe_frame, frame, iterations=10000)
        assert_latency(stats, THRESHOLDS.memory_store_record, "GridMemoryStore.observe_frame")


# ============================================================
# DQN Benchmarks
# ============================================================

class TestDQNLatency:
    """Benchmarks for DQNAgent."""

    def test_dqn_select_action_latency(self) -> None:
        """Benchmark DQN action selection."""
        agent = DQNAgent(input_size=40, action_count=6, seed=42)
        state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True,) * 6

        stats = benchmark_function(agent.select_action, state, action_mask, explore=False, iterations=1000)
        assert_latency(stats, THRESHOLDS.dqn_select_action, "DQNAgent.select_action")

    def test_dqn_train_step_latency(self) -> None:
        """Benchmark DQN training step."""
        agent = DQNAgent(input_size=40, action_count=6, batch_size=32, warmup_steps=64, seed=42)
        action_mask = (True,) * 6

        # Fill replay buffer
        for i in range(100):
            state = tuple(np.random.rand(40).astype(np.float32))
            next_state = tuple(np.random.rand(40).astype(np.float32))
            agent.remember(state, i % 6, float(i % 3), next_state, action_mask, i == 99)

        stats = benchmark_function(agent.train_step, iterations=10)
        assert_latency(stats, THRESHOLDS.dqn_train_step, "DQNAgent.train_step")


# ============================================================
# End-to-End Latency
# ============================================================

class TestEndToEndLatency:
    """End-to-end latency benchmarks."""

    def test_full_pipeline_latency(self) -> None:
        """Benchmark full observation -> decision -> action pipeline."""
        # Setup minimal pipeline
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
            # Observe
            frame = game.observe()
            # Update world model
            model = GridWorldModel()
            state = model.update(frame)
            # Plan
            action_candidate = planner.choose_action(state, memory)
            if action_candidate:
                # Execute
                action = Action(name=action_candidate.name)
                game.execute(action)

        stats = benchmark_function(full_pipeline, iterations=100)
        # Full pipeline should be under 20ms
        assert_latency(stats, 20.0, "Full pipeline (observe -> plan -> act)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])