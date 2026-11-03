"""Shared test fixtures and configuration for Game Master tests."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock
from typing import Generator, Any, Dict, List
from dataclasses import dataclass

from core.types import Action, ActionResult, GameState, Observation
from core.memory import MemoryStore
from core.engine import GameEngine
from adapters.grid_30x30.game import GridGame
from adapters.grid_30x30.planner import GridPlanner
from adapters.grid_30x30.world_model import GridWorldModel, WorldState
from adapters.grid_30x30.memory import GridMemoryStore
from learning.dqn import DQNAgent


# ============================================================
# Pytest Configuration
# ============================================================

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
    config.addinivalue_line("markers", "unit: marks unit tests")


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Auto-mark tests based on name patterns."""
    for item in items:
        # Mark benchmark tests
        if "benchmark" in item.nodeid.lower():
            item.add_marker(pytest.mark.benchmark)
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "directml" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "engine_integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        # Mark slow tests
        if "slow" in item.name or "transfer" in item.name or "consolidation_training" in item.name:
            item.add_marker(pytest.mark.slow)


# ============================================================
# Core Fixtures
# ============================================================

@pytest.fixture
def mock_policy() -> Mock:
    """Create a mock decision policy."""
    policy = Mock()
    policy.decide = Mock(return_value=Action(name="test_action", params={}))
    policy.observe_result = Mock(return_value=None)
    return policy


@pytest.fixture
def mock_executor() -> Mock:
    """Create a mock action executor."""
    executor = Mock()
    executor.execute = Mock(return_value=ActionResult(success=True, details={"status": "executed"}))
    return executor


@pytest.fixture
def mock_adapter() -> Mock:
    """Create a mock observation adapter."""
    adapter = Mock()
    adapter.get_frame = Mock(return_value={"hud": {"value": 1}})
    adapter.extract = Mock(return_value=GameState(hud={"value": 1}, confidence=1.0))
    adapter.reset = Mock()
    adapter.is_real_time = False
    adapter.frame_shape = (0, 0, 0)
    return adapter


@pytest.fixture
def game_engine(mock_policy: Mock, mock_executor: Mock, mock_adapter: Mock) -> GameEngine:
    """Create a GameEngine with mocked components."""
    return GameEngine(mock_policy, mock_executor, mock_adapter, max_steps=10)


@pytest.fixture
def memory_store() -> MemoryStore:
    """Create a fresh MemoryStore."""
    return MemoryStore()


@pytest.fixture
def grid_game() -> GridGame:
    """Create a GridGame with fixed seed."""
    return GridGame(seed=42)


@pytest.fixture
def grid_planner() -> GridPlanner:
    """Create a GridPlanner with default strategy."""
    return GridPlanner()


@pytest.fixture
def grid_world_model() -> GridWorldModel:
    """Create a GridWorldModel."""
    return GridWorldModel()


@pytest.fixture
def grid_memory_store() -> GridMemoryStore:
    """Create a GridMemoryStore."""
    return GridMemoryStore(game_id="test_game")


@pytest.fixture
def dqn_agent() -> DQNAgent:
    """Create a DQNAgent for testing."""
    return DQNAgent(input_size=40, action_count=6, seed=42, warmup_steps=10, batch_size=16)


# ============================================================
# Observation Fixtures
# ============================================================

@pytest.fixture
def sample_observation() -> Dict[str, Any]:
    """Sample observation dictionary."""
    return {
        "objects": ["player", "enemy", "item"],
        "controls": ["move_up", "move_down", "move_left", "move_right"],
        "hud": {
            "health": 100,
            "score": 1000,
            "level": 5,
        },
        "goal": "reach_exit",
        "confidence": 0.95,
    }


@pytest.fixture
def sample_grid_observation() -> Dict[str, Any]:
    """Sample GridGame observation."""
    return {
        "objects": ["player", "goal", "obstacle"],
        "controls": [f"control_{i}" for i in range(1, 7)],
        "hud": {
            "status": "playing",
            "health": 20,
            "grid": {
                "width": 30,
                "height": 30,
                "position": [10, 10],
                "vision_radius": 4,
                "visible_cells": [[10, 10], [11, 10], [10, 11]],
                "target": [20, 20],
                "obstacles": [[12, 10]],
                "entities": [
                    {"id": "entity-1", "position": [11, 10], "kind": "obstacle"},
                    {"id": "entity-2", "position": [15, 15], "kind": "enemy", "health": 8},
                ],
            },
        },
        "goal": "reach_goal",
        "confidence": 1.0,
    }


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Sample frame as numpy array."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_hud_frame() -> np.ndarray:
    """Sample HUD region frame."""
    return np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)


# ============================================================
# State Fixtures
# ============================================================

@pytest.fixture
def sample_game_state() -> GameState:
    """Sample GameState."""
    return GameState(
        objects=["player", "enemy"],
        hud={"health": 100, "score": 500},
        goal="defeat_enemy",
        confidence=0.9,
        controls=["attack", "defend", "move"],
    )


@pytest.fixture
def sample_world_state() -> WorldState:
    """Sample WorldState."""
    return WorldState(
        current_position=(10, 10),
        target_position=(20, 20),
        obstacle_positions=[(12, 10), (13, 10)],
        grid_size=(30, 30),
        confidence=1.0,
        goal="reach_target",
    )


# ============================================================
# Mock Fixtures for External Dependencies
# ============================================================

@pytest.fixture
def mock_cv2(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock OpenCV."""
    mock = Mock()
    mock.cvtColor = Mock(return_value=np.zeros((224, 224, 3), dtype=np.uint8))
    mock.resize = Mock(return_value=np.zeros((224, 224, 3), dtype=np.uint8))
    mock.threshold = Mock(return_value=(None, np.zeros((100, 100), dtype=np.uint8)))
    mock.findContours = Mock(return_value=([], None))
    mock.boundingRect = Mock(return_value=(0, 0, 10, 10))
    mock.contourArea = Mock(return_value=100)
    mock.getStructuringElement = Mock(return_value=np.ones((5, 5), dtype=np.uint8))
    mock.morphologyEx = Mock(side_effect=lambda *args, **kwargs: args[0])
    mock.COLOR_BGR2RGB = 1
    mock.COLOR_BGR2GRAY = 1
    mock.COLOR_GRAY2BGR = 1
    mock.INTER_LINEAR = 1
    mock.INTER_CUBIC = 1
    mock.RETR_EXTERNAL = 1
    mock.CHAIN_APPROX_SIMPLE = 1
    mock.MORPH_ELLIPSE = 1
    mock.MORPH_OPEN = 1
    mock.MORPH_CLOSE = 1
    mock.THRESH_BINARY = 1
    mock.THRESH_OTSU = 1
    mock.fastNlMeansDenoising = Mock(return_value=np.zeros((100, 100), dtype=np.uint8))
    mock.createCLAHE = Mock(return_value=Mock(apply=Mock(return_value=np.zeros((100, 100), dtype=np.uint8))))

    monkeypatch.setattr("cv2", mock)
    monkeypatch.setattr("perception.capture.cv2", mock)
    monkeypatch.setattr("perception.preprocessing.cv2", mock)
    monkeypatch.setattr("perception.ocr.cv2", mock)
    monkeypatch.setattr("perception.detection.cv2", mock)
    return mock


@pytest.fixture
def mock_torch(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock PyTorch for CPU-only testing."""
    # Don't mock torch entirely, just ensure CPU device
    original_device = torch.device
    def cpu_device(*args, **kwargs):
        if args and args[0] in ("cuda", "directml", "privateuseone"):
            return torch.device("cpu")
        return original_device(*args, **kwargs)
    monkeypatch.setattr(torch, "device", cpu_device)
    return torch


@pytest.fixture
def mock_ort(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock ONNX Runtime."""
    mock = Mock()
    mock_session = Mock()
    mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 3, 640, 640])]
    mock_session.get_outputs.return_value = [Mock(name="output")]
    mock_session.get_modelmeta.return_value = Mock(custom_metadata_map={})
    mock_session.run.return_value = [np.random.rand(1, 84, 8400).astype(np.float32)]
    mock.InferenceSession = Mock(return_value=mock_session)
    mock.SessionOptions = Mock(return_value=Mock())
    mock.GraphOptimizationLevel = Mock(ORT_ENABLE_ALL=1)
    mock.get_available_providers = Mock(return_value=["CPUExecutionProvider"])

    monkeypatch.setattr("onnxruntime", mock)
    monkeypatch.setattr("perception.detection.ort", mock)
    return mock


@pytest.fixture
def mock_paddleocr(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock PaddleOCR."""
    mock = Mock()
    mock_ocr = Mock()
    mock_ocr.ocr.return_value = [[
        [[[10, 20], [110, 20], [110, 50], [10, 50]], ("HP: 100/100", 0.95)],
    ]]
    mock.PaddleOCR = Mock(return_value=mock_ocr)

    monkeypatch.setattr("paddleocr", mock)
    monkeypatch.setattr("perception.ocr.PaddleOCR", mock.PaddleOCR)
    monkeypatch.setattr("perception.ocr.PADDLEOCR_AVAILABLE", True)
    return mock


@pytest.fixture
def mock_tesseract(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock Tesseract."""
    mock = Mock()
    mock.get_tesseract_version = Mock(return_value="5.0.0")
    mock.image_to_data = Mock(return_value={
        "text": ["HP:", "100/100"],
        "conf": ["95", "95"],
        "left": [10, 40],
        "top": [20, 20],
        "width": [30, 60],
        "height": [30, 30],
    })
    mock.Output = Mock(DICT="dict")

    monkeypatch.setattr("pytesseract", mock)
    monkeypatch.setattr("perception.ocr.pytesseract", mock)
    monkeypatch.setattr("perception.ocr.TESSERACT_AVAILABLE", True)
    return mock


@pytest.fixture
def mock_dxcam(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock dxcam."""
    mock = Mock()
    mock_camera = Mock()
    mock_camera.grab.return_value = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    mock_camera.stop = Mock()
    mock.create = Mock(return_value=mock_camera)

    monkeypatch.setattr("dxcam", mock)
    monkeypatch.setattr("perception.capture.dxcam", mock)
    monkeypatch.setattr("perception.capture.DXCAM_AVAILABLE", True)
    return mock


@pytest.fixture
def mock_mss(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock mss."""
    mock = Mock()
    mock_instance = Mock()
    mock_instance.monitors = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
    ]
    mock_instance.grab.return_value = np.random.randint(0, 255, (1080, 1920, 4), dtype=np.uint8)
    mock_instance.close = Mock()
    mock.mss = Mock(return_value=mock_instance)

    monkeypatch.setattr("mss", mock)
    monkeypatch.setattr("perception.capture.mss", mock)
    monkeypatch.setattr("perception.capture.MSS_AVAILABLE", True)
    return mock


# ============================================================
# Benchmark Fixtures
# ============================================================

@pytest.fixture
def latency_thresholds() -> Dict[str, float]:
    """Latency thresholds for benchmark assertions."""
    return {
        "engine_process_frame": 10.0,
        "screen_capture_get_frame": 5.0,
        "preprocessing_single": 5.0,
        "preprocessing_batch_4": 10.0,
        "detection_inference": 50.0,
        "grid_game_execute": 2.0,
        "grid_planner_choose_action": 5.0,
        "world_model_update": 2.0,
        "bfs_pathfinding": 1.0,
        "memory_record": 0.5,
        "dqn_select_action": 2.0,
        "dqn_train_step": 50.0,
    }


@pytest.fixture
def throughput_thresholds() -> Dict[str, float]:
    """Throughput thresholds for benchmark assertions."""
    return {
        "engine_process_frame_fps": 100.0,
        "preprocessing_single_fps": 200.0,
        "preprocessing_batch4_fps": 100.0,
        "grid_game_execute_fps": 500.0,
        "planner_choose_action_fps": 200.0,
        "dqn_select_action_fps": 500.0,
        "dqn_train_step_fps": 20.0,
    }


# ============================================================
# Test Data Fixtures
# ============================================================

@pytest.fixture
def transitions_batch() -> List[Dict[str, Any]]:
    """Batch of transitions for DQN testing."""
    return [
        {
            "state": tuple(np.random.rand(40).astype(np.float32)),
            "action": np.random.randint(0, 6),
            "reward": float(np.random.randint(-1, 2)),
            "next_state": tuple(np.random.rand(40).astype(np.float32)),
            "next_action_mask": (True,) * 6,
            "done": False,
        }
        for _ in range(32)
    ]


@pytest.fixture
def demo_samples() -> List[tuple]:
    """Demonstration samples for behavior cloning."""
    return [
        (tuple(np.random.rand(40).astype(np.float32)), np.random.randint(0, 6), (True,) * 6)
        for _ in range(10)
    ]


# ============================================================
# Cleanup Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def cleanup_torch_cuda():
    """Cleanup CUDA cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield


# ============================================================
# Helper Functions
# ============================================================

def assert_latency_below(stats: Dict[str, float], threshold: float, name: str) -> None:
    """Assert mean latency is below threshold."""
    assert stats["mean"] < threshold, (
        f"{name} latency too high: mean={stats['mean']:.2f}ms > {threshold}ms "
        f"(p95={stats['p95']:.2f}ms, p99={stats['p99']:.2f}ms)"
    )


def assert_throughput_above(stats: Dict[str, float], threshold: float, name: str) -> None:
    """Assert throughput is above threshold."""
    actual = stats["ops_per_sec"]
    assert actual >= threshold, (
        f"{name} throughput too low: {actual:.1f} ops/sec < {threshold} ops/sec "
        f"(mean latency: {stats['mean_latency_ms']:.2f}ms)"
    )


# ============================================================
# Pytest Hooks
# ============================================================

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip GPU tests if no GPU available."""
    if item.get_closest_marker("gpu"):
        # Check for GPU availability
        has_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
            try:
                import torch_directml
                if torch_directml.is_available():
                    has_gpu = True
            except ImportError:
                pass
        except ImportError:
            pass

        if not has_gpu:
            pytest.skip("GPU not available")


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Print custom summary."""
    if exitstatus == 0:
        terminalreporter.write_sep("=", "All tests passed!")
    else:
        terminalreporter.write_sep("=", "Some tests failed!")