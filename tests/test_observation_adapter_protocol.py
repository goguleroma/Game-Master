"""Tests for ObservationAdapter protocol compliance."""

import pytest
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from core.types import GameState, Observation
from core.engine import ObservationAdapter
from core.observation_adapter import ObservationAdapter as ObservationAdapterProtocol


class MockObservationAdapter:
    """Mock adapter that fully implements the ObservationAdapter protocol."""

    def __init__(self) -> None:
        self._frame: Any = None
        self._extract_result: Optional[GameState] = None
        self._reset_count = 0

    def get_frame(self) -> Any:
        return self._frame

    def extract(self, observation: Observation) -> GameState:
        if self._extract_result is not None:
            return self._extract_result
        # Default extraction
        image = observation.image
        if isinstance(image, dict):
            return GameState(
                objects=image.get("objects", ["normalized"]),
                hud=image.get("hud", {"value": "adapted"}),
                goal=image.get("goal"),
                confidence=image.get("confidence", 0.5),
                controls=image.get("controls", []),
            )
        return GameState(objects=["normalized"], hud={"value": "adapted"})

    def reset(self) -> None:
        self._reset_count += 1
        self._frame = None
        self._extract_result = None

    @property
    def is_real_time(self) -> bool:
        return False

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return (0, 0, 0)


class MinimalObservationAdapter:
    """Minimal adapter implementing only required methods."""

    def get_frame(self) -> Any:
        return {"test": "data"}

    def extract(self, observation: Observation) -> GameState:
        return GameState(objects=["extracted"])

    def reset(self) -> None:
        pass

    @property
    def is_real_time(self) -> bool:
        return False

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return (480, 640, 3)


def test_observation_adapter_protocol_structure() -> None:
    """Test that the protocol has all required methods and properties."""
    # Check the protocol defines the expected interface
    assert hasattr(ObservationAdapterProtocol, 'get_frame')
    assert hasattr(ObservationAdapterProtocol, 'extract')
    assert hasattr(ObservationAdapterProtocol, 'reset')
    assert hasattr(ObservationAdapterProtocol, 'is_real_time')
    assert hasattr(ObservationAdapterProtocol, 'frame_shape')


def test_mock_adapter_implements_protocol() -> None:
    """Test that MockObservationAdapter implements all protocol methods."""
    adapter = MockObservationAdapter()

    # Test get_frame
    adapter._frame = {"test": "frame"}
    assert adapter.get_frame() == {"test": "frame"}

    # Test extract
    obs = Observation(image={"objects": ["obj1"], "hud": {"hp": 100}}, timestamp=1.0, metadata={})
    state = adapter.extract(obs)
    assert isinstance(state, GameState)
    assert state.objects == ["obj1"]
    assert state.hud == {"hp": 100}

    # Test reset
    adapter.reset()
    assert adapter._reset_count == 1
    assert adapter.get_frame() is None

    # Test properties
    assert adapter.is_real_time is False
    assert adapter.frame_shape == (0, 0, 0)


def test_minimal_adapter_implements_protocol() -> None:
    """Test that MinimalObservationAdapter implements all protocol methods."""
    adapter = MinimalObservationAdapter()

    # Test get_frame
    frame = adapter.get_frame()
    assert frame == {"test": "data"}

    # Test extract
    obs = Observation(image="anything", timestamp=1.0, metadata={})
    state = adapter.extract(obs)
    assert isinstance(state, GameState)
    assert state.objects == ["extracted"]

    # Test reset (should not raise)
    adapter.reset()

    # Test properties
    assert adapter.is_real_time is False
    assert adapter.frame_shape == (480, 640, 3)


def test_observation_adapter_with_engine() -> None:
    """Test that adapters work with GameEngine."""
    from core.engine import GameEngine
    from core.memory import MemoryStore
    from core.types import Action, ActionResult

    class TestPolicy:
        def decide(self, observation: GameState, memory: Any) -> Action | None:
            return Action(name="test_action", params={"value": 42})

        def observe_result(self, result: ActionResult, memory: Any) -> None:
            pass

    class TestExecutor:
        def execute(self, action: Action) -> ActionResult:
            return ActionResult(success=True, details={"executed": action.name})

    adapter = MockObservationAdapter()
    adapter._frame = {"objects": ["test"], "hud": {"value": 123}}
    adapter.set_extract_result = lambda s: setattr(adapter, '_extract_result', s)
    adapter.set_extract_result(GameState(
        objects=["test"],
        hud={"value": 123},
        goal="test_goal",
        confidence=1.0,
    ))

    engine = GameEngine(TestPolicy(), TestExecutor(), adapter, max_steps=1)
    result = engine.process_frame({"objects": ["test"], "hud": {"value": 123}}, source="test")

    assert result is not None
    assert result.success
    assert result.details["executed"] == "test_action"


def test_observation_adapter_extract_handles_different_inputs() -> None:
    """Test extract handles various observation formats."""
    adapter = MockObservationAdapter()

    # Test with dict image
    obs1 = Observation(image={"objects": ["a", "b"], "hud": {"score": 10}}, timestamp=1.0, metadata={})
    state1 = adapter.extract(obs1)
    assert state1.objects == ["a", "b"]
    assert state1.hud == {"score": 10}

    # Test with non-dict image
    obs2 = Observation(image=np_array_mock(), timestamp=1.0, metadata={})
    state2 = adapter.extract(obs2)
    assert state2.objects == ["normalized"]
    assert state2.hud == {"value": "adapted"}

    # Test with custom extract result
    custom_state = GameState(objects=["custom"], hud={"custom": True}, confidence=0.9)
    adapter._extract_result = custom_state
    obs3 = Observation(image={}, timestamp=1.0, metadata={})
    state3 = adapter.extract(obs3)
    assert state3.objects == ["custom"]
    assert state3.confidence == 0.9


def test_observation_adapter_reset_clears_state() -> None:
    """Test that reset clears internal state."""
    adapter = MockObservationAdapter()
    adapter._frame = {"data": "test"}
    adapter._extract_result = GameState(objects=["test"])

    adapter.reset()

    assert adapter.get_frame() is None
    assert adapter._extract_result is None
    assert adapter._reset_count == 1


# Helper for testing
class np_array_mock:
    """Mock numpy array for testing."""
    def __init__(self):
        self.shape = (480, 640, 3)
    def __array__(self):
        return self


def test_observation_adapter_frame_shape_property() -> None:
    """Test frame_shape property returns expected format."""
    adapter = MockObservationAdapter()
    shape = adapter.frame_shape
    assert isinstance(shape, tuple)
    assert len(shape) >= 2  # At least (H, W) or (H, W, C)


def test_observation_adapter_is_real_time_property() -> None:
    """Test is_real_time property returns boolean."""
    adapter = MockObservationAdapter()
    assert isinstance(adapter.is_real_time, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])