from core.engine import GameEngine
from core.types import Action, GameState, Observation


def test_engine_initialization() -> None:
    engine = GameEngine()
    assert engine is not None
    assert engine.memory is not None
    assert engine.planner is not None
    assert engine.input_adapter is not None


def test_mocked_state_produces_click_action() -> None:
    engine = GameEngine()
    frame = {"objects": ["start_button"], "hud": {"status": "menu"}, "goal": "start", "confidence": 0.9}
    observation = Observation(image=frame, timestamp=0.0)

    state = engine.state_extractor.extract(observation)
    action = engine.planner.choose_action(state)

    assert state.objects == ["start_button"]
    assert action is not None
    assert action.name == "click"
    assert action.params["x"] == 500


def test_mocked_loop_keeps_input_adapter_valid() -> None:
    engine = GameEngine()
    action = Action(name="move_left", params={"steps": 2})
    result = engine.input_adapter.execute(action)

    assert result.success is True
    assert result.details["status"] == "moved_left"
    assert result.details["steps"] == 2


def test_no_frame_returns_unknown_state() -> None:
    engine = GameEngine()
    observation = Observation(image=None, timestamp=0.0)
    state = engine.state_extractor.extract(observation)

    assert "unknown" in state.objects
    assert state.goal is None
