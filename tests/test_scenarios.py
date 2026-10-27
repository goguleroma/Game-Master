from core.engine import GameEngine
from core.types import Observation


def test_scenario_menu_start_button() -> None:
    engine = GameEngine()
    frame = {
        "objects": ["start_button"],
        "hud": {"status": "menu"},
        "goal": "start",
        "confidence": 0.95,
    }

    state = engine.state_extractor.extract(Observation(image=frame, timestamp=0.0))
    action = engine.planner.choose_action(state)
    result = engine.input_adapter.execute(action)

    assert state.goal == "start"
    assert action is not None
    assert action.name == "click"
    assert result.success is True
    assert result.details["status"] == "clicked"


def test_scenario_no_data() -> None:
    engine = GameEngine()
    state = engine.state_extractor.extract(Observation(image=None, timestamp=0.0))
    action = engine.planner.choose_action(state)

    assert "unknown" in state.objects
    assert action is not None
    assert action.name == "wait"


def test_scenario_move_left_goal() -> None:
    engine = GameEngine()
    frame = {
        "objects": ["player"],
        "hud": {"status": "playing"},
        "goal": "move_left",
        "confidence": 0.9,
    }

    state = engine.state_extractor.extract(Observation(image=frame, timestamp=0.0))
    action = engine.planner.choose_action(state)
    result = engine.input_adapter.execute(action)

    assert state.goal == "move_left"
    assert action is not None
    assert action.name == "move_left"
    assert result.success is True
    assert result.details["status"] == "moved_left"


def test_scenario_unknown_object() -> None:
    engine = GameEngine()
    frame = {
        "objects": ["unknown"],
        "hud": {"status": "unclear"},
        "goal": None,
        "confidence": 0.1,
    }

    state = engine.state_extractor.extract(Observation(image=frame, timestamp=0.0))
    action = engine.planner.choose_action(state)

    assert "unknown" in state.objects
    assert action is not None
    assert action.name == "wait"


def test_unseen_mixed_signals_produce_and_record_a_response() -> None:
    engine = GameEngine()
    frame = {
        "objects": ["player", "start_button", "key", "hazard"],
        "hud": {"status": "playing", "health": 1},
        "goal": "move_left",
        "confidence": 0.55,
    }

    result = engine.process_frame(frame, source="test")

    assert result is not None
    assert len(engine.memory.records) == 1
    record = engine.memory.records[0]
    assert record["state"].objects == frame["objects"]
    assert record["state"].hud == frame["hud"]
    assert record["action"] is not None
    assert record["result"] is result


def test_memory_records_actions() -> None:
    engine = GameEngine()
    action = engine.planner.choose_action(
        engine.state_extractor.extract(
            Observation(
                image={"objects": ["start_button"], "goal": "start", "confidence": 1.0},
                timestamp=0.0,
            )
        )
    )
    result = engine.input_adapter.execute(action)
    engine.memory.record({"action": action, "result": result})

    assert len(engine.memory.records) >= 1
    assert engine.memory.records[-1]["action"].name == "click"
    assert engine.memory.records[-1]["result"].success is True
