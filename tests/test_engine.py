from core.engine import GameEngine


def test_engine_initialization() -> None:
    engine = GameEngine()
    assert engine is not None
    assert engine.memory is not None
    assert engine.planner is not None
    assert engine.input_adapter is not None
