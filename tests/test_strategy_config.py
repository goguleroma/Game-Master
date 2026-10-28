import json

import pytest

from simulated_game.agent.planner import GridStrategyParameters
from simulated_game.strategy_config import load_strategy, save_strategy


def test_strategy_config_round_trips_and_rejects_wrong_scenario(tmp_path) -> None:
    path = tmp_path / "strategy.json"
    expected = GridStrategyParameters(0.7, 1.2, 2.0)

    save_strategy(path, "dynamic", expected)

    assert load_strategy(path, "dynamic") == expected
    with pytest.raises(ValueError, match="mismatched"):
        load_strategy(path, "static")


def test_strategy_config_rejects_out_of_range_parameters(tmp_path) -> None:
    path = tmp_path / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "scenario": "dynamic",
                "parameters": {
                    "visit_penalty": 3,
                    "information_gain_bonus": 0.5,
                    "goal_change_threshold": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="between 0 and 2"):
        load_strategy(path, "dynamic")


def test_strategy_config_rejects_fractional_evidence_threshold() -> None:
    with pytest.raises(ValueError, match="whole number"):
        GridStrategyParameters(goal_change_threshold=2.5)
