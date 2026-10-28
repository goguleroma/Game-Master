import json
from pathlib import Path
import tempfile

from simulated_game.agent.planner import GridStrategyParameters


def load_strategy(
    path: Path,
    scenario: str,
) -> GridStrategyParameters:
    if not path.exists():
        return GridStrategyParameters()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or payload.get("scenario") != scenario
        or not isinstance(payload.get("parameters"), dict)
    ):
        raise ValueError(f"Invalid or mismatched strategy file: {path}")
    parameters = payload["parameters"]
    expected = {
        "visit_penalty",
        "information_gain_bonus",
        "goal_change_threshold",
    }
    if set(parameters) != expected or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in parameters.values()
    ):
        raise ValueError(f"Invalid strategy parameters in {path}")
    return GridStrategyParameters.from_mapping(parameters)


def save_strategy(
    path: Path,
    scenario: str,
    strategy: GridStrategyParameters,
) -> None:
    payload = {
        "version": 1,
        "scenario": scenario,
        "parameters": strategy.to_mapping(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temp_path = Path(temporary.name)
    temp_path.replace(path)
