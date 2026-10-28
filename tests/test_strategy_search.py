import math

import pytest

from core.strategy_search import StrategySearch
from simulated_game.strategy_search import _promotes


def test_search_reproducibly_generates_and_compares_variants() -> None:
    def evaluate(genome: dict[str, float], seeds: list[int]) -> list[float]:
        return [-((genome["weight"] - 0.7) ** 2) for _ in seeds]

    def run():
        return StrategySearch({"weight": (0.0, 1.0)}, seed=12).search(
            {"weight": 0.0},
            [2, 5, 8],
            evaluate,
            generations=4,
            population_size=6,
        )

    first = run()
    repeat = run()

    assert first == repeat
    assert first.evaluated > 1
    assert len(first.leaderboard) == first.evaluated
    assert first.best_scores[0] >= first.baseline_scores[0]
    assert 0.0 <= first.best["weight"] <= 1.0


def test_search_rejects_missing_seeds_and_invalid_scores() -> None:
    search = StrategySearch({"weight": (0.0, 1.0)})

    with pytest.raises(ValueError, match="seed"):
        search.search({"weight": 0.5}, [], lambda genome, seeds: [], 1, 2)

    with pytest.raises(ValueError, match="finite"):
        search.search(
            {"weight": 0.5},
            [1],
            lambda genome, seeds: [math.nan],
            1,
            2,
        )


def test_search_keeps_declared_integer_parameters_discrete() -> None:
    result = StrategySearch(
        {"threshold": (1.0, 5.0)},
        seed=7,
        integer_parameters=("threshold",),
    ).search(
        {"threshold": 1.0},
        [1, 2],
        lambda genome, seeds: [genome["threshold"] for _ in seeds],
        generations=3,
        population_size=5,
    )

    assert result.best["threshold"] == 5.0
    assert all(
        candidate.genome["threshold"].is_integer()
        for candidate in result.leaderboard
    )


def test_promotion_requires_paired_gains_without_regressions() -> None:
    baseline = {
        "episodes": 20,
        "wins": 8,
        "losses": 0,
        "mean_turns_to_win": 90.0,
        "outcomes_by_seed": {str(seed): "timeout" for seed in range(20)},
    }
    candidate = {
        **baseline,
        "wins": 10,
        "outcomes_by_seed": dict(baseline["outcomes_by_seed"]),
    }
    candidate["outcomes_by_seed"]["0"] = "win"
    candidate["outcomes_by_seed"]["1"] = "win"

    assert _promotes(candidate, baseline)

    candidate["outcomes_by_seed"]["2"] = "win"
    baseline["outcomes_by_seed"]["2"] = "win"
    baseline["outcomes_by_seed"]["3"] = "win"
    candidate["wins"] = 3
    baseline["wins"] = 2

    assert not _promotes(candidate, baseline)
