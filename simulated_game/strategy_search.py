import argparse
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

from core.strategy_search import Genome, StrategySearch
from simulated_game.agent.planner import GridStrategyParameters
from simulated_game.benchmark import _play_episode
from simulated_game.strategy_config import load_strategy, save_strategy


PARAMETER_BOUNDS = {
    "visit_penalty": (0.0, 2.0),
    "information_gain_bonus": (0.0, 2.0),
    "goal_change_threshold": (1.0, 5.0),
}


def _fitness(episode: dict[str, Any], max_turns: int) -> float:
    if episode["outcome"] == "win":
        return 1.0 + (max_turns - episode["turns"]) / (max_turns * 100)
    if episode["outcome"] == "loss":
        return -1.0 + 0.1 * episode["coverage"]
    return 0.1 * episode["coverage"]


def _evaluate(
    genome: Genome,
    seeds: Sequence[int],
    scenario: str,
    max_turns: int,
) -> list[float]:
    strategy = GridStrategyParameters.from_mapping(genome)
    return [
        _fitness(
            _play_episode(seed, max_turns, None, scenario, strategy),
            max_turns,
        )
        for seed in seeds
    ]


def _summarize(
    genome: Genome,
    seeds: Sequence[int],
    scenario: str,
    max_turns: int,
) -> dict[str, Any]:
    strategy = GridStrategyParameters.from_mapping(genome)
    episodes = [
        _play_episode(seed, max_turns, None, scenario, strategy)
        for seed in seeds
    ]
    wins = sum(episode["outcome"] == "win" for episode in episodes)
    losses = sum(episode["outcome"] == "loss" for episode in episodes)
    turns_to_win = [
        episode["turns"] for episode in episodes if episode["outcome"] == "win"
    ]
    scores = [_fitness(episode, max_turns) for episode in episodes]
    return {
        "episodes": len(episodes),
        "wins": wins,
        "losses": losses,
        "timeouts": sum(episode["outcome"] == "timeout" for episode in episodes),
        "stalled": sum(episode["outcome"] == "stalled" for episode in episodes),
        "mean_coverage": round(
            statistics.mean(episode["coverage"] for episode in episodes), 4
        ),
        "mean_turns_to_win": round(statistics.mean(turns_to_win), 2)
        if turns_to_win
        else None,
        "mean_score": round(statistics.mean(scores), 5),
        "outcomes_by_seed": {
            str(seed): episode["outcome"]
            for seed, episode in zip(seeds, episodes)
        },
    }


def _promotes(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    candidate_outcomes = candidate["outcomes_by_seed"]
    baseline_outcomes = baseline["outcomes_by_seed"]
    if candidate_outcomes.keys() != baseline_outcomes.keys():
        raise ValueError("Candidate and baseline must use identical holdout seeds.")
    gains = sum(
        candidate_outcomes[seed] == "win"
        and baseline_outcomes[seed] != "win"
        for seed in candidate_outcomes
    )
    regressions = sum(
        baseline_outcomes[seed] == "win"
        and candidate_outcomes[seed] != "win"
        for seed in candidate_outcomes
    )
    if gains >= max(2, math.ceil(candidate["episodes"] * 0.05)):
        return regressions == 0 and candidate["losses"] <= baseline["losses"]
    if candidate["wins"] != baseline["wins"] or candidate["losses"] != baseline["losses"]:
        return False
    if candidate_outcomes != baseline_outcomes:
        return False
    candidate_turns = candidate["mean_turns_to_win"]
    baseline_turns = baseline["mean_turns_to_win"]
    return (
        candidate_turns is not None
        and baseline_turns is not None
        and candidate_turns <= baseline_turns * 0.95
    )


def run_search(
    scenario: str,
    train_episodes: int,
    holdout_episodes: int,
    max_turns: int,
    generations: int,
    population: int,
    seed: int,
    strategy_path: Path,
) -> dict[str, Any]:
    training_seeds = list(range(seed, seed + train_episodes))
    holdout_seeds = list(range(1_000_000 + seed, 1_000_000 + seed + holdout_episodes))
    baseline = load_strategy(strategy_path, scenario).to_mapping()
    search = StrategySearch(
        PARAMETER_BOUNDS,
        seed=seed,
        integer_parameters=("goal_change_threshold",),
    )
    result = search.search(
        baseline,
        training_seeds,
        lambda genome, seeds: _evaluate(genome, seeds, scenario, max_turns),
        generations=generations,
        population_size=population,
    )

    baseline_holdout = _summarize(
        result.baseline, holdout_seeds, scenario, max_turns
    )
    candidate_holdout = _summarize(
        result.best, holdout_seeds, scenario, max_turns
    )
    promoted = result.best != result.baseline and _promotes(
        candidate_holdout, baseline_holdout
    )
    if promoted:
        save_strategy(
            strategy_path,
            scenario,
            GridStrategyParameters.from_mapping(result.best),
        )

    return {
        "scenario": scenario,
        "training_seeds": training_seeds,
        "holdout_seeds": holdout_seeds,
        "evaluated_variants": result.evaluated,
        "training_baseline_mean_score": round(
            statistics.mean(result.baseline_scores), 5
        ),
        "training_best_mean_score": round(
            statistics.mean(result.best_scores), 5
        ),
        "best_training_candidate": result.best,
        "top_training_variants": [
            {
                "parameters": candidate.genome,
                "mean_score": round(statistics.mean(candidate.scores), 5),
            }
            for candidate in result.leaderboard[:5]
        ],
        "holdout_baseline": baseline_holdout,
        "holdout_candidate": candidate_holdout,
        "promoted": promoted,
        "saved_strategy": str(strategy_path) if promoted else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate, compare, and validate grid strategy variants."
    )
    parser.add_argument(
        "--scenario",
        choices=("static", "dynamic"),
        default="dynamic",
    )
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--holdout-episodes", type=int, default=40)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4815)
    parser.add_argument("--strategy-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if min(
        args.train_episodes,
        args.holdout_episodes,
        args.max_turns,
        args.generations,
    ) <= 0 or args.population < 2:
        parser.error(
            "episode counts, max-turns, generations must be positive "
            "and population >= 2"
        )
    if args.holdout_episodes < 20:
        parser.error("holdout-episodes must be at least 20 for promotion checks")

    logging.disable(logging.CRITICAL)
    report = run_search(
        args.scenario,
        args.train_episodes,
        args.holdout_episodes,
        args.max_turns,
        args.generations,
        args.population,
        args.seed,
        args.strategy_dir / f".gm-strategy-{args.scenario}.json",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
