import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

from core.types import Action, ActionResult, GameState
from simulated_game.agent.exploration_policy import ExplorationPolicy
from simulated_game.agent.memory import GridMemoryStore
from simulated_game.agent.neural_policy import NeuralGridPolicy
from simulated_game.agent.planner import GridStrategyParameters
from simulated_game.benchmark import _play_episode
from simulated_game.game import GridGame
from simulated_game.runtime import create_grid_engine


class _BaselineDemonstrationPolicy:
    def __init__(self) -> None:
        self.baseline = ExplorationPolicy(GridStrategyParameters())
        self.encoder = NeuralGridPolicy()
        self.samples: list[tuple[tuple[float, ...], int, tuple[bool, ...]]] = []

    def decide(self, observation: GameState, memory: GridMemoryStore) -> Action | None:
        action = self.baseline.decide(observation, memory)
        if action is None or action.name not in observation.controls:
            return action
        features = self.encoder._encode(
            self.encoder._normalize(observation),
            memory,
            update_memory=False,
        )
        action_index = observation.controls.index(action.name)
        action_mask = tuple(
            index < len(observation.controls)
            for index in range(self.encoder.ACTION_COUNT)
        )
        self.samples.append((features, action_index, action_mask))
        return action

    def observe_result(
        self,
        result: ActionResult,
        memory: GridMemoryStore,
    ) -> None:
        self.baseline.observe_result(result, memory)


def _play_neural_episode(
    seed: int,
    max_turns: int,
    policy: NeuralGridPolicy,
    training: bool,
) -> dict[str, Any]:
    game = GridGame(seed=seed, moving_target=True, pursuer_enabled=True)
    memory = GridMemoryStore()
    policy.training = training
    policy.begin_episode()
    engine = create_grid_engine(game, memory=memory, policy=policy)
    engine.logger.logger.disabled = True
    outcome = "timeout"
    turns = 0
    for turns in range(1, max_turns + 1):
        result = engine.process_frame(game.observe(), source="neural_experiment")
        if result is None:
            continue
        status = result.details.get("status")
        if status in {"goal_reached", "won"}:
            outcome = "win"
            break
        if status in {"caught", "lost"}:
            outcome = "loss"
            break
        if status == "waiting_for_visible_target":
            outcome = "stalled"
            break
    if training and outcome in {"timeout", "stalled"}:
        policy.finish_episode(game.observe(), memory)
    return {
        "outcome": outcome,
        "turns": turns,
        "coverage": len(memory.spatial.grid) / (game.width * game.height),
    }


def _collect_baseline_demonstrations(
    seed: int,
    max_turns: int,
    collector: _BaselineDemonstrationPolicy,
) -> dict[str, Any]:
    game = GridGame(seed=seed, moving_target=True, pursuer_enabled=True)
    memory = GridMemoryStore()
    engine = create_grid_engine(game, memory=memory, policy=collector)
    engine.logger.logger.disabled = True
    outcome = "timeout"
    turns = 0
    for turns in range(1, max_turns + 1):
        result = engine.process_frame(
            game.observe(),
            source="neural_demonstration",
        )
        if result is None:
            continue
        status = result.details.get("status")
        if status in {"goal_reached", "won"}:
            outcome = "win"
            break
        if status in {"caught", "lost"}:
            outcome = "loss"
            break
        if status == "waiting_for_visible_target":
            outcome = "stalled"
            break
    return {
        "outcome": outcome,
        "turns": turns,
        "coverage": len(memory.spatial.grid) / (game.width * game.height),
    }


def _score(episode: dict[str, Any], max_turns: int) -> float:
    if episode["outcome"] == "win":
        return 1.0 + (max_turns - episode["turns"]) / (max_turns * 100)
    if episode["outcome"] == "loss":
        return -1.0
    return episode["coverage"] * 0.1


def _summary(episodes: list[dict[str, Any]], max_turns: int) -> dict[str, Any]:
    if not episodes:
        raise ValueError("Cannot summarize an empty episode set.")
    wins = sum(episode["outcome"] == "win" for episode in episodes)
    turns_to_win = [
        episode["turns"] for episode in episodes if episode["outcome"] == "win"
    ]
    return {
        "episodes": len(episodes),
        "wins": wins,
        "losses": sum(episode["outcome"] == "loss" for episode in episodes),
        "timeouts": sum(episode["outcome"] == "timeout" for episode in episodes),
        "stalled": sum(episode["outcome"] == "stalled" for episode in episodes),
        "mean_turns_to_win": round(statistics.mean(turns_to_win), 2)
        if turns_to_win
        else None,
        "mean_score": round(
            statistics.mean(_score(episode, max_turns) for episode in episodes),
            5,
        ),
        "outcomes": [episode["outcome"] for episode in episodes],
    }


def _paired_outcomes(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> dict[str, int]:
    if len(candidate) != len(baseline):
        raise ValueError("Paired evaluations must contain the same episode count.")
    return {
        "improvements": sum(
            base["outcome"] != "win" and current["outcome"] == "win"
            for base, current in zip(baseline, candidate)
        ),
        "regressions": sum(
            base["outcome"] == "win" and current["outcome"] != "win"
            for base, current in zip(baseline, candidate)
        ),
    }


def _evaluate_baseline(
    seeds: list[int],
    max_turns: int,
) -> list[dict[str, Any]]:
    return [
        _play_episode(
            seed,
            max_turns,
            None,
            "dynamic",
            GridStrategyParameters(),
        )
        for seed in seeds
    ]


def _evaluate_neural(
    seeds: list[int],
    max_turns: int,
    policy: NeuralGridPolicy,
) -> list[dict[str, Any]]:
    policy.begin_episode()
    novelty_counts = policy._novel_states.copy()
    try:
        return [
            _play_neural_episode(seed, max_turns, policy, training=False)
            for seed in seeds
        ]
    finally:
        policy._novel_states = novelty_counts


def _meets_paired_gain_gate(paired: dict[str, int]) -> bool:
    return paired["improvements"] >= 2 and paired["regressions"] == 0


def _best_checkpoint_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}.best{model_path.suffix}")


def run_experiment(
    train_episodes: int,
    holdout_episodes: int,
    max_turns: int,
    seed: int,
    model_path: Path,
    eval_interval: int = 50,
    eval_episodes: int = 20,
    resume: bool = True,
    initialization: str = "scratch",
    imitation_epochs: int = 5,
    imitation_episodes: int | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    if imitation_episodes is None:
        imitation_episodes = train_episodes
    if min(
        train_episodes,
        holdout_episodes,
        max_turns,
        eval_interval,
        eval_episodes,
        imitation_episodes,
    ) <= 0:
        raise ValueError(
            "Episode counts, max_turns, and eval_interval must be positive."
        )
    if holdout_episodes < 20 or eval_episodes < 20:
        raise ValueError("Holdout and evaluation sets must each contain at least 20 episodes.")
    if initialization not in {"scratch", "baseline"}:
        raise ValueError("initialization must be 'scratch' or 'baseline'.")
    if imitation_epochs <= 0:
        raise ValueError("imitation_epochs must be positive.")

    policy = NeuralGridPolicy(
        seed=seed,
        training=True,
        epsilon_start=0.2 if initialization == "baseline" else 1.0,
    )
    training_mode = (
        "baseline_warm_start" if initialization == "baseline" else "dqn_from_scratch"
    )
    metadata: dict[str, Any] = {}
    if resume and model_path.is_file():
        metadata = policy.agent.load(model_path)
        if metadata.get("training_mode", "dqn_from_scratch") != training_mode:
            raise ValueError(
                "Checkpoint initialization mode differs from the requested mode; "
                "choose another --model path."
            )
        if metadata.get("seed_base", seed) != seed:
            raise ValueError("Cannot resume training with a different seed base.")
        if metadata.get("max_turns", max_turns) != max_turns:
            raise ValueError("Cannot resume training with a different max-turns value.")
        if metadata.get("eval_episodes", eval_episodes) != eval_episodes:
            raise ValueError("Cannot resume with a different validation seed set.")
        if (
            initialization == "baseline"
            and not metadata.get("imitation_complete", False)
        ):
            policy = NeuralGridPolicy(
                seed=seed,
                training=True,
                epsilon_start=0.2,
            )
            metadata = {}
    completed_before = int(metadata.get("episodes_completed", 0))
    best_validation_score = metadata.get("best_validation_score")
    best_validation = metadata.get("best_validation")
    evaluation_history = list(metadata.get("evaluation_history", []))
    imitation_report = metadata.get("behavior_cloning")
    used_holdout_seeds = list(metadata.get("holdout_seeds_used", []))
    if "holdout_seeds_used" not in metadata and completed_before:
        used_holdout_seeds.extend(
            range(1_000_000 + seed, 1_000_000 + seed + holdout_episodes)
        )
    best_model_path = _best_checkpoint_path(model_path)

    validation_seeds = [2_000_000 + seed + index for index in range(eval_episodes)]
    training_seeds = set(
        range(seed, seed + completed_before + train_episodes)
    )
    if training_seeds.intersection(validation_seeds):
        raise ValueError("Training and validation seed sets must be disjoint.")
    used_holdout_set = set(used_holdout_seeds)
    if training_seeds.intersection(used_holdout_set):
        raise ValueError("Training seeds overlap a previously used holdout.")
    holdout_start = 1_000_000 + seed + completed_before
    while True:
        holdout_seeds = list(range(holdout_start, holdout_start + holdout_episodes))
        current_holdout_set = set(holdout_seeds)
        if (
            not current_holdout_set.intersection(used_holdout_set)
            and not current_holdout_set.intersection(validation_seeds)
            and not current_holdout_set.intersection(training_seeds)
        ):
            break
        holdout_start += holdout_episodes
    baseline_validation = _evaluate_baseline(validation_seeds, max_turns)
    baseline_validation_summary = _summary(baseline_validation, max_turns)

    training_results: list[dict[str, Any]] = []
    losses_at_start = len(policy.losses)
    completed_this_run = 0
    interrupted = False
    last_validation: dict[str, Any] | None = None

    def checkpoint_metadata(episodes_completed: int) -> dict[str, Any]:
        return {
            "episodes_completed": episodes_completed,
            "best_validation_score": best_validation_score,
            "best_validation": best_validation,
            "evaluation_history": evaluation_history,
            "training_mode": training_mode,
            "seed_base": seed,
            "max_turns": max_turns,
            "eval_episodes": eval_episodes,
            "imitation_complete": initialization != "baseline"
            or imitation_report is not None,
            "behavior_cloning": imitation_report,
            "holdout_seeds_used": sorted(used_holdout_seeds),
        }

    def progress(message: str) -> None:
        if show_progress:
            print(message, file=sys.stderr, flush=True)

    try:
        if (
            initialization == "baseline"
            and imitation_report is None
        ):
            collector = _BaselineDemonstrationPolicy()
            demonstration_results: list[dict[str, Any]] = []
            for index in range(imitation_episodes):
                demonstration_results.append(
                    _collect_baseline_demonstrations(
                        seed + completed_before + index,
                        max_turns,
                        collector,
                    )
                )
                if (index + 1) % 10 == 0:
                    progress(
                        f"Baseline demonstrations: {index + 1}/{imitation_episodes} "
                        f"episodes, {len(collector.samples)} actions."
                    )
            imitation_loss = policy.agent.behavior_clone(
                collector.samples,
                epochs=imitation_epochs,
            )
            imitation_report = {
                "episodes": len(demonstration_results),
                "samples": len(collector.samples),
                "epochs": imitation_epochs,
                "mean_loss": round(imitation_loss, 5),
                "baseline_demonstrations": _summary(
                    demonstration_results,
                    max_turns,
                ),
            }
            validation_results = _evaluate_neural(
                validation_seeds,
                max_turns,
                policy,
            )
            validation_summary = _summary(validation_results, max_turns)
            paired = _paired_outcomes(validation_results, baseline_validation)
            last_validation = {
                "episodes_completed": completed_before,
                "seed_range": [validation_seeds[0], validation_seeds[-1]],
                "candidate": validation_summary,
                "baseline": baseline_validation_summary,
                "paired_improvements": paired["improvements"],
                "paired_regressions": paired["regressions"],
                "stage": "behavior_cloning_before_dqn",
            }
            evaluation_history.append(last_validation)
            best_validation_score = validation_summary["mean_score"]
            best_validation = last_validation
            policy.agent.save(
                best_model_path,
                metadata={
                    "episodes_completed": completed_before,
                    "validation": last_validation,
                    "training_mode": training_mode,
                    "seed_base": seed,
                    "max_turns": max_turns,
                    "eval_episodes": eval_episodes,
                    "imitation_complete": True,
                    "behavior_cloning": imitation_report,
                    "imitation_episodes": (
                        imitation_report["episodes"] if imitation_report is not None else 0
                    ),
                },
                include_training_state=False,
            )
            policy.agent.save(
                model_path,
                metadata=checkpoint_metadata(completed_before),
            )
            progress(
                f"Behavior cloning complete: {len(collector.samples)} actions, "
                f"mean loss {imitation_loss:.5f}; validation "
                f"{validation_summary['wins']}/{eval_episodes} wins versus "
                f"baseline {baseline_validation_summary['wins']}/{eval_episodes}."
            )

        for episode_index in range(train_episodes):
            episode_seed = seed + completed_before + episode_index
            result = _play_neural_episode(
                episode_seed,
                max_turns,
                policy,
                training=True,
            )
            training_results.append(result)
            completed_this_run += 1
            cumulative_episodes = completed_before + completed_this_run

            if (
                cumulative_episodes % eval_interval == 0
                or episode_index == train_episodes - 1
            ):
                validation_results = _evaluate_neural(
                    validation_seeds,
                    max_turns,
                    policy,
                )
                validation_summary = _summary(validation_results, max_turns)
                paired = _paired_outcomes(validation_results, baseline_validation)
                last_validation = {
                    "episodes_completed": cumulative_episodes,
                    "seed_range": [validation_seeds[0], validation_seeds[-1]],
                    "candidate": validation_summary,
                    "baseline": baseline_validation_summary,
                    "paired_improvements": paired["improvements"],
                    "paired_regressions": paired["regressions"],
                }
                evaluation_history.append(last_validation)
                progress(
                    f"Training episodes {cumulative_episodes}: validation "
                    f"{validation_summary['wins']}/{eval_episodes} wins; "
                    f"baseline {baseline_validation_summary['wins']}/{eval_episodes}."
                )

                if (
                    best_validation_score is None
                    or validation_summary["mean_score"] > best_validation_score
                ):
                    best_validation_score = validation_summary["mean_score"]
                    best_validation = last_validation
                    policy.agent.save(
                        best_model_path,
                        metadata={
                            "episodes_completed": cumulative_episodes,
                            "validation": last_validation,
                            "training_mode": training_mode,
                            "seed_base": seed,
                            "max_turns": max_turns,
                            "eval_episodes": eval_episodes,
                            "imitation_complete": initialization != "baseline"
                            or imitation_report is not None,
                            "behavior_cloning": imitation_report,
                        },
                        include_training_state=False,
                    )

                policy.agent.save(
                    model_path,
                    metadata=checkpoint_metadata(cumulative_episodes),
                )
    except KeyboardInterrupt:
        interrupted = True

    cumulative_episodes = completed_before + completed_this_run
    if interrupted or completed_this_run == 0:
        policy.agent.save(
            model_path,
            metadata=checkpoint_metadata(cumulative_episodes),
        )

    if completed_this_run and last_validation is None:
        validation_results = _evaluate_neural(validation_seeds, max_turns, policy)
        validation_summary = _summary(validation_results, max_turns)
        paired = _paired_outcomes(validation_results, baseline_validation)
        last_validation = {
            "episodes_completed": cumulative_episodes,
            "seed_range": [validation_seeds[0], validation_seeds[-1]],
            "candidate": validation_summary,
            "baseline": baseline_validation_summary,
            "paired_improvements": paired["improvements"],
            "paired_regressions": paired["regressions"],
        }
        evaluation_history.append(last_validation)
        if (
            best_validation_score is None
            or validation_summary["mean_score"] > best_validation_score
        ):
            best_validation_score = validation_summary["mean_score"]
            best_validation = last_validation
            policy.agent.save(
                best_model_path,
                metadata={
                    "episodes_completed": cumulative_episodes,
                    "validation": last_validation,
                    "training_mode": training_mode,
                    "seed_base": seed,
                    "max_turns": max_turns,
                    "eval_episodes": eval_episodes,
                    "imitation_complete": initialization != "baseline"
                    or imitation_report is not None,
                    "behavior_cloning": imitation_report,
                },
                include_training_state=False,
            )

    policy.agent.save(
        model_path,
        metadata=checkpoint_metadata(cumulative_episodes),
    )

    selected_checkpoint = (
        best_model_path if best_model_path.is_file() else model_path
    )
    holdout_policy = NeuralGridPolicy(
        seed=seed,
        training=False,
        epsilon_start=0.2 if initialization == "baseline" else 1.0,
    )
    holdout_policy.agent.load(selected_checkpoint, resume_training=False)
    neural_holdout = _evaluate_neural(holdout_seeds, max_turns, holdout_policy)
    baseline_holdout = _evaluate_baseline(holdout_seeds, max_turns)
    used_holdout_seeds.extend(holdout_seeds)
    policy.agent.save(
        model_path,
        metadata=checkpoint_metadata(cumulative_episodes),
    )
    holdout_summary = _summary(neural_holdout, max_turns)
    baseline_holdout_summary = _summary(baseline_holdout, max_turns)
    holdout_paired = _paired_outcomes(neural_holdout, baseline_holdout)
    validation_gate = bool(
        last_validation
        and _meets_paired_gain_gate(
            {
                "improvements": last_validation["paired_improvements"],
                "regressions": last_validation["paired_regressions"],
            }
        )
    )
    holdout_gate = _meets_paired_gain_gate(holdout_paired)
    return {
        "scenario": "dynamic",
        "initialization": initialization,
        "training_mode": "resumed" if completed_before else "from_scratch",
        "training_episodes_this_run": completed_this_run,
        "training_episodes_total": cumulative_episodes,
        "training_seed_range_this_run": (
            [seed + completed_before, seed + cumulative_episodes - 1]
            if completed_this_run
            else None
        ),
        "validation_seeds": validation_seeds,
        "holdout_seeds": holdout_seeds,
        "training_results": _summary(training_results, max_turns)
        if training_results
        else None,
        "behavior_cloning": imitation_report,
        "training_updates_total": policy.agent.training_steps,
        "mean_training_loss_this_run": round(
            statistics.mean(policy.losses[losses_at_start:]),
            5,
        )
        if len(policy.losses) > losses_at_start
        else None,
        "validation_baseline": baseline_validation_summary,
        "last_validation": last_validation,
        "best_validation": best_validation,
        "evaluation_count": len(evaluation_history),
        "final_neural_holdout": holdout_summary,
        "final_baseline_holdout": baseline_holdout_summary,
        "final_paired_improvements": holdout_paired["improvements"],
        "final_paired_regressions": holdout_paired["regressions"],
        "validation_gain_gate": validation_gate,
        "final_holdout_gain_gate": holdout_gate,
        "repeatable_improvement_observed": validation_gate and holdout_gate,
        "latest_training_checkpoint": str(model_path),
        "best_candidate_checkpoint": str(best_model_path),
        "holdout_checkpoint": str(selected_checkpoint),
        "promoted": False,
        "interrupted": interrupted,
        "note": (
            "Validation is repeated on fixed validation seeds; the final holdout "
            "is disjoint, used only after training, and never reused by resumed "
            "runs. The baseline is never replaced automatically."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistently train and compare a hybrid-curiosity neural policy."
    )
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--holdout-episodes", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument(
        "--initialization",
        choices=("scratch", "baseline"),
        default="scratch",
        help="use random weights or imitate the baseline before DQN updates",
    )
    parser.add_argument("--imitation-epochs", type=int, default=5)
    parser.add_argument(
        "--imitation-episodes",
        type=int,
        default=100,
        help="baseline episodes used to create behavior-cloning demonstrations",
    )
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7319)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(".gm-neural-training.pt"),
        help="latest resumable training checkpoint",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start from random weights; refuses to overwrite an existing checkpoint",
    )
    args = parser.parse_args()
    if min(
        args.train_episodes,
        args.holdout_episodes,
        args.eval_episodes,
        args.eval_interval,
        args.imitation_epochs,
        args.imitation_episodes,
        args.max_turns,
    ) <= 0:
        parser.error("episode counts, eval interval, and max-turns must be positive")
    if args.holdout_episodes < 20 or args.eval_episodes < 20:
        parser.error("holdout-episodes and eval-episodes must each be at least 20")
    if args.fresh and (
        args.model.exists() or _best_checkpoint_path(args.model).exists()
    ):
        parser.error(
            "The latest or best checkpoint already exists; choose a different "
            "--model path to start fresh"
        )

    logging.disable(logging.CRITICAL)
    report = run_experiment(
        train_episodes=args.train_episodes,
        holdout_episodes=args.holdout_episodes,
        max_turns=args.max_turns,
        seed=args.seed,
        model_path=args.model,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        resume=not args.fresh,
        initialization=args.initialization,
        imitation_epochs=args.imitation_epochs,
        imitation_episodes=args.imitation_episodes,
        show_progress=True,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
