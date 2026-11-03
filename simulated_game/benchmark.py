import argparse
import json
import logging
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from core.types import Action, ActionResult
from adapters.grid_30x30 import (
    GridExplorationPolicy,
    GridMemoryStore,
    GridStrategyParameters,
    GridGame,
    WorldModelLearner,
)
from simulated_game.runtime import create_grid_engine


class _BenchmarkInputAdapter:
    def __init__(self, game: GridGame, policy: GridExplorationPolicy, memory: GridMemoryStore) -> None:
        self.game = game
        self.learner = policy.control_learner
        self.memory = memory
        self.predictions = 0
        self.correct_predictions = 0
        self.predicted_movement_attempts = 0
        self.predicted_movement_without_progress = 0

    def execute(self, action: Action) -> ActionResult:
        prediction = None
        if action.name.startswith("control_"):
            # Build context from current game state
            observation = self.game.observe()
            hud = observation.get("hud", {})
            grid = hud.get("grid", {})
            position = grid.get("position", [0, 0])
            target = grid.get("target")
            
            target_direction = None
            if target is not None:
                dx = target[0] - position[0]
                dy = target[1] - position[1]
                target_direction = [
                    0 if dx == 0 else (1 if dx > 0 else -1),
                    0 if dy == 0 else (1 if dy > 0 else -1),
                ]
            
            known_obstacles = set(tuple(obs) for obs in grid.get("obstacles", []))
            adjacent_obstacles = []
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                cell = (position[0] + dx, position[1] + dy)
                if cell in known_obstacles:
                    adjacent_obstacles.append(True)
                else:
                    adjacent_obstacles.append(False)
            
            # Get health
            health = hud.get("health", 10)
            health_bucket = 0 if health <= 3 else (1 if health <= 6 else 2)
            
            # Get entity distance
            entity_distances = []
            for entity in grid.get("entities", []):
                if isinstance(entity, dict) and entity.get("id") and entity.get("position"):
                    epos = tuple(entity["position"])
                    dist = abs(epos[0] - position[0]) + abs(epos[1] - position[1])
                    entity_distances.append(dist)
            nearest_entity = min(entity_distances) if entity_distances else None
            entity_bucket = None
            if nearest_entity is not None:
                if nearest_entity <= 1:
                    entity_bucket = 0
                elif nearest_entity <= 2:
                    entity_bucket = 1
                elif nearest_entity <= 4:
                    entity_bucket = 2
                else:
                    entity_bucket = 3
            
            frontier_count = len(self.memory.spatial.frontier) if self.memory else 0
            
            context = {
                "goal_visible": target is not None,
                "goal_known": target is not None,
                "target_direction": target_direction,
                "goal_dynamic": self.memory.spatial.goal_is_dynamic if self.memory else False,
                "adjacent_obstacles": adjacent_obstacles,
                "entity_distance_bucket": entity_bucket,
                "health_bucket": health_bucket,
                "frontier_bucket": min(frontier_count, 2),
            }
            
            # Convert context to key
            required_keys = [
                "goal_visible", "goal_known", "target_direction", "goal_dynamic",
                "adjacent_obstacles", "entity_distance_bucket", "health_bucket", "frontier_bucket",
            ]
            key_parts = []
            for k in required_keys:
                v = context.get(k)
                if v is None:
                    key_parts.append(-1)
                elif isinstance(v, bool):
                    key_parts.append(1 if v else 0)
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if item is None:
                            key_parts.append(-1)
                        else:
                            key_parts.append(int(item))
                elif isinstance(v, int):
                    key_parts.append(v)
                else:
                    key_parts.append(0)
            context_key = tuple(key_parts)
            
            prediction = self.learner.predicted_displacement(action.name, context_key)
        
        previous_position = self.game.position
        result = self.game.execute(action)
        if prediction is not None:
            self.predictions += 1
            actual = (
                self.game.position[0] - previous_position[0],
                self.game.position[1] - previous_position[1],
            )
            if prediction != (0.0, 0.0):
                self.predicted_movement_attempts += 1
                if actual == (0, 0):
                    self.predicted_movement_without_progress += 1
            if prediction == actual:
                self.correct_predictions += 1
        return result


def _play_episode(
    seed: int,
    max_turns: int,
    database_path: Path | None,
    scenario: str,
    strategy: GridStrategyParameters | None = None,
) -> dict[str, Any]:
    is_static = scenario == "static"
    game = GridGame(
        seed=seed,
        moving_target=not is_static,
        pursuer_enabled=not is_static,
    )
    memory = (
        GridMemoryStore(game_id="grid_30x30_v1", database_path=database_path)
        if database_path is not None
        else GridMemoryStore()
    )
<<<<<<< Updated upstream
    policy = ExplorationPolicy(strategy)
    adapter = _BenchmarkInputAdapter(game, policy)
    engine = create_grid_engine(game, memory, adapter, policy)
=======
    policy = GridExplorationPolicy(strategy)
    adapter = _BenchmarkInputAdapter(game, policy, memory)
    engine = create_grid_engine(
        game,
        memory,
        adapter,
        policy,
        enable_episodic_retrieval=enable_episodic_retrieval,
    )
>>>>>>> Stashed changes

    turns = 0
    blocked = 0
    stationary = 0
    completed_without_movement = 0
    controls_without_prediction = 0
    control_actions = 0
    target_sightings = 0
    target_relocations = 0
    distance_progress = 0
    statuses: Counter[str] = Counter()
    outcome = "timeout"
    recent_stationary: list[bool] = []
    for turns in range(1, max_turns + 1):
        observation = game.observe()
        target_sightings += observation["hud"]["grid"]["target"] is not None
        previous_position = game.position
        previous_target = game.target
        result = engine.process_frame(observation, source="benchmark")
        if result is None:
            continue
        status = result.details.get("status")
        statuses[str(status)] += 1
        blocked += isinstance(status, str) and status.startswith("blocked_")
        did_not_move = game.position == previous_position
        stationary += did_not_move
        completed_without_movement += did_not_move and status == "completed"
        action = memory.experiences[-1]["action"]
        if action.name.startswith("control_"):
            control_actions += 1
            # Build context from current game state (same as _BenchmarkInputAdapter)
            observation = game.observe()
            hud = observation.get("hud", {})
            grid = hud.get("grid", {})
            position = grid.get("position", [0, 0])
            target = grid.get("target")
            
            target_direction = None
            if target is not None:
                dx = target[0] - position[0]
                dy = target[1] - position[1]
                target_direction = [
                    0 if dx == 0 else (1 if dx > 0 else -1),
                    0 if dy == 0 else (1 if dy > 0 else -1),
                ]
            
            known_obstacles = set(tuple(obs) for obs in grid.get("obstacles", []))
            adjacent_obstacles = []
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                cell = (position[0] + dx, position[1] + dy)
                if cell in known_obstacles:
                    adjacent_obstacles.append(True)
                else:
                    adjacent_obstacles.append(False)
            
            # Get health
            health = hud.get("health", 10)
            health_bucket = 0 if health <= 3 else (1 if health <= 6 else 2)
            
            # Get entity distance
            entity_distances = []
            for entity in grid.get("entities", []):
                if isinstance(entity, dict) and entity.get("id") and entity.get("position"):
                    epos = tuple(entity["position"])
                    dist = abs(epos[0] - position[0]) + abs(epos[1] - position[1])
                    entity_distances.append(dist)
            nearest_entity = min(entity_distances) if entity_distances else None
            entity_bucket = None
            if nearest_entity is not None:
                if nearest_entity <= 1:
                    entity_bucket = 0
                elif nearest_entity <= 2:
                    entity_bucket = 1
                elif nearest_entity <= 4:
                    entity_bucket = 2
                else:
                    entity_bucket = 3
            
            frontier_count = len(memory.spatial.frontier) if memory else 0
            
            context = {
                "goal_visible": target is not None,
                "goal_known": target is not None,
                "target_direction": target_direction,
                "goal_dynamic": memory.spatial.goal_is_dynamic if memory else False,
                "adjacent_obstacles": adjacent_obstacles,
                "entity_distance_bucket": entity_bucket,
                "health_bucket": health_bucket,
                "frontier_bucket": min(frontier_count, 2),
            }
            
            # Convert context to key
            required_keys = [
                "goal_visible", "goal_known", "target_direction", "goal_dynamic",
                "adjacent_obstacles", "entity_distance_bucket", "health_bucket", "frontier_bucket",
            ]
            key_parts = []
            for k in required_keys:
                v = context.get(k)
                if v is None:
                    key_parts.append(-1)
                elif isinstance(v, bool):
                    key_parts.append(1 if v else 0)
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if item is None:
                            key_parts.append(-1)
                        else:
                            key_parts.append(int(item))
                elif isinstance(v, int):
                    key_parts.append(v)
                else:
                    key_parts.append(0)
            context_key = tuple(key_parts)
            
            controls_without_prediction += (
                did_not_move
                and engine.policy.control_learner.predicted_displacement(
                    action.name, context_key
                )
                is None
            )
        recent_stationary.append(did_not_move)
        recent_stationary = recent_stationary[-8:]
        if not did_not_move:
            before_distance = abs(previous_position[0] - previous_target[0]) + abs(
                previous_position[1] - previous_target[1]
            )
            after_distance = abs(game.position[0] - previous_target[0]) + abs(
                game.position[1] - previous_target[1]
            )
            distance_progress += before_distance - after_distance
        target_relocations += game.target != previous_target
        if game.won or game.caught:
            outcome = "win" if game.won else "loss"
            break
        if status == "waiting_for_visible_target":
            outcome = "stalled"
            break

    return {
        "outcome": outcome,
        "turns": turns,
        "blocked_actions": blocked,
        "stationary_actions": stationary,
        "completed_without_movement": completed_without_movement,
        "control_actions": control_actions,
        "controls_without_prediction_and_without_movement": controls_without_prediction,
        "stationary_streak_8": len(recent_stationary) == 8
        and all(recent_stationary),
        "target_sightings": target_sightings,
        "target_relocations": target_relocations,
        "target_change_evidence": memory.spatial.goal_change_count,
        "target_inferred_dynamic": memory.spatial.goal_is_dynamic,
        "distance_progress": distance_progress,
        "action_status_counts": dict(statuses),
        "coverage": len(memory.spatial.grid) / (game.width * game.height),
        "predictions": adapter.predictions,
        "correct_predictions": adapter.correct_predictions,
        "predicted_movement_attempts": adapter.predicted_movement_attempts,
        "predicted_movement_without_progress": (
            adapter.predicted_movement_without_progress
        ),
        "retrieval": memory.retrieval_summary(),
    }


def _summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(episode["outcome"] for episode in episodes)
    predictions = sum(episode["predictions"] for episode in episodes)
    correct_predictions = sum(
        episode["correct_predictions"] for episode in episodes
    )
    count = len(episodes)
    action_status_counts: Counter[str] = Counter()
    retrieval_queries = 0
    retrieval_queries_with_matches = 0
    retrieved_cases = 0
    memory_guided_decisions = 0
    available_decision_cases = []
    for episode in episodes:
        action_status_counts.update(episode["action_status_counts"])
        retrieval = episode["retrieval"]
        retrieval_queries += retrieval["queries"]
        retrieval_queries_with_matches += retrieval["queries_with_matches"]
        retrieved_cases += retrieval["matched_cases"]
        memory_guided_decisions += retrieval["guided_decisions"]
        available_decision_cases.append(retrieval["available_cases"])
    return {
        "episodes": count,
        "wins": outcomes["win"],
        "losses": outcomes["loss"],
        "timeouts": outcomes["timeout"],
        "stalled": outcomes["stalled"],
        "mean_turns": round(sum(item["turns"] for item in episodes) / count, 2),
        "blocked_actions": sum(item["blocked_actions"] for item in episodes),
        "stationary_actions": sum(
            item["stationary_actions"] for item in episodes
        ),
        "completed_without_movement": sum(
            item["completed_without_movement"] for item in episodes
        ),
        "control_actions": sum(item["control_actions"] for item in episodes),
        "controls_without_prediction_and_without_movement": sum(
            item["controls_without_prediction_and_without_movement"]
            for item in episodes
        ),
        "episodes_with_eight_turn_stall": sum(
            item["stationary_streak_8"] for item in episodes
        ),
        "mean_target_sightings": round(
            sum(item["target_sightings"] for item in episodes) / count,
            2,
        ),
        "mean_target_relocations": round(
            sum(item["target_relocations"] for item in episodes) / count,
            2,
        ),
        "target_change_evidence": sum(
            item["target_change_evidence"] for item in episodes
        ),
        "episodes_with_dynamic_target_hypothesis": sum(
            item["target_inferred_dynamic"] for item in episodes
        ),
        "mean_coverage": round(
            sum(item["coverage"] for item in episodes) / count,
            3,
        ),
        "distance_progress": sum(
            item["distance_progress"] for item in episodes
        ),
        "action_status_counts": dict(action_status_counts),
        "prediction_samples": predictions,
        "prediction_accuracy": round(
            correct_predictions / predictions,
            3,
        )
        if predictions
        else None,
        "predicted_movement_attempts": sum(
            item["predicted_movement_attempts"] for item in episodes
        ),
        "predicted_movement_without_progress": sum(
            item["predicted_movement_without_progress"]
            for item in episodes
        ),
        "memory_retrieval": {
            "queries": retrieval_queries,
            "queries_with_matches": retrieval_queries_with_matches,
            "matched_cases": retrieved_cases,
            "guided_decisions": memory_guided_decisions,
            "mean_available_cases": round(
                sum(available_decision_cases) / count, 2
            ),
        },
    }


def run_benchmark(
    scenario: str,
    episodes: int,
    max_turns: int,
    sequences: int,
    order_seed: int,
) -> dict[str, Any]:
    reports = []
    for sequence in range(sequences):
        seed_order = list(range(episodes))
        random.Random(order_seed + sequence).shuffle(seed_order)
        volatile = [
            _play_episode(seed, max_turns, None, scenario)
            for seed in seed_order
        ]
        with tempfile.TemporaryDirectory(prefix="gm-benchmark-") as directory:
            database_path = Path(directory) / "memory.sqlite3"
            persistent = [
                _play_episode(seed, max_turns, database_path, scenario)
                for seed in seed_order
            ]
        changed = sum(
            first["outcome"] != second["outcome"]
            for first, second in zip(volatile, persistent)
        )
        reports.append(
            {
                "sequence": sequence + 1,
                "seed_order": seed_order,
                "volatile": _summarize(volatile),
                "persistent": _summarize(persistent),
                "paired_outcomes_changed": changed,
            }
        )
    return {
        "scenario": scenario,
        "max_turns": max_turns,
        "order_seed": order_seed,
        "sequences": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare volatile and persistent control learning."
    )
    parser.add_argument(
        "--scenario",
        choices=("static", "dynamic", "all"),
        default="all",
    )
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=9204)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_turns <= 0 or args.sequences <= 0:
        parser.error("episodes, max-turns, and sequences must be positive")

    logging.disable(logging.CRITICAL)
    scenarios = ("static", "dynamic") if args.scenario == "all" else (args.scenario,)
    started = time.perf_counter()
    report = {
        scenario: run_benchmark(
            scenario,
            args.episodes,
            args.max_turns,
            args.sequences,
            args.order_seed,
        )
        for scenario in scenarios
    }
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()