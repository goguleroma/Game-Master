from pathlib import Path

import pytest
import torch

pytest.importorskip("numpy")
pytest.importorskip("torch")

from core.types import GameState
from learning.dqn import DQNAgent
from simulated_game.agent.memory import GridMemoryStore
from simulated_game.agent.neural_policy import NeuralGridPolicy
from simulated_game.game import GridGame
from simulated_game.neural_experiment import run_experiment
from simulated_game.runner import run_game


def test_dqn_respects_action_mask_and_decays_by_environment_action() -> None:
    agent = DQNAgent(
        input_size=2,
        action_count=3,
        seed=3,
        epsilon_decay_steps=2,
        warmup_steps=2,
        batch_size=2,
    )
    state = (0.0, 1.0)

    assert agent.select_action(state, (False, True, False)) == 1
    assert agent.environment_steps == 1
    assert agent.epsilon == pytest.approx(0.525)
    assert agent.select_action(state, (True, False, True)) in {0, 2}
    assert agent.environment_steps == 2
    assert agent.epsilon == pytest.approx(0.05)


def test_dqn_trains_and_round_trips_checkpoint(tmp_path: Path) -> None:
    agent = DQNAgent(
        input_size=2,
        action_count=2,
        seed=7,
        warmup_steps=2,
        batch_size=2,
    )
    for action in (0, 1):
        agent.remember(
            (0.0, 1.0),
            action,
            0.1,
            (1.0, 0.0),
            (True, True),
            done=False,
        )

    loss = agent.train_step()
    checkpoint = tmp_path / "candidate.pt"
    agent.save(checkpoint)
    restored = DQNAgent(
        input_size=2,
        action_count=2,
        seed=9,
        warmup_steps=2,
        batch_size=2,
    )
    restored.load(checkpoint)

    assert loss is not None
    assert loss >= 0
    assert checkpoint.is_file()
    assert len(restored.replay) == 2
    assert restored.training_steps == agent.training_steps
    assert restored.select_action(
        (0.0, 1.0), (True, True), explore=True
    ) == agent.select_action((0.0, 1.0), (True, True), explore=True)


def test_dqn_behavior_cloning_trains_only_allowed_actions() -> None:
    agent = DQNAgent(
        input_size=2,
        action_count=3,
        seed=17,
        epsilon_start=0.2,
    )
    samples = [
        ((0.0, 1.0), 0, (True, True, False)),
        ((1.0, 0.0), 1, (True, True, False)),
    ]

    loss = agent.behavior_clone(samples, epochs=2, batch_size=2)

    assert loss >= 0
    assert agent.training_steps == 2
    assert agent.epsilon == pytest.approx(0.2)


def test_legacy_weights_only_checkpoint_remains_inference_compatible(
    tmp_path: Path,
) -> None:
    source = DQNAgent(input_size=2, action_count=2, seed=5)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "format_version": 1,
            "input_size": 2,
            "action_count": 2,
            "online_state_dict": source.online.state_dict(),
        },
        checkpoint,
    )
    restored = DQNAgent(input_size=2, action_count=2, seed=6)

    restored.load(checkpoint, resume_training=False)

    with pytest.raises(ValueError, match="policy weights only"):
        restored.load(checkpoint)


def test_neural_policy_encodes_observations_and_updates_spatial_memory() -> None:
    game = GridGame(seed=11, moving_target=False, pursuer_enabled=False)
    frame = game.observe()
    observation = GameState(
        objects=frame["objects"],
        hud=frame["hud"],
        goal=frame["goal"],
        confidence=frame["confidence"],
        controls=frame["controls"],
    )
    memory = GridMemoryStore()
    policy = NeuralGridPolicy(seed=11)

    action = policy.decide(observation, memory)

    assert action is not None
    assert len(policy._pending[0]) == policy.FEATURE_COUNT
    assert memory.spatial.grid[game.position].visited
    assert memory.spatial.grid[(2, 1)].obstacle
    assert memory.spatial.frontier


def test_neural_features_include_planner_direction_intent() -> None:
    observed_directions = []
    for seed, target in ((21, (0, 0)), (22, (1, 1))):
        game = GridGame(seed=seed, moving_target=False, pursuer_enabled=False)
        game.target = target
        frame = game.observe()
        observation = GameState(
            objects=frame["objects"],
            hud=frame["hud"],
            goal=frame["goal"],
            confidence=frame["confidence"],
            controls=frame["controls"],
        )
        policy = NeuralGridPolicy(seed=seed)
        features = policy._encode(
            policy._normalize(observation),
            GridMemoryStore(),
        )
        observed_directions.append(features[-9:-5])
        assert features[-5] == 0.0

    assert observed_directions == [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
    ]


def test_neural_planner_hint_uses_retrieved_memory_without_changing_encoder_size() -> None:
    frame = {
        "objects": ["player", "goal"],
        "controls": [f"control_{index}" for index in range(1, 7)],
        "goal": "reach_goal",
        "confidence": 1.0,
        "hud": {
            "health": 10,
            "grid": {
                "width": 5,
                "height": 5,
                "position": [1, 1],
                "target": [3, 2],
                "visible_cells": [[x, y] for y in range(5) for x in range(5)],
                "obstacles": [],
                "entities": [],
            },
        },
    }
    observation = GameState(
        objects=frame["objects"],
        hud=frame["hud"],
        goal=frame["goal"],
        confidence=frame["confidence"],
        controls=frame["controls"],
    )
    policy = NeuralGridPolicy(seed=31)
    normalized = policy._normalize(observation)
    state = policy.world_model.update(normalized)
    memory = GridMemoryStore()
    visible_cells = [(x, y) for y in range(5) for x in range(5)]
    memory.update_working_memory(normalized, None, None)
    for cell in visible_cells:
        memory.mark_cell(cell, seen=True, obstacle=False)
    memory.observe_goal(state.target_position, visible_cells)
    memory.update_frontier(5, 5)
    context = memory.build_decision_context(state, normalized)
    for _ in range(3):
        memory.record_decision_case(context, (0, 1), 0.5, "moved")

    features = policy._encode(normalized, memory)

    assert len(features) == policy.FEATURE_COUNT == 40
    assert features[-9:-5] == (0.0, 0.0, 1.0, 0.0)
    assert memory.retrieval_summary()["guided_decisions"] == 1


def test_neural_reward_tracks_planner_progress_when_target_is_hidden() -> None:
    policy = NeuralGridPolicy(seed=23)
    before = {
        "hud": {
            "health": 10,
            "grid": {
                "position": [5, 5],
                "visible_cells": [],
                "target": None,
            },
        }
    }
    aligned_after = {
        "hud": {
            "health": 10,
            "grid": {
                "position": [5, 4],
                "visible_cells": [],
                "target": None,
            },
        }
    }
    stalled_after = {
        "hud": {
            "health": 10,
            "grid": {
                "position": [5, 5],
                "visible_cells": [],
                "target": None,
            },
        }
    }
    features = [0.0] * policy.FEATURE_COUNT
    features[policy.PLANNED_DIRECTION_START] = 1.0
    feature_tuple = tuple(features)

    aligned_reward = policy._reward(
        before,
        aligned_after,
        feature_tuple,
        {"status": "moved"},
    )
    stalled_reward = policy._reward(
        before,
        stalled_after,
        feature_tuple,
        {"status": "blocked_by_boundary"},
    )

    assert aligned_reward > stalled_reward
    assert aligned_reward == pytest.approx(0.43)
    assert stalled_reward == pytest.approx(0.08)


def test_neural_experiment_smoke_saves_unpromoted_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "candidate.pt"

    report = run_experiment(
        train_episodes=1,
        holdout_episodes=20,
        max_turns=3,
        seed=13,
        model_path=checkpoint,
    )

    assert checkpoint.is_file()
    assert report["training_episodes_this_run"] == 1
    assert report["final_neural_holdout"]["episodes"] == 20
    assert report["final_baseline_holdout"]["episodes"] == 20
    assert report["promoted"] is False
    resumed = run_experiment(
        train_episodes=1,
        holdout_episodes=20,
        max_turns=3,
        seed=13,
        model_path=checkpoint,
    )
    assert resumed["training_mode"] == "resumed"
    assert resumed["training_episodes_total"] == 2
    assert resumed["training_seed_range_this_run"] == [14, 14]
    assert set(report["holdout_seeds"]).isdisjoint(resumed["holdout_seeds"])
    best_checkpoint = tmp_path / "candidate.best.pt"
    assert best_checkpoint.is_file()
    inference_policy = NeuralGridPolicy(seed=13)
    inference_policy.agent.load(best_checkpoint, resume_training=False)

    monkeypatch.chdir(tmp_path)
    assert (
        run_game(
            max_steps=1,
            seed=13,
            policy_kind="neural",
            model_path=best_checkpoint,
        )
        is False
    )

    imitation_report = run_experiment(
        train_episodes=1,
        holdout_episodes=20,
        max_turns=3,
        seed=15,
        model_path=tmp_path / "imitation.pt",
        initialization="baseline",
        imitation_epochs=1,
    )
    assert imitation_report["initialization"] == "baseline"
    assert imitation_report["behavior_cloning"]["episodes"] == 1
    assert imitation_report["behavior_cloning"]["samples"] > 0
    imitation_resumed = run_experiment(
        train_episodes=1,
        holdout_episodes=20,
        max_turns=3,
        seed=15,
        model_path=tmp_path / "imitation.pt",
        initialization="baseline",
        imitation_epochs=1,
    )
    assert imitation_resumed["training_mode"] == "resumed"
    assert imitation_resumed["training_episodes_total"] == 2
    assert imitation_resumed["behavior_cloning"]["episodes"] == 1
    assert set(imitation_report["holdout_seeds"]).isdisjoint(
        imitation_resumed["holdout_seeds"]
    )
