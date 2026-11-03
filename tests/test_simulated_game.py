from unittest.mock import patch

from core.types import Action, ActionResult
from adapters.grid_30x30 import GridExplorationPolicy as ExplorationPolicy
from adapters.grid_30x30 import GridMemoryStore as MemoryStore
from adapters.grid_30x30 import GridGame
from simulated_game.runtime import create_grid_engine


def test_agent_reaches_grid_goal_around_obstacle() -> None:
    game = GridGame(seed=7)
    game.target = (1, 1)
    engine = create_grid_engine(game)

    for _ in range(game.width * game.height):
        frame = game.observe()
        frame["controls"] = []
        engine.process_frame(frame, source="test_grid_game")
        if game.won:
            break

    assert game.won
    assert 1 <= len(engine.memory.records) <= game.width * game.height
    assert engine.memory.records[-1]["result"].details["status"] == "goal_reached"


def test_random_target_is_valid_distant_and_seed_reproducible() -> None:
    first = GridGame(seed=42)
    repeat = GridGame(seed=42)

    assert first.width == first.height == 30
    assert first.target == repeat.target
    assert first.target != first.position
    assert first.target not in first.obstacles
    distance = abs(first.target[0] - first.position[0]) + abs(
        first.target[1] - first.position[1]
    )
    assert distance >= 4


def test_observation_hides_target_outside_four_cell_vision_radius() -> None:
    game = GridGame(seed=42)
    game.target = (9, 9)

    frame = game.observe()
    grid = frame["hud"]["grid"]

    assert grid["vision_radius"] == 4
    assert grid["target"] is None
    assert [9, 9] not in grid["visible_cells"]
    assert "goal" not in frame["objects"]


def test_agent_explores_when_target_is_hidden_without_receiving_its_coordinates() -> None:
    game = GridGame(seed=42)
    game.target = (9, 9)
    engine = create_grid_engine(game)

    frame = game.observe()
    frame["controls"] = []
    result = engine.process_frame(frame, source="test_limited_vision")

    assert result is not None
    assert result.success is True
    assert result.details["status"] == "moved"
    assert game.position != (0, 1)
    assert frame["hud"]["grid"]["target"] is None
    assert engine.memory.experiences[-1]["action"].name.startswith("move_")
    first_destination = game.position
    engine.process_frame(game.observe(), source="test_limited_vision")
    assert engine.memory.spatial.grid[first_destination].visited


def test_policy_reuses_a_recently_seen_goal_after_it_leaves_current_vision() -> None:
    policy = ExplorationPolicy()
    memory = MemoryStore()
    visible_goal = {
        "controls": [],
        "goal": "reach_goal",
        "confidence": 1.0,
        "hud": {
            "grid": {
                "width": 4,
                "height": 3,
                "position": [1, 1],
                "target": [2, 1],
                "visible_cells": [[1, 1], [2, 1]],
                "obstacles": [],
            }
        },
    }
    hidden_goal = {
        "controls": [],
        "goal": "reach_goal",
        "confidence": 1.0,
        "hud": {
            "grid": {
                "width": 4,
                "height": 3,
                "position": [1, 1],
                "target": None,
                "visible_cells": [[1, 1]],
                "obstacles": [],
            }
        },
    }

    policy.decide(visible_goal, memory)
    action = policy.decide(hidden_goal, memory)

    assert memory.spatial.known_goal_positions == [(2, 1)]
    assert action.name == "move_right"


def test_policy_forgets_a_goal_when_its_location_is_visible_and_empty() -> None:
    policy = ExplorationPolicy()
    memory = MemoryStore()
    visible_goal = {
        "controls": [],
        "confidence": 1.0,
        "hud": {
            "grid": {
                "width": 4,
                "height": 3,
                "position": [1, 1],
                "target": [2, 1],
                "visible_cells": [[1, 1], [2, 1]],
                "obstacles": [],
            }
        },
    }
    observed_empty = {
        "controls": [],
        "confidence": 1.0,
        "hud": {
            "grid": {
                "width": 4,
                "height": 3,
                "position": [1, 1],
                "target": None,
                "visible_cells": [[1, 1], [2, 1]],
                "obstacles": [],
            }
        },
    }

    policy.decide(visible_goal, memory)
    policy.decide(observed_empty, memory)

    assert memory.spatial.known_goal_positions == []
    assert memory.spatial.grid[(2, 1)].value != "target"


def test_policy_recovers_from_a_blocked_planned_transition() -> None:
    policy = ExplorationPolicy()
    memory = MemoryStore()
    observation = {
        "controls": [],
        "confidence": 1.0,
        "hud": {
            "grid": {
                "width": 4,
                "height": 3,
                "position": [1, 1],
                "target": [3, 1],
                "visible_cells": [
                    [x, y] for y in range(3) for x in range(4)
                ],
                "obstacles": [],
            }
        },
    }

    first_action = policy.decide(observation, memory)
    assert first_action.name == "move_right"
    policy.observe_result(
        ActionResult(success=False, details={"status": "blocked_by_obstacle"}),
        memory,
    )
    recovery_action = policy.decide(observation, memory)

    assert recovery_action.name in {"move_up", "move_down"}


def test_policy_does_not_blame_planned_step_for_unmapped_control_failure() -> None:
    policy = ExplorationPolicy()
    memory = MemoryStore()
    observation = {
        "controls": ["opaque_input"],
        "confidence": 1.0,
        "hud": {
            "grid": {
                "width": 4,
                "height": 3,
                "position": [1, 1],
                "target": [3, 1],
                "visible_cells": [
                    [x, y] for y in range(3) for x in range(4)
                ],
                "obstacles": [],
            }
        },
    }

    action = policy.decide(observation, memory)
    policy.observe_result(
        ActionResult(success=False, details={"status": "blocked_by_obstacle"}),
        memory,
    )

    assert action.name == "opaque_input"
    assert policy.planner._blocked_transitions == {}


def test_enemy_can_be_seen_without_a_new_observation_field() -> None:
    game = GridGame(seed=42)
    game.enemy_position = (3, 1)

    frame = game.observe()
    grid = frame["hud"]["grid"]

    # Enemy should NOT appear in obstacles (only static/moving obstacles)
    assert [3, 1] not in grid["obstacles"]
    assert "enemy" not in frame
    assert "enemy" not in frame["objects"]
    # Enemy should appear in entities
    assert {
        "id": "entity-2",
        "position": [3, 1],
        "health": 8,  # ENEMY_MAX_HEALTH
    } in grid["entities"]


def test_game_master_tracks_and_categorizes_anonymous_moving_entity() -> None:
    game = GridGame(seed=42)
    game.enemy_position = (4, 1)
    engine = create_grid_engine(game)

    for _ in range(3):
        frame = game.observe()
        frame["controls"] = []
        engine.process_frame(frame, source="test_entity_categories")

    entity = engine.memory.entities["entity-2"]
    assert entity.category in {"mobile", "approaches_agent"}
    assert entity.observation_count >= 2
    assert entity.position != (4, 1)


def test_enemy_alerts_within_vision_and_pursues_player() -> None:
    game = GridGame(seed=42)
    game.enemy_position = (4, 2)
    player_start = game.position
    enemy_start = game.enemy_position

    result = game.execute(Action(name="move_down"))

    assert game.enemy_alerted
    assert game.enemy_position != enemy_start
    assert result.details["status"] == "moved"
    assert abs(game.enemy_position[0] - game.position[0]) + abs(
        game.enemy_position[1] - game.position[1]
    ) < abs(enemy_start[0] - player_start[0]) + abs(
        enemy_start[1] - player_start[1]
    )


def test_agent_outcomes_with_active_pursuer_across_multiple_seeds() -> None:
    outcomes = []
    for seed in range(10):
        game = GridGame(seed=seed)
        engine = create_grid_engine(game)
        engine.logger.logger.disabled = True

        for _ in range(200):
            engine.process_frame(game.observe(), source="test_seeded_exploration")
            if game.won or game.caught:
                break

        assert game.won or game.caught or len(engine.memory.experiences) == 200
        outcomes.append((
            game.won,
            game.caught,
            game.player_health,
            game.enemy_health,
            game.enemy_alerted,
            sum(
                experience["action"].name.startswith("control_")
                for experience in engine.memory.experiences
            ),
        ))

    assert any(won for won, _, _, _, _, _ in outcomes)
    assert any(alerted for _, _, _, _, alerted, _ in outcomes)
    assert any(control_trials for _, _, _, _, _, control_trials in outcomes)


def test_obstacle_moves_after_each_directional_action_without_overlapping() -> None:
    game = GridGame(seed=42)
    target = game.target

    for action in (Action(name="move_right"), Action(name="move_down")):
        previous_obstacle = next(iter(game.obstacles))
        result = game.execute(action)
        obstacle = next(iter(game.obstacles))

        assert obstacle != previous_obstacle
        assert obstacle != game.position
        assert obstacle != target
        assert "obstacle" not in result.details


def test_target_moves_after_every_second_directional_action() -> None:
    game = GridGame(seed=42)
    initial_target = game.target

    first_result = game.execute(Action(name="move_right"))
    assert game.target == initial_target
    assert first_result.details["move_count"] == 1

    second_result = game.execute(Action(name="move_up"))
    assert game.target != initial_target
    assert game.target != game.position
    assert game.target not in game.obstacles
    observed_target = game.observe()["hud"]["grid"]["target"]
    assert observed_target == (list(game.target) if game.is_visible(game.target) else None)
    assert second_result.details["move_count"] == 2


def test_static_goal_scenario_disables_target_motion_and_pursuer() -> None:
    game = GridGame(seed=42, moving_target=False, pursuer_enabled=False)
    target = game.target

    game.execute(Action(name="control_2"))
    game.execute(Action(name="control_3"))

    assert game.target == target
    assert game.enemy_position is None


def test_grid_game_reports_blocked_move_without_changing_position() -> None:
    game = GridGame()
    game.position = (1, 1)
    start = game.position

    result = game.execute(Action(name="move_right"))

    assert result.success is False
    assert result.details["status"] == "blocked_by_obstacle"
    assert game.position == start
    assert "obstacle" not in result.details


def test_health_is_visible_without_an_enemy_label_or_combat_instructions() -> None:
    game = GridGame(seed=42)
    game.position = (3, 1)
    game.enemy_position = (4, 1)

    frame = game.observe()

    assert frame["hud"]["health"] == 20  # PLAYER_MAX_HEALTH
    assert {"id": "entity-2", "position": [4, 1], "health": 8} in frame["hud"]["grid"]["entities"]  # ENEMY_MAX_HEALTH
    assert frame["controls"] == [f"control_{index}" for index in range(1, 7)]
    assert "enemy" not in frame
    assert "attack" not in str(frame).lower()
    assert "defend" not in str(frame).lower()


def test_six_anonymous_controls_are_exposed_and_mapped_to_game_inputs() -> None:
    game = GridGame(seed=42)
    game.position = (10, 10)
    game.target = (25, 25)
    game.obstacles = {(29, 29)}
    game.enemy_position = (20, 20)
    expected_positions = {
        "control_1": (10, 9),
        "control_2": (11, 10),
        "control_3": (10, 11),
        "control_4": (9, 10),
    }

    frame = game.observe()
    assert frame["controls"] == [f"control_{index}" for index in range(1, 7)]
    assert all(
        name not in str(frame).lower()
        for name in ("move_up", "move_right", "move_down", "move_left", "attack", "defend")
    )

    for control, expected_position in expected_positions.items():
        game.position = (10, 10)
        game._move_count = 0
        game.execute(Action(name=control))
        assert game.position == expected_position


def test_generic_control_policy_learns_effects_from_observation_changes() -> None:
    policy = ExplorationPolicy()
    memory = MemoryStore()
    observation = lambda player_health, entity_health: {
        "objects": ["player", "obstacle"],
        "controls": ["control_1", "control_2"],
        "hud": {
            "health": player_health,
            "grid": {
                "width": 30,
                "height": 30,
                "position": [1, 1],
                "visible_cells": [[1, 1], [2, 1]],
                "obstacles": [[2, 1]],
                "entities": [
                    {"id": "entity-2", "position": [2, 1], "health": entity_health}
                ],
            },
        },
        "confidence": 1.0,
    }

    state = policy.world_model.update(observation(10, 10))
    state.entities[0].position = (2, 1)
    observations = (
        observation(10, 10),
        observation(7, 7),
        observation(7, 7),
        observation(4, 7),
        observation(4, 7),
    )
    actions = []
    for frame in observations:
        normalized = frame
        control = policy.control_learner.choose(
            state,
            normalized,
            normalized["controls"],
            memory,
        )
        if control is not None:
            actions.append(control)

    assert actions
    trials = policy.control_learner.trials[(1, 1)]
    assert len(trials) <= 6
    assert all(control.startswith("control_") for control in trials)
    assert sum(map(len, trials.values())) == len(observations) - 1
    assert all("mean_reward" in inference for inference in (policy.control_learner.latest_inference,))


def test_defense_prevents_damage_and_an_attacking_entity_retreats() -> None:
    game = GridGame(seed=42)
    game.position = (1, 1)
    game.enemy_position = (2, 1)
    game.obstacles = {(9, 9)}
    with patch.object(game, "_enemy_combat_choice", return_value="attack"):
        result = game.execute(Action(name="control_6"))

    assert game.player_health == 20  # PLAYER_MAX_HEALTH
    assert game.enemy_health == 8  # ENEMY_MAX_HEALTH
    assert game.enemy_position is not None
    assert game._distance(game.position, game.enemy_position) == 2
    assert result.details == {"status": "completed"}


def test_an_attacking_entity_is_not_damaged_when_it_defends() -> None:
    game = GridGame(seed=42)
    game.position = (1, 1)
    game.enemy_position = (2, 1)
    game.obstacles = {(9, 9)}
    with patch.object(game, "_enemy_combat_choice", return_value="defend"):
        result = game.execute(Action(name="control_5"))

    assert game.player_health == 20  # PLAYER_MAX_HEALTH
    assert game.enemy_health == 8  # ENEMY_MAX_HEALTH
    assert game.enemy_position == (2, 1)
    assert result.details == {"status": "completed"}


def test_combat_attacks_deal_three_damage() -> None:
    game = GridGame(seed=42)
    game.position = (1, 1)
    game.enemy_position = (2, 1)
    game.obstacles = {(9, 9)}
    with patch.object(game, "_enemy_combat_choice", return_value="attack"):
        game.execute(Action(name="control_5"))

<<<<<<< Updated upstream
    assert game.player_health == 7
    assert game.enemy_health == 7
=======
    assert game.player_health == 17  # 20 - 3 (ATTACK_DAMAGE)
    assert game.enemy_health == 5  # 8 - 3 (ATTACK_DAMAGE)
>>>>>>> Stashed changes


def test_player_attack_can_remove_entity_as_a_regular_obstacle() -> None:
    game = GridGame(seed=42)
    game.position = (1, 1)
    game.enemy_position = (2, 1)
    game.obstacles = {(9, 9)}
    with patch.object(game, "_enemy_combat_choice", return_value="attack"):
<<<<<<< Updated upstream
        for _ in range(4):
=======
        for _ in range(3):  # 8 health, 3 damage per attack = 3 attacks needed
>>>>>>> Stashed changes
            game.enemy_position = (2, 1)
            game.execute(Action(name="control_5"))

    # The specific enemy at (2, 1) should be dead and removed
    assert not any(e["position"] == (2, 1) for e in game.enemies)
    assert (2, 1) in game.static_obstacles
    frame = game.observe()
    assert [2, 1] in frame["hud"]["grid"]["obstacles"]
    assert all(entity["id"] != "entity-2" or entity["position"] != [2, 1] for entity in frame["hud"]["grid"]["entities"])


def test_player_death_ends_game_as_a_loss() -> None:
    game = GridGame(seed=42)
    game.position = (1, 1)
    game.enemy_position = (2, 1)
    game.obstacles = {(9, 9)}
    game.player_health = 1
    with patch.object(game, "_enemy_combat_choice", return_value="attack"):
        result = game.execute(Action(name="control_5"))

    assert game.player_health == 0
    assert game.caught
    assert result.details == {"status": "caught"}
    assert game.observe()["hud"]["status"] == "lost"
