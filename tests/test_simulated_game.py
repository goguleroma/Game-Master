from core.engine import GameEngine
from core.types import Action
from simulated_game.game import GridGame, GridGameInputAdapter


def test_agent_reaches_grid_goal_around_obstacle() -> None:
    game = GridGame(seed=7)
    game.target = (1, 1)
    engine = GameEngine()
    engine.input_adapter = GridGameInputAdapter(game)

    for _ in range(game.width * game.height):
        engine.process_frame(game.observe(), source="test_grid_game")
        if game.won:
            break

    assert game.won
    assert 1 <= len(engine.memory.records) <= game.width * game.height
    assert engine.memory.records[-1]["result"].details["status"] == "goal_reached"


def test_random_target_is_valid_distant_and_seed_reproducible() -> None:
    first = GridGame(seed=42)
    repeat = GridGame(seed=42)

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


def test_agent_waits_when_target_is_hidden_instead_of_getting_hidden_coordinates() -> None:
    game = GridGame(seed=42)
    game.target = (9, 9)
    engine = GameEngine()
    engine.input_adapter = GridGameInputAdapter(game)

    result = engine.process_frame(game.observe(), source="test_limited_vision")

    assert result is not None
    assert result.success is False
    assert result.details["status"] == "waiting_for_visible_target"
    assert game.position == (0, 1)


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
        assert result.details["obstacle"] == list(obstacle)


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
    assert second_result.details["target"] == list(game.target)
    assert second_result.details["move_count"] == 2


def test_grid_game_reports_blocked_move_without_changing_position() -> None:
    game = GridGame()
    game.position = (1, 1)
    start = game.position

    result = game.execute(Action(name="move_right"))

    assert result.success is False
    assert result.details["status"] == "blocked_by_obstacle"
    assert game.position == start
    assert result.details["obstacle"] != [2, 1]
