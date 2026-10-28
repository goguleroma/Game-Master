from types import SimpleNamespace

from simulated_game.agent.memory import GridMemoryStore as MemoryStore
from simulated_game.agent.control_learner import ControlLearner
from simulated_game.agent.exploration_policy import ExplorationPolicy
from simulated_game.agent.planner import GridStrategyParameters, Planner
from simulated_game.agent.world_model import WorldState


def test_memory_supports_spatial_and_knowledge_tracking() -> None:
    memory = MemoryStore()
    memory.mark_cell((2, 2), seen=True, visited=False, obstacle=False)
    memory.mark_cell((2, 3), seen=True, visited=True, obstacle=False)
    memory.update_frontier()
    memory.add_knowledge("visible_goal_strategy", "frontier_exploration", 0.8, {"context": "partial_observation"})

    assert (2, 2) in memory.spatial.grid
    assert memory.spatial.frontier
    assert memory.knowledge[0].key == "visible_goal_strategy"
    assert memory.knowledge[0].confidence > 0.5


def test_exploration_policy_uses_frontier_when_goal_is_missing() -> None:
    policy = ExplorationPolicy()
    memory = MemoryStore()
    memory.mark_cell((0, 0), seen=True, visited=True)
    memory.mark_cell((0, 1), seen=True, visited=False)
    memory.update_frontier()

    action = policy.decide({"visible_cells": [(0, 0)], "unknown_cells": [(0, 1)], "frontier_cells": [(0, 1)]}, memory)

    assert action is not None
    assert action.name == "explore_frontier"


def test_planner_searches_instead_of_chasing_a_goal_inferred_to_be_mobile() -> None:
    memory = MemoryStore()
    memory.observe_goal((3, 1), [])
    memory.observe_goal((2, 1), [])
    memory.mark_cell((1, 1), seen=True, obstacle=False)
    memory.mark_cell((1, 0), seen=True, obstacle=False)
    memory.spatial.frontier = [(1, 0)]
    state = WorldState(
        current_position=(1, 1),
        grid_size=(4, 3),
    )

    action = Planner().choose_action(state, memory)

    assert memory.spatial.goal_is_dynamic
    assert memory.spatial.goal_change_count == 1
    assert action is not None
    assert action.name == "move_up"


def test_strategy_parameters_change_frontier_selection() -> None:
    memory = MemoryStore()
    memory.mark_cell((1, 1), seen=True, visited=True)
    memory.mark_cell((0, 1), seen=True, visited=True)
    memory.mark_cell((2, 1), seen=True, visited=False)
    for position in ((2, 0), (2, 2), (3, 1)):
        memory.mark_cell(position, seen=True)
    memory.spatial.frontier = [(0, 1), (2, 1)]
    state = WorldState(current_position=(1, 1), grid_size=(5, 3))

    visit_averse = Planner(GridStrategyParameters(1.0, 0.25)).choose_action(
        state, memory
    )
    information_seeking = Planner(
        GridStrategyParameters(0.0, 2.0)
    ).choose_action(state, memory)

    assert visit_averse is not None
    assert information_seeking is not None
    assert visit_averse.name == "move_right"
    assert information_seeking.name == "move_left"


def test_strategy_can_vary_how_much_mobility_evidence_to_require() -> None:
    memory = MemoryStore()
    memory.observe_goal((3, 1), [])
    memory.observe_goal((4, 1), [])
    for x in range(1, 5):
        memory.mark_cell((x, 1), seen=True)
    state = WorldState(current_position=(1, 1), grid_size=(6, 3))

    immediate_abandonment = Planner(
        GridStrategyParameters(goal_change_threshold=1.0)
    ).choose_action(state, memory)
    wait_for_evidence = Planner(
        GridStrategyParameters(goal_change_threshold=2.0)
    ).choose_action(state, memory)

    assert immediate_abandonment is not None
    assert wait_for_evidence is not None
    assert immediate_abandonment.name == "wait"
    assert wait_for_evidence.name == "move_right"


def test_memory_categorizes_anonymous_entity_from_repeated_motion() -> None:
    memory = MemoryStore()
    agent_position = (1, 1)

    memory.observe_entity("entity-7", (6, 1), agent_position, turn=0)
    memory.observe_entity("entity-7", (5, 1), agent_position, turn=1)
    memory.observe_entity("entity-7", (4, 1), agent_position, turn=2)
    memory.observe_entity("entity-7", (3, 1), agent_position, turn=3)
    entity = memory.observe_entity("entity-7", (2, 1), agent_position, turn=4)

    assert entity.category == "mobile"
    assert entity.behavior_hypothesis == "approaches_agent"
    assert entity.approach_evidence == 4
    assert entity.behavior_confidence == 2 / 3
    assert entity.position == (2, 1)


def test_memory_distinguishes_stationary_and_mobile_entities_without_semantic_labels() -> None:
    memory = MemoryStore()
    memory.observe_entity("entity-a", (2, 2), (0, 0), turn=0)
    stationary = memory.observe_entity("entity-a", (2, 2), (0, 0), turn=1)
    assert stationary.category == "unclassified"
    memory.observe_entity("entity-a", (2, 2), (0, 0), turn=2)
    stationary = memory.observe_entity("entity-a", (2, 2), (0, 0), turn=3)
    memory.observe_entity("entity-b", (4, 4), (0, 0), turn=0)
    mobile = memory.observe_entity("entity-b", (5, 4), (0, 0), turn=1)

    assert stationary.category == "stationary"
    assert mobile.category == "mobile"


def test_behavior_hypothesis_is_cleared_when_new_evidence_disagrees() -> None:
    memory = MemoryStore()
    agent_position = (1, 1)
    for turn, position in enumerate(((6, 1), (5, 1), (4, 1), (3, 1), (2, 1))):
        entity = memory.observe_entity("entity-3", position, agent_position, turn)
    assert entity.behavior_hypothesis == "approaches_agent"

    entity = memory.observe_entity("entity-3", (3, 1), agent_position, turn=5)

    assert entity.behavior_hypothesis is None
    assert entity.behavior_confidence == 0.0


def test_short_term_memory_is_bounded_by_frames_and_age() -> None:
    from core.memory import ShortTermMemory

    memory = ShortTermMemory(max_frames=3, max_age_seconds=2)
    memory.add("frame-1", timestamp=0)
    memory.add("frame-2", timestamp=1)
    memory.add("frame-3", timestamp=2)
    memory.add("frame-4", timestamp=3)

    assert [frame for _, frame in memory.frames] == ["frame-2", "frame-3", "frame-4"]

    memory.add("frame-5", timestamp=5.1)

    assert [frame for _, frame in memory.frames] == ["frame-5"]


def test_game_control_learning_persists_per_game(tmp_path) -> None:
    database_path = tmp_path / "memory.sqlite3"
    memory = MemoryStore(game_id="grid-game", database_path=database_path)
    learner = ControlLearner()
    state = SimpleNamespace(
        current_position=(1, 1),
        target_position=(100, 1),
        grid_size=(200, 3),
        entities=[
            SimpleNamespace(
                entity_id="entity-1",
                position=(2, 1),
                metadata={},
            )
        ],
    )
    health = 10
    for _ in range(80):
        chosen = learner.choose(
            state,
            {"hud": {"health": health}},
            ["control_1", "control_2"],
            memory,
        )
        if chosen == "control_2":
            state.current_position = (state.current_position[0] + 1, 1)
            health += 1
        else:
            state.current_position = (state.current_position[0] - 1, 1)
            health -= 1
        state.entities[0].position = (state.current_position[0] + 1, 1)
    learner.choose(
        state,
        {"hud": {"health": health}},
        ["control_1", "control_2"],
        memory,
    )

    restored_learner = ControlLearner()
    assert restored_learner.choose(
        state,
        {"hud": {"health": health}},
        ["control_1", "control_2"],
        MemoryStore(game_id="grid-game", database_path=database_path),
    ) == "control_2"

    other_game_learner = ControlLearner()
    assert other_game_learner.choose(
        state,
        {"hud": {"health": health}},
        ["control_1", "control_2"],
        MemoryStore(game_id="another-game", database_path=database_path),
    ) in {"control_1", "control_2"}


def test_learner_stops_repeating_controls_with_no_observed_effects() -> None:
    learner = ControlLearner()
    state = SimpleNamespace(
        current_position=(1, 1),
        entities=[
            SimpleNamespace(
                entity_id="entity-1",
                position=(2, 1),
                metadata={},
            )
        ],
    )
    observation = {"hud": {"health": 10}}
    choices = []
    for _ in range(10):
        choice = learner.choose(
            state,
            observation,
            ["control_1", "control_2"],
        )
        if choice is None:
            break
        choices.append(choice)

    assert len(choices) == 6
    assert set(choices) == {"control_1", "control_2"}


def test_learner_discovers_control_effects_and_selects_toward_requested_direction() -> None:
    learner = ControlLearner()
    state = SimpleNamespace(current_position=(0, 0), entities=[])
    controls = ["opaque_a", "opaque_b"]

    first = learner.choose(
        state,
        {"position": list(state.current_position)},
        controls,
        desired_direction=(1, 0),
    )
    assert first is not None
    state.current_position = (-1, 0)
    second = learner.choose(
        state,
        {"position": list(state.current_position)},
        controls,
        desired_direction=(1, 0),
    )
    assert second is not None and second != first
    state.current_position = (0, 0)
    third = learner.choose(
        state,
        {"position": list(state.current_position)},
        controls,
        desired_direction=(1, 0),
    )

    assert third == second
    assert learner.latest_inference is not None
    assert learner.latest_inference["mean_effects"]["__agent_dx"] == 1.0
    assert learner.predicted_displacement(second) == (1.0, 0.0)


def test_learner_does_not_repeat_stationary_controls_when_direction_is_unknown() -> None:
    learner = ControlLearner()
    stationary_samples = [
        {
            "__agent_dx": 0.0,
            "__agent_dy": 0.0,
            "__agent_x": float(x),
            "__agent_y": 0.0,
        }
        for x in range(4)
    ]
    blocked_samples = stationary_samples[:2]
    learner.trials = {
        (): {
            "move_up": [{
                "__agent_dx": 0.0,
                "__agent_dy": -1.0,
                "__agent_x": 0.0,
                "__agent_y": 0.0,
            }],
            "move_right": [{
                "__agent_dx": 1.0,
                "__agent_dy": 0.0,
                "__agent_x": 0.0,
                "__agent_y": 0.0,
            }],
            "move_down": [{
                "__agent_dx": 0.0,
                "__agent_dy": 1.0,
                "__agent_x": 0.0,
                "__agent_y": 0.0,
            }],
            "blocked_left": blocked_samples,
            "stationary": stationary_samples,
        }
    }

    choice = learner.choose(
        SimpleNamespace(current_position=(3, 0), entities=[]),
        {},
        ["move_up", "move_right", "move_down", "blocked_left", "stationary"],
        desired_direction=(-1, 0),
    )

    assert choice == "blocked_left"


def test_policy_can_execute_controls_not_hardcoded_in_action_registry() -> None:
    policy = ExplorationPolicy()
    action = policy.decide(
        {"controls": ["unfamiliar_control"], "confidence": 0.5},
        MemoryStore(),
    )

    assert action is not None
    assert action.name == "unfamiliar_control"
