from core.memory import MemoryStore
from decision.exploration_policy import ExplorationPolicy


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
