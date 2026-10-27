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
