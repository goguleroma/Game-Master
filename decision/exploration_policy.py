from core.memory import MemoryStore
from core.world_model import WorldModel
from decision.action_manager import ActionManager
from decision.planner import Planner


class ExplorationPolicy:
    def __init__(self) -> None:
        self.planner = Planner()
        self.world_model = WorldModel()
        self.action_manager = ActionManager()

    def decide(self, observation: dict, memory: MemoryStore):
        state = self.world_model.update(observation, memory.working.last_position)
        action = self.planner.choose_action(state, memory)

        if action is None:
            return self.action_manager.get_action("wait")

        if action.name == "explore_frontier":
            frontier = action.params["frontier"]
            return self.action_manager.get_action("explore_frontier", frontier=frontier)

        return self.action_manager.get_action(action.name, **action.params)
