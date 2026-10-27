from core.memory import MemoryStore
from core.world_model import WorldModel
from decision.action_manager import ActionManager
from decision.control_learner import ControlLearner
from decision.planner import Planner


class ExplorationPolicy:
    def __init__(self) -> None:
        self.planner = Planner()
        self.world_model = WorldModel()
        self.action_manager = ActionManager()
        self.control_learner = ControlLearner()

    def decide(self, observation: object, memory: MemoryStore):
        if hasattr(observation, "hud"):
            normalized_observation = {
                "objects": observation.objects,
                "hud": observation.hud,
                "goal": observation.goal,
                "confidence": observation.confidence,
                "controls": observation.controls,
            }
        elif isinstance(observation, dict):
            normalized_observation = observation
        else:
            return self.action_manager.get_action("wait")

        state = self.world_model.update(normalized_observation)
        if state.grid_size is not None:
            current_turn = len(memory.experiences)
            visible = set(state.visible_cells)
            for cell in visible:
                memory.mark_cell(
                    cell,
                    seen=True,
                    obstacle=False,
                    value="",
                    turn=current_turn,
                )
            for obstacle in state.obstacle_positions:
                memory.mark_cell(
                    obstacle,
                    seen=True,
                    obstacle=True,
                    value="obstacle",
                    turn=current_turn,
                )
            if state.target_position is not None:
                memory.mark_cell(
                    state.target_position,
                    seen=True,
                    obstacle=False,
                    value="target",
                    turn=current_turn,
                )
            if state.current_position is not None:
                memory.mark_cell(
                    state.current_position,
                    seen=True,
                    visited=True,
                    obstacle=False,
                    value="agent",
                    turn=current_turn,
                )
            for entity in state.entities:
                if entity.entity_id is not None and entity.position is not None:
                    memory.observe_entity(
                        entity.entity_id,
                        entity.position,
                        state.current_position,
                        current_turn,
                    )
            memory.update_frontier(*state.grid_size)
            memory.update_working_memory(
                normalized_observation,
                memory.working.last_action,
                memory.working.last_result,
                state.current_position,
            )

        available_controls = normalized_observation.get("controls", [])
        control = self.control_learner.choose(
            state,
            normalized_observation,
            available_controls,
            set(self.action_manager.action_registry),
        )
        if control is not None:
            return self.action_manager.get_action(control)

        # Preserve the original rule-based behavior for non-spatial scenarios.
        planner_state = state if state.grid_size is not None else observation
        action = self.planner.choose_action(planner_state, memory)

        if action is None:
            return self.action_manager.get_action("wait")

        if action.name == "explore_frontier":
            frontier = action.params["frontier"]
            return self.action_manager.get_action("explore_frontier", frontier=frontier)

        return self.action_manager.get_action(action.name, **action.params)
