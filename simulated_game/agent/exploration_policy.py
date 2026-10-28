from core.types import ActionResult
from .action_manager import ActionManager
from .control_learner import ControlLearner
from .memory import GridMemoryStore
from .planner import GridStrategyParameters, Planner
from .world_model import WorldModel


class ExplorationPolicy:
    def __init__(
        self,
        strategy: GridStrategyParameters | None = None,
    ) -> None:
        self.planner = Planner(strategy)
        self.world_model = WorldModel()
        self.action_manager = ActionManager()
        self.control_learner = ControlLearner()
        self._pending_step: tuple[tuple[int, int], tuple[int, int]] | None = None

    def observe_result(self, result: ActionResult, memory: GridMemoryStore) -> None:
        details = result.details
        self.control_learner.observe_action_result(details, memory)
        if (
            self._pending_step is not None
            and details.get("status") in {"blocked_by_boundary", "blocked_by_obstacle"}
        ):
            self.planner.record_blocked_transition(*self._pending_step)
        self._pending_step = None

    def decide(self, observation: object, memory: GridMemoryStore):
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
            memory.observe_goal(state.target_position, list(visible))
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
            )

        planner_state = state if state.grid_size is not None else observation
        action = self.planner.choose_action(planner_state, memory)
        available_controls = normalized_observation.get("controls", [])
        desired_direction = (
            {
                "move_up": (0, -1),
                "move_right": (1, 0),
                "move_down": (0, 1),
                "move_left": (-1, 0),
            }.get(action.name)
            if action is not None
            else None
        )
        control = None
        if state.grid_size is None or desired_direction is not None:
            control = self.control_learner.choose(
                state,
                normalized_observation,
                available_controls,
                memory,
                desired_direction,
            )
        if control is not None:
            predicted_direction = self.control_learner.predicted_displacement(control)
            self._pending_step = (
                (state.current_position, desired_direction)
                if state.current_position is not None
                and desired_direction is not None
                and predicted_direction
                == (float(desired_direction[0]), float(desired_direction[1]))
                else None
            )
            return self.action_manager.get_control_action(
                control,
                available_controls,
            )

        self._pending_step = (
            (state.current_position, desired_direction)
            if state.current_position is not None and desired_direction is not None
            else None
        )
        if action is None:
            return self.action_manager.get_action("wait")

        if action.name == "explore_frontier":
            frontier = action.params["frontier"]
            return self.action_manager.get_action("explore_frontier", frontier=frontier)

        return self.action_manager.get_action(action.name, **action.params)
