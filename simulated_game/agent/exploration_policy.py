from typing import Any, Tuple

from core.types import ActionResult
from .action_manager import ActionManager
from .control_learner import ControlLearner, WorldModelLearner
from .memory import GridMemoryStore
from .path_tree import PathTreePlanner
from .planner import GridStrategyParameters, Planner, ActionCandidate
from .world_model import WorldModel, WorldState


class ExplorationPolicy:
    def __init__(
        self,
        strategy: GridStrategyParameters | None = None,
    ) -> None:
        self.strategy = strategy or GridStrategyParameters()
        self.planner = Planner(self.strategy)  # BFS fallback
        self.world_model = WorldModel()
        self.action_manager = ActionManager()
        self.control_learner = ControlLearner()
        self.path_tree = PathTreePlanner(
            max_depth=7,
            num_simulations=30,
            exploration_constant=1.414,
            num_particles=25,
            control_learner=self.control_learner,
            world_model_learner=self.control_learner,  # ControlLearner IS a WorldModelLearner
            memory=None,  # Set in set_memory
            seed=42,  # Deterministic per episode
        )
        self._memory: GridMemoryStore | None = None
        self._pending_step: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._pending_decision_case: tuple[
            dict[str, object],
            tuple[int, int],
            tuple[int, int],
            tuple[int, int] | None,
        ] | None = None
        # Store state before action for recording control trials
        self._before_state: WorldState | None = None
        self._before_observation: dict[str, Any] | None = None
        self._before_action: ActionCandidate | None = None

    def set_memory(self, memory: GridMemoryStore) -> None:
        """Set the memory store and wire it to PathTreePlanner and ControlLearner."""
        self._memory = memory
        self.path_tree.memory = memory
        self.control_learner.load_trials(memory.load_control_trials())

    def _context_to_key(self, context: dict[str, Any]) -> tuple[int, ...]:
        """Convert decision context dict to tuple key for control_learner."""
        # Extract features in consistent order matching retrieve_direction_preferences
        required_keys = [
            "goal_visible",
            "goal_known",
            "target_direction",
            "goal_dynamic",
            "adjacent_obstacles",
            "entity_distance_bucket",
            "health_bucket",
            "frontier_bucket",
        ]
        key_parts: list[int] = []
        for k in required_keys:
            v = context.get(k)
            if v is None:
                key_parts.append(-1)
            elif isinstance(v, bool):
                key_parts.append(1 if v else 0)
            elif isinstance(v, (list, tuple)):
                # Flatten list/tuple (e.g., target_direction [1, 0] -> 1, 0)
                for item in v:
                    if item is None:
                        key_parts.append(-1)
                    else:
                        key_parts.append(int(item))
            elif isinstance(v, int):
                key_parts.append(v)
            else:
                key_parts.append(0)
        return tuple(key_parts)

    def _compute_effects(
        self,
        before: WorldState,
        after: WorldState,
        details: dict[str, Any],
    ) -> dict[str, float]:
        """Compute effects from state transition for control learning."""
        effects: dict[str, float] = {}

        # Displacement
        if before.current_position is not None and after.current_position is not None:
            effects["displacement_x"] = float(after.current_position[0] - before.current_position[0])
            effects["displacement_y"] = float(after.current_position[1] - before.current_position[1])

        # Status
        status = details.get("status")
        if isinstance(status, str):
            effects["status"] = float(hash(status) % 1000)  # Hash for numeric representation

        # Target distance change
        if before.target_position is not None and before.current_position is not None:
            before_dist = abs(before.current_position[0] - before.target_position[0]) + abs(
                before.current_position[1] - before.target_position[1]
            )
            if after.current_position is not None:
                after_dist = abs(after.current_position[0] - before.target_position[0]) + abs(
                    after.current_position[1] - before.target_position[1]
                )
                effects["target_distance_delta"] = float(before_dist - after_dist)

        return effects

    def _extract_after_state(
        self,
        before_state: WorldState,
        details: dict[str, Any],
    ) -> WorldState | None:
        """Extract after state from action result details."""
        position = details.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            return None
        after_state = WorldState(
            current_position=(int(position[0]), int(position[1])),
            target_position=before_state.target_position,
            obstacle_positions=before_state.obstacle_positions,
            grid_size=before_state.grid_size,
            visible_cells=before_state.visible_cells,
            entities=before_state.entities,
        )
        return after_state

    def observe_result(self, result: ActionResult, memory: GridMemoryStore) -> None:
        details = result.details
        pending_case = self._pending_decision_case
        self._pending_decision_case = None
        if pending_case is not None:
            context, direction, position, target = pending_case
            record_outcome = getattr(memory, "record_decision_outcome", None)
            if callable(record_outcome):
                record_outcome(context, direction, position, target, details)
        self.control_learner.observe_action_result(details, memory)
        
        # Record control trial for learning
        if (
            self._before_action is not None
            and self._before_state is not None
            and self._before_action.name.startswith("control_")
        ):
            # Extract after_state from details
            after_state = self._extract_after_state(self._before_state, details)
            if after_state is not None:
                self.record_result(self._before_action, self._before_state, after_state, details)
        
        if (
            self._pending_step is not None
            and details.get("status") in {"blocked_by_boundary", "blocked_by_obstacle"}
        ):
            self.planner.record_blocked_transition(*self._pending_step)
            # Mark the blocked cell as obstacle in memory for PathTreePlanner
            start_pos, direction = self._pending_step
            blocked_pos = (start_pos[0] + direction[0], start_pos[1] + direction[1])
            memory.mark_cell(blocked_pos, seen=True, obstacle=True, value="obstacle")
        self._pending_step = None
        self._before_state = None
        self._before_observation = None
        self._before_action = None

    def decide(self, observation: object, memory: GridMemoryStore):
        if self._memory is None:
            self.set_memory(memory)

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
                # Don't overwrite known obstacles
                existing = memory.spatial.grid.get(cell)
                is_known_obstacle = existing is not None and existing.obstacle is True
                memory.mark_cell(
                    cell,
                    seen=True,
                    obstacle=False if not is_known_obstacle else True,
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
                    # Also update the world model learner's entity dynamics
                    self.control_learner.observe_entity(
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

        # --- PATH TREE PLANNER FOR WORLDSTATE ---
        if isinstance(state, WorldState) and state.grid_size is not None and self._memory is not None:
            available_controls = normalized_observation.get("controls", [])
            
            # If controls are opaque (not standard control_1..control_6), just execute first one
            standard_controls = {"control_1", "control_2", "control_3", "control_4", "control_5", "control_6"}
            has_standard = any(c in standard_controls for c in available_controls)
            has_opaque = available_controls and not has_standard
            
            if has_opaque:
                # Opaque controls - execute first one directly without planning
                control = available_controls[0]
                self._pending_step = None
                self._pending_decision_case = None
                return self.action_manager.get_control_action(control, available_controls)
            
            # Build context for retrieval priors
            context = self._memory.build_decision_context(state, normalized_observation)
            direction_preferences = self._memory.retrieve_direction_preferences(context)

            # Update root retrieval priors for this turn
            self.path_tree.memory = self._memory

            # Use PathTreePlanner (MCTS)
            candidate = self.path_tree.plan(state, self._memory)

            # Fallback to BFS planner if PathTree returns None or wait
            if candidate is None or candidate.name == "wait":
                candidate = self.planner.choose_action(state, self._memory, direction_preferences)

            # Record for learning
            self._memory.update_working_memory(
                observation=normalized_observation,
                action=candidate.name if candidate else None,
                result="pending",
            )

            desired_direction = (
                {
                    "move_up": (0, -1),
                    "move_right": (1, 0),
                    "move_down": (0, 1),
                    "move_left": (-1, 0),
                }.get(candidate.name)
                if candidate is not None
                else None
            )

            self._pending_decision_case = None
            if (
                state.grid_size is not None
                and state.current_position is not None
                and desired_direction is not None
            ):
                self._pending_decision_case = (
                    self._memory.build_decision_context(state, normalized_observation),
                    desired_direction,
                    state.current_position,
                    state.target_position,
                )

            # Try to use control_learner to map desired_direction to a control
            control = None
            if desired_direction is not None:
                control = self.control_learner.choose(
                    state,
                    normalized_observation,
                    available_controls,
                    self._memory,
                    desired_direction,
                )
            
            # Check if the control actually produces the desired direction
            use_control = False
            if control is not None:
                # Use the same context that control_learner uses internally
                context_key = WorldModelLearner._context_from_state(state)
                predicted_direction = self.control_learner.predicted_displacement(control, context_key)
                # Treat (0.0, 0.0) as "no prediction" - use control to learn
                has_prediction = (
                    predicted_direction is not None
                    and predicted_direction != (0.0, 0.0)
                )
                # Use control if prediction matches desired direction OR if we have no prediction yet (to learn)
                if has_prediction:
                    pred_dir = (int(round(predicted_direction[0])), int(round(predicted_direction[1])))
                    if pred_dir == desired_direction:
                        use_control = True
                else:
                    # No prediction yet - use control to learn its effects
                    use_control = True
                
                if use_control:
                    self._pending_step = (
                        (state.current_position, desired_direction)
                        if state.current_position is not None
                        else None
                    )
            
            if use_control:
                action = self.action_manager.get_control_action(
                    control,
                    available_controls,
                )
                self._before_state = state
                self._before_observation = normalized_observation
                self._before_action = action
                return action
            
            # Fallback: use cardinal action directly if control mapping is unreliable
            self._pending_step = (
                (state.current_position, desired_direction)
                if state.current_position is not None and desired_direction is not None
                else None
            )
            if candidate is None:
                return self.action_manager.get_action("wait")

            if candidate.name == "explore_frontier":
                frontier = candidate.params["frontier"]
                action = self.action_manager.get_action("explore_frontier", frontier=frontier)
            else:
                action = self.action_manager.get_action(candidate.name, **candidate.params)
            self._before_state = state
            self._before_observation = normalized_observation
            self._before_action = action
            return action

        # For non-WorldState (menu, etc): use BFS planner
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
        self._pending_decision_case = None
        if (
            state.grid_size is not None
            and state.current_position is not None
            and desired_direction is not None
        ):
            self._pending_decision_case = (
                memory.build_decision_context(state, normalized_observation),
                desired_direction,
                state.current_position,
                state.target_position,
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
            # Use the same context that control_learner uses internally
            prediction_context_key = WorldModelLearner._context_from_state(state)
            predicted_direction = self.control_learner.predicted_displacement(control, prediction_context_key)
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

    def record_result(
        self,
        action: ActionCandidate,
        before_state: WorldState,
        after_state: WorldState,
        details: dict[str, Any],
    ) -> None:
        """Record control trial for learning."""
        if action.name.startswith("control_") and self._memory is not None:
            context = self._memory.build_decision_context(before_state)
            context_key = self._context_to_key(context)
            effects = self._compute_effects(before_state, after_state, details)
            self.control_learner.record_trial(context_key, action.name, effects)
            # Persist if memory has database
            if self._memory.database_path is not None:
                self._memory.save_control_trial(context_key, action.name, effects)