from core.types import Action, GameState


class Planner:
    def choose_action(self, state: GameState) -> Action | None:
        if state is None:
            return None

        objects = state.objects or []
        objective = state.goal

        if "start_button" in objects:
            return Action(name="click", params={"x": 500, "y": 400})

        if objective == "move_left":
            return Action(name="move_left", params={"steps": 1})

        if objective == "observe":
            return Action(name="wait", params={"duration": 0.1})

        if "unknown" in objects:
            return Action(name="wait", params={"duration": 0.1})

        return Action(name="wait", params={"duration": 0.1})
