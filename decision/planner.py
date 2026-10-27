from core.types import Action, GameState


class Planner:
    def choose_action(self, state: GameState) -> Action | None:
        if not state:
            return None

        if "unknown" in state.objects:
            return Action(name="wait", params={"duration": 0.1})

        return Action(name="wait", params={"duration": 0.1})
