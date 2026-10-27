from core.types import Action, ActionResult


class InputAdapter:
    def execute(self, action: Action) -> ActionResult:
        if action.name == "wait":
            return ActionResult(success=True, details={"status": "waiting", "duration": action.params.get("duration", 0.1)})

        if action.name == "click":
            return ActionResult(success=True, details={"status": "clicked", "x": action.params.get("x"), "y": action.params.get("y")})

        if action.name == "move_left":
            return ActionResult(success=True, details={"status": "moved_left", "steps": action.params.get("steps", 1)})

        return ActionResult(success=False, details={"unknown_action": action.name})
