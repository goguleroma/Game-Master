from core.types import Action, ActionResult


class InputAdapter:
    def execute(self, action: Action) -> ActionResult:
        if action.name == "wait":
            return ActionResult(success=True, details={"status": "waiting"})

        return ActionResult(success=False, details={"unknown_action": action.name})
