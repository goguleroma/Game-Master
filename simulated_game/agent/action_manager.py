from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.types import Action


@dataclass
class ActionSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


class ActionManager:
    def __init__(self) -> None:
        self.action_registry: Dict[str, ActionSpec] = {
            "wait": ActionSpec(name="wait", params={"duration": 0.1}),
            "click": ActionSpec(name="click", params={"x": 0, "y": 0}),
            "move_left": ActionSpec(name="move_left", params={"steps": 1}),
            "move_up": ActionSpec(name="move_up", params={"steps": 1}),
            "move_right": ActionSpec(name="move_right", params={"steps": 1}),
            "move_down": ActionSpec(name="move_down", params={"steps": 1}),
            "explore_frontier": ActionSpec(name="explore_frontier", params={"frontier": (0, 0)}),
        }

    def get_action(self, name: str, **kwargs) -> Action:
        spec = self.action_registry.get(name)
        if spec is None:
            raise ValueError(f"Unknown action: {name}")

        action = Action(name=spec.name, params=dict(spec.params))
        action.params.update(kwargs)
        return action

    @staticmethod
    def get_control_action(name: str, available_controls: object) -> Action:
        if (
            not isinstance(available_controls, list)
            or not isinstance(name, str)
            or name not in available_controls
        ):
            raise ValueError(f"Unavailable control: {name}")
        return Action(name=name)
