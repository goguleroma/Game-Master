from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ActionCandidate:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    utility: float = 0.0
    confidence: float = 0.0


class Planner:
    def __init__(self) -> None:
        self.last_goal: Optional[Tuple[int, int]] = None

    def choose_action(self, state: Any, memory: Any = None) -> Optional[ActionCandidate]:
        if state is None:
            return None

        if hasattr(state, "goal") and state.goal is not None:
            if state.goal == "move_left":
                return ActionCandidate(name="move_left", params={"steps": 1}, utility=1.0, confidence=0.9)
            if state.goal == "start":
                return ActionCandidate(name="click", params={"x": 500, "y": 400}, utility=1.0, confidence=0.9)

        if hasattr(state, "objects"):
            if "start_button" in state.objects:
                return ActionCandidate(name="click", params={"x": 500, "y": 400}, utility=1.9, confidence=0.9)
            if "unknown" in state.objects:
                return ActionCandidate(name="wait", params={"duration": 0.1}, utility=0.2, confidence=0.2)

        if memory is not None and hasattr(memory, "spatial"):
            frontier = getattr(memory.spatial, "frontier", [])
            if frontier:
                return ActionCandidate(name="explore_frontier", params={"frontier": frontier[0]}, utility=0.8, confidence=0.7)

        return ActionCandidate(name="wait", params={"duration": 0.1}, utility=0.1, confidence=0.1)
