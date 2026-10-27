from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Observation:
    image: Any
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameState:
    objects: List[str] = field(default_factory=list)
    hud: Dict[str, Any] = field(default_factory=dict)
    goal: str | None = None
    confidence: float = 0.0


@dataclass
class Action:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
