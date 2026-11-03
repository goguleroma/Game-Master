"""
Grid 30x30 adapter configuration.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict


@dataclass(frozen=True)
class GridStrategyParameters:
    """Strategy parameters for grid-based planners."""
    visit_penalty: float = 1.0
    information_gain_bonus: float = 0.25
    goal_change_threshold: float = 1.0

    def __post_init__(self) -> None:
        for name, value in self.to_mapping().items():
            low, high = (1, 5) if name == "goal_change_threshold" else (0, 2)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(
                    f"{name} must be a finite value between {low} and {high}."
                )
            if name == "goal_change_threshold" and not float(value).is_integer():
                raise ValueError(f"{name} must be a whole number.")

    @classmethod
    def from_mapping(cls, values: Dict[str, float]) -> "GridStrategyParameters":
        return cls(
            visit_penalty=float(values["visit_penalty"]),
            information_gain_bonus=float(values["information_gain_bonus"]),
            goal_change_threshold=float(values["goal_change_threshold"]),
        )

    def to_mapping(self) -> Dict[str, float]:
        return {
            "visit_penalty": self.visit_penalty,
            "information_gain_bonus": self.information_gain_bonus,
            "goal_change_threshold": self.goal_change_threshold,
        }


@dataclass
class GridGameConfig:
    """Configuration for GridGame."""
    width: int = 30
    height: int = 30
    vision_radius: int = 4
    enemy_vision_radius: int = 4
    moving_target: bool = True
    pursuer_enabled: bool = True
    seed: int | None = None