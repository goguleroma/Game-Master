"""
Observation adapter for Grid 30x30 game.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from core.types import GameState, Observation
from core.engine import ObservationAdapter


class GridObservationAdapter:
    """ObservationAdapter implementation for GridGame.
    
    Extracts GameState from raw observation dictionary or Observation object.
    """
    
    def extract(self, observation: Union[Observation, Dict[str, Any]]) -> GameState:
        # Handle plain dict (from game.observe())
        if isinstance(observation, dict):
            return self._extract_from_dict(observation)
        # Handle Observation object
        image = observation.image
        if isinstance(image, dict):
            return self._extract_from_dict(image)
        # Fallback for other image types
        return GameState()
    
    def _extract_from_dict(self, obs: Dict[str, Any]) -> GameState:
        objects = obs.get("objects", [])
        hud = obs.get("hud", {})
        goal = obs.get("goal")
        confidence = obs.get("confidence", 0.0)
        controls = obs.get("controls", [])
        
        return GameState(
            objects=objects,
            hud=hud,
            goal=goal,
            confidence=confidence,
            controls=controls,
        )