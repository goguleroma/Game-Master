from core.types import GameState, Observation


class StateExtractor:
    def extract(self, observation: Observation) -> GameState:
        return GameState(
            objects=["unknown"],
            hud={"status": "no_data"},
            goal=None,
            confidence=0.0,
        )
