from core.types import GameState, Observation


class StateExtractor:
    def extract(self, observation: Observation) -> GameState:
        image = observation.image

        if image is None:
            return GameState(
                objects=["unknown"],
                hud={"status": "no_data"},
                goal=None,
                confidence=0.0,
            )

        if isinstance(image, dict):
            objects = image.get("objects", [])
            hud = image.get("hud", {})
            goal = image.get("goal")
            confidence = image.get("confidence", 0.0)
            return GameState(
                objects=objects,
                hud=hud,
                goal=goal,
                confidence=confidence,
            )

        return GameState(
            objects=["visible_scene"],
            hud={"status": "unknown"},
            goal="observe",
            confidence=0.3,
        )
