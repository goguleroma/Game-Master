import random
from typing import Any


class ControlLearner:
    def __init__(self) -> None:
        self.trials: dict[
            tuple[int, ...], dict[str, list[dict[str, float]]]
        ] = {}
        self._pending: tuple[
            tuple[int, ...], str, dict[str, float]
        ] | None = None
        self._rng = random.Random(0)
        self.latest_inference: dict[str, Any] | None = None

    def choose(
        self,
        state: object,
        observation: dict[str, object],
        available_controls: object,
        action_names: set[str],
    ) -> str | None:
        self._observe_transition(observation)
        position = getattr(state, "current_position", None)
        if position is None or not isinstance(available_controls, list):
            return None

        controls = [
            control
            for control in available_controls
            if isinstance(control, str) and control in action_names
        ]
        if not controls:
            return None

        context = self._context(state)
        context_trials = self.trials.setdefault(context, {})
        uncertainties = {
            control: 1 / (len(context_trials.get(control, [])) + 1)
            for control in controls
        }
        best_uncertainty = max(uncertainties.values())
        if best_uncertainty < 1 / 3:
            return None

        candidates = [
            control
            for control, uncertainty in uncertainties.items()
            if uncertainty == best_uncertainty
        ]
        control = self._rng.choice(candidates)
        self._pending = (context, control, self._numeric_snapshot(observation))
        return control

    def _observe_transition(self, observation: dict[str, object]) -> None:
        if self._pending is None:
            return
        context, control, before = self._pending
        after = self._numeric_snapshot(observation)
        effects = {
            key: after.get(key, 0.0) - before.get(key, 0.0)
            for key in before.keys() | after.keys()
            if after.get(key, 0.0) != before.get(key, 0.0)
        }
        samples = self.trials.setdefault(context, {}).setdefault(control, [])
        samples.append(effects)
        features = set().union(*(sample.keys() for sample in samples))
        mean_effects = {
            feature: sum(sample.get(feature, 0.0) for sample in samples) / len(samples)
            for feature in features
        }
        self.latest_inference = {
            "control": control,
            "context": context,
            "samples": len(samples),
            "mean_effects": {
                feature: effect
                for feature, effect in mean_effects.items()
                if effect != 0
            },
        }
        self._pending = None

    @staticmethod
    def _numeric_snapshot(observation: dict[str, object]) -> dict[str, float]:
        snapshot: dict[str, float] = {}

        def visit(value: object, path: str) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, (int, float)):
                snapshot[path] = float(value)
                snapshot[f"{path}#present"] = 1.0
            elif isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{path}.{key}" if path else str(key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")

        visit(observation, "")
        return snapshot

    @staticmethod
    def _context(state: object) -> tuple[int, ...]:
        position = getattr(state, "current_position", None)
        if position is None:
            return ()
        distances = [
            abs(entity.position[0] - position[0])
            + abs(entity.position[1] - position[1])
            for entity in getattr(state, "entities", [])
            if entity.entity_id is not None and entity.position is not None
        ]
        return tuple(sorted(distances))
