import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _DirectionEvidence:
    dx_total: float = 0.0
    dy_total: float = 0.0
    moving_samples: int = 0
    origins: set[tuple[int, int]] = field(default_factory=set)

    def observe(self, sample: dict[str, float]) -> None:
        if not all(
            feature in sample
            for feature in ("__agent_dx", "__agent_dy", "__agent_x", "__agent_y")
        ):
            return
        self.origins.add(
            (int(sample["__agent_x"]), int(sample["__agent_y"]))
        )
        dx = sample["__agent_dx"]
        dy = sample["__agent_dy"]
        if dx != 0 or dy != 0:
            self.dx_total += dx
            self.dy_total += dy
            self.moving_samples += 1


class ControlLearner:
    def __init__(self) -> None:
        self.trials: dict[
            tuple[int, ...], dict[str, list[dict[str, float]]]
        ] = {}
        self.outcomes: dict[tuple[int, ...], dict[str, list[float]]] = {}
        self._pending: tuple[
            tuple[int, ...],
            str,
            dict[str, float],
            tuple[float | None, float],
            tuple[int, int] | None,
        ] | None = None
        self._rng = random.Random(0)
        self.latest_inference: dict[str, Any] | None = None
        self._memory_loaded = False
        self._direction_evidence: dict[str, _DirectionEvidence] = {}

    def choose(
        self,
        state: object,
        observation: dict[str, object],
        available_controls: object,
        memory: object | None = None,
        desired_direction: tuple[int, int] | None = None,
    ) -> str | None:
        if not self._memory_loaded and memory is not None:
            load_trials = getattr(memory, "load_control_trials", None)
            if callable(load_trials):
                self.trials = load_trials()
            load_outcomes = getattr(memory, "load_control_outcomes", None)
            if callable(load_outcomes):
                self.outcomes = load_outcomes()
            self._direction_evidence.clear()
            for context_trials in self.trials.values():
                for control, samples in context_trials.items():
                    for sample in samples:
                        self._add_direction_evidence(control, sample)
            self._memory_loaded = True
        self._observe_transition(observation, state, memory)
        if not isinstance(available_controls, list):
            return None

        controls = [
            control
            for control in available_controls
            if isinstance(control, str) and control
        ]
        if not controls:
            return None

        context = self._context(state)
        if desired_direction is not None:
            control = self._choose_toward_direction(
                controls,
                desired_direction,
                self._position(state),
            )
            if control is not None:
                self._pending = (
                    context,
                    control,
                    self._numeric_snapshot(observation),
                    self._utility(state, memory),
                    self._position(state),
                )
                return control

        context_trials = self.trials.setdefault(context, {})
        if all(
            len(context_trials.get(control, [])) >= 3
            and all(
                not any(
                    effect != 0
                    for feature, effect in sample.items()
                    if feature not in {"__agent_x", "__agent_y"}
                )
                for sample in context_trials.get(control, [])
            )
            for control in controls
        ):
            return None

        context_outcomes = self.outcomes.setdefault(context, {})
        counts = {
            control: len(context_outcomes.get(control, []))
            for control in controls
        }
        total = sum(counts.values())
        scores = {
            control: (
                (
                    sum(context_outcomes.get(control, [])) / counts[control]
                    if counts[control]
                    else 0.0
                )
                + 0.35
                * math.sqrt(math.log(total + 2) / (counts[control] + 1))
            )
            for control in controls
        }
        best_score = max(scores.values())
        candidates = [
            control for control, score in scores.items() if score == best_score
        ]
        control = self._rng.choice(candidates)
        snapshot = self._numeric_snapshot(observation)
        self._pending = (
            context,
            control,
            snapshot,
            self._utility(state, memory),
            self._position(state),
        )
        return control

    def _choose_toward_direction(
        self,
        controls: list[str],
        desired_direction: tuple[int, int],
        current_position: tuple[int, int] | None,
    ) -> str | None:
        learned: dict[str, tuple[float, float]] = {}
        unexplored: list[str] = []
        for control in controls:
            evidence = self._direction_evidence.get(control)
            if evidence is None or not evidence.origins:
                unexplored.append(control)
                continue
            if evidence.moving_samples:
                learned[control] = (
                    evidence.dx_total / evidence.moving_samples,
                    evidence.dy_total / evidence.moving_samples,
                )
                continue
            if len(evidence.origins) >= 4:
                learned[control] = (0.0, 0.0)
            elif current_position not in evidence.origins:
                unexplored.append(control)
        aligned = [
            (
                direction[0] * desired_direction[0]
                + direction[1] * desired_direction[1],
                control,
            )
            for control, direction in learned.items()
            if direction[0] * desired_direction[0]
            + direction[1] * desired_direction[1]
            > 0
        ]
        if aligned:
            return max(aligned, key=lambda item: (item[0], item[1]))[1]
        if unexplored:
            return self._rng.choice(unexplored)
        if learned:
            return max(
                controls,
                key=lambda control: (
                    learned.get(control, (0.0, 0.0))[0] * desired_direction[0]
                    + learned.get(control, (0.0, 0.0))[1] * desired_direction[1],
                    abs(learned.get(control, (0.0, 0.0))[0])
                    + abs(learned.get(control, (0.0, 0.0))[1]),
                    control,
                ),
            )
        return None

    def _add_direction_evidence(
        self,
        control: str,
        sample: dict[str, float],
    ) -> None:
        self._direction_evidence.setdefault(
            control,
            _DirectionEvidence(),
        ).observe(sample)

    def predicted_displacement(
        self,
        control: str,
    ) -> tuple[float, float] | None:
        evidence = self._direction_evidence.get(control)
        if evidence is None or evidence.moving_samples == 0:
            return None
        return (
            evidence.dx_total / evidence.moving_samples,
            evidence.dy_total / evidence.moving_samples,
        )

    def _observe_transition(
        self,
        observation: dict[str, object],
        state: object,
        memory: object | None,
    ) -> None:
        if self._pending is None:
            return
        context, control, before, before_utility, before_position = self._pending
        after = self._numeric_snapshot(observation)
        effects = {
            key: after.get(key, 0.0) - before.get(key, 0.0)
            for key in before.keys() | after.keys()
            if after.get(key, 0.0) != before.get(key, 0.0)
        }
        after_position = self._position(state)
        if before_position is not None and after_position is not None:
            effects["__agent_dx"] = float(after_position[0] - before_position[0])
            effects["__agent_dy"] = float(after_position[1] - before_position[1])
            effects["__agent_x"] = float(before_position[0])
            effects["__agent_y"] = float(before_position[1])
        samples = self.trials.setdefault(context, {}).setdefault(control, [])
        samples.append(effects)
        self._add_direction_evidence(control, effects)
        after_utility = self._utility(state, memory)
        if before_utility[0] is not None and after_utility[0] is not None:
            reward = before_utility[0] - after_utility[0]
        else:
            reward = after_utility[1] - before_utility[1]
        self._record_outcome(context, control, reward, memory)
        save_trial = getattr(memory, "save_control_trial", None)
        if callable(save_trial):
            save_trial(context, control, effects)
        features = set().union(*(sample.keys() for sample in samples))
        mean_effects = {
            feature: sum(sample.get(feature, 0.0) for sample in samples) / len(samples)
            for feature in features
        }
        self.latest_inference = {
            "control": control,
            "context": context,
            "samples": len(samples),
            "mean_reward": sum(self.outcomes[context][control])
            / len(self.outcomes[context][control]),
            "mean_effects": {
                feature: effect
                for feature, effect in mean_effects.items()
                if effect != 0 and feature not in {"__agent_x", "__agent_y"}
            },
        }
        self._pending = None

    def observe_action_result(
        self,
        details: dict[str, object],
        memory: object | None,
    ) -> None:
        if self._pending is None:
            return
        status = details.get("status")
        if status not in {"goal_reached", "won", "caught", "lost"}:
            return
        context, control, _, _, before_position = self._pending
        reward = 1.0 if status in {"goal_reached", "won"} else -1.0
        position = details.get("position")
        after_position = self._position_value(position)
        if before_position is not None and after_position is not None:
            effects = {
                "__agent_dx": float(after_position[0] - before_position[0]),
                "__agent_dy": float(after_position[1] - before_position[1]),
                "__agent_x": float(before_position[0]),
                "__agent_y": float(before_position[1]),
            }
            self.trials.setdefault(context, {}).setdefault(control, []).append(effects)
            self._add_direction_evidence(control, effects)
            save_trial = getattr(memory, "save_control_trial", None)
            if callable(save_trial):
                save_trial(context, control, effects)
        self._record_outcome(context, control, reward, memory)
        self._pending = None

    def _record_outcome(
        self,
        context: tuple[int, ...],
        control: str,
        reward: float,
        memory: object | None,
    ) -> None:
        outcomes = self.outcomes.setdefault(context, {}).setdefault(control, [])
        outcomes.append(reward)
        save_outcome = getattr(memory, "save_control_outcome", None)
        if callable(save_outcome):
            save_outcome(context, control, reward)

    @staticmethod
    def _utility(
        state: object,
        memory: object | None,
    ) -> tuple[float | None, float]:
        position = getattr(state, "current_position", None)
        target = getattr(state, "target_position", None)
        distance = ControlLearner._manhattan_distance(position, target)
        goal_progress: float | None = None
        if distance is not None:
            goal_progress = float(distance)
        grid_size = getattr(state, "grid_size", None)
        if (
            isinstance(grid_size, tuple)
            and len(grid_size) == 2
            and all(isinstance(value, int) and value > 0 for value in grid_size)
        ):
            area = max(grid_size[0] * grid_size[1], 1)
            if goal_progress is not None:
                goal_progress /= max(sum(grid_size), 1)
        else:
            area = 1
        spatial = getattr(memory, "spatial", None)
        known_cells = getattr(spatial, "grid", {})
        coverage = len(known_cells) / area if isinstance(known_cells, dict) else 0.0
        return goal_progress, coverage

    @staticmethod
    def _manhattan_distance(first: object, second: object) -> int | None:
        if (
            isinstance(first, (list, tuple))
            and len(first) == 2
            and all(isinstance(part, int) for part in first)
            and isinstance(second, (list, tuple))
            and len(second) == 2
            and all(isinstance(part, int) for part in second)
        ):
            return abs(first[0] - second[0]) + abs(first[1] - second[1])
        return None

    @staticmethod
    def _position(state: object) -> tuple[int, int] | None:
        return ControlLearner._position_value(
            getattr(state, "current_position", None)
        )

    @staticmethod
    def _position_value(position: object) -> tuple[int, int] | None:
        if (
            isinstance(position, (list, tuple))
            and len(position) == 2
            and all(isinstance(part, int) for part in position)
        ):
            return position[0], position[1]
        return None

    @staticmethod
    def _numeric_snapshot(observation: dict[str, object]) -> dict[str, float]:
        snapshot: dict[str, float] = {}

        def visit(value: object, path: str) -> None:
            if isinstance(value, bool):
                snapshot[path] = float(value)
                snapshot[f"{path}#present"] = 1.0
            elif isinstance(value, (int, float)):
                snapshot[path] = float(value)
                snapshot[f"{path}#present"] = 1.0
            elif isinstance(value, str):
                snapshot[f"{path}={value}"] = 1.0
            elif value is None:
                snapshot[f"{path}#null"] = 1.0
            elif isinstance(value, dict):
                snapshot[f"{path}#keys={','.join(sorted(value))}"] = 1.0
                for key, child in value.items():
                    visit(child, f"{path}.{key}" if path else str(key))
            elif isinstance(value, list):
                snapshot[f"{path}#length"] = float(len(value))
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")

        visit(observation, "")
        return snapshot

    @staticmethod
    def _context(state: object) -> tuple[int, ...]:
        position = getattr(state, "current_position", None)
        if position is None:
            return ()
        nearby_entities = [
            (
                abs(entity.position[0] - position[0])
                + abs(entity.position[1] - position[1]),
                sum(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in entity.metadata.values()
                ),
            )
            for entity in getattr(state, "entities", [])
            if entity.entity_id is not None and entity.position is not None
        ]
        if not nearby_entities:
            return ()
        nearest_distance = min(distance for distance, _ in nearby_entities)
        observable_features = sum(
            feature_count
            for distance, feature_count in nearby_entities
            if distance == nearest_distance
        )
        return nearest_distance, observable_features
