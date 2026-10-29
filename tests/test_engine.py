from core.engine import GameEngine
from core.memory import MemoryStore
from core.types import Action, ActionResult, GameState, Observation


class RecordingPolicy:
    def __init__(self) -> None:
        self.observations: list[GameState] = []
        self.results: list[ActionResult] = []

    def decide(self, observation: GameState, memory: MemoryStore) -> Action:
        self.observations.append(observation)
        return Action(name="opaque-action", params={"value": observation.hud["value"]})

    def observe_result(self, result: ActionResult, memory: MemoryStore) -> None:
        self.results.append(result)


class RecordingExecutor:
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> ActionResult:
        self.actions.append(action)
        return ActionResult(success=True, details={"status": "executed"})


class FixedObservationAdapter:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def extract(self, observation: Observation) -> GameState:
        self.observations.append(observation)
        return GameState(objects=["normalized"], hud={"value": "adapted"})


def test_engine_delegates_decision_and_execution_without_action_assumptions() -> None:
    policy = RecordingPolicy()
    executor = RecordingExecutor()
    engine = GameEngine(policy, executor)

    result = engine.process_frame(
        {
            "objects": ["unclassified-object"],
            "hud": {"value": "opaque-state"},
            "goal": "uninterpreted-intent",
            "confidence": 0.8,
        },
        source="contract-test",
    )

    assert result is not None
    assert result.success
    assert policy.observations[0].hud == {"value": "opaque-state"}
    assert executor.actions[0].name == "opaque-action"
    assert policy.results == [result]
    assert engine.memory.records[0]["action"] == executor.actions[0]


def test_engine_uses_injected_observation_adapter() -> None:
    policy = RecordingPolicy()
    executor = RecordingExecutor()
    adapter = FixedObservationAdapter()
    engine = GameEngine(policy, executor, state_extractor=adapter)

    result = engine.process_frame({"raw": "environment-data"}, source="adapter-test")

    assert result is not None
    assert adapter.observations[0].metadata["source"] == "adapter-test"
    assert policy.observations[0].objects == ["normalized"]
    assert policy.observations[0].hud == {"value": "adapted"}


def test_engine_records_observation_when_policy_declines_to_act() -> None:
    class NoActionPolicy(RecordingPolicy):
        def decide(self, observation: GameState, memory: MemoryStore) -> None:
            self.observations.append(observation)
            return None

    policy = NoActionPolicy()
    executor = RecordingExecutor()
    engine = GameEngine(policy, executor)

    result = engine.process_frame({"hud": {"value": 1}}, source="contract-test")

    assert result is None
    assert len(policy.observations) == 1
    assert executor.actions == []
    assert engine.memory.records == []


def test_engine_run_stops_after_max_steps() -> None:
    policy = RecordingPolicy()
    executor = RecordingExecutor()
    engine = GameEngine(policy, executor, max_steps=2)
    engine.capture.set_frame({"hud": {"value": "frame"}})

    engine.run()

    assert len(policy.observations) == 2
    assert len(executor.actions) == 2


def test_engine_run_stops_when_should_stop_is_requested() -> None:
    policy = RecordingPolicy()
    executor = RecordingExecutor()
    engine = GameEngine(policy, executor, should_stop=lambda: True)
    engine.capture.set_frame({"hud": {"value": "frame"}})

    engine.run()

    assert len(policy.observations) == 0
    assert len(executor.actions) == 0
