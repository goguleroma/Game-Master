from typing import Any, Callable, Protocol

from core.memory import MemoryStore
from core.logger import Logger
from core.types import Action, ActionResult, GameState, Observation


class DecisionPolicy(Protocol):
    def decide(self, observation: GameState, memory: Any) -> Action | None: ...

    def observe_result(self, result: ActionResult, memory: Any) -> None: ...


class AgentMemory(Protocol):
    @property
    def records(self) -> list[dict[str, Any]]: ...

    def observe_frame(self, frame: object) -> None: ...

    def record(self, entry: dict[str, Any]) -> None: ...

    def update_working_memory(
        self,
        observation: object,
        action: str | None,
        result: str | None,
    ) -> None: ...


class ActionExecutor(Protocol):
    def execute(self, action: Action) -> ActionResult: ...


class ObservationAdapter(Protocol):
    def extract(self, observation: Observation) -> GameState: ...


class GameEngine:
    def __init__(
        self,
        policy: DecisionPolicy,
        input_adapter: ActionExecutor,
        observation_adapter: "ObservationAdapter",
        memory: AgentMemory | None = None,
        max_steps: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.observation_adapter = observation_adapter
        self.policy = policy
        self.input_adapter = input_adapter
        self.memory = memory if memory is not None else MemoryStore()
        self.logger = Logger()
        self.max_steps = max_steps
        self._should_stop = should_stop

    def run(self) -> None:
        self.logger.info("Game engine started")
        steps = 0
        while True:
            if self._should_stop is not None and self._should_stop():
                self.logger.info("Stop requested; engine loop exiting.")
                return
            if self.max_steps is not None and steps >= self.max_steps:
                self.logger.info(f"Step limit of {self.max_steps} reached; engine loop exiting.")
                return
            frame = self.observation_adapter.get_frame()
            if frame is None:
                self.logger.warning("No frame captured; retrying.")
                continue

            self.process_frame(frame)
            if self.max_steps is not None:
                steps += 1

    def process_frame(self, frame: object, source: str = "capture") -> ActionResult | None:
        observation = Observation(
            image=frame,
            timestamp=0.0,
            metadata={"source": source},
        )
        self.logger.info(f"Observation received: {frame}")

        state = self.observation_adapter.extract(observation)
        self.memory.observe_frame(state)
        self.logger.info(f"Extracted state with confidence={state.confidence}")

        action = self.policy.decide(state, self.memory)
        if action is None:
            self.logger.info("Policy selected no action.")
            return None

        self.logger.info(
            f"Policy decision: {action.name}, params={action.params}"
        )
        result = self.input_adapter.execute(action)
        self.policy.observe_result(result, self.memory)
        self.memory.record({
            "state": state,
            "action": action,
            "result": result,
        })
        self.memory.update_working_memory(
            state,
            action.name,
            result.details.get("status"),
        )
        self.logger.info(
            f"Action result: success={result.success}, details={result.details}"
        )
        return result