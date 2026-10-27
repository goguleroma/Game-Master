from core.memory import MemoryStore
from core.logger import Logger
from core.types import Action, ActionResult, GameState, Observation
from decision.planner import Planner
from perception.capture import ScreenCapture
from perception.state_extractor import StateExtractor
from control.input_adapter import InputAdapter


class GameEngine:
    def __init__(self) -> None:
        self.capture = ScreenCapture()
        self.state_extractor = StateExtractor()
        self.planner = Planner()
        self.input_adapter = InputAdapter()
        self.memory = MemoryStore()
        self.logger = Logger()

    def run(self) -> None:
        self.logger.info("Game engine started")

        while True:
            frame = self.capture.get_frame()
            if frame is None:
                self.logger.warning("No frame captured; retrying.")
                continue

            self.process_frame(frame)

    def process_frame(self, frame: object, source: str = "capture") -> ActionResult | None:
        observation = Observation(
            image=frame,
            timestamp=0.0,
            metadata={"source": source},
        )
        self.logger.info(f"Observation received: {frame}")

        state = self.state_extractor.extract(observation)
        self.logger.info(
            f"Extracted state: objects={state.objects}, hud={state.hud}, "
            f"goal={state.goal}, confidence={state.confidence}"
        )

        action = self.planner.choose_action(state)
        if action is None:
            self.logger.info("Planner selected no action.")
            return None

        self.logger.info(
            f"Planner decision (rule-based): {action.name}, params={action.params}"
        )
        result = self.input_adapter.execute(action)
        self.memory.record({
            "state": state,
            "action": action,
            "result": result,
        })
        self.logger.info(
            f"Action result: success={result.success}, details={result.details}"
        )
        return result
