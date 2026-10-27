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

            observation = Observation(
                image=frame,
                timestamp=0.0,
                metadata={"source": "capture"},
            )

            state = self.state_extractor.extract(observation)
            action = self.planner.choose_action(state)

            if action is None:
                self.logger.info("No action selected; waiting.")
                continue

            result = self.input_adapter.execute(action)
            self.memory.record({
                "state": state,
                "action": action,
                "result": result,
            })

            self.logger.info(f"Action executed: {action.name} -> {result.success}")
