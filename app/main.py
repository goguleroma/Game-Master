import sys
from core.engine import GameEngine
from core.memory import MemoryStore
from core.logger import Logger
from core.types import Action, ActionResult, GameState, Observation
from perception.capture import ScreenCapture, ScreenCaptureAdapter
from perception.state_extractor import SimpleStateExtractor


class DemoPolicy:
    """Simple demo policy that logs observations and takes a fixed action."""

    def __init__(self) -> None:
        self.step_count = 0

    def decide(self, observation: GameState, memory: MemoryStore) -> Action | None:
        self.step_count += 1
        print(f"[DemoPolicy] Step {self.step_count}: Observed {observation.objects}, confidence={observation.confidence}")
        if self.step_count >= 3:
            return None
        return Action(name="demo_action", params={"step": self.step_count})

    def observe_result(self, result: ActionResult, memory: MemoryStore) -> None:
        print(f"[DemoPolicy] Result: success={result.success}, details={result.details}")


class DemoExecutor:
    """Simple demo action executor."""

    def execute(self, action: Action) -> ActionResult:
        print(f"[DemoExecutor] Executing: {action.name} with params={action.params}")
        return ActionResult(success=True, details={"status": "executed", "action": action.name})


class TestCapture(ScreenCapture):
    """Test capture that allows frame injection for demos."""

    def __init__(self) -> None:
        # Don't call parent init - we don't want real capture
        self._frame = None
        self._backend = "test"
        self._current_fps = 0.0
        self.config = type('Config', (), {'height': 480, 'width': 640})()

    def set_frame(self, frame: object) -> None:
        self._frame = frame

    def get_frame(self) -> object | None:
        return self._frame

    def start(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def extract(self, observation: Observation) -> GameState:
        from core.types import GameState
        return GameState(
            objects=["captured_frame"],
            hud={"backend": self._backend, "fps": self._current_fps},
            goal="capture",
            confidence=1.0 if self._frame is not None else 0.0,
        )


def create_observation_adapter() -> ScreenCaptureAdapter:
    """Create and configure the observation adapter with real components."""
    capture = TestCapture()
    extractor = SimpleStateExtractor()
    return ScreenCaptureAdapter(capture=capture, state_extractor=extractor)


def main() -> None:
    print("Game Master - Demo Mode", file=sys.stderr)
    print("=" * 40, file=sys.stderr)

    # Create components
    policy = DemoPolicy()
    executor = DemoExecutor()
    observation_adapter = create_observation_adapter()
    memory = MemoryStore()
    logger = Logger()

    # Inject a test frame for demonstration
    test_frame = {
        "objects": ["player", "enemy", "coin"],
        "hud": {"health": 100, "score": 0},
        "goal": "collect_coin",
        "confidence": 0.95,
        "controls": ["move_left", "move_right", "jump"],
    }
    observation_adapter._capture.set_frame(test_frame)

    # Create and run engine
    engine = GameEngine(
        policy=policy,
        input_adapter=executor,
        observation_adapter=observation_adapter,
        memory=memory,
        max_steps=5,
    )

    print("Starting engine loop...", file=sys.stderr)
    engine.run()
    print("Engine stopped.", file=sys.stderr)

    print("\nFinal memory records:", len(memory.records))
    for i, record in enumerate(memory.records):
        print(f"  {i+1}. action={record['action'].name}, result={record['result'].success}")


if __name__ == "__main__":
    main()