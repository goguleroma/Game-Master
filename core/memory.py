from collections import deque
from dataclasses import dataclass
import time
from typing import Any


class ShortTermMemory:
    def __init__(
        self,
        max_frames: int = 60,
        max_age_seconds: float = 12.0,
    ) -> None:
        self.max_frames = max_frames
        self.max_age_seconds = max_age_seconds
        self.frames: deque[tuple[float, object]] = deque(maxlen=max_frames)

    def add(self, frame: object, timestamp: float | None = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        self.frames.append((now, frame))
        while self.frames and now - self.frames[0][0] > self.max_age_seconds:
            self.frames.popleft()


@dataclass
class WorkingMemory:
    current_observation: object | None = None
    last_action: str | None = None
    last_result: str | None = None


class MemoryStore:
    """Environment-neutral short-term and action-history memory."""

    def __init__(self) -> None:
        self.short_term = ShortTermMemory()
        self.working = WorkingMemory()
        self.experiences: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        return self.experiences

    def observe_frame(self, frame: object) -> None:
        self.short_term.add(frame)

    def record(self, entry: dict[str, Any]) -> None:
        self.experiences.append(entry)

    def update_working_memory(
        self,
        observation: object,
        action: str | None,
        result: str | None,
    ) -> None:
        self.working.current_observation = observation
        self.working.last_action = action
        self.working.last_result = result
