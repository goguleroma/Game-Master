from typing import Any


class ScreenCapture:
    def __init__(self, frame: Any | None = None) -> None:
        self._frame = frame

    def set_frame(self, frame: Any | None) -> None:
        self._frame = frame

    def get_frame(self) -> Any | None:
        return self._frame
