from typing import Any, Protocol

from core.types import GameState, Observation


class ObservationAdapter(Protocol):
    """Protocol defining the interface for observation adapters.

    An observation adapter bridges the gap between raw environment frames
    and the GameState representation used by the engine's decision policy.
    """

    def get_frame(self) -> Any | None:
        """Capture and return a raw frame/image from the environment.

        Returns:
            Raw frame data (e.g., numpy array, PIL Image, or dict), or None if unavailable.
        """
        ...

    def extract(self, observation: Observation) -> GameState:
        """Convert a raw observation into a structured GameState.

        Args:
            observation: Raw observation containing image data and metadata.

        Returns:
            Structured GameState with objects, HUD, goal, confidence, and controls.
        """
        ...

    def reset(self) -> None:
        """Reset the adapter's internal state (e.g., clear buffers, reconnect)."""
        ...

    @property
    def is_real_time(self) -> bool:
        """Whether this adapter provides real-time capture (True) or simulated (False)."""
        ...

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """Expected shape of frames returned by get_frame(), e.g., (H, W, C)."""
        ...