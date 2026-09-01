"""Camera implementation facade (design doc 01-core §5.1).

LeRobot-style Camera ABC: implementations own a background capture thread;
consumers poll :meth:`latest` and treat ``None`` as degraded-but-valid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..state import CameraFrame


class CameraInterface(ABC):
    """One RGB camera stream."""

    @abstractmethod
    def start(self) -> None:
        """Spawn the capture thread; raises :class:`CameraInitError` on failure."""

    @abstractmethod
    def stop(self) -> None:
        """Join the capture thread; idempotent."""

    @abstractmethod
    def latest(self) -> CameraFrame | None:
        """Non-blocking newest frame; ``None`` if never captured or stale."""

    @property
    @abstractmethod
    def camera_id(self) -> str: ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """(width, height) pixels."""

    @property
    @abstractmethod
    def fps(self) -> float: ...
