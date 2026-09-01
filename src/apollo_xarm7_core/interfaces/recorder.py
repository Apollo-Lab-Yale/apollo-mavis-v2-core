"""Episode recorder facade (design doc 01-core §5.2).

The runtime implements this over LeRobotDataset v3 (lerobot lives THERE,
never in core).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EpisodeRecorder(ABC):
    """Buffered episode recording with explicit save/discard."""

    @abstractmethod
    def start(self, meta: dict[str, object]) -> None:
        """Open a new episode buffer."""

    @abstractmethod
    def add_frame(self, frame: dict[str, object]) -> None:
        """Append one frame (-> ``LeRobotDataset.add_frame``)."""

    @abstractmethod
    def save(self) -> int:
        """Commit the buffer; returns the episode_index (``save_episode``)."""

    @abstractmethod
    def discard(self) -> None:
        """Drop the buffer (``clear_episode_buffer``)."""

    @abstractmethod
    def finalize(self) -> None:
        """MANDATORY at session end (writes parquet footers)."""

    @property
    @abstractmethod
    def recording(self) -> bool:
        """True while an episode buffer is open."""
