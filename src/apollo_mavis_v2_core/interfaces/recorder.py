"""Episode recorder facade (design doc 01-core §5.2).

The runtime implements this over the episode-directory store (10-frames §11:
one ``episodes/<episode_id>/`` per saved episode; LeRobot v3 is a derived
export). lerobot lives THERE, never in core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EpisodeRecorder(ABC):
    """Buffered episode recording with explicit save/discard."""

    @abstractmethod
    def start(self, meta: dict[str, object]) -> None:
        """Open a new episode buffer (mints the ``episode_id``; no filesystem op)."""

    def prepare(self) -> None:
        """Optional: open the episode's temp directory + video encoder BEFORE the
        first frame — called on the recorder thread right after ``episode_new`` so
        the encoder's start stall lands there, not under a moving arm (04-runtime
        §10.1). Default: nothing; ``add_frame`` must then open lazily."""
        return None

    @abstractmethod
    def add_frame(self, frame: dict[str, object]) -> None:
        """Append one frame: buffer the rows, feed the video encoder."""

    @abstractmethod
    def save(self, sidecar: dict[str, object], audio: object | None = None) -> tuple[int, str]:
        """Publish ``episodes/<episode_id>/`` atomically: videos, ``frames.parquet``,
        stats, ``audio.wav`` (from ``audio``), then ``episode.json`` = ``sidecar`` +
        the video / audio / stats blocks LAST, then the rename. Returns
        ``(ordinal in capture order, episode_id)``."""

    @abstractmethod
    def discard(self) -> None:
        """Cancel the encoder and remove the temp directory; nothing is left."""

    @abstractmethod
    def finalize(self) -> None:
        """Idempotent close: discard an open episode, close the encoder, sweep
        stale ``.tmp-*`` directories."""

    @property
    @abstractmethod
    def recording(self) -> bool:
        """True while an episode buffer is open."""

    @property
    @abstractmethod
    def episode_id(self) -> str | None:
        """The open episode's id (10-frames §11.3), None when idle."""
