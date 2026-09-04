"""Microphone REST model and status vocabulary (design doc 01-core §12; phase-11).

``GET /api/microphones -> list[MicrophoneInfo]`` (04-runtime §13.1). The
``MicStatus`` literal is shared with ``MicrophoneTelemetry`` (§11) so the REST
snapshot and the telemetry block spell device state identically. This module
is a dependency-free leaf (pure pydantic) so ``protocol.telemetry`` can import
it without cycles.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

MicStatus = Literal["no_backend", "starting", "absent", "live", "stalled", "error"]
# no_backend: microphone disabled / no capture backend importable
# starting:   capture thread opening the source
# absent:     configured but the device is not present (unplugged)
# live:       frames arriving within stale_s
# stalled:    device present, no frame for > stale_s
# error:      open/read failure (runtime retries with backoff)


class MicrophoneInfo(BaseModel):
    """GET /api/microphones row (RØDE NT-USB Mini on the view arm, 05-ui §8.1)."""

    mic_id: str  # e.g. "mic_view"
    label: str  # operator-facing name
    kind: Literal["pulse", "fake", "none"]  # capture route actually in use
    source: str | None  # PulseAudio source name (None when unresolved / fake / none)
    sample_rate: int  # Hz (48000 for the NT-USB Mini)
    channels: int = 1
    live: bool  # status == "live" (waveform preview available pre-session)
    status: MicStatus
    detail: str = ""  # human-readable reason for absent/error/no_backend


__all__ = [
    "MicStatus",
    "MicrophoneInfo",
]
