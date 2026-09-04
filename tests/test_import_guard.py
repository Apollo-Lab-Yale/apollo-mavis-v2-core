"""Import guard (design doc 01-core §18): core pulls in no heavy deps."""

from __future__ import annotations

import json
import subprocess
import sys

FORBIDDEN = {"mujoco", "xarm", "fastapi", "torch", "lerobot", "cv2", "mink", "zmq", "websockets"}

_CHILD = """
import importlib.util
import json
import sys
import time

t0 = time.perf_counter()
import apollo_mavis_v2_core  # noqa: F401
for name in ("apollo_mavis_v2_core.protocol", "apollo_mavis_v2_core.dagger.types"):
    if importlib.util.find_spec(name) is not None:
        __import__(name)
elapsed = time.perf_counter() - t0
print(json.dumps({"elapsed_s": elapsed, "modules": sorted(sys.modules)}))
"""


def test_import_guard() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)

    top_level = {m.split(".")[0] for m in payload["modules"]}
    leaked = top_level & FORBIDDEN
    assert not leaked, f"core imported banned modules: {sorted(leaked)}"

    assert payload["elapsed_s"] < 0.5, f"import took {payload['elapsed_s']:.3f}s (budget 0.5s)"
