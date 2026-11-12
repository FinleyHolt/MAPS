"""The segmenter's startup and shutdown run through one lifespan handler: the
module imports without FastAPI's deprecation warning, and the shutdown hands
the model's memory back."""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import sys, types, warnings
warnings.simplefilter("error", DeprecationWarning)
torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch
import serving.sam_engine as engine
import serving.sam_server as server
from fastapi.testclient import TestClient
released = []
engine.release = lambda: released.append(1)
with TestClient(server.app):
    assert not released
assert released == [1], released
"""


def test_the_server_starts_and_stops_through_its_lifespan():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
