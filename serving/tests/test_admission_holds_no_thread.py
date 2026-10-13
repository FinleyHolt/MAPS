"""A request waiting at the segmenter's admission gate holds no thread, so a
crowd of waiters cannot starve admitted forwards or ``/health`` of the pool
they share."""

from __future__ import annotations

import ast
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "sam_server.py"


def test_the_admission_gate_waits_on_the_event_loop() -> None:
    tree = ast.parse(SERVER.read_text())
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert isinstance(fns["_admission"], ast.AsyncFunctionDef)
    built = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_inflight_sem" for t in n.targets)]
    assert len(built) == 1 and ast.unparse(built[0].func) == "asyncio.BoundedSemaphore"
