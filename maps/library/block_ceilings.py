"""The liveness ceilings of a running block, charged to the block's own
lines and nothing else, and the block line an exception was raised on."""

from __future__ import annotations

import contextlib
import sys
import threading
import traceback
import types
from typing import TYPE_CHECKING

from maps.library.block_scan import BLOCK_FILENAME

if TYPE_CHECKING:
    from maps.library.interpreter import MissionPython


# ``sys.monitoring`` delivers LINE events for exactly the blocks' code
# objects, so the executor beneath the block runs untraced.

_MON = sys.monitoring
_MON_LOCK = threading.Lock()
_MON_TOOL: list = [None]  # the tool id, once claimed
_MON_ACTIVE: dict = {}  # id(code object) -> the MissionPython running it


def code_objects(code) -> list:
    """A code object and every code object nested in its constants: the
    block's lambdas, comprehensions and defs all count as its own lines."""
    out = [code]
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out.extend(code_objects(c))
    return out


def _on_line(code, _line):
    mp = _MON_ACTIVE.get(id(code))
    if mp is not None:
        mp._tick()


def _monitor_tool() -> int:
    with _MON_LOCK:
        if _MON_TOOL[0] is None:
            tid = next(i for i in (3, 4, 5, 2, 1, 0) if _MON.get_tool(i) is None)
            _MON.use_tool_id(tid, "maps-block-ceilings")
            _MON.register_callback(tid, _MON.events.LINE, _on_line)
            _MON_TOOL[0] = tid
        return _MON_TOOL[0]


@contextlib.contextmanager
def charging_lines(mp: MissionPython):
    """Charge ``mp._tick`` for every line of the mission's blocks while one
    runs."""
    tid = _monitor_tool()
    codes = mp._codes
    with _MON_LOCK:
        for c in codes:
            _MON_ACTIVE[id(c)] = mp
            _MON.set_local_events(tid, c, _MON.events.LINE)
    try:
        yield
    finally:
        with _MON_LOCK:
            for c in codes:
                _MON.set_local_events(tid, c, 0)
                _MON_ACTIVE.pop(id(c), None)


def block_line(codes: set) -> int:
    """The failing line of the block the model just sent (``codes``, its
    code objects), off the live traceback."""
    _, _, tb = sys.exc_info()
    line = 0
    for fr, lineno in traceback.walk_tb(tb):
        if fr.f_code in codes:
            line = lineno or line
    return line


def outer_block_line() -> tuple[int, int | None]:
    """The (line, column) the block's top-level code was running when it
    raised: the first block frame of the live traceback. The column tells
    apart two statements on one line; ``None`` without position info."""
    _, _, tb = sys.exc_info()
    for fr in traceback.extract_tb(tb):
        if fr.filename == BLOCK_FILENAME:
            return fr.lineno or 0, fr.colno
    return 0, None
