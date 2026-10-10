"""Test config: the code under
test is this checkout's, never an editable-install skew (the benchmark skew
guard, same pattern as maps/search/tests)."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
