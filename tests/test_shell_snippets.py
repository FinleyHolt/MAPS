"""Every repository import a shell script makes resolves: no test runs the data-prep scripts, so a moved name would break them unseen."""

from __future__ import annotations

import importlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORT = re.compile(r"from ((?:benchmarks|maps|station|serving)[\w.]*) import ([\w, ]+?)(?: as \w+)?(?=[;\n'\"]|$)", re.M)


def test_every_name_a_shell_script_imports_exists():
    scripts = subprocess.run(["git", "ls-files", "*.sh"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    found = [(s, mod, name.strip()) for s in scripts for mod, names in IMPORT.findall((ROOT / s).read_text())
             for name in names.split(",")]
    assert len(found) >= 4  # vacuity floor: the bake, the fetch and the setup scripts each import the bench
    missing = [(s, mod, name) for s, mod, name in found if not hasattr(importlib.import_module(mod), name)]
    assert missing == []
