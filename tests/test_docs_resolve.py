"""Every tracked Markdown file points at things that exist: its relative links, the modules its ``python -m`` commands name, and the source files it names by path."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("maps", "benchmarks", "station", "serving", "tests")
#: ``python -m`` targets that are not this repository's.
OUTSIDE = {"venv", "pytest"}

LINK = re.compile(r"\]\(([^)\s]+)\)")
MODULE = re.compile(r"python3? -m ([\w.]+)")
SOURCE_PATH = re.compile(r"`([\w./-]+/[\w.-]+\.(?:py|sh))`")


def _markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ROOT / line for line in out.split()]


def _module_exists(name: str) -> bool:
    stem = ROOT.joinpath(*name.split("."))
    return stem.with_suffix(".py").is_file() or (stem / "__main__.py").is_file()


def test_every_relative_link_resolves():
    links = [
        (md, target)
        for md in _markdown()
        for target in LINK.findall(md.read_text())
        if not target.startswith(("http://", "https://", "mailto:", "#"))
    ]
    assert len(links) >= 40  # vacuity floor: the README alone links every package and document
    broken = [
        (str(md.relative_to(ROOT)), target)
        for md, target in links
        if not (md.parent / target.split("#")[0]).exists()
    ]
    assert broken == []


def test_every_python_m_names_a_module_in_the_tree():
    found = [
        (md, name)
        for md in _markdown()
        for name in MODULE.findall(md.read_text())
        if name not in OUTSIDE
    ]
    assert len(found) >= 40  # vacuity floor: the experiment, data and flight commands
    missing = [
        (str(md.relative_to(ROOT)), name)
        for md, name in found
        if name.split(".")[0] not in PACKAGES or not _module_exists(name)
    ]
    assert missing == []


def test_every_source_path_names_a_file():
    """A ``.py`` or ``.sh`` path in backticks is read from the repository root, or from the document's own directory or one above it."""
    found = [(md, path) for md in _markdown() for path in SOURCE_PATH.findall(md.read_text())]
    assert len(found) >= 80  # vacuity floor: maps/README.md alone names the engine's modules
    missing = [
        (str(md.relative_to(ROOT)), path)
        for md, path in found
        if not any(
            (base / path).is_file()
            for base in (ROOT, md.parent, *md.parent.parents)
            if ROOT in (base, *base.parents)
        )
    ]
    assert missing == []
