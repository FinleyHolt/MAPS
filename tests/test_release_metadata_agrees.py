"""The package metadata, the citation file and the README's BibTeX name one release.

``CITATION.cff``'s version and license are the package's, and the README's
BibTeX entry carries the citation file's title, authors, version and
repository.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _bibtex() -> dict[str, str]:
    readme = (_ROOT / "README.md").read_text()
    (entry,) = re.findall(r"```bibtex\n(.*?)```", readme, re.S)
    return dict(re.findall(r"^\s*(\w+)\s*=\s*\{(.*)\},?\s*$", entry, re.M))


def test_citation_pyproject_and_readme_agree():
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    cff = yaml.safe_load((_ROOT / "CITATION.cff").read_text())
    bib = _bibtex()
    assert cff["cff-version"] == "1.2.0"
    assert str(cff["version"]) == project["version"] == bib["version"]
    assert cff["license"] == project["license"]["text"] == "MIT"
    assert cff["title"] == bib["title"] == "MAPS"
    assert cff["repository-code"] == bib["url"]
    authors = " and ".join(f"{a['family-names']}, {a['given-names']}" for a in cff["authors"])
    assert authors == bib["author"]
    assert "Copyright (c) 2026 Finley Holt and the MAPS authors" in (_ROOT / "LICENSE").read_text()
