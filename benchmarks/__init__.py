"""The ``benchmarks`` package root: the checkout every bench path defaults
under, and which code ran (``git_provenance``: the git commit and dirty flag).
"""

from __future__ import annotations

import pathlib
import subprocess

#: The checkout every bench path defaults under.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


#: A copy deployed without git (an rsync or a git archive export) names its commit here.
COMMIT_STAMP = REPO_ROOT / ".commit"


def git_provenance() -> dict:
    """Commit and dirty flag for this checkout, None where git cannot say. A
    tree that is not its own git work tree (no git, or an export inside
    another repository) reads its commit from ``COMMIT_STAMP``; dirty stays
    None."""
    out: dict = {"git_commit": None, "git_dirty": None}

    def _git(*args):
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=5
        )

    try:
        head = _git("rev-parse", "--show-toplevel", "HEAD")
        lines = head.stdout.splitlines()
        if (
            head.returncode == 0
            and len(lines) == 2
            and pathlib.Path(lines[0]).resolve() == REPO_ROOT.resolve()
        ):
            out["git_commit"] = lines[1]
            status = _git("status", "--porcelain")
            if status.returncode == 0:
                out["git_dirty"] = bool(status.stdout.strip())
    except Exception:
        pass  # git absent / not a repo -> leave None
    if out["git_commit"] is None and COMMIT_STAMP.is_file():
        out["git_commit"] = COMMIT_STAMP.read_text().strip() or None
    return out
