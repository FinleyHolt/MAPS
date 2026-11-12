"""The ``benchmarks`` package root: the run provenance every bench entry point records.

``repo_provenance`` says which code ran (git commit and dirty flag) and under
what run context.
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
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True, timeout=5)
    try:
        head = _git("rev-parse", "--show-toplevel", "HEAD")
        lines = head.stdout.splitlines()
        if head.returncode == 0 and len(lines) == 2 \
                and pathlib.Path(lines[0]).resolve() == REPO_ROOT.resolve():
            out["git_commit"] = lines[1]
            status = _git("status", "--porcelain")
            if status.returncode == 0:
                out["git_dirty"] = bool(status.stdout.strip())
    except Exception:
        pass  # git absent / not a repo -> leave None
    if out["git_commit"] is None and COMMIT_STAMP.is_file():
        out["git_commit"] = COMMIT_STAMP.read_text().strip() or None
    return out


def repo_provenance(
    *,
    config_path: str | pathlib.Path | None = None,
    split: str | None = None,
    search_flags: dict | None = None,
    served_model: str | None = None,
    served_build: str | None = None,
) -> dict:
    """The provenance record: git, and each run fact given."""
    prov: dict = git_provenance()
    if config_path is not None:
        prov["config"] = str(config_path)
        from benchmarks.run_record.config import config_sha256  # noqa: PLC0415

        prov["config_sha256"] = config_sha256(config_path)
    # One served alias can name two quantizations; the tokenizer (served_build) tells them apart.
    for key, value in (
        ("split", split),
        ("search_env_flags", None if search_flags is None else dict(search_flags)),
        ("served_model", served_model),
        ("served_build", served_build),
    ):
        if value is not None:
            prov[key] = value
    return prov
