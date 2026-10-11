"""The ``benchmarks`` package root: the run provenance every bench entry point records.

``repo_provenance`` says which code ran (git commit and dirty flag, and the
path each installed package resolves to, so ``maps`` and ``benchmarks`` bound
to different checkouts show) and under what run context; ``stamp_repo_provenance``
logs the code half as a grep-able ``EVAL-PROVENANCE`` block.
"""

from __future__ import annotations

import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _git_provenance() -> dict:
    """Commit and dirty flag for this checkout, None where git cannot say."""
    import subprocess

    out: dict = {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if commit.returncode == 0:
            out["git_commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if status.returncode == 0:
            out["git_dirty"] = bool(status.stdout.strip())
    except Exception:
        pass  # git absent / not a repo -> leave None; the package path still pins code
    return out


def _resolved_package_paths() -> dict:
    """Each package's origin as ``import <pkg>`` will bind it, found without importing it."""
    import importlib.util

    out: dict = {}
    for pkg in ("maps", "benchmarks"):
        try:
            spec = importlib.util.find_spec(pkg)
            out[pkg] = spec.origin if spec is not None else None
        except Exception:
            out[pkg] = None
    return out


def repo_provenance(
    *,
    config_path: str | pathlib.Path | None = None,
    split: str | None = None,
    seed: int | None = None,
    limit: int | None = None,
    search_flags: dict | None = None,
    served_model: str | None = None,
    served_build: str | None = None,
) -> dict:
    """The provenance record: git, package paths, and each run fact given."""
    prov: dict = {
        "repo_root": str(_REPO_ROOT),
        **_git_provenance(),
        "packages": _resolved_package_paths(),
    }
    if config_path is not None:
        prov["config"] = str(config_path)
        from benchmarks.citynav.run_benchmark.config import config_sha256  # noqa: PLC0415

        prov["config_sha256"] = config_sha256(config_path)
    # One served alias can name two quantizations; the tokenizer (served_build) tells them apart.
    for key, value in (
        ("split", split),
        ("seed", seed),
        ("limit", limit),
        ("search_env_flags", None if search_flags is None else dict(search_flags)),
        ("served_model", served_model),
        ("served_build", served_build),
    ):
        if value is not None:
            prov[key] = value
    return prov


def stamp_repo_provenance() -> None:
    """Log the ``EVAL-PROVENANCE`` block (commit, dirty flag, package paths) to run.log."""
    import logging

    prov = repo_provenance()
    lines = [
        "EVAL-PROVENANCE "
        f"commit={prov['git_commit']} dirty={prov['git_dirty']} "
        f"repo_root={prov['repo_root']}",
    ]
    lines += [f"  {name}.__file__={path}" for name, path in prov["packages"].items()]
    logging.getLogger("benchmarks.provenance").info("\n".join(lines))
