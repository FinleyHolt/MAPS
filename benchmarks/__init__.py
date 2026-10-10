"""The ``benchmarks`` package root: run-provenance shared across the eval harnesses.

The CityNav harness (per-episode search loop, flight executor, ``maps-bench``
CLI) lives in ``benchmarks.citynav``; the canonical SR readers in
``benchmarks.citynav.analysis``. This package root holds only the provenance stamp.

Provenance is the audit surface behind every SR number. The repo ships two
installed packages (``maps`` / ``benchmarks``), so an editable install could in
principle bind them to different checkouts; ``stamp_repo_provenance()`` records
each package's resolved path so cross-package skew is visible rather than
silent. What it records is *which code ran*: the git commit and dirty flag, the
resolved package paths, and the run's config hash / split / seed / served model. That triple -- ``(sha, run,
config)`` -- is what lets any number be reproduced or contested. Call
``stamp_repo_provenance()`` at the top of each entry point; it logs a grep-able
``EVAL-PROVENANCE`` block to run.log and returns the provenance dict.
"""

from __future__ import annotations

import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _git_provenance() -> dict:
    """Best-effort commit + dirty flag for this checkout. Never raises."""
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
    """Resolve each installed package's origin WITHOUT importing it.

    ``importlib.util.find_spec`` walks the same ``sys.path`` a future import will,
    so each path is what ``import <pkg>`` will bind to -- making cross-package
    editable skew (e.g. ``maps`` and ``benchmarks`` bound to different checkouts)
    visible in the provenance record rather than silent."""
    import importlib.util

    out: dict = {}
    for pkg in ("maps", "benchmarks"):
        try:
            spec = importlib.util.find_spec(pkg)
            out[pkg] = spec.origin if spec is not None else None
        except Exception:
            out[pkg] = None
    return out


def sha256_file(path: str | pathlib.Path) -> str | None:
    """Hex sha256 of a file's bytes (used to pin config identity). None on error."""
    import hashlib

    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except Exception:
        return None


def _resolved_look_tool() -> bool:
    """Whether the reasoner will actually be shown images this run.

    Not the env flag: ``search_env_flags`` records ``SEARCH_LOOK_TOOL`` only
    when the launching shell exported it, so with the tool on by default the
    flag is absent precisely when the tool is ON. The audit needs the resolved
    answer, because "did the agent have eyes" is not readable from anything
    else on the row — a blind run and a seeing run differ in no other field.
    """
    from maps.search.search_loop import SearchLoopConfig

    return bool(SearchLoopConfig.from_env().look_tool)


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
    """The full provenance record: git + resolved package path + (optional) the
    run's config hash, split/seed/limit, flag set, and served model. Pure data
    (no logging) so it can be embedded into episodes.jsonl rows as well as
    logged. Cheap; safe to call once per run."""
    prov: dict = {
        "repo_root": str(_REPO_ROOT),
        **_git_provenance(),
        "packages": _resolved_package_paths(),
    }
    if config_path is not None:
        prov["config"] = str(config_path)
        from benchmarks.citynav.run_benchmark.config import config_sha256  # noqa: PLC0415

        prov["config_sha256"] = config_sha256(config_path)
    if split is not None:
        prov["split"] = split
    if seed is not None:
        prov["seed"] = seed
    if limit is not None:
        prov["limit"] = limit
    if search_flags is not None:
        prov["search_env_flags"] = dict(search_flags)
    if served_model is not None:
        prov["served_model"] = served_model
    # The served ALIAS is not the served build. `inkling-small` named both
    # `thinkingmachines/Inkling-Small-NVFP4` and `RedHatAI/Inkling-Small-FP8-
    # dynamic` on the same cluster in the same week, and the pair read as a
    # NULL through comparability_issues -- -8.7 pts at McNemar p=0.009,
    # quoted as what identical code does. The tokenizer identity is the
    # finest-grained build fact the endpoint spec measures.
    if served_build is not None:
        prov["served_build"] = served_build
    prov["look_tool"] = _resolved_look_tool()
    return prov


def stamp_repo_provenance(*, logger=None, **kwargs) -> dict:
    """Log a loud, grep-able ``EVAL-PROVENANCE`` block (the post-hoc audit surface)
    and return the provenance dict. Call at the top of every bench entry point;
    pass whatever run context is known (``config_path``/``split``/``seed``/
    ``limit``/``search_flags``/``served_model``). Logging lands in run.log via
    the runner's basicConfig."""
    import logging

    prov = repo_provenance(**kwargs)
    log = logger or logging.getLogger("benchmarks.provenance")
    pkgs = prov["packages"]
    lines = [
        "EVAL-PROVENANCE "
        f"commit={prov.get('git_commit')} dirty={prov.get('git_dirty')} "
        f"repo_root={prov['repo_root']}",
    ]
    for name, path in pkgs.items():
        lines.append(f"  {name}.__file__={path}")
    if "config" in prov:
        lines.append(f"  config={prov['config']} sha256={prov.get('config_sha256')}")
    ctx = {k: prov[k] for k in ("split", "seed", "limit", "served_model") if k in prov}
    if ctx:
        lines.append("  " + "  ".join(f"{k}={v}" for k, v in ctx.items()))
    if "search_env_flags" in prov:
        lines.append(f"  search_env_flags={prov['search_env_flags']}")
    lines.append(
        f"  look_tool={prov.get('look_tool')} "
        f"({'reasoner sees crops' if prov.get('look_tool') else 'reasoner is TEXT-ONLY'})"
    )
    log.info("\n".join(lines))
    return prov
