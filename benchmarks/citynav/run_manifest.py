"""The run's ONE machine-written record: ``run_manifest.json`` at the run root.

Emitted after argv, endpoint discovery and episode selection resolve, before
the first episode flies. Identity fields are measured (the served model and
window off the installed endpoint spec, the bake off the tile files' bytes)
and the config is inlined, so the record is self-contained. The field station
writes one at the sortie root and one per mission workdir (``mission=``),
carrying the sampling that mission flew under.

Write-once: the first emit writes the file; a re-entry into the same root (a
resume) verifies its resolution against it and REFUSES on identity drift
(``IDENTITY_KEYS``): a resume under another bake/model/config/selection is
two runs wearing one run id. Git-sha drift is recorded as a ``resumes``
entry, not refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

from benchmarks import repo_provenance
from benchmarks.citynav.episodes import EPISODES_JSONL

SCHEMA = "maps.run_manifest/v1"
MANIFEST_NAME = "run_manifest.json"

# The experiment-arm flags every row carries (``arm_env_stamp``: dispatch
# stamps rows with it, ab_report refuses to pair across it). Set-only, except
# the tile bake, which is resolved.
ARM_ENV_FLAGS = (
    "SEARCH_CITYNAV_TILES",
    # The imagery raster the ortho crops read (the Cal City bridge resolves
    # SEARCH_CALCITY_RGBD_DIR onto it): two arms over two rasters of one
    # block share a bake and a split and differ on exactly this.
    "SEARCH_CITYNAV_RGBD_DIR",
    "SEARCH_SEED_BASE",
    # The envelope sweep's knobs (maps/llm/sampling.py); reasoner turns
    # sample at temperature 1.0, so these move the score directly.
    "SEARCH_SAMPLING_TEMPERATURE",
    "SEARCH_SAMPLING_PRESENCE_PENALTY",
    # Unset sends none and flies at the serve's default: a row without it
    # and a row with it are two arms.
    "SEARCH_SAMPLING_REASONING_EFFORT",
)
# Set keys under these prefixes join the stamp the same way (compaction
# budgets change reasoning behavior).
ARM_ENV_PREFIXES = ("SEARCH_COMPACT_",)

# The audited ambient env: the arm flags, plus the vars that select what runs
# (config, data roots, bridge/runner) and the endpoints (the two override
# vars fabricate served-model facts without probing). The manifest records
# every SET key; argv facts (split/seed/limit/workers) are explicit fields.
AUDITED_ENV = ARM_ENV_FLAGS + (
    "BENCH_SAMPLE_SEED",
    "CITYNAV_BENCH_CONFIG",
    "CITYNAV_ROOT",
    "SEARCH_BENCHMARK_BRIDGE",
    "SEARCH_BENCHMARK_NAME",
    "SEARCH_BENCHMARK_RUNNER_MODULE",
    "SEARCH_CITYNAV_DATA",
    "SEARCH_CITYNAV_SPLITS",
    "SAM_SERVER_URL",
    "SAM_VIDEO_SERVER_URL",
    "SEARCH_ENDPOINT_OFFLINE",
    "SEARCH_ENDPOINT_SPEC_OVERRIDE",
    "SEARCH_REASONER_URL",
)
AUDITED_ENV_PREFIXES = ARM_ENV_PREFIXES + ("SEARCH_CALCITY_",)
# Never recorded, ever. `.env` is the only home for secrets.
SECRET_ENV = ("SEARCH_REASONER_API_KEY",)

# Drift in any of these between the stored manifest and a re-entering
# process refuses the re-entry. Dotted paths into the manifest dict.
IDENTITY_KEYS = (
    ("config_sha256",),
    ("split",),
    ("selection", "sample_seed"),
    ("selection", "sample_offset"),
    ("selection", "selection_sha"),
    ("endpoints", "model_id"),
    ("endpoints", "max_model_len"),
    ("perception", "bake_sha256"),
    ("search_env_flags",),
)
# Arm flags that name a host path, not the arm: one bake at two paths is one
# arm, and ``perception.bake_sha256`` is its identity.
_HOST_ARM_FLAGS = ("SEARCH_CITYNAV_TILES",)

class ManifestIdentityError(RuntimeError):
    """A re-entering run's resolution contradicts the stored manifest."""


def arm_env_stamp() -> dict:
    """The set arm flags, plus the tile bake resolved: an unset
    SEARCH_CITYNAV_TILES records ``paths.TILES_DIR``, so a default-bake row
    proves its bake instead of reading as unrecorded."""
    stamp = {k: os.environ[k] for k in ARM_ENV_FLAGS if os.environ.get(k, "") != ""}
    stamp.update({k: v for k, v in os.environ.items()
                  if k.startswith(ARM_ENV_PREFIXES) and v != ""})
    if "SEARCH_CITYNAV_TILES" not in stamp:
        from benchmarks.citynav import paths

        stamp["SEARCH_CITYNAV_TILES"] = str(paths.TILES_DIR)
    return stamp


_EFFORT = "SEARCH_SAMPLING_REASONING_EFFORT"


def settle_effort(model_id: str) -> None:
    """Write the arm's reasoning effort back to ``os.environ`` as the
    requests carry it (stripped; blank is unset), after the served card's
    check (``model_cards.check_effort``, which raises ``ValueError``), so
    the stamp and every reader see one spelling. Called once, after
    endpoint discovery and before any stamp or spawn."""
    from maps.llm.model_cards import check_effort  # noqa: PLC0415

    raw = os.environ.get(_EFFORT)
    if raw is None:
        return
    effort = raw.strip()
    if not effort:
        del os.environ[_EFFORT]
        return
    check_effort(model_id, effort)
    os.environ[_EFFORT] = effort


def _audited_env() -> dict:
    keys = set(AUDITED_ENV) | {k for k in os.environ if k.startswith(AUDITED_ENV_PREFIXES)}
    return {k: os.environ[k] for k in sorted(keys - set(SECRET_ENV)) if k in os.environ}


def _endpoint_section() -> dict:
    """The installed endpoint spec: served facts as validated at startup.
    Raises if no spec is installed — the emit point sits after discovery,
    so an uninstalled spec is a caller bug, not a degradable state."""
    from maps.llm.endpoint_spec import spec

    s = spec()
    return {
        "base_url": s.base_url,
        "model_id": s.model_id,
        "max_model_len": s.max_model_len,
        "server_declared_window": s.server_declared_window,
        "tokenizer": s.tokenizer_name,
        "output_reserve": s.output_reserve,
        "input_budget": s.input_budget,
        "max_output_tokens": s.max_output_tokens,
        "sam_url": s.sam_url,
        "sam_video_url": s.sam_video_url,
        "offline": os.environ.get("SEARCH_ENDPOINT_OFFLINE") or None,
        "spec_override": os.environ.get("SEARCH_ENDPOINT_SPEC_OVERRIDE") or None,
    }


def _perception_section() -> dict:
    """The tile bake this process resolves, by path and by content.

    ``bake_sha256`` is the one content check, so a reader can tell two
    same-named bakes apart after the fact.
    """
    from benchmarks.citynav import paths

    return {
        "tiles_dir_resolved": str(paths.TILES_DIR),
        "bake_sha256": paths.bake_sha256(),
    }


def build_manifest(
    *,
    out_root: Path,
    config_path: Path,
    split: str,
    sample_seed: int,
    sample_offset: int,
    limit: int,
    workers: int,
    allowlist_path: Path | None,
    episodes: list[str],
    mission: dict | None = None,
) -> dict:
    """Compose the complete resolved record. Every field is committed
    config, argv selection, runtime measurement, or derived — never a bare
    ambient copy (ambient values appear only in the ``env`` map).

    ``mission`` is the field station's per-mission grain: the mission's own
    identity and sampling (``episode_id``, ``run_id``, ``run_no``,
    ``reasoning_effort``), recorded whole under
    ``mission``. A ``reasoning_effort`` it names REPLACES the arm's in
    ``search_env_flags``, because that is what the mission's requests
    carried; None leaves the arm's stamp standing."""
    arm = arm_env_stamp()
    if mission is not None and mission.get("reasoning_effort") is not None:
        arm["SEARCH_SAMPLING_REASONING_EFFORT"] = str(mission["reasoning_effort"])
    endpoints = _endpoint_section()
    prov = repo_provenance(
        config_path=str(config_path),
        split=split,
        seed=sample_seed,
        limit=limit,
        search_flags=arm,
        served_model=endpoints["model_id"],
        served_build=endpoints["tokenizer"],
    )
    import yaml  # noqa: PLC0415

    from benchmarks.citynav.run_benchmark.config import resolve_config  # noqa: PLC0415

    resolved = resolve_config(config_path)[0]
    config_yaml = yaml.safe_dump(resolved, sort_keys=False)
    return {
        "schema": SCHEMA,
        "created_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        # A run IS the directory its ledger and this manifest sit in, so its
        # id is that directory's name: a sortie stamp on a flight day, the
        # run root's name on the bench, the name typed at submission on the
        # cluster. Nothing mints a second one.
        "run_id": out_root.name,
        "out_root": str(out_root),
        **prov,
        "config_yaml": config_yaml,
        # The arm's withholds (maps.search.withhold); readers group by it.
        "withhold": list(resolved["withhold"]),
        "selection": {
            "sample_seed": sample_seed,
            "sample_offset": sample_offset,
            "limit": limit,
            "workers": workers,
            "allowlist": str(allowlist_path) if allowlist_path else None,
            "episodes_n": len(episodes),
            "selection_sha": hashlib.sha1(
                "\n".join(episodes).encode()
            ).hexdigest()[:12],
        },
        "endpoints": endpoints,
        "perception": _perception_section(),
        "env": _audited_env(),
        **({"mission": dict(mission)} if mission is not None else {}),
        "resumes": [],
    }


def _get(d: dict, keypath: tuple) -> object:
    cur: object = d
    for k in keypath:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _identity(d: dict, keypath: tuple) -> object:
    v = _get(d, keypath)
    if keypath == ("search_env_flags",) and isinstance(v, dict):
        return {k: x for k, x in v.items() if k not in _HOST_ARM_FLAGS}
    return v


def identity_drift(prior: dict, current: dict) -> list[str]:
    """The IDENTITY_KEYS on which two manifests disagree, named for the
    refusal message. Empty = same run identity."""
    return [
        f"{'.'.join(kp)}: manifest={_get(prior, kp)!r} now={_get(current, kp)!r}"
        for kp in IDENTITY_KEYS
        if _identity(prior, kp) != _identity(current, kp)
    ]


def emit_or_verify(out_root: Path, manifest: dict, log=print) -> Path:
    """First entry writes the manifest (write-once); re-entry into the same
    run root verifies identity and appends a ``resumes`` entry. Raises
    ``ManifestIdentityError`` on drift, on a manifest that does not read, and
    on a root holding a ledger but no manifest — the caller refuses the run."""
    # Lazy: maps.search imports the engine, which no reader of this module needs.
    from maps.search.io.atomic import atomic_write_json  # noqa: PLC0415

    path = Path(out_root) / MANIFEST_NAME
    if not path.exists():
        if (Path(out_root) / EPISODES_JSONL).exists():
            raise ManifestIdentityError(
                f"{out_root} holds episodes.jsonl and no run_manifest.json: "
                "nothing records what flew those rows, so a resume cannot be "
                "checked against them. Use a new root."
            )
        atomic_write_json(path, manifest)
        log(f"run_manifest: wrote {path}")
        return path
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestIdentityError(f"{path} unreadable: {exc}") from None
    drift = identity_drift(prior, manifest)
    if drift:
        raise ManifestIdentityError(
            "run_manifest identity drift — a resumed run under a different "
            "bake/model/config/selection is two runs wearing one run id. "
            "Reconcile by hand (new out_root for a new identity):\n  "
            + "\n  ".join(drift)
            + f"\n  manifest: {path}"
        )
    prior.setdefault("resumes", []).append(
        {
            "ts": manifest["created_ts"],
            "git_commit": manifest.get("git_commit"),
            "git_dirty": manifest.get("git_dirty"),
            "hostname": manifest.get("hostname"),
        }
    )
    atomic_write_json(path, prior)
    log(f"run_manifest: identity verified, resume #{len(prior['resumes'])} recorded")
    return path
