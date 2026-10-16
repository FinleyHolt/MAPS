"""The run's record: ``run_manifest.json`` at the run root, written before the first episode flies.

Identity fields are measured (served model and window off the installed
endpoint spec, the bake off its bytes) and the config is inlined. The flight
mission writes one at the sortie root and one per mission workdir
(``mission=``). Write-once: a resume into the same root is refused on
``IDENTITY_KEYS`` drift and records git drift as a ``resumes`` entry.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import socket
import time
from pathlib import Path

from benchmarks import repo_provenance
from benchmarks.citynav.episodes import EPISODES_JSONL, MANIFEST_NAME, atomic_write_json
from maps.llm.model_cards import effort

SCHEMA = "maps.run_manifest/v1"

# The experiment-arm flags every row carries (``arm_stamp``). Set-only,
# except the tile bake, which is resolved.
ARM_ENV_FLAGS = (
    "SEARCH_CITYNAV_TILES",
    # The imagery raster the ortho crops read: two rasters of one block are two arms.
    "SEARCH_CITYNAV_RGBD_DIR",
    "SEARCH_SEED_BASE",
    # Unset flies at the serve's default, which is its own arm.
    "SEARCH_SAMPLING_REASONING_EFFORT",
)
# Set keys under these prefixes join the stamp the same way (compaction
# budgets change reasoning behavior).
ARM_ENV_PREFIXES = ("SEARCH_COMPACT_",)

# The ambient env the manifest records when set: the arm flags, the vars that
# select what runs, and the endpoints.
AUDITED_ENV = ARM_ENV_FLAGS + (
    "CITYNAV_BENCH_CONFIG",
    "CITYNAV_ROOT",
    "SEARCH_BENCHMARK_NAME",
    "SEARCH_CITYNAV_DATA",
    "SEARCH_CITYNAV_SPLITS",
    "SAM_SERVER_URL",
    "SEARCH_ENDPOINT_OFFLINE",
    "SEARCH_ENDPOINT_SPEC_OVERRIDE",
    "SEARCH_REASONER_MODELS_URL",
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
    ("selection", "arrivals"),
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


def arm_stamp(policy: str | None) -> dict:
    """The arm flags: :func:`arm_env_stamp`, with a baseline ``policy``'s fixed
    dial (``baselines.seam.REASONING_EFFORT``; ``none`` is the dial off,
    whichever switch the card takes) in place of the environment's effort. The
    manifest and every row stamp this."""
    stamp = arm_env_stamp()
    if policy:
        # By name: benchmarks carries no static import of baselines.
        fixed = importlib.import_module("baselines.seam").REASONING_EFFORT.get(policy)
        if fixed is not None:
            stamp[_EFFORT] = fixed
    return stamp


def settle_effort() -> None:
    """Write the arm's reasoning effort back to ``os.environ`` as the
    requests carry it (stripped; blank is unset). Endpoint discovery has
    checked it against the served card; called once after it, before any
    stamp or spawn."""
    if chosen := effort():
        os.environ[_EFFORT] = chosen
    else:
        os.environ.pop(_EFFORT, None)


class EndpointRefused(RuntimeError):
    """The serve cannot fly this process: discovery failed, or a baseline
    arm's model has no dial-off."""


def install_endpoint(policy: str | None = None) -> None:
    """Discover the serve, install its spec and settle the effort, once per
    process before any inference or perception call; a baseline ``policy``
    (every arm makes dial-off calls) is refused here when its model has no
    dial-off. Raises :class:`EndpointRefused`."""
    from maps.llm import endpoint_spec

    try:
        spec = endpoint_spec.discover_from_env()
    except endpoint_spec.EndpointDiscoveryError as exc:
        raise EndpointRefused(f"endpoint discovery failed: {exc}") from exc
    endpoint_spec.install_spec(spec)
    settle_effort()
    if policy:
        # By name: benchmarks carries no static import of baselines.
        seam = importlib.import_module("baselines.seam")
        try:
            seam.dial_off(spec.model_id)
        except seam.NoDialOff as exc:
            raise EndpointRefused(str(exc)) from exc


def _audited_env() -> dict:
    keys = set(AUDITED_ENV) | {k for k in os.environ if k.startswith(AUDITED_ENV_PREFIXES)}
    return {k: os.environ[k] for k in sorted(keys - set(SECRET_ENV)) if k in os.environ}


def _endpoint_section() -> dict:
    """The installed endpoint spec's served facts; raises if none is installed."""
    from maps.llm.endpoint_spec import spec

    s = spec()
    return {
        "base_url": s.base_url,
        "model_id": s.model_id,
        "max_model_len": s.max_model_len,
        "tokenizer": s.tokenizer_name,
        "input_budget": s.input_budget,
        "max_output_tokens": s.max_output_tokens,
        "sam_url": ",".join(s.sam_urls),
    }


def _perception_section() -> dict:
    """The tile bake this process resolves, by path and by content (``bake_sha256``)."""
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
    limit: int,
    allowlist_path: Path | None,
    episodes: list[str],
    mission: dict | None = None,
    arrivals: str | None = None,
) -> dict:
    """The resolved record. ``mission`` is the flight mission's per-mission
    stamp, recorded whole; a ``reasoning_effort`` it names replaces the arm's
    in ``search_env_flags`` (:func:`arm_stamp`), since the mission's requests
    carried it. ``arrivals`` is a capacity session's stream sha256: the
    stream, not a seed, selects its episodes."""
    import yaml  # noqa: PLC0415

    from benchmarks.citynav.run_benchmark.config import resolve_config  # noqa: PLC0415

    resolved = resolve_config(config_path)[0]
    arm = arm_stamp(resolved.get("policy"))
    if mission is not None and mission.get("reasoning_effort") is not None:
        arm[_EFFORT] = str(mission["reasoning_effort"])
    endpoints = _endpoint_section()
    prov = repo_provenance(
        config_path=str(config_path),
        split=split,
        search_flags=arm,
        served_model=endpoints["model_id"],
        served_build=endpoints["tokenizer"],
    )
    config_yaml = yaml.safe_dump(resolved, sort_keys=False)
    return {
        "schema": SCHEMA,
        "created_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        # A run is the directory its ledger and manifest sit in.
        "run_id": out_root.name,
        "out_root": str(out_root),
        **prov,
        "config_yaml": config_yaml,
        # The arm's withholds (maps.search.withhold); readers group by it.
        "withhold": list(resolved["withhold"]),
        "selection": {
            **({"arrivals": arrivals} if arrivals is not None else {"sample_seed": sample_seed}),
            "limit": limit,
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
    """First entry writes the manifest; re-entry verifies identity and
    appends a ``resumes`` entry. Raises ``ManifestIdentityError`` on drift, an
    unreadable manifest, or a root holding a ledger but no manifest."""
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
            "run_manifest identity drift: a resumed run under a different "
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
