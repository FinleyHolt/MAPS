"""The run's ONE machine-written record: ``run_manifest.json`` at the run root.

A config authored before a run is a promise; this manifest is evidence. It is
emitted at flight start — after argv, endpoint discovery, and episode
selection have all resolved, before the first episode flies — and its
identity fields are MEASURED, not copied from declarations: the served model
and window come off ``/v1/models`` (the installed endpoint spec), the bake
identity off the tile files' own bytes. The record is self-contained (the
config YAML is inlined, every path resolved), so an appendix cites
``(manifest, git sha)`` and a reader chases nothing.

This module also owns the run-identity env inventory:

- ``ARM_ENV_FLAGS`` — the experiment-arm flags dispatch stamps onto every
  predictions row (``arm_env_stamp``); the ledger pairs and ``ab_report``
  refuses on these.
- ``AUDITED_ENV`` — every ambient var read between entry and first episode
  that can change what runs or where it runs. Set keys are recorded; unset
  keys are absent. Secrets never appear — an unverifiable field is a place
  the appendix can lie, a secret is a leak.

The field station writes it at two grains. The sortie's manifest sits at
the sortie root, written by the first mission that flies and describing the
arm as that mission found it. Each mission then writes its own into its
workdir (``mission=``), one deeper, carrying the sampling THAT mission flew
under: the reasoning effort the console named, or the arm's when it named
none. A reader chasing a row walks up from the mission's artifacts and lands
on the mission's record first (``analysis.provenance.resolve``), so a
per-mission fact is never read off the sortie's arm-wide stamp.

Write-once semantics (the commit_state "first dump wins" idiom): the first
emit writes the file; every re-entry into the same run root (a resume)
verifies the current resolution against it and REFUSES on
identity drift — a resumed run under a different bake/model/config/selection
is two runs wearing one run id. Git-sha drift on resume is recorded (a
``resumes`` entry), not refused: mid-campaign code fixes are routine and the
entry makes them visible.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

from benchmarks import _git_provenance, repo_provenance, sha256_file
from benchmarks.citynav.episodes import EPISODES_JSONL

SCHEMA = "maps.run_manifest/v1"
MANIFEST_NAME = "run_manifest.json"

# A/B arm auditability: the load-bearing launcher pins must be readable off
# the results row itself, not inferred from side-effect artifacts. Set-only
# (an unset flag stays absent), except the tile bake, which is resolved —
# see ``arm_env_stamp``. Readers: run_benchmark.dispatch (row stamp),
# analysis.ab_report (pairing refusal).
ARM_ENV_FLAGS = (
    "SEARCH_CITYNAV_TILES",
    # The imagery raster the ortho crops read. The Cal City bridge resolves
    # SEARCH_CALCITY_RGBD_DIR onto this name, default and override alike,
    # so two arms over two rasters of one block -- the NAIP release and the
    # program's own map -- share a bake and a split and differ on exactly
    # this. Set-only: a CityNav row never carries it.
    "SEARCH_CITYNAV_RGBD_DIR",
    # The schematic's delivered size. It sets what the model can READ off
    # the map -- chip glyphs are absolute pixels, so a smaller canvas makes
    # them relatively larger while fitting fewer of them -- and it sets the
    # per-map image-token bill directly (1 token per 1024 px, no
    # max_pixels bound on the serve). A row flown at a different value is a
    # different arm and has to say so.
    "SEARCH_MAP_MAX_PX",
    "SEARCH_LOOK_TOOL",
    # Per-k resample roots differ only by their pinned seed base.
    "SEARCH_SEED_BASE",
    # Score-affecting toggles: a row flown with any of these set is a
    # different arm, and ab_report must refuse to pair it against a row
    # without. Set-only, so rows that never set them still pair.
    "SEARCH_STOP_AFTER",
    "SEARCH_WALLED_UNNAMED_TYPE",
    # The envelope sweep's two knobs (maps/llm/sampling.py). Reasoner turns
    # sample at temperature 1.0, so these move the score directly. They are
    # read through a helper that takes the key as a parameter, which hides
    # them from the env-inventory scan, so they are listed here by name.
    # An arm flown at a different temperature is not a null.
    "SEARCH_SAMPLING_TEMPERATURE",
    "SEARCH_SAMPLING_PRESENCE_PENALTY",
    # How hard the model was asked to think, per request. It moves decode
    # tokens per episode and the score with them, and an unset arm inherits
    # whatever the served chat template defaults to — so a row that carries
    # no effort is a row flown at the SERVE's value, and two rows differing
    # only here are two arms.
    "SEARCH_SAMPLING_REASONING_EFFORT",
)
# Set keys under these prefixes join the stamp the same way (compaction
# budgets change reasoning behavior).
ARM_ENV_PREFIXES = ("SEARCH_COMPACT_",)

# The audited ambient-env inventory: vars that select WHAT runs (config,
# data roots, bridge/runner, seeds), toggle an arm, fabricate endpoint
# facts, or place the work. The manifest records every SET key;
# argv-selected facts (split/seed/limit/workers) are explicit
# manifest fields and do not ride this map.
AUDITED_ENV = (
    # selects what runs
    "BENCH_SAMPLE_SEED",
    "BENCH_SAMPLE_OFFSET",
    "CITYNAV_BENCH_CONFIG",
    "CITYNAV_BENCH_MAX_WALL_S",
    "CITYNAV_ROOT",
    "SEARCH_AGENTIC_REPO_ROOT",
    "SEARCH_BENCHMARK_AUTO_BAKE",
    "SEARCH_BENCHMARK_BRIDGE",
    "SEARCH_BENCHMARK_DEFAULT_CONFIG",
    "SEARCH_BENCHMARK_DEFAULT_SPLIT",
    "SEARCH_BENCHMARK_EPISODE_EXAMPLE",
    "SEARCH_BENCHMARK_LABEL",
    "SEARCH_BENCHMARK_LAUNCH_MODULE",
    "SEARCH_BENCHMARK_NAME",
    "SEARCH_BENCHMARK_OUTPUT_ROOT",
    "SEARCH_BENCHMARK_RUNNER_MODULE",
    "SEARCH_CITYNAV_AUTO_BAKE",
    "SEARCH_CITYNAV_DATA",
    "SEARCH_CITYNAV_RGBD_DIR",
    "SEARCH_CITYNAV_SPLITS",
    "SEARCH_CITYNAV_TILES",
    "SEARCH_SEED_BASE",
    # arm toggles
    "SEARCH_LOOK_TOOL",
    "SEARCH_SAMPLING_PRESENCE_PENALTY",
    "SEARCH_SAMPLING_REASONING_EFFORT",
    "SEARCH_SAMPLING_TEMPERATURE",
    "SEARCH_STOP_AFTER",
    "SEARCH_WALLED_UNNAMED_TYPE",
    # endpoint identity (the two override vars FABRICATE served-model facts
    # without probing — they must be on the record whenever set)
    "SAM_SERVER_URL",
    "SAM_VIDEO_SERVER_URL",
    "SEARCH_ENDPOINT_OFFLINE",
    "SEARCH_ENDPOINT_SPEC_OVERRIDE",
    "SEARCH_REASONER_URL",
    # placement / throughput (wall clock, not scores)
    "CITYNAV_BENCH_EPISODE_TIMEOUT_S",
)
AUDITED_ENV_PREFIXES = (
    "SEARCH_COMPACT_",
    "SEARCH_CALCITY_",
)
# Never recorded, ever. `.env` is the only home for secrets.
SECRET_ENV = ("SEARCH_REASONER_API_KEY",)

# Declared-but-not-recorded classes. Every identity-prefixed env read in the
# run path must belong to SOME class here (test_env_inventory.py enforces
# it), so a knob added without classifying itself fails a test the day it is
# written. When unsure, a new var goes in AUDITED_ENV — recording is free,
# an invisible knob is not.
#
# MECHANICS: wall clock, placement gates, retries, coordination transport —
# never what is measured. Ambient by design; not worth a manifest line each.
MECHANICS_ENV = (
    "CITYNAV_BENCH_ABORT_FRACTION",
    "CITYNAV_BENCH_ABORT_WINDOW",
    "CITYNAV_BENCH_RECYCLE_AFTER",
    "CITYNAV_BENCH_RECYCLE_MAPS",
    "SEARCH_LLM_ROUTING_KEY_HEADER",
    "SEARCH_REASONER_TIMEOUT_S",
    "SEARCH_REASONER_WARMUP_TIMEOUT_S",
    "SEARCH_REASONER_WARMUP_TOTAL_S",
)
# DERIVED: written by the harness itself mid-run (driver→worker and
# launcher→runner transport). An ambient value is meaningless; the facts
# they carry land in the manifest as explicit fields (split, seed, run id).
DERIVED_ENV = (
    "CITYNAV_BENCH_PARENT_PID",
    "CITYNAV_BENCH_SAMPLE_SEED",
    "CITYNAV_BENCH_SPLIT",
    "SEARCH_CITYNAV_RUNNER_CMD_JSON",
    "SEARCH_CITYNAV_STARTUP_CONFIG",
    "SEARCH_CITYNAV_STARTUP_MAP",
    "SEARCH_CITYNAV_STARTUP_REPO",
    "SEARCH_COMPILE_WORKDIR",
    "SEARCH_LLM_ROUTING_KEY",
)

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
    """The set arm flags, plus the ONE resolution the ledger's fatal
    comparability class stands on: the tile bake. An unset
    SEARCH_CITYNAV_TILES resolves to ``paths.TILES_DIR`` and the stamp
    records that RESOLUTION — a default-bake run must prove its bake on
    the row, not read as "unrecorded" (a 255-episode arm paired against
    nothing tonight for exactly this)."""
    stamp = {
        k: os.environ[k] for k in ARM_ENV_FLAGS if os.environ.get(k, "") != ""
    }
    stamp.update(
        {
            k: v
            for k, v in os.environ.items()
            if k.startswith(ARM_ENV_PREFIXES) and v != ""
        }
    )
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
    keys = {k for k in AUDITED_ENV}
    keys.update(
        k for k in os.environ if k.startswith(AUDITED_ENV_PREFIXES)
    )
    out: dict = {}
    for k in sorted(keys - set(SECRET_ENV)):
        v = os.environ.get(k)
        if v is not None:
            out[k] = v
    return out


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


def load_manifest(path: str | Path) -> dict:
    """A stored manifest, schema-checked. Loud on anything else — a replay
    fed a random JSON must not half-work."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ManifestIdentityError(
            f"{path} is not a run manifest (schema="
            f"{data.get('schema') if isinstance(data, dict) else None!r}, "
            f"expected {SCHEMA!r})"
        )
    return data


def assert_replay_git(source: dict) -> None:
    """A replay refuses on git-sha mismatch: the recorded number is only
    reproducible by the code that produced it. The human reconciles
    (checkout the recorded sha); no auto-checkout."""
    cur = str(_git_provenance().get("git_commit") or "")
    rec = str(source.get("git_commit") or "")
    # The two derivations may differ in length (a short deploy stamp vs a
    # full rev-parse); prefix equality at sha length is the same commit.
    same = bool(cur and rec) and (cur.startswith(rec) or rec.startswith(cur))
    if not same:
        raise ManifestIdentityError(
            f"replay git mismatch: the manifest was flown at "
            f"{rec or '<unrecorded>'}, this checkout is at "
            f"{cur or '<no git>'}. Check out the recorded commit to replay "
            f"its run (no auto-checkout)."
        )


def apply_replay_args(args) -> dict:
    """``--from-manifest``: the record becomes an input again. Loads the
    source manifest, refuses on git mismatch, and overwrites the selection
    argv (split/seed/offset/limit/allowlist) so the identical episode set
    replays. The live parts (endpoints, bake) are re-derived
    at startup and gated against the source by ``verify_replay``."""
    source = load_manifest(args.from_manifest)
    assert_replay_git(source)
    sel = source.get("selection") or {}
    args.split = source.get("split") or args.split
    args.sample_seed = int(sel.get("sample_seed") or 0)
    args.sample_offset = int(sel.get("sample_offset") or 0)
    args.limit = int(sel.get("limit") or 0)
    allow = sel.get("allowlist")
    if allow:
        if not Path(allow).is_file():
            raise ManifestIdentityError(
                f"replay: the manifest's episode allowlist is missing: {allow}"
            )
        args.episode_allowlist = allow
    return source


def verify_replay(source: dict, fresh: dict) -> None:
    """Gate a replay's re-derived resolution against the source record.
    selection_sha proves the identical episode set; config sha, bake and
    served model prove the identical measurement."""
    drift = identity_drift(source, fresh)
    if drift:
        raise ManifestIdentityError(
            "replay drift — the re-derived run does not reproduce the "
            "manifest (fix the environment or serve, never the record):\n  "
            + "\n  ".join(drift)
        )


def emit_or_verify(out_root: Path, manifest: dict, log=print) -> Path:
    """First entry writes the manifest (write-once); re-entry into the same
    run root verifies identity and appends a ``resumes`` entry. Raises
    ``ManifestIdentityError`` on drift, on a manifest that does not read, and
    on a root holding a ledger but no manifest — the caller refuses the run."""
    path = Path(out_root) / MANIFEST_NAME
    if not path.exists():
        if (Path(out_root) / EPISODES_JSONL).exists():
            raise ManifestIdentityError(
                f"{out_root} holds episodes.jsonl and no run_manifest.json: "
                "nothing records what flew those rows, so a resume cannot be "
                "checked against them. Use a new root."
            )
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
    path.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
    log(f"run_manifest: identity verified, resume #{len(prior['resumes'])} recorded")
    return path
