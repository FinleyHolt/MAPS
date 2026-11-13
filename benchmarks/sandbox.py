"""OS isolation for the processes that run model-written Python: a bench attempt and the station.

The run tool executes a model's block inside the process that flies the
mission, and the interpreter's checks are no security boundary (README
"Security"). The OS adds:

| Layer | Mechanism | Effect |
| --- | --- | --- |
| Memory | ``RLIMIT_DATA``, soft and hard alike | an allocation past the cap raises ``MemoryError`` in the block, served as the block's error |
| Memory, processes | a bench attempt's transient systemd user scope, ``MemoryMax`` the cap | shared memory and child processes count; the scope is killed with the attempt |
| File system | bubblewrap: ``/`` read-only; home, ``/tmp``, ``/dev/shm`` and ``/run`` empty; the run's inputs bound read-only, its output and cache directories writable | a block reads only the run's inputs and writes only its own directories |
| Processes | bubblewrap's PID namespace | the code sees and signals its own process tree alone |

``RLIMIT_DATA`` counts a process's private writable memory (heap, anonymous
maps, thread stacks), so the read-only ortho cache a mission maps
(gigapixels on Cal City) and the shared libraries stay outside the cap, which
``RLIMIT_AS`` would count.

ponytail: the network stays shared, since the mission reaches the reasoner and
the segmenter over HTTP; holding egress to those endpoints takes a network
namespace with firewall rules (root) or a filtering proxy. A shared network
namespace also keeps the host's abstract Unix sockets (an X server's among
them) reachable, and the reasoner's key rides the process environment.

ponytail: an allocation failure in native code (a Rust tokenizer's abort, the
scope's OOM kill) ends the attempt with a signal, which the driver books as a
crash and re-queues as infrastructure, though a block pinned the memory;
telling the two apart takes a marker the attempt writes before each block.

ponytail: the attempt scores itself (its frames hold the target, the driver
trusts its ``episode.json`` and stdout), so a block that escapes the library can
rewrite its own score; benchmark numbers from an untrusted model need a
container or a virtual machine that scores outside it.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import resource
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

#: A bench attempt's memory cap, GiB: measured CityNav attempts of 23 turns and
#: 20 flights (the Stanford block with the served model's tokenizer;
#: birmingham_block_1 building its ortho cache) peak at 1.1 GiB of data
#: segment; hardware and blocks vary, so the driver takes ``--attempt-memory-gib``.
ATTEMPT_MEMORY_GIB = 4.0
#: A mission process's cap (the station, the twin CLI), GiB: the station
#: flying the Stanford episode from the seat peaks at 2.6 GiB of data segment,
#: its server threads' stacks included; the rest is headroom for the Cal City
#: twin and the console's basemaps, neither measured.
STATION_MEMORY_GIB = 16.0
#: Set in a process this module re-executed under bubblewrap.
SANDBOXED_ENV = "MAPS_SANDBOXED"
#: The repository root: code and configs an attempt reads.
REPO = Path(__file__).resolve().parents[1]


#: The environment model code runs under keeps these names and prefixes alone, so a
#: token in the launching shell stays out of a block's reach: MAPS's own variables
#: (the run flags the manifest records, the endpoints, the reasoner's key), the
#: Hugging Face settings, the locale, and what the interpreter, the proxies and the
#: native thread pools read.
_ENV_NAMES = frozenset(
    [
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "TERM",
        "TZ",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "PYTHONHASHSEED",
        "MALLOC_ARENA_MAX",
        "OPENCV_FOR_THREADS_NUM",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_OFFLINE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
    ]
)
_ENV_PREFIXES = ("SEARCH_", "CITYNAV_", "SAM_", "MAPS_", "HF_", "LC_")
#: Hugging Face's token variables, dropped though they match ``HF_``.
_ENV_SECRETS = frozenset({"HF_TOKEN", "HF_TOKEN_PATH", "HUGGING_FACE_HUB_TOKEN"})


#: What ``systemd-run --user`` and ``systemctl --user`` find the user manager by.
USER_BUS_ENV = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")


#: One thread per native pool (BLAS, OpenCV, Rayon) and two malloc arenas. Uncapped,
#: each pool starts a thread per core, and ``RLIMIT_DATA`` counts every thread's
#: reserved stack and OpenBLAS's per-thread buffers: numpy, OpenCV and SciPy reserve
#: 6.7 GiB at import on a 32-core node (0.1 GiB resident), past any cap. A value the
#: shell set wins.
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENCV_FOR_THREADS_NUM": "1",
    "RAYON_NUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
}


def child_env(environ) -> dict[str, str]:
    """``environ`` cut to what a process running model code may see (:data:`_ENV_NAMES`,
    :data:`_ENV_PREFIXES`, any ``*_NUM_THREADS``), Hugging Face's token excepted, over
    :data:`THREAD_ENV`."""
    return THREAD_ENV | {
        k: v
        for k, v in environ.items()
        if (k in _ENV_NAMES or k.startswith(_ENV_PREFIXES) or k.endswith("_NUM_THREADS"))
        and k not in _ENV_SECRETS
    }


def cap_memory(gib: float) -> None:
    """Cap this process (and every child it starts) at ``gib`` of data segment; the hard
    limit too, so code running here cannot raise it."""
    n = int(gib * 2**30)
    resource.setrlimit(resource.RLIMIT_DATA, (n, n))


def bwrap_refusal() -> str | None:
    """Why bubblewrap cannot build the sandbox :func:`bwrap_argv` describes here, or
    None when it can: ``bwrap`` absent, or its probe refused (a host blocking
    unprivileged user namespaces, e.g. Ubuntu's AppArmor default, or a ``/proc``
    mount)."""
    if shutil.which("bwrap") is None:
        return "bwrap is not installed"
    probe = subprocess.run(bwrap_argv([]) + ["true"], capture_output=True, text=True)
    if probe.returncode == 0:
        return None
    said = (probe.stderr or "").strip().splitlines()
    return "bwrap cannot build the sandbox here" + (f" ({said[0]})" if said else "")


def scope_works() -> bool:
    """True when a systemd user manager starts transient scopes here (a compute node
    without a user session has none)."""
    if shutil.which("systemd-run") is None:
        return False
    probe = ["systemd-run", "--user", "--scope", "-q", "--", "true"]
    return subprocess.run(probe, capture_output=True).returncode == 0


def scope_argv(unit: str, gib: float) -> list[str]:
    """``systemd-run`` up to the command: a transient user scope named ``unit`` whose
    every process (shared memory and tmpfs pages included) counts against ``gib``."""
    n = int(gib * 2**30)
    return ["systemd-run", "--user", "--scope", "-q", f"--unit={unit}",
            "-p", f"MemoryMax={n}", "-p", "MemorySwapMax=0",
            # An OOM-killed scope ends "failed"; unloaded, it leaves no unit per crash.
            "-p", "CollectMode=inactive-or-failed", "--"]  # fmt: skip


def kill_scope(unit: str) -> None:
    """SIGKILL every process left in the scope ``unit``, a daemon that left the
    attempt's session among them; a scope already gone is a no-op."""
    subprocess.run(
        ["systemctl", "--user", "kill", "--signal=SIGKILL", f"{unit}.scope"], capture_output=True
    )


def ortho_caches(raster_dirs: Iterable[str | Path | None]) -> list[Path]:
    """The ortho cache directory beside each raster directory, the one a mission
    writes on a cache miss (``camera_model._mmap_dir``)."""
    return [Path(d) / "ortho" for d in raster_dirs if d is not None]


def hf_hub_cache() -> str:
    """The Hugging Face hub cache, where the served model's tokenizer lives."""
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE


def _secret_files() -> list[str]:
    """The files bubblewrap masks wherever ``/`` exposes them: the repo's ``.env`` and
    the Hugging Face token files (under ``HF_HOME``, which need not lie in home)."""
    from huggingface_hub import constants

    paths = [REPO / ".env", constants.HF_TOKEN_PATH, constants.HF_STORED_TOKENS_PATH]
    return [os.path.realpath(p) for p in paths if os.path.isfile(p)]


def bench_inputs(bench) -> list[Path]:
    """What a mission on ``bench`` reads: the data root, splits, catalog, rasters and
    tile bake, and a tokenizer named by local directory."""
    tok = bench.environ.get("SEARCH_COMPACT_TOKENIZER_NAME", "")
    return [
        bench.root,
        bench.splits_dir,
        Path(bench.objects_json).parent,
        bench.rgbd_dir,
        bench.tiles_dir,
        *([Path(tok)] if tok and os.path.isdir(tok) else []),
    ]


def _existing(paths: Iterable[str | Path], *, make: bool = False) -> list[str]:
    """``paths`` resolved, deduplicated, and kept where they exist; with ``make``,
    directories created first (one that cannot be made is left out, so what writes
    there fails as on any read-only path)."""
    out: list[str] = []
    for p in paths:
        p = os.path.realpath(p)
        if make:
            with contextlib.suppress(OSError):
                os.makedirs(p, exist_ok=True)
        if os.path.exists(p) and p not in out:
            out.append(p)
    return out


def _hops(path: str | Path) -> list[tuple[str, str]]:
    """The symlinks met on the way from ``path`` to what it names, as ``(target, link)``."""
    hops: list[tuple[str, str]] = []
    todo = os.path.abspath(path)
    for _ in range(40):  # the kernel's own bound on a symlink chain
        cur, parts = "/", Path(todo).parts[1:]
        for i, part in enumerate(parts):
            cur = os.path.join(cur, part)
            if os.path.islink(cur):
                target = os.readlink(cur)
                hops.append((target, cur))
                todo = os.path.normpath(os.path.join(os.path.dirname(cur), target, *parts[i + 1 :]))
                break
        else:
            break
    return hops


def bwrap_argv(
    writable: Iterable[str | Path],
    readable: Iterable[str | Path] = (),
    *,
    new_session: bool = True,
) -> list[str]:
    """``bwrap`` up to the command. ``/`` is read-only; home, ``/tmp``, ``/dev/shm`` and
    ``/run`` are empty (the user's secrets, other sessions' scratch, D-Bus and
    every desktop socket gone), the name resolver's directory excepted; the repo,
    the interpreter and ``readable`` come back read-only, the repo's ``.env`` and
    the Hugging Face token masked, and ``writable`` comes back writable. The command gets its own PID
    namespace, dies with its parent, and shares the network. ``new_session`` takes
    it off the terminal, which a terminal's keystroke injection (TIOCSTI) needs;
    the terminal's Ctrl-C then stops reaching it."""
    home = os.path.realpath(os.path.expanduser("~"))
    empty = [d for d in ("/tmp", "/dev/shm", "/run", home) if d != "/" and os.path.isdir(d)]
    resolver = os.path.realpath("/etc/resolv.conf")
    if resolver.startswith("/run/"):
        readable = [os.path.dirname(resolver), *readable]
    readable, writable = [REPO, sys.prefix, sys.base_prefix, *readable], list(writable)
    ro, rw = _existing(readable), _existing(writable, make=True)
    masked = _secret_files()
    # Each path is bound at its real location; a symlink on the way that lies in an
    # emptied directory is made again, so the path as named still resolves.
    links = {
        hop: None
        for p in [sys.executable, *readable, *writable]
        for hop in _hops(p)
        if any(hop[1].startswith(d + "/") for d in empty)
    }
    return [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        *(a for d in empty for a in ("--tmpfs", d)),
        *(a for t, link in links for a in ("--dir", os.path.dirname(link), "--symlink", t, link)),
        *(a for d in ro for a in ("--ro-bind", d, d)),
        *(a for p in masked for a in ("--ro-bind", "/dev/null", str(p))),
        *(a for d in rw for a in ("--bind", d, d)),
        "--setenv", "TMPDIR", "/tmp",
        "--unshare-pid",
        "--die-with-parent",
        *(["--new-session"] if new_session else []),
        "--",
    ]  # fmt: skip


def mode_line(refusal: str | None, gib: float, *, scope: bool | None = None) -> str:
    """The line a launcher logs for the isolation it runs code under: bubblewrap when
    ``refusal`` (:func:`bwrap_refusal`) is None, else the refusal; ``scope`` is
    whether each bench attempt runs in a systemd scope (None: a mission process,
    which runs in none)."""
    cap = f"memory cap {gib:g} GiB of data segment"
    if scope:
        cap += " and of the attempt's scope (shared memory and children included)"
    elif scope is False:
        cap += "; no systemd user manager: shared memory and child processes escape the cap"
    if refusal is None:
        return (
            "sandbox: bubblewrap, / read-only, home /tmp /run empty, readable the run's "
            f"inputs, writable only the run's output and caches; network shared; {cap}"
        )
    return f"sandbox: {refusal}; model code runs without file-system isolation; {cap}"


def _confined() -> bool:
    """True inside a bubblewrap sandbox this module built: ``/`` read-only and PID 1
    bubblewrap's init."""
    try:
        init = Path("/proc/1/comm").read_text().strip()
    except OSError:
        return False
    return bool(os.statvfs("/").f_flag & os.ST_RDONLY) and init == "bwrap"


def _cache_tokenizer(log: Callable[[str], None]) -> None:
    """Discover the reasoner once before confining, so the tokenizer the confined
    process loads is already in the hub cache it reads offline; a reasoner that does
    not answer yet leaves the confined process to report it."""
    from maps.llm import endpoint_spec
    from maps.settings import from_environ

    try:
        endpoint_spec.discover_from_settings(from_environ(os.environ), probe_sam=False)
    except Exception as exc:  # the confined process discovers again and refuses by name
        log(f"sandbox: reasoner tokenizer not cached before confining ({type(exc).__name__})")


def confine(
    writable: list[Path],
    gib: float,
    log: Callable[[str], None],
    readable: Iterable[str | Path] = (),
    *,
    flies: bool = False,
) -> None:
    """Confine this process before it serves or flies anything: cap its memory,
    restart it under :func:`child_env`'s environment and, where bubblewrap runs,
    under bubblewrap with ``writable`` its writable directories and ``readable``
    besides the repo and interpreter; log which mode it runs in. A process that
    ``flies`` caches the reasoner's tokenizer first and reads the hub cache read-only
    and offline. The restarted process keeps the terminal, so the operator's Ctrl-C
    still shuts it down cleanly. An inherited marker without the sandbox around it is
    ignored.

    ponytail: on a host with ``dev.tty.legacy_tiocsti=1`` code in the process can
    inject keystrokes into its terminal; run it under a supervisor with no terminal
    there. Relative paths resolve inside the sandbox only from a bound working
    directory (the repo root)."""
    cap_memory(gib)
    if os.environ.pop(SANDBOXED_ENV, None) == "1" and _confined():
        log(mode_line(None, gib))
        return
    env, argv = child_env(os.environ), [sys.executable, *sys.orig_argv[1:]]
    if (why := bwrap_refusal()) is not None:
        if env != dict(os.environ):
            os.execve(sys.executable, argv, env)
        log(mode_line(why, gib))
        return
    if flies:
        _cache_tokenizer(log)
        readable = [*readable, hf_hub_cache()]
        env["HF_HUB_OFFLINE"] = "1"
    env[SANDBOXED_ENV] = "1"
    argv = bwrap_argv(writable, readable, new_session=False) + argv
    os.execvpe(argv[0], argv, env)


#: The smallest memory cap a flag takes, GiB: an attempt's interpreter and imports
#: alone come near it.
MIN_MEMORY_GIB = 1.0


def gib(raw: str) -> float:
    """An argparse type for a memory cap: a finite number of GiB, at least
    :data:`MIN_MEMORY_GIB` (-1 or 0 would lift or zero the cap, nan fails every check)."""
    value = float(raw)
    if not math.isfinite(value) or value < MIN_MEMORY_GIB:
        raise argparse.ArgumentTypeError(f"must be at least {MIN_MEMORY_GIB:g} GiB, got {raw}")
    return value


def add_memory_flag(ap, default: float = STATION_MEMORY_GIB) -> None:
    """``--memory-gib`` on a mission CLI's parser."""
    ap.add_argument(
        "--memory-gib",
        type=gib,
        default=default,
        help="the process's memory cap in GiB (default %(default)s): a block that allocates "
        "past it raises MemoryError, served as the block's error",
    )


def confine_cli(args, bench, log: Callable[[str], None], *, flies: bool = True) -> None:
    """:func:`confine` for a mission CLI: writable its ``--out`` and, when it ``flies``,
    the ortho caches beside the bench's and the ``--ortho``/``--register`` rasters;
    readable the bench's inputs and every path argument the operator named."""
    args.out.mkdir(parents=True, exist_ok=True)
    caches = ortho_caches([bench.rgbd_dir, args.ortho, args.register])
    named = [v for v in vars(args).values() if isinstance(v, (str, Path)) and os.path.exists(v)]
    confine(
        [args.out, *(caches if flies else [])],
        args.memory_gib,
        log,
        [*bench_inputs(bench), *named],
        flies=flies,
    )
