"""OS isolation for the processes that run model-written Python: a bench attempt and the station.

The run tool executes a model's block inside the process that flies the
mission, and the interpreter's checks are no security boundary (README
"Security"). The OS adds:

| Layer | Mechanism | Effect |
| --- | --- | --- |
| Memory | ``RLIMIT_DATA``, soft and hard alike | an allocation past the cap raises ``MemoryError`` in the block, served as the block's error |
| File system | bubblewrap: ``/`` bound read-only, each output and cache directory bound writable | writes land only in the run's own directories |
| Processes | bubblewrap's PID namespace | the code sees and signals its own process tree alone |

``RLIMIT_DATA`` counts a process's private writable memory (heap, anonymous
maps, thread stacks), so the read-only ortho cache a mission maps
(gigapixels on Cal City) and the shared libraries stay outside the cap, which
``RLIMIT_AS`` would count.

ponytail: the network stays shared, since the mission reaches the reasoner and
the segmenter over HTTP; holding egress to those endpoints takes a network
namespace with firewall rules (root) or a filtering proxy.
"""

from __future__ import annotations

import contextlib
import os
import resource
import shutil
import subprocess
import sys
import tempfile
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


def cap_argv(gib: float) -> list[str]:
    """The argv prefix that starts a command under the memory cap (util-linux ``prlimit``)."""
    n = int(gib * 2**30)
    return ["prlimit", f"--data={n}:{n}", "--"]


def cap_memory(gib: float) -> None:
    """Cap this process (and every child it starts) at ``gib`` of data segment; the hard
    limit too, so code running here cannot raise it."""
    n = int(gib * 2**30)
    resource.setrlimit(resource.RLIMIT_DATA, (n, n))


def bwrap_works() -> bool:
    """True when ``bwrap`` is installed and can make its namespaces here (a host
    may block unprivileged user namespaces, e.g. Ubuntu's AppArmor default)."""
    exe = shutil.which("bwrap")
    if exe is None:
        return False
    probe = [exe, "--ro-bind", "/", "/", "--unshare-pid", "--die-with-parent", "true"]
    return subprocess.run(probe, capture_output=True).returncode == 0


def ortho_caches(raster_dirs: Iterable[str | Path | None]) -> list[Path]:
    """The ortho cache directory beside each raster directory, the one a mission
    writes on a cache miss (``camera_model._mmap_dir``)."""
    return [Path(d) / "ortho" for d in raster_dirs if d is not None]


def _hf_caches() -> list[str]:
    from huggingface_hub import constants

    return [constants.HF_HOME, constants.HF_HUB_CACHE]


def bwrap_argv(writable: Iterable[str | Path], *, new_session: bool = True) -> list[str]:
    """``bwrap`` up to the command: ``/`` read-only, ``writable`` plus TMPDIR and the
    Hugging Face cache writable, its own PID namespace, killed with its parent, the
    network shared. ``new_session`` takes the command off the terminal, which a
    terminal's keystroke injection (TIOCSTI) needs; the terminal's Ctrl-C then
    stops reaching it."""
    dirs: list[str] = []
    for d in [*writable, tempfile.gettempdir(), *_hf_caches()]:
        d = os.path.realpath(d)
        # A directory that cannot be made is left read-only: what writes there
        # fails as it would on any read-only path.
        with contextlib.suppress(OSError):
            os.makedirs(d, exist_ok=True)
        if os.path.isdir(d) and d not in dirs:
            dirs.append(d)
    return [
        "bwrap",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        *(a for d in dirs for a in ("--bind", d, d)),
        "--unshare-pid",
        "--die-with-parent",
        *(["--new-session"] if new_session else []),
        "--",
    ]


def mode_line(bwrap: bool, gib: float) -> str:
    """The line a launcher logs for the isolation it runs code under."""
    cap = f"memory cap {gib:g} GiB"
    if bwrap:
        return (
            "sandbox: bubblewrap, / read-only, writable only the run's output, the ortho "
            f"cache, TMPDIR and the HF cache; network shared; {cap}"
        )
    return (
        "sandbox: bwrap absent or user namespaces blocked; model code runs without "
        f"file-system isolation; {cap}"
    )


def confine(writable: list[Path], gib: float, log: Callable[[str], None]) -> None:
    """Confine this process before it serves or flies anything: cap its memory and,
    where bubblewrap runs, re-execute it under bubblewrap with ``writable`` its
    writable directories; log which mode it runs in. The re-executed process keeps
    the terminal, so the operator's Ctrl-C still shuts it down cleanly.

    ponytail: on a host with ``dev.tty.legacy_tiocsti=1`` code in the process can
    inject keystrokes into its terminal; run it under a supervisor with no terminal
    there."""
    cap_memory(gib)
    if os.environ.get(SANDBOXED_ENV) == "1":
        log(mode_line(True, gib))
        return
    if not bwrap_works():
        log(mode_line(False, gib))
        return
    os.environ[SANDBOXED_ENV] = "1"
    argv = bwrap_argv(writable, new_session=False) + [sys.executable, *sys.orig_argv[1:]]
    os.execvp(argv[0], argv)
