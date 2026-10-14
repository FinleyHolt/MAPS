"""The pinned upstream identity of every released baseline this package runs.

One table, read by the setup script (``pins``) to clone each repository at
its commit and by ``outcome_rows`` to stamp that commit on every row. The
checkpoint's own identity is measured at setup time (a file hash or a Hugging
Face revision written beside the weights) and handed to the converter, never
recorded here.

Usage::

    python -m baselines.released.upstream pins
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

#: Every row this package writes says it ran the authors' released
#: evaluator and action budget, not this harness's.
PROTOCOL = "upstream_released"


@dataclass(frozen=True)
class Release:
    method: str
    repo: str
    sha: str
    #: The interpreter the upstream README builds its environment on.
    python: str


RELEASES: dict[str, Release] = {
    "mgp": Release(
        method="mgp",
        repo="https://github.com/water-cookie/citynav",
        sha="372ecbd1df58b46ffaad91c118c4838b88b87710",
        python="3.10",
    ),
    "hett": Release(
        method="hett",
        repo="https://github.com/crotonyl/HETT",
        sha="51eb86c3df1f4a5ce3ba1b1750d027971efa378e",
        python="3.10",
    ),
    "flightgpt": Release(
        method="flightgpt",
        repo="https://github.com/Uavln/FlightGPT",
        sha="a1dd6760b7064a3b63e0ec4f75ae5921b3c95f92",
        python="3.11",
    ),
}


def main(argv: list[str]) -> int:
    if argv[1:] != ["pins"]:
        print(__doc__, file=sys.stderr)
        return 2
    for r in RELEASES.values():
        print(r.method, r.repo, r.sha, r.python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
