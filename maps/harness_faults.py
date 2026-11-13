"""The faults a mission call raises that are the harness's, never the
model's: a run block re-raises them past its own ``except``, and the bench's
infra rule (``benchmarks/citynav/run_benchmark/dispatch.is_infra_fault``)
re-queues the attempt. Beside the layers, so perception raises them and the
library reads them without importing each other."""

from __future__ import annotations

import httpx


class HarnessFault(RuntimeError):
    """A serve the mission depends on failed out from under the call."""


#: A harness fault, or a transport fault (timeouts included) of the HTTP
#: client every inference serve is reached through.
HARNESS_FAULTS = (HarnessFault, httpx.TransportError)
