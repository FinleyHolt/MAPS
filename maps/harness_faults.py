"""The faults a mission call raises that are the harness's, never the
model's: a run block re-raises them past its own ``except``, and the bench's
infra rule (``benchmarks/citynav/run_benchmark/dispatch.is_infra_fault``)
re-queues the attempt. Beside the layers, so perception raises them and the
library reads them without importing each other."""

from __future__ import annotations


class HarnessFault(RuntimeError):
    """A serve the mission depends on failed out from under the call."""


#: What a run block re-raises. The segmenter client is a block's one network
#: path and keeps its transport faults: a frame no tile answered raises
#: ``SamServerOutage``.
HARNESS_FAULTS = (HarnessFault,)
