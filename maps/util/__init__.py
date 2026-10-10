"""Cross-cutting plumbing any layer may call.

Home for utilities not tied to one domain package. Currently the detection
mask codec (``mask_codec``), the one wire format for masks between the SAM
server, the render daemon and the runtime. Consumers import the submodule
directly (``from maps.util.mask_codec import ...``).
"""
