"""Transport plumbing for the reasoning engine.

OpenAI-compatible HTTP client, SSE streaming + retries, the flight
transcript and its row summary, endpoint discovery, sampling envelopes, and
the HF tokenizer factory.

Installed as the supported ``maps/llm`` package so
benchmark and agent packages can declare it like any other runtime
dependency.
"""
