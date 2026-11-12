"""Two reasoner turns per effort against the configured endpoint, before a run.

    python -m maps.mission.endpoint_preflight [--effort medium --effort high]

Discovers the endpoint as a run does (``SEARCH_REASONER_URL``,
``SEARCH_REASONER_API_KEY``, ``SEARCH_ENDPOINT_SPEC_OVERRIDE``,
``SEARCH_REASONER_MODELS_URL``; ``SEARCH_ENDPOINT_OFFLINE=1`` skips the
model listing and takes the override as the endpoint's facts), without the
SAM probe, then flies a short
conversation through the reasoner's own request path: the served system turn
and ``run`` tool, a picture, the effort, the model's tool call, and its
result. The first turn is a blocking request, as the bench sends; the second
streams, as the flight mission sends, and carries the assistant tool-call and
tool-result messages a long mission is made of. A refusal prints the
endpoint's own words (``maps.llm.http`` logs the body it refused); a pass
prints each reply's shape and token meter. With no ``--effort`` it sends
``SEARCH_SAMPLING_REASONING_EFFORT``, or none when that is unset.

On a hosted (OpenAI-wire) card each conversation rides one
``prompt_cache_key``, and the second turn, which opens with the first turn's
bytes, must read a cache hit (``usage.prompt_tokens_details.cached_tokens``
above zero); a miss fails the effort, since a run would then pay full price
for every turn's prefix.

It writes the transcript to a temporary directory and nothing else: no run
root, no SAM call.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tempfile
import threading
from pathlib import Path

from maps.llm import endpoint_spec
from maps.llm.model_cards import check_effort, effort, speaks_openai
from maps.mission.served_text import SERVED_TOOLS
from maps.settings import Settings, from_process


def _picture() -> bytes:
    """A 512 px JPEG: grey ground with one red square, so the reply can say
    whether the model saw it."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    im = Image.new("RGB", (512, 512), (110, 110, 110))
    ImageDraw.Draw(im).rectangle((192, 192, 320, 320), fill=(200, 30, 30))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


def _shape(out: dict) -> dict:
    return {"kind": out["kind"], "finish_reason": out["finish_reason"],
            "tool_calls": [c["function"] for c in out.get("tool_calls") or []],
            "content": out["content"][:300]}


def _conversation(spec: endpoint_spec.EndpointSpec, effort: str | None, workdir: Path,
                  settings: Settings = Settings()) -> dict:
    # The mission stack loads transformers: not at --help.
    from maps.llm.flight_summary import summarize  # noqa: PLC0415
    from maps.mission.reasoner import Reasoner  # noqa: PLC0415

    # one prompt-cache key per conversation, as one per episode
    reasoner = Reasoner(endpoint_spec=spec, settings=settings, workdir=workdir,
                        routing_key=f"preflight-{workdir.name}", seed=1234, reasoning_effort=effort)
    reasoner.append_vision_user(
        "Endpoint check. Name the colour of the square in this picture, then call run with the code "
        "print('ok').", [_picture()])
    first = reasoner.step_tool_call(SERVED_TOOLS)
    after_first = summarize(workdir / "flight", settings.endpoint.tokenizer_name)
    for call in first.get("tool_calls") or []:
        reasoner.append_tool_result(call["id"], call["function"]["name"], "ok")
    if not first.get("tool_calls"):
        reasoner.append_user("Call run with the code print('ok').")
    reasoner.cancel_event = threading.Event()  # never set: the call streams, as a flight's does
    second = reasoner.step_tool_call(SERVED_TOOLS)
    meter = summarize(workdir / "flight", settings.endpoint.tokenizer_name)
    out = {"first": _shape(first), "second": _shape(second),
           "tokens": {k: meter[k] for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens",
                                            "cached_tokens", "cache_write_tokens")},
           "served_model": meter["served_model"], "latency_s": meter["latency_s_sum"]}
    if speaks_openai(spec.model_id):
        out["second_turn_cached_tokens"] = _second_turn_cached(after_first, meter)
    return out


class NoCacheHit(RuntimeError):
    """A hosted endpoint's second turn read nothing from its prompt cache."""


def _second_turn_cached(after_first: dict, after_second: dict) -> int:
    """The second turn's cache hit; raises :class:`NoCacheHit` when the
    endpoint reported none, or reported a zero."""
    total = after_second["cached_tokens"]
    if total is None:
        raise NoCacheHit("the endpoint reported no usage.prompt_tokens_details.cached_tokens")
    hit = total - (after_first["cached_tokens"] or 0)
    if hit <= 0:
        raise NoCacheHit("turn 2 opens with turn 1's bytes but read 0 cached tokens")
    return hit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--effort", action="append", default=[],
                    help="a reasoning effort to send; repeat for several conversations")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = from_process()
    try:
        spec = endpoint_spec.discover_from_settings(settings, probe_sam=False)
    except endpoint_spec.EndpointDiscoveryError as exc:
        print(f"discovery refused: {exc}", file=sys.stderr)
        return 2
    print(f"discovered {spec.model_id} at {spec.base_url}: window {spec.max_model_len}, "
          f"input budget {spec.input_budget}, output cap {spec.max_output_tokens}, "
          f"tokenizer {spec.tokenizer_name}")

    efforts = args.effort or [effort(None, settings.sampling)]
    failed = 0
    for asked in efforts:
        with tempfile.TemporaryDirectory(prefix="maps-preflight-") as tmp:
            try:
                if asked is not None:
                    check_effort(spec.model_id, asked)
                result = _conversation(spec, asked, Path(tmp), settings)
            except Exception as exc:  # noqa: BLE001 — a refusal is the finding
                failed += 1
                print(f"effort {asked!r}: REFUSED {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
        print(f"effort {asked!r}: " + json.dumps(result, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
