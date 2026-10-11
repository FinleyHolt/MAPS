"""Two reasoner turns per effort against the configured endpoint, before a run.

    python -m maps.llm.endpoint_preflight [--effort medium --effort high]

Discovers the endpoint as a run does (``SEARCH_REASONER_URL``,
``SEARCH_REASONER_API_KEY``, ``SEARCH_ENDPOINT_SPEC_OVERRIDE``,
``SEARCH_REASONER_MODELS_URL``), without the SAM probe, then flies a short
conversation through the reasoner's own request path: the served system turn
and ``run`` tool, a picture, the effort, the model's tool call, and its
result. The first turn is a blocking request, as the bench sends; the second
streams, as the flight mission sends, and carries the assistant tool-call and
tool-result messages a long mission is made of. A refusal prints the
endpoint's own words (``maps.llm.http`` logs the body it refused); a pass
prints each reply's shape and token meter. With no ``--effort`` it sends
``SEARCH_SAMPLING_REASONING_EFFORT``, or none when that is unset.

It writes the transcript to a temporary directory and nothing else: no run
root, no SAM call.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

from maps.llm import endpoint_spec
from maps.llm.flight_summary import summarize
from maps.llm.model_cards import check_effort


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


def _conversation(spec: endpoint_spec.EndpointSpec, effort: str | None, workdir: Path) -> dict:
    from maps.search.parse_compile.prompts import served_tools  # noqa: PLC0415
    from maps.search.parse_compile.reasoner_agent import Reasoner  # noqa: PLC0415

    reasoner = Reasoner(reasoner_url=spec.base_url, endpoint_spec=spec,
                        api_key=os.environ.get("SEARCH_REASONER_API_KEY", "EMPTY"),
                        seed=1234, reasoning_effort=effort)
    reasoner.append_vision_user(
        "Endpoint check. Name the colour of the square in this picture, then call run with the code "
        "print('ok').", [_picture()])
    first = reasoner.step_tool_call(served_tools())
    for call in first.get("tool_calls") or []:
        reasoner.append_tool_result(call["id"], call["function"]["name"], "ok")
    if not first.get("tool_calls"):
        reasoner.append_user("Call run with the code print('ok').")
    reasoner.cancel_event = threading.Event()  # never set: the call streams, as a flight's does
    second = reasoner.step_tool_call(served_tools())
    meter = summarize(workdir / "flight")
    return {"first": _shape(first), "second": _shape(second),
            "tokens": {k: meter[k] for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")},
            "latency_s": meter["latency_s_sum"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--effort", action="append", default=[],
                    help="a reasoning effort to send; repeat for several conversations")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    override = os.environ.get("SEARCH_ENDPOINT_SPEC_OVERRIDE", "").strip()
    try:
        spec = endpoint_spec.discover(
            os.environ.get("SEARCH_REASONER_URL", "http://127.0.0.1:8001/v1"),
            override=endpoint_spec.parse_override(override) if override else None)
    except endpoint_spec.EndpointDiscoveryError as exc:
        print(f"discovery refused: {exc}", file=sys.stderr)
        return 2
    print(f"discovered {spec.model_id} at {spec.base_url}: window {spec.max_model_len}, "
          f"input budget {spec.input_budget}, output cap {spec.max_output_tokens}, "
          f"tokenizer {spec.tokenizer_name}")

    efforts = args.effort or [os.environ.get("SEARCH_SAMPLING_REASONING_EFFORT", "").strip() or None]
    workdir_before = os.environ.get("SEARCH_COMPILE_WORKDIR")
    failed = 0
    try:
        for effort in efforts:
            with tempfile.TemporaryDirectory(prefix="maps-preflight-") as tmp:
                os.environ["SEARCH_COMPILE_WORKDIR"] = tmp
                try:
                    if effort is not None:
                        check_effort(spec.model_id, effort)
                    result = _conversation(spec, effort, Path(tmp))
                except Exception as exc:  # noqa: BLE001 — a refusal is the finding
                    failed += 1
                    print(f"effort {effort!r}: REFUSED {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
            print(f"effort {effort!r}: " + json.dumps(result, default=str))
    finally:
        if workdir_before is None:
            os.environ.pop("SEARCH_COMPILE_WORKDIR", None)
        else:
            os.environ["SEARCH_COMPILE_WORKDIR"] = workdir_before
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
