"""Print the served LLM contract: system text + the mission tools.

What the model is served at this sha, from the owners a run serves through
(``prompts.served_system_prompt`` and ``prompts.served_tools``); the flight
log records what a real run sent. Diff two shas by dumping at each.

  python -m maps.search.parse_compile.contract
  python -m maps.search.parse_compile.contract --sensor <rig> [--agl-m <m>]   # Cal City's height by default
  python -m maps.search.parse_compile.contract --withhold look,images
  python -m maps.search.parse_compile.contract --rendered   # template bytes

``--rendered`` prints the chat-template render of the system turn, one
sample instruction and the tools, with its token count on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from maps.perception.geometry.sensor_spec import list_sensors, resolve_sensor, served_description
from maps.search.parse_compile import prompts

_SAMPLE_INSTRUCTION = "<operator cue>"
_CALCITY = Path(__file__).resolve().parents[3] / "configs" / "calcity.yaml"


def assemble(sensor: str | None = None, agl_m: float | None = None) -> dict:
    """The contract parts exactly as served: system text (with the optional
    sensor block fused the way ``Reasoner`` fuses it, its footprint at
    ``agl_m``) and the mission tools."""
    description = ""
    if sensor:
        try:
            spec = resolve_sensor(sensor)
        except ValueError:
            raise SystemExit(
                f"contract: unknown sensor '{sensor}' -- one of: "
                + ", ".join(list_sensors())) from None
        try:
            description = served_description(spec, agl_m=agl_m)
        except ValueError as exc:
            raise SystemExit(f"contract: {exc} (pass --agl-m)") from exc
    system = prompts.served_system_prompt(description)
    return {"system": system, "tools": prompts.served_tools()}


def surface_lines() -> list[str]:
    """Every call the namespace serves, with its real signature.

    Read off the executing functions themselves (``MissionPython._call_*``
    over ``_LIBRARY_CALLS``), so a call the legend advertises and the
    surface does not serve shows up as a row that reads wrong rather than
    as nothing at all.
    """
    import inspect

    from maps.search.parse_compile.python_surface import (
        MissionPython, Rows, served_calls)

    out = []
    for call in served_calls():
        sig = inspect.signature(getattr(MissionPython, f"_call_{call}"))
        params = ", ".join(
            str(p) for n, p in sig.parameters.items() if n != "self")
        out.append(f"  {call}({params})")
    # A cut and a sort are METHODS on a result, not _LIBRARY_CALLS.
    for name in ("filter", "order", "distance"):
        sig = inspect.signature(getattr(Rows, name))
        params = ", ".join(
            str(p) for n, p in sig.parameters.items() if n != "self")
        out.append(f"  <rows>.{name}({params})")
    return out


def dump_text(sensor: str | None = None, agl_m: float | None = None) -> str:
    c = assemble(sensor=sensor, agl_m=agl_m)
    tools_json = json.dumps(c["tools"], indent=2, ensure_ascii=False)
    names = ", ".join(t["function"]["name"] for t in c["tools"])
    return "\n".join([
        f"=== SYSTEM ({len(c['system'])} chars) ===",
        c["system"].rstrip("\n"),
        "",
        f"=== TOOLS [{names}] ({len(tools_json)} chars JSON) ===",
        tools_json,
        "",
        "=== SURFACE (the calls the namespace serves) ===",
        *surface_lines(),
        "",
        "=== USER TURN ===",
        "The operator's instruction, verbatim. Nothing wraps or precedes it:",
        "the tile is discovered through `find`, not pushed.",
        "",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="maps.search.parse_compile.contract",
        description=__doc__.splitlines()[0])
    ap.add_argument("--sensor", metavar="RIG",
                    help="append this rig's sensor block, as a run does; "
                         "one of the sensor_spec registry names")
    ap.add_argument("--agl-m", type=float,
                    default=yaml.safe_load(_CALCITY.read_text())["start_altitude_agl_m"],
                    help="the height above ground the run flies, for a rig that "
                         "states its footprint (default: configs/calcity.yaml's "
                         "start_altitude_agl_m)")
    ap.add_argument("--rendered", action="store_true",
                    help="print the chat-template render instead of the "
                         "sectioned dump; token count goes to stderr")
    ap.add_argument("--model", default="inkling-small",
                    help="served-model alias for the --rendered tokenizer "
                         "(SEARCH_COMPACT_TOKENIZER_NAME overrides)")
    ap.add_argument("--withhold", metavar="NAMES", default=None,
                    help="dump what a bench arm withholding these serves "
                         "(comma separated; maps.search.withhold)")
    args = ap.parse_args(argv)
    if args.withhold is not None:
        from maps.search.withhold import ENV, active  # noqa: PLC0415
        os.environ[ENV] = args.withhold
        active()

    if not args.rendered:
        sys.stdout.write(dump_text(sensor=args.sensor, agl_m=args.agl_m))
        return 0

    # Lazy: maps.llm.tokenizer eagerly imports transformers.
    from maps.llm.tokenizer import _tokenizer

    c = assemble(sensor=args.sensor, agl_m=args.agl_m)
    tok = _tokenizer(args.model)
    rendered = tok.apply_chat_template(
        [{"role": "system", "content": c["system"]},
         {"role": "user", "content": _SAMPLE_INSTRUCTION}],
        tools=c["tools"], add_generation_prompt=True, tokenize=False)
    sys.stdout.write(rendered)
    n = len(tok(rendered)["input_ids"])
    print(f"tokens: {n} (system + tools + sample instruction + "
          "generation prompt)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
