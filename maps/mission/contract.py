"""Print the served LLM contract: system text + the mission tools.

What the model is served at this sha, read from the owners a run serves
through (``served_text``); diff two shas by dumping at each.

  python -m maps.mission.contract
  python -m maps.mission.contract --sensor <rig> [--agl-m <m>]   # Cal City's height by default
  python -m maps.mission.contract --rendered   # template bytes
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import yaml

from maps.library.block_scan import LIBRARY_CALLS
from maps.library.calls import Library
from maps.library.rows import Rows
from maps.mission import served_text
from maps.perception.geometry.sensor_spec import list_sensors, resolve_sensor
from maps.perception.ortho.sensor_frame import served_description
from maps.settings import from_process

_SAMPLE_INSTRUCTION = "<operator cue>"
_CALCITY = Path(__file__).resolve().parents[2] / "configs" / "calcity.yaml"


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
                f"contract: unknown sensor '{sensor}' -- one of: " + ", ".join(list_sensors())
            ) from None
        try:
            description = served_description(spec, agl_m=agl_m)
        except ValueError as exc:
            raise SystemExit(f"contract: {exc} (pass --agl-m)") from exc
    system = served_text.served_system_prompt(description)
    return {"system": system, "tools": served_text.SERVED_TOOLS}


def surface_lines() -> list[str]:
    """Every call the namespace serves, with its real signature, read off
    the executing functions themselves (the ``Library`` method of each
    served call, then the ``Rows`` methods)."""

    def served_signature(fn) -> str:
        # A parameter that exists only to refuse (find's extra queries, a
        # cut's unknown keywords, show_map's missing centre) is not served.
        sig = inspect.signature(fn)
        ps = [
            p.replace(default=p.empty) if p.default is None and p.name == "around" else p
            for n, p in sig.parameters.items()
            if n != "self"
            and p.kind is not p.VAR_KEYWORD
            and not (p.kind is p.VAR_POSITIONAL and p.name == "more")
        ]
        return str(sig.replace(parameters=ps, return_annotation=sig.empty))[1:-1]

    out = [f"  {call}({served_signature(getattr(Library, call))})" for call in LIBRARY_CALLS]
    # A cut, a sort and a distance are methods on a result.
    return out + [
        f"  <rows>.{name}({served_signature(getattr(Rows, name))})"
        for name in ("filter", "order", "distance")
    ]


def dump_text(sensor: str | None = None, agl_m: float | None = None) -> str:
    """The sectioned dump: system text, tools, the served surface, the user turn."""
    c = assemble(sensor=sensor, agl_m=agl_m)
    tools_json = json.dumps(c["tools"], indent=2, ensure_ascii=False)
    names = ", ".join(t["function"]["name"] for t in c["tools"])
    return "\n".join(
        [
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
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="maps.mission.contract", description=__doc__.splitlines()[0])
    ap.add_argument(
        "--sensor",
        metavar="RIG",
        help="append this rig's sensor block, as a run does; one of the sensor_spec registry names",
    )
    ap.add_argument(
        "--agl-m",
        type=float,
        default=yaml.safe_load(_CALCITY.read_text())["start_altitude_agl_m"],
        help="the height above ground the run flies, for a rig that "
        "states its footprint (default: configs/calcity.yaml's "
        "start_altitude_agl_m)",
    )
    ap.add_argument(
        "--rendered",
        action="store_true",
        help="print the chat-template render of the system turn, "
        "one sample instruction and the tools instead of the "
        "sectioned dump; token count goes to stderr",
    )
    ap.add_argument(
        "--model",
        default="inkling-small",
        help="served-model alias for the --rendered tokenizer "
        "(SEARCH_COMPACT_TOKENIZER_NAME overrides)",
    )
    args = ap.parse_args(argv)
    if not args.rendered:
        sys.stdout.write(dump_text(sensor=args.sensor, agl_m=args.agl_m))
        return 0

    # Lazy: maps.llm.tokenizer eagerly imports transformers.
    from maps.llm.tokenizer import load_tokenizer

    c = assemble(sensor=args.sensor, agl_m=args.agl_m)
    # The one read of the environment: the tokenizer it names.
    tok = load_tokenizer(args.model, from_process().endpoint.tokenizer_name)
    rendered = tok.apply_chat_template(
        [
            {"role": "system", "content": c["system"]},
            {"role": "user", "content": _SAMPLE_INSTRUCTION},
        ],
        tools=c["tools"],
        add_generation_prompt=True,
        tokenize=False,
    )
    sys.stdout.write(rendered)
    n = len(tok(rendered)["input_ids"])
    print(f"tokens: {n} (system + tools + sample instruction + generation prompt)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
