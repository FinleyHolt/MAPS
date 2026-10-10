"""Print the served LLM contract: system text + the mission tools.

The audit artifact — what the model gets at THIS sha, assembled from the
same single owners the flight path serves (``served_system_text`` and
``served_tools``, which serve the flown constants unless a bench arm
withholds a part). A DERIVATION from the
constants; the flight log records what a real run actually sent (the
capture side). Diff two shas by dumping at each.

  python -m maps.search.parse_compile.contract
  python -m maps.search.parse_compile.contract --sensor <rig> [--agl-m <m>]
  python -m maps.search.parse_compile.contract --rendered   # template bytes
  python -m maps.search.parse_compile.contract --withhold look,images

``--rendered`` prints the chat-template render of [system + one sample
operator instruction] with the tools attached — the byte stream the
model, xgrammar and the prefix cache see — and the token count on stderr.

THE CODE ARM's contract. There is no crops-off variant and no ``--look``:
the code legend states the look()/show_map() capability unconditionally
and the loop asserts the wiring that makes that true. There is no
LANGUAGE section either — Python is not EBNF-able, the arm builds no
decoder mask, and ``CODE_DOC`` IS the language, so it rides the TOOLS
section where the model reads it. A dump that kept the sentence grammar
and lexicon beside a code tool would be the same
two-contradictory-surfaces artifact the system prompt was wired to end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from maps.perception.geometry.sensor_spec import list_sensors, resolve_sensor, served_description
from maps.search.parse_compile import prompts
from maps.search.parse_compile import reasoner_agent as ra

_SAMPLE_INSTRUCTION = "<operator cue>"


def assemble(sensor: str | None = None, agl_m: float | None = None) -> dict:
    """The contract parts exactly as served: system text (with the optional
    sensor block fused the way ``Reasoner`` fuses it, its footprint at
    ``agl_m``) and the mission tools."""
    description = ""
    if sensor:
        try:
            spec = resolve_sensor(sensor)
        except Exception:
            spec = None
        if spec is None:
            raise SystemExit(
                f"contract: unknown sensor '{sensor}' -- one of: "
                + ", ".join(list_sensors()))
        try:
            description = served_description(spec, agl_m)
        except ValueError as exc:
            raise SystemExit(f"contract: {exc} (pass --agl-m)") from exc
    system = ra.served_system_text(sensor_description=description)
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
    # A cut and a sort are METHODS on a result, so they are not in
    # _LIBRARY_CALLS and a derivation over that tuple alone would serve a
    # surface with no way to narrow or rank -- reading as a capability the
    # tree lost rather than as one it moved.
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
    # The SURFACE section replaces the sentence dump's LANGUAGE + LEXICON.
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
    ap.add_argument("--agl-m", type=float, default=None,
                    help="the height above ground the run flies, for a rig that "
                         "states its footprint (configs/calcity.yaml: 914.4)")
    ap.add_argument("--rendered", action="store_true",
                    help="print the chat-template render instead of the "
                         "sectioned dump; token count goes to stderr")
    ap.add_argument("--model", default="qwen36-35b-a3b",
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
