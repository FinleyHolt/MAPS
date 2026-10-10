"""Print the served LLM contract: system text + the mission tools.

The audit artifact — what the model gets at THIS sha, assembled from the
same single owners the flight path serves (``served_system_text`` over
``CODE_SYSTEM_PROMPT``, and ``CODE_TOOLS``). A DERIVATION from the
constants; the flight log records what a real run actually sent (the
capture side). Diff two shas by dumping at each.

  python -m maps.search.parse_compile.contract
  python -m maps.search.parse_compile.contract --sensor <rig>
  python -m maps.search.parse_compile.contract --rendered   # template bytes

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
import sys

from maps.perception.geometry.sensor_spec import list_sensors, resolve_sensor
from maps.search.parse_compile import prompts
from maps.search.parse_compile import reasoner_agent as ra

_SAMPLE_INSTRUCTION = "<operator cue>"


def assemble(sensor: str | None = None) -> dict:
    """The contract parts exactly as served: system text (with the optional
    sensor block fused the way ``Reasoner`` fuses it) and the mission
    tools."""
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
        description = spec.description
    system = ra.served_system_text(
                                   sensor_description=description,
                                   base=prompts.CODE_SYSTEM_PROMPT)
    return {"system": system, "tools": prompts.CODE_TOOLS}


def surface_lines() -> list[str]:
    """Every call the namespace serves, with its real signature.

    Read off the executing functions themselves (``MissionPython._call_*``
    over ``_LIBRARY_CALLS``), so a call the legend advertises and the
    surface does not serve shows up as a row that reads wrong rather than
    as nothing at all.
    """
    import inspect

    from maps.search.parse_compile.python_surface import (
        _LIBRARY_CALLS, MissionPython, Rows)

    out = []
    for call in _LIBRARY_CALLS:
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


def dump_text(sensor: str | None = None) -> str:
    c = assemble(sensor=sensor)
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
    ap.add_argument("--rendered", action="store_true",
                    help="print the chat-template render instead of the "
                         "sectioned dump; token count goes to stderr")
    ap.add_argument("--model", default="qwen36-35b-a3b",
                    help="served-model alias for the --rendered tokenizer "
                         "(SEARCH_COMPACT_TOKENIZER_NAME overrides)")
    args = ap.parse_args(argv)

    if not args.rendered:
        sys.stdout.write(dump_text(sensor=args.sensor))
        return 0

    # Lazy: maps.llm.tokenizer eagerly imports transformers.
    from maps.llm.tokenizer import _tokenizer

    c = assemble(sensor=args.sensor)
    tok = _tokenizer(args.model)
    if tok is None:
        print(f"contract: tokenizer for '{args.model}' unavailable "
              "(offline, or set SEARCH_COMPACT_TOKENIZER_NAME)",
              file=sys.stderr)
        return 2
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
