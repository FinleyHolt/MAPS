"""The contract dump serves the real surfaces, not a drifted copy.

Vacuity floors ride each check: the dump must carry a plausible amount of
text, exactly the one tool, the surface section, and the same objects the
flight path serves — an empty or stubbed dump fails loudly.

The say-legend checks below stay: the sentence surface's constants are
untouched by the code arm and its lexicon law (one spelling per relation,
no word served that the parser refuses) is what a rebuild would have to
honour. They pin the constants, not the dump — nothing here claims the
say tool is served.
"""

import pytest

from maps.perception.geometry.sensor_spec import list_sensors
from maps.search.parse_compile import contract, prompts
from maps.search.parse_compile import reasoner_agent as ra


def test_the_dump_is_the_code_arm_the_flight_path_serves():
    """THE contract, and there is only one wiring of it.

    A dump that showed the say surface while flights sent Python would be
    the same two-contradictory-surfaces artifact the system prompt was
    wired to end -- and this is what the lab route and a reviewer read to
    answer "what was served at this sha".
    """
    c = contract.assemble()
    assert c["system"] == ra.served_system_text(
        base=prompts.CODE_SYSTEM_PROMPT)
    assert c["tools"] is prompts.CODE_TOOLS
    assert [t["function"]["name"] for t in c["tools"]] == ["run"]
    for t in c["tools"]:
        assert len(t["function"]["description"]) > 300

    text = contract.dump_text()
    assert "=== MISSION MAP ===" in text
    assert "=== WORLD CONVENTIONS ===" in text
    assert "=== SURFACE" in text
    # The mission contract has no packet builder: the instruction IS the
    # user turn, and the dump says so instead of rendering one.
    assert "STOP PACKET" not in text
    # No sentence surface anywhere in it: no decoder mask (Python is not
    # EBNF-able and the arm builds none), no closed-class lexicon.
    assert "=== LANGUAGE" not in text and "=== LEXICON" not in text
    assert "root ::= sentence" not in text
    # Vacuity floor: the whole served surface, not a header skeleton.
    assert len(text) > 6_000


def test_sensor_block_fuses_like_a_run():
    name = list_sensors()[0]
    c = contract.assemble(sensor=name)
    assert "Sensor rig (active for this run):" in c["system"]
    # Fused into the ONE system string, never a second message.
    assert c["system"].startswith(ra.served_system_text(
        base=prompts.CODE_SYSTEM_PROMPT))


def test_unknown_sensor_dies_loud_naming_the_registry():
    try:
        contract.assemble(sensor="not_a_rig")
    except SystemExit as exc:
        assert "not_a_rig" in str(exc.code) and list_sensors()[0] in str(exc.code)
    else:
        raise AssertionError("unknown sensor did not raise")










def test_the_dump_surface_section_names_every_call_the_namespace_serves():
    """The code arm's replacement for the LEXICON section.

    Off the EXECUTING functions -- ``MissionPython._call_*`` over
    ``_LIBRARY_CALLS`` -- because the surface has no parser tables to read:
    what the namespace binds and what each ``_call_`` signature accepts is
    the whole of what the model can write.

    Read the SURFACE SECTION, not the whole dump: the same trap the
    lexicon check fell into (scanning the whole dump matched ordinary
    English in the legend, so emptying a table entirely would not have
    failed it).
    """
    import inspect

    from maps.search.parse_compile.python_surface import (
        _LIBRARY_CALLS, MissionPython)
    text = contract.dump_text()
    assert "=== SURFACE" in text
    section = text.split("=== SURFACE", 1)[1].split("=== USER TURN", 1)[0]
    # Vacuity floor. Moved 12 -> 11 when side() was deleted: ordering is
    # global and cuts are anchored, so a free anchored measuring call was
    # the wrong arity for either job, and its number rides the signed
    # `<dir> of <ref>` column instead.
    assert len(_LIBRARY_CALLS) >= 11, _LIBRARY_CALLS
    kws = 0
    for call in _LIBRARY_CALLS:
        line = next((ln for ln in section.splitlines()
                     if ln.strip().startswith(f"{call}(")), None)
        assert line, call
        sig = inspect.signature(getattr(MissionPython, f"_call_{call}"))
        for n, p in sig.parameters.items():
            if n == "self":
                continue
            # every argument the call accepts is named on its own row
            assert n in line, (call, n)
            kws += p.kind is p.KEYWORD_ONLY or p.default is not p.empty
    assert kws >= 9, kws                              # vacuity floor
