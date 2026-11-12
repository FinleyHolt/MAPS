"""The contract dump serves the real surfaces, not a drifted copy."""

from maps.mission import contract, served_text
from maps.perception.geometry.sensor_spec import list_sensors


def test_the_dump_is_the_code_arm_the_flight_path_serves():
    c = contract.assemble()
    assert c["system"] == served_text.SYSTEM_PROMPT
    assert c["tools"] == served_text.SERVED_TOOLS
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
    # EBNF-able and the run builds none), no closed-class lexicon.
    assert "=== LANGUAGE" not in text and "=== LEXICON" not in text
    assert "root ::= sentence" not in text
    # Vacuity floor: the whole served surface, not a header skeleton.
    assert len(text) > 6_000


def test_sensor_block_fuses_like_a_run():
    name = list_sensors()[0]
    c = contract.assemble(sensor=name)
    assert "Sensor rig (active for this run):" in c["system"]
    # Fused into the ONE system string, never a second message.
    assert c["system"].startswith(served_text.served_system_prompt())


def test_unknown_sensor_dies_loud_naming_the_registry():
    try:
        contract.assemble(sensor="not_a_rig")
    except SystemExit as exc:
        assert "not_a_rig" in str(exc.code) and list_sensors()[0] in str(exc.code)
    else:
        raise AssertionError("unknown sensor did not raise")


def test_the_dump_surface_section_names_every_call_the_namespace_serves():
    import inspect

    from maps.library.block_scan import LIBRARY_CALLS
    from maps.library.calls import Library

    text = contract.dump_text()
    assert "=== SURFACE" in text
    section = text.split("=== SURFACE", 1)[1].split("=== USER TURN", 1)[0]
    # Vacuity floor.
    assert len(LIBRARY_CALLS) >= 11, LIBRARY_CALLS
    kws = 0
    for call in LIBRARY_CALLS:
        line = next((ln for ln in section.splitlines() if ln.strip().startswith(f"{call}(")), None)
        assert line, call
        sig = inspect.signature(getattr(Library, call))
        for n, p in sig.parameters.items():
            # A parameter that only refuses is not served (find's *more).
            if n == "self" or p.kind is p.VAR_POSITIONAL:
                continue
            # every argument the call accepts is named on its own row
            assert n in line, (call, n)
            kws += p.kind is p.KEYWORD_ONLY or p.default is not p.empty
    assert kws >= 9, kws  # vacuity floor


def test_a_footprint_rig_dumps_at_the_cal_city_height_by_default(capsys):
    assert contract.main(["--sensor", "oryx_orx32_25mm_nadir"]) == 0
    assert "at the 3,000 ft AGL this run flies" in capsys.readouterr().out
