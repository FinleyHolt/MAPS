"""A find matched as a NAME counts its sub-threshold rows apart from the
rows the name band cut: each row is counted once, in the line that says
why it is not in the set."""

from maps.query import program as GP


def _row(fid, name, x, text):
    return {"fid": fid, "name": name, "xy": [x, 0.0], "tags": {}, "search_text": text}


def test_band_cut_rows_are_not_also_sub_threshold():
    rows = [
        _row("prior_1", "Neuralia Road", 0, "neuralia road"),
        _row("prior_2", "Mendiburu Road", 10, "mendiburu road"),
    ]
    rows += [_row(f"prior_{9 + i}", f"Lot {i}", 50.0 + i, f"lot {i}") for i in range(10)]
    out = GP.run_program(
        [], rows, {"ops": [{"ref": "r0", "op": "find", "args": {"query": "Neuralia Road"}}]}
    )
    notes = out["results"]["r0"].notes
    # 12 rows: 1 matched, 1 cut by the name band ("... Road"), 10 below.
    assert notes[0].startswith("1 of 12 rows matched")
    assert "; 10 sub-threshold row(s) NOT in this set)." in notes[0]
    assert "matched as a NAME: 1 row(s) under the name tier" in notes[1]
