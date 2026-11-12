"""A census served once for two sets folds the second into a census_ref
only when its named count is the same too, so no set loses its count."""

from maps.query.op_io import Result

from maps.query.result_tables import compact_results
def _rows(names):
    return [{"fid": f"prior_{i}", "name": nm, "xy": [float(i), 0.0],
             "tags": {"building": "house"}} for i, nm in enumerate(names)]


def test_same_census_other_named_count_serves_both():
    out = compact_results({"r0": Result(_rows(["A", "", ""])),
                              "r1": Result(_rows(["", "", ""])),
                              "r2": Result(_rows(["", "B", ""]))})
    assert out["r0"]["named"] == 1 and out["r1"]["named"] == 0
    assert "census_ref" not in out["r1"]
    assert out["r2"]["census_ref"] == (
        "census identical to r0's (served once above)")
