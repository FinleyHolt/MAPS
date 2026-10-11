"""An episode id names one split row: ``<map>::<ann>#<row>`` that exact row,
a bare ``<map>::<ann>`` the first row carrying it."""

from benchmarks.citynav.adapter import find_row


def _row(block, desc):
    return {"area": "birmingham", "block": block, "ann_ids": [0], "descriptions": [desc],
            "target_positions": [[1.0, 2.0, 0.0]], "trajectory": [[0.0, 0.0, 50.0]]}


ROWS = [_row(1, "a"), _row(1, "b"), _row(2, "c")]


def test_an_indexed_id_names_that_row_and_only_if_it_matches():
    assert find_row(ROWS, "birmingham_block_1::0#1") == (1, ROWS[1])
    assert find_row(ROWS, "birmingham_block_1::0#2") is None
    assert find_row(ROWS, "birmingham_block_1::0#9") is None


def test_a_bare_id_names_the_first_row_carrying_it():
    assert find_row(ROWS, "birmingham_block_1::0") == (0, ROWS[0])
    assert find_row(ROWS, "birmingham_block_2::0") == (2, ROWS[2])
    assert find_row(ROWS, "birmingham_block_3::0") is None
