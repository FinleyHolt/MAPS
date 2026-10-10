from maps.util.xy_token import is_xy


def test_is_xy_takes_two_numbers_only():
    assert is_xy([1, 2.5]) and is_xy((-3.0, 4))
    for v in ([1], [1, 2, 3], [True, 2], ["1", 2], "[1, 2]", None, [[1, 2], [3, 4]]):
        assert not is_xy(v)
