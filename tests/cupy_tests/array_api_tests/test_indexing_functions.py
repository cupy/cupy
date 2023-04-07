import pytest

from cupy import array_api as xp


@pytest.mark.parametrize(
    "obj, ind, axis, expected",
    [
        ([0, 1, 2, 3], [1, 3], -1, [1, 3]),
        ([0, 1, 2, 3], [2, 0], 0, [2, 0]),
        ([[0, 1, 2, 3]], [0, 0, 0], 0, [[0, 1, 2, 3]] * 3),
        ([[0, 1, 2, 3]], [1, 2, 1], 1, [[1, 2, 1]]),
    ],
)
def test_take(obj, ind, axis, expected):
    """
    Tests xp.take function
    """
    x = xp.asarray(obj)
    ind = xp.asarray(ind)
    out = xp.take(x, ind, axis=axis)
    assert xp.all(out == xp.asarray(expected))
