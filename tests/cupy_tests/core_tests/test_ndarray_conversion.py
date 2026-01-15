from __future__ import annotations

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing


@pytest.mark.parametrize('shape', [(), (1,), (1, 1, 1)])
@testing.for_all_dtypes()
@testing.numpy_cupy_equal()
def test_ndarray_item(shape, xp, dtype):
    a = xp.full(shape, 3, dtype)
    return a.item()


@pytest.mark.parametrize('shape', [(0,), (2, 3), (1, 0, 1)])
def test_ndarray_item_raise(shape):
    for xp in (numpy, cupy):
        a = testing.shaped_arange(shape, xp, xp.float32)
        with pytest.raises(ValueError):
            a.item()


@pytest.mark.parametrize(
    ('shape', 'order'),
    [
        ((), None),
        ((1,), None),
        ((2, 3), None),
        ((2, 3), 'C'),
        ((2, 3), 'F'),
    ],
)
@testing.for_all_dtypes()
@testing.numpy_cupy_equal()
def test_item(shape, order, xp, dtype):
    if (runtime.is_hip and
            (shape == (1,) or
             (shape == (2, 3) and order is None))):
        pytest.xfail('ROCm/HIP may have a bug')
    a = testing.shaped_arange(shape, xp, dtype)
    if order is not None:
        return a.tobytes(order)
    else:
        return a.tobytes()
