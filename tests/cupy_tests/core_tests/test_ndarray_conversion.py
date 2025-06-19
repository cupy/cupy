import unittest

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing


@testing.parameterize(
    {'shape': ()},
    {'shape': (1,)},
    {'shape': (1, 1, 1)},
)
class TestNdarrayItem(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_item(self, xp, dtype):
        a = xp.full(self.shape, 3, dtype)
        return a.item()


@testing.parameterize(
    {'shape': (0,)},
    {'shape': (2, 3)},
    {'shape': (1, 0, 1)},
)
class TestNdarrayItemRaise(unittest.TestCase):

    def test_item(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, xp.float32)
            with pytest.raises(ValueError):
                a.item()


@testing.parameterize(
    {'shape': ()},
    {'shape': (1,)},
    {'shape': (2, 3)},
    {'shape': (2, 3), 'order': 'C'},
    {'shape': (2, 3), 'order': 'F'},
)
class TestNdarrayToBytes(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_item(self, xp, dtype):
        if (runtime.is_hip and
            (self.shape == (1,) or
             (self.shape == (2, 3) and not hasattr(self, 'order')))):
            pytest.xfail('ROCm/HIP may have a bug')
        a = testing.shaped_arange(self.shape, xp, dtype)
        if hasattr(self, 'order'):
            return a.tobytes(self.order)
        else:
            return a.tobytes()
