import unittest
from unittest import mock

import numpy

import cupy
from cupy import testing


class CreateMock(object):

    def __init__(self, target):
        self.target = eval(target)
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.target(*args, **kwargs)

    def check_call_count(self, xp, count):
        # TODO(asi1024): Uncomment after replace fusion implementaiton.

        # assert xp in (numpy, cupy)
        # assert isinstance(count, int)
        # if xp is cupy:
        #     assert self.call_count == count
        pass


def mock_fusion_history():
    def wrapper(impl):
        def new_impl(self):
            target = 'cupy._core._fusion_trace.TraceImpl'
            with mock.patch(target, CreateMock(target)) as m:
                numpy_result = impl(self, numpy, m)
            with mock.patch(target, CreateMock(target)) as m:
                cupy_result = impl(self, cupy, m)
            testing.assert_array_list_equal(numpy_result, cupy_result)
        return new_impl
    return wrapper


@testing.gpu
class TestFusionCache(unittest.TestCase):

    @mock_fusion_history()
    def test_same_array(self, xp, m):
        @cupy.fuse()
        def f(x, y):
            return x + y

        result = []
        m.check_call_count(xp, 0)

        x = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=1)
        result.append(f(x, y))
        m.check_call_count(xp, 1)

        result.append(f(x, y))
        m.check_call_count(xp, 1)

        return result

    @mock_fusion_history()
    def test_dtype_combinations(self, xp, m):
        @cupy.fuse()
        def f(x, y):
            return x + y

        result = []
        m.check_call_count(xp, 0)

        x = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=1)
        result.append(f(x, y))
        m.check_call_count(xp, 1)

        x = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=2)
        y = testing.shaped_random((3, 4), xp, 'int16', scale=10, seed=3)
        result.append(f(x, y))
        m.check_call_count(xp, 2)

        x = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=4)
        y = testing.shaped_random((3, 4), xp, 'int16', scale=10, seed=5)
        result.append(f(x, y))
        m.check_call_count(xp, 2)

        x = testing.shaped_random((3, 4), xp, 'int16', scale=10, seed=6)
        y = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=7)
        result.append(f(x, y))
        m.check_call_count(xp, 3)

        x = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=8)
        y = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=9)
        result.append(f(x, y))
        m.check_call_count(xp, 3)

        return result

    @mock_fusion_history()
    def test_shape_combinations(self, xp, m):
        @cupy.fuse()
        def f(x, y):
            return x + y

        result = []
        m.check_call_count(xp, 0)

        x = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=1)
        result.append(f(x, y))
        m.check_call_count(xp, 1)

        x = testing.shaped_random((4, 5), xp, 'int32', scale=10, seed=2)
        y = testing.shaped_random((4, 5), xp, 'int32', scale=10, seed=3)
        result.append(f(x, y))
        m.check_call_count(xp, 1)

        x = testing.shaped_random((5,), xp, 'int32', scale=10, seed=4)
        y = testing.shaped_random((4, 5), xp, 'int32', scale=10, seed=5)
        result.append(f(x, y))
        m.check_call_count(xp, 2)

        x = testing.shaped_random((4, 3), xp, 'int32', scale=10, seed=6)
        y = testing.shaped_random((4, 3), xp, 'int32', scale=10, seed=7)
        result.append(f(x, y))
        m.check_call_count(xp, 2)

        x = testing.shaped_random((4, 1), xp, 'int32', scale=10, seed=8)
        y = testing.shaped_random((4, 5), xp, 'int32', scale=10, seed=9)
        result.append(f(x, y))
        m.check_call_count(xp, 3)

        x = testing.shaped_random((1, 1), xp, 'int32', scale=10, seed=8)
        y = testing.shaped_random((1, 1), xp, 'int32', scale=10, seed=9)
        result.append(f(x, y))
        m.check_call_count(xp, 3)

        x = testing.shaped_random((2, 5), xp, 'int32', scale=10, seed=10)
        y = testing.shaped_random((4, 5), xp, 'int32', scale=10, seed=11)
        with self.assertRaises(ValueError, msg='could not be broadcast'):
            f(x, y)
        m.check_call_count(xp, 4)

        return result

    @mock_fusion_history()
    def test_memoryspace_combinations(self, xp, m):
        @cupy.fuse()
        def f(x, y):
            return x + y

        result = []
        m.check_call_count(xp, 0)

        x = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=0)
        y = x
        result.append(f(x, y))
        m.check_call_count(xp, 1)

        x = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=2)
        y = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=3)
        result.append(f(x, y))
        m.check_call_count(xp, 2)

        x = testing.shaped_random((3, 3), xp, 'int32', scale=10, seed=4)
        y = x
        result.append(f(x, y))
        m.check_call_count(xp, 2)

        x = testing.shaped_random((3, 3), xp, 'int32', scale=10, seed=6)
        y = x.T
        result.append(f(x, y))
        m.check_call_count(xp, 3)

        return result
