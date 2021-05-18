import unittest

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing


@testing.gpu
class TestDims(unittest.TestCase):

    def check_atleast(self, func, xp):
        a = testing.shaped_arange((), xp)
        b = testing.shaped_arange((2,), xp)
        c = testing.shaped_arange((2, 2), xp)
        d = testing.shaped_arange((4, 3, 2), xp)
        e = 1
        f = numpy.float32(1)
        return func(a, b, c, d, e, f)

    @testing.numpy_cupy_array_equal()
    def test_atleast_1d1(self, xp):
        return self.check_atleast(xp.atleast_1d, xp)

    @testing.numpy_cupy_array_equal()
    def test_atleast_1d2(self, xp):
        a = testing.shaped_arange((1, 3, 2), xp)
        return xp.atleast_1d(a)

    @testing.numpy_cupy_array_equal()
    def test_atleast_2d1(self, xp):
        return self.check_atleast(xp.atleast_2d, xp)

    @testing.numpy_cupy_array_equal()
    def test_atleast_2d2(self, xp):
        a = testing.shaped_arange((1, 3, 2), xp)
        return xp.atleast_2d(a)

    @testing.numpy_cupy_array_equal()
    def test_atleast_3d1(self, xp):
        return self.check_atleast(xp.atleast_3d, xp)

    @testing.numpy_cupy_array_equal()
    def test_atleast_3d2(self, xp):
        a = testing.shaped_arange((1, 3, 2), xp)
        return xp.atleast_3d(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_broadcast_to(self, xp, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((3, 1, 4), xp, dtype)
        b = xp.broadcast_to(a, (2, 3, 3, 4))
        return b

    @testing.for_all_dtypes()
    def test_broadcast_to_fail(self, dtype):
        for xp in (numpy, cupy):
            # Note that broadcast_to is only supported on numpy>=1.10
            a = testing.shaped_arange((3, 1, 4), xp, dtype)
            with pytest.raises(ValueError):
                xp.broadcast_to(a, (1, 3, 4))

    @testing.for_all_dtypes()
    def test_broadcast_to_short_shape(self, dtype):
        for xp in (numpy, cupy):
            # Note that broadcast_to is only supported on numpy>=1.10
            a = testing.shaped_arange((1, 3, 4), xp, dtype)
            with pytest.raises(ValueError):
                xp.broadcast_to(a, (3, 4))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_broadcast_to_numpy19(self, xp, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((3, 1, 4), xp, dtype)
        if xp is cupy:
            b = xp.broadcast_to(a, (2, 3, 3, 4))
        else:
            dummy = xp.empty((2, 3, 3, 4))
            b, _ = xp.broadcast_arrays(a, dummy)
        return b

    @testing.for_all_dtypes()
    def test_broadcast_to_fail_numpy19(self, dtype):
        for xp in (numpy, cupy):
            # Note that broadcast_to is only supported on numpy>=1.10
            a = testing.shaped_arange((3, 1, 4), xp, dtype)
            with pytest.raises(ValueError):
                xp.broadcast_to(a, (1, 3, 4))

    @testing.for_all_dtypes()
    def test_broadcast_to_short_shape_numpy19(self, dtype):
        for xp in (numpy, cupy):
            # Note that broadcast_to is only supported on numpy>=1.10
            a = testing.shaped_arange((1, 3, 4), xp, dtype)
            with pytest.raises(ValueError):
                xp.broadcast_to(a, (3, 4))

    @testing.numpy_cupy_array_equal()
    def test_expand_dims0(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, 0)

    @testing.numpy_cupy_array_equal()
    def test_expand_dims1(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, 1)

    @testing.numpy_cupy_array_equal()
    def test_expand_dims2(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, 2)

    @testing.numpy_cupy_array_equal()
    def test_expand_dims_negative1(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, -2)

    @testing.with_requires('numpy>=1.18')
    def test_expand_dims_negative2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3), xp)
            with pytest.raises(numpy.AxisError):
                xp.expand_dims(a, -4)

    @testing.with_requires('numpy>=1.18')
    @testing.numpy_cupy_array_equal()
    def test_expand_dims_tuple_axis(self, xp):
        a = testing.shaped_arange((2, 2, 2), xp)
        return [xp.expand_dims(a, axis) for axis in [
            (0, 1, 2),
            (0, -1, -2),
            (0, 3, 5),
            (0, -3, -5),
            (),
            (1,),
        ]]

    @testing.with_requires('numpy>=1.18')
    def test_expand_dims_out_of_range(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 2, 2), xp)
            for axis in [(1, -6), (1, 5)]:
                with pytest.raises(numpy.AxisError):
                    xp.expand_dims(a, axis)

    @testing.with_requires('numpy>=1.18')
    def test_expand_dims_repeated_axis(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 2, 2), xp)
            with pytest.raises(ValueError):
                xp.expand_dims(a, (1, 1))

    @testing.numpy_cupy_array_equal()
    def test_squeeze1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze()

    @testing.numpy_cupy_array_equal()
    def test_squeeze2(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.squeeze()

    @testing.numpy_cupy_array_equal()
    def test_squeeze_int_axis1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=2)

    @testing.numpy_cupy_array_equal()
    def test_squeeze_int_axis2(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=-3)

    def test_squeeze_int_axis_failure1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
            with pytest.raises(ValueError):
                a.squeeze(axis=-9)

    def test_squeeze_int_axis_failure2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
            with pytest.raises(numpy.AxisError):
                a.squeeze(axis=-9)

    @testing.numpy_cupy_array_equal()
    def test_squeeze_tuple_axis1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=(2, 4))

    @testing.numpy_cupy_array_equal()
    def test_squeeze_tuple_axis2(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=(-4, -3))

    @testing.numpy_cupy_array_equal()
    def test_squeeze_tuple_axis3(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=(4, 2))

    @testing.numpy_cupy_array_equal()
    def test_squeeze_tuple_axis4(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=())

    def test_squeeze_tuple_axis_failure1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
            with pytest.raises(ValueError):
                a.squeeze(axis=(-9,))

    def test_squeeze_tuple_axis_failure2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
            with pytest.raises(ValueError):
                a.squeeze(axis=(2, 2))

    def test_squeeze_tuple_axis_failure3(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
            with pytest.raises(numpy.AxisError):
                a.squeeze(axis=(-9,))

    @testing.numpy_cupy_array_equal()
    def test_squeeze_scalar1(self, xp):
        a = testing.shaped_arange((), xp)
        return a.squeeze(axis=0)

    @testing.numpy_cupy_array_equal()
    def test_squeeze_scalar2(self, xp):
        a = testing.shaped_arange((), xp)
        return a.squeeze(axis=-1)

    def test_squeeze_scalar_failure1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), xp)
            with pytest.raises(ValueError):
                a.squeeze(axis=-2)

    def test_squeeze_scalar_failure2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), xp)
            with pytest.raises(ValueError):
                a.squeeze(axis=1)

    def test_squeeze_scalar_failure3(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), xp)
            with pytest.raises(numpy.AxisError):
                a.squeeze(axis=-2)

    def test_squeeze_scalar_failure4(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), cupy)
            with pytest.raises(numpy.AxisError):
                a.squeeze(axis=1)

    def test_squeeze_failure(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 1, 3, 4), xp)
            with pytest.raises(ValueError):
                a.squeeze(axis=2)

    @testing.numpy_cupy_array_equal()
    def test_external_squeeze(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return xp.squeeze(a)


@testing.parameterize(
    {'shapes': [(), ()]},
    {'shapes': [(0,), (0,)]},
    {'shapes': [(1,), (1,)]},
    {'shapes': [(2,), (2,)]},
    {'shapes': [(0,), (1,)]},
    {'shapes': [(2, 3), (1, 3)]},
    {'shapes': [(2, 1, 3, 4), (3, 1, 4)]},
    {'shapes': [(4, 3, 2, 3), (2, 3)]},
    {'shapes': [(2, 0, 1, 1, 3), (2, 1, 0, 0, 3)]},
    {'shapes': [(0, 1, 1, 3), (2, 1, 0, 0, 3)]},
    {'shapes': [(0, 1, 1, 0, 3), (5, 2, 0, 1, 0, 0, 3), (2, 1, 0, 0, 0, 3)]},
)
@testing.gpu
class TestBroadcast(unittest.TestCase):

    def _broadcast(self, xp, dtype, shapes):
        arrays = [
            testing.shaped_arange(s, xp, dtype) for s in shapes]
        return xp.broadcast(*arrays)

    @testing.for_all_dtypes()
    def test_broadcast(self, dtype):
        broadcast_np = self._broadcast(numpy, dtype, self.shapes)
        broadcast_cp = self._broadcast(cupy, dtype, self.shapes)
        assert broadcast_np.shape == broadcast_cp.shape
        assert broadcast_np.size == broadcast_cp.size
        assert broadcast_np.nd == broadcast_cp.nd

    def _hip_skip_invalid_broadcast(self):
        invalid_shapes = [
            [(1,), (1,)],
            [(2,), (2,)],
        ]
        if runtime.is_hip and self.shapes in invalid_shapes:
            pytest.xfail('HIP/ROCm may have a bug')

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_broadcast_arrays(self, xp, dtype):
        self._hip_skip_invalid_broadcast()
        arrays = [
            testing.shaped_arange(s, xp, dtype) for s in self.shapes]
        return xp.broadcast_arrays(*arrays)


@testing.parameterize(
    {'shapes': [(3,), (2,)]},
    {'shapes': [(3, 2), (2, 3,)]},
    {'shapes': [(3, 2), (3, 4,)]},
    {'shapes': [(0,), (2,)]},
)
@testing.gpu
class TestInvalidBroadcast(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_invalid_broadcast(self, dtype):
        for xp in (numpy, cupy):
            arrays = [testing.shaped_arange(s, xp, dtype) for s in self.shapes]
            with pytest.raises(ValueError):
                xp.broadcast(*arrays)

    @testing.for_all_dtypes()
    def test_invalid_broadcast_arrays(self, dtype):
        for xp in (numpy, cupy):
            arrays = [testing.shaped_arange(s, xp, dtype) for s in self.shapes]
            with pytest.raises(ValueError):
                xp.broadcast_arrays(*arrays)
