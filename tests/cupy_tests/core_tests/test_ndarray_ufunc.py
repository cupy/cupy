import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
class TestArrayUfunc:

    @testing.for_all_dtypes()
    def test_unary_op(self, dtype):
        a = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        outa = numpy.sin(a)
        # numpy operation produced a cupy array
        assert isinstance(outa, cupy.ndarray)
        b = a.get()
        outb = numpy.sin(b)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_unary_op_out(self, dtype):
        a = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        b = a.get()
        outb = numpy.sin(b)
        # pre-make output with same type as input
        outa = cupy.array(numpy.array([0, 1, 2]), dtype=outb.dtype)
        numpy.sin(a, out=outa)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_op(self, dtype):
        a1 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        outa = numpy.add(a1, a2)
        # numpy operation produced a cupy array
        assert isinstance(outa, cupy.ndarray)
        b1 = a1.get()
        b2 = a2.get()
        outb = numpy.add(b1, b2)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_op_out(self, dtype):
        a1 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        outa = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        numpy.add(a1, a2, out=outa)
        b1 = a1.get()
        b2 = a2.get()
        outb = numpy.add(b1, b2)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_mixed_op(self, dtype):
        a1 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype).get()
        with pytest.raises(TypeError):
            # attempt to add cupy and numpy arrays
            numpy.add(a1, a2)
        with pytest.raises(TypeError):
            # check reverse order
            numpy.add(a2, a1)
        with pytest.raises(TypeError):
            # reject numpy output from cupy
            numpy.add(a1, a1, out=a2)
        with pytest.raises(TypeError):
            # reject cupy output from numpy
            numpy.add(a2, a2, out=a1)
        with pytest.raises(ValueError):
            # bad form for out=
            # this is also an error with numpy array
            numpy.sin(a1, out=())
        with pytest.raises(ValueError):
            # bad form for out=
            # this is also an error with numpy array
            numpy.sin(a1, out=(a1, a1))

    @testing.numpy_cupy_array_equal()
    def test_indexing(self, xp):
        a = cupy.testing.shaped_arange((3, 1), xp)[:, :, None]
        b = cupy.testing.shaped_arange((3, 2), xp)[:, None, :]
        return a * b

    @testing.numpy_cupy_array_equal()
    def test_shares_memory(self, xp):
        a = cupy.testing.shaped_arange((1000, 1000), xp, 'int64')
        b = xp.transpose(a)
        a += b
        return a


class TestUfunc:
    @pytest.mark.parametrize('ufunc', [
        'add',
        'sin',
    ])
    @testing.numpy_cupy_equal()
    def test_types(self, xp, ufunc):
        types = getattr(xp, ufunc).types
        if xp == numpy:
            assert isinstance(types, list)
            types = list(dict.fromkeys(  # remove dups: numpy/numpy#7897
                sig for sig in types
                # CuPy does not support the following dtypes:
                # (c)longdouble, datetime, timedelta, and object.
                if not any(t in sig for t in 'GgMmO')
            ))
        return types
