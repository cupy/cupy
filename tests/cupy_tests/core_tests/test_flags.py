import unittest

import numpy
import pytest

import cupy
from cupy._core import flags
from cupy import testing


class TestFlags(unittest.TestCase):

    def setUp(self):
        class DummyArray:
            def __init__(self):
                self._c_contiguous = 1
                self._f_contiguous = 2
                self.base = 3
                self._writeable = 4

        self.flags = flags.Flags(DummyArray())

    def test_c_contiguous(self):
        assert 1 == self.flags['C_CONTIGUOUS']

    def test_f_contiguous(self):
        assert 2 == self.flags['F_CONTIGUOUS']

    def test_owndata(self):
        assert False is self.flags['OWNDATA']

    def test_writeable(self):
        assert 4 == self.flags['WRITEABLE']

    def test_key_error(self):
        with self.assertRaises(KeyError):
            self.flags['unknown key']

    def test_repr(self):
        assert '\n'.join([
            '  C_CONTIGUOUS : 1',
            '  F_CONTIGUOUS : 2',
            '  OWNDATA : False',
            '  WRITEABLE : 4'
        ]) == repr(self.flags)


@testing.parameterize(
    *testing.product({
        'order': ['C', 'F', 'non-contiguous'],
        'shape': [(8, ), (4, 8)],
    })
)
class TestContiguityFlags(unittest.TestCase):

    def setUp(self):
        self.flags = None

    def init_flags(self, xp):
        if self.order == 'non-contiguous':
            a = xp.empty(self.shape)[::2]
        else:
            a = xp.empty(self.shape, order=self.order)
        self.flags = a.flags

    @testing.numpy_cupy_equal()
    def test_fnc(self, xp):
        self.init_flags(xp)
        return self.flags.fnc

    @testing.numpy_cupy_equal()
    def test_forc(self, xp):
        self.init_flags(xp)
        return self.flags.forc

    @testing.numpy_cupy_equal()
    def test_f_contiguous(self, xp):
        self.init_flags(xp)
        return self.flags.f_contiguous

    @testing.numpy_cupy_equal()
    def test_c_contiguous(self, xp):
        self.init_flags(xp)
        return self.flags.c_contiguous

    def test_c_contiguous_setter(self):
        for xp in (numpy, cupy):
            self.init_flags(xp)
            with pytest.raises(AttributeError):
                self.flags.c_contiguous = True

    def test_f_contiguous_setter(self):
        for xp in (numpy, cupy):
            self.init_flags(xp)
            with pytest.raises(AttributeError):
                self.flags.f_contiguous = True

    def test_owndata_setter(self):
        for xp in (numpy, cupy):
            self.init_flags(xp)
            with pytest.raises(AttributeError):
                self.flags.owndata = True


class TestWriteableFlags:

    def test_writeable(self):
        for xp in (numpy, cupy):
            x = xp.array([1, 2, 3])
            assert x.flags.writeable is True
            x.flags.writeable = False
            assert x.flags.writeable is False
            x.flags.writeable = True
            assert x.flags.writeable is True

    def test_writeable_false_view(self):
        for xp in (numpy, cupy):
            x = xp.array([1, 2, 3])
            assert x.flags.writeable is True
            x.flags.writeable = False
            assert x.flags.writeable is False
            y = x.view()
            assert x.flags.writeable is False
            assert y.flags.writeable is False

    def test_writeable_view_x_false(self):
        for xp in (numpy, cupy):
            x = xp.array([1, 2, 3])
            assert x.flags.writeable is True
            y = x.view()
            x.flags.writeable = False
            assert x.flags.writeable is False
            assert y.flags.writeable is True

    def test_writeable_view_y_false(self):
        for xp in (numpy, cupy):
            x = xp.array([1, 2, 3])
            assert x.flags.writeable is True
            y = x.view()
            y.flags.writeable = False
            assert x.flags.writeable is True
            assert y.flags.writeable is False

    def test_writeable_set_to_view(self):
        for xp in (numpy, cupy):
            x = xp.array([1, 2, 3])
            x.flags.writeable = False
            y = x.view()
            assert x.flags.writeable is False
            assert y.flags.writeable is False
            with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
                y.flags.writeable = True
            assert x.flags.writeable is False
            assert y.flags.writeable is False
            y.flags.writeable = False
            assert x.flags.writeable is False
            assert y.flags.writeable is False
            x.flags.writeable = True
            assert x.flags.writeable is True
            assert y.flags.writeable is False
            y.flags.writeable = True
            assert x.flags.writeable is True
            assert y.flags.writeable is True

    def test_writeable_set_to_view_of_view(self):
        for xp in (numpy, cupy):
            x = xp.array([1, 2, 3])
            y = x.view()
            y.flags.writeable = False
            z = y.view()
            assert x.flags.writeable is True
            assert y.flags.writeable is False
            assert z.flags.writeable is False
            z.flags.writeable = True
            assert x.flags.writeable is True
            assert y.flags.writeable is False
            assert z.flags.writeable is True


class TestWriteableFalse:

    def test_ufunc_out(self):
        for xp in (numpy, cupy):
            out = xp.array([0, 0, 0])
            out.flags.writeable = False
            x = xp.array([1, 2, 3])
            with pytest.raises(ValueError, match='read-only'):
                xp.negative(x, out=out)

    def test_setitem(self):
        for xp in (numpy, cupy):
            out = xp.array([1, 2, 3])
            out.flags.writeable = False
            with pytest.raises(ValueError, match='read-only'):
                out[0] = 100

    def test_copy(self):
        for xp in (numpy, cupy):
            out = xp.array([0, 0, 0])
            out.flags.writeable = False
            x = xp.array([1, 2, 3])
            with pytest.raises(ValueError, match='read-only'):
                out[:] = x

    def test_inplace_add(self):
        for xp in (numpy, cupy):
            out = xp.array([1, 2, 3])
            out.flags.writeable = False
            with pytest.raises(ValueError, match='read-only'):
                out += 100

    def test_reduction_out(self):
        for xp in (numpy, cupy):
            out = xp.array([0, 0, 0])
            out.flags.writeable = False
            x = xp.array([[1, 2, 3], [4, 5, 6]])
            with pytest.raises(ValueError, match='read-only'):
                xp.sum(x, axis=0, out=out)

    def test_matmul(self):
        for xp in (numpy, cupy):
            out = xp.zeros((10, 15), dtype=numpy.float32)
            out.flags.writeable = False
            x1 = xp.zeros((10, 12), dtype=numpy.float32)
            x2 = xp.zeros((12, 15), dtype=numpy.float32)
            with pytest.raises(ValueError, match='read-only'):
                xp.matmul(x1, x2, out=out)
