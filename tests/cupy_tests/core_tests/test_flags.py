import unittest

from cupy._core import flags
from cupy import testing


class TestFlags(unittest.TestCase):

    def setUp(self):
        self.flags = flags.Flags(1, 2, 3)

    def test_c_contiguous(self):
        assert 1 == self.flags['C_CONTIGUOUS']

    def test_f_contiguous(self):
        assert 2 == self.flags['F_CONTIGUOUS']

    def test_owndata(self):
        assert 3 == self.flags['OWNDATA']

    def test_key_error(self):
        with self.assertRaises(KeyError):
            self.flags['unknown key']

    def test_repr(self):
        assert '''  C_CONTIGUOUS : 1
  F_CONTIGUOUS : 2
  OWNDATA : 3''' == repr(self.flags)


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
