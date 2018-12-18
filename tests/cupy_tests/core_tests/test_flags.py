import unittest

import cupy
from cupy.core import flags
from cupy import testing


class TestFlags(unittest.TestCase):

    def setUp(self):
        self.flags = flags.Flags(1, 2, 3)

    def test_c_contiguous(self):
        self.assertEqual(1, self.flags['C_CONTIGUOUS'])

    def test_f_contiguous(self):
        self.assertEqual(2, self.flags['F_CONTIGUOUS'])

    def test_owndata(self):
        self.assertEqual(3, self.flags['OWNDATA'])

    def test_key_error(self):
        with self.assertRaises(KeyError):
            self.flags['unknown key']

    def test_repr(self):
        self.assertEqual('''  C_CONTIGUOUS : 1
  F_CONTIGUOUS : 2
  OWNDATA : 3''', repr(self.flags))

    def test_fnc(self):
        a_1d_c = testing.shaped_random((4, ), cupy)
        a_2d_c = testing.shaped_random((4, 4), cupy)
        a_2d_f = cupy.asfortranarray(testing.shaped_random((4, 4), cupy))
        a_2d_noncontig = testing.shaped_random((4, 8), cupy)[:, ::2]
        assert a_1d_c.flags.fnc is False
        assert a_2d_c.flags.fnc is False
        assert a_2d_f.flags.fnc is True
        assert a_2d_noncontig.flags.fnc is False

    def test_forc(self):
        a_1d_c = testing.shaped_random((4, ), cupy)
        a_2d_c = testing.shaped_random((4, 4), cupy)
        a_2d_f = cupy.asfortranarray(testing.shaped_random((4, 4), cupy))
        a_2d_noncontig = testing.shaped_random((4, 8), cupy)[:, ::2]
        assert a_1d_c.flags.forc is True
        assert a_2d_c.flags.forc is True
        assert a_2d_f.flags.forc is True
        assert a_2d_noncontig.flags.forc is False
