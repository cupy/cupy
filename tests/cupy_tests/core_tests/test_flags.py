import unittest

from cupy.core import flags


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
