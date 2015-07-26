import unittest

from cupy import testing


@testing.gpu
class TestArrayContiguity(unittest.TestCase):

    _multiprocess_can_split_ = True

    def test_is_contiguous(self):
        a = testing.shaped_arange((2, 3, 4))
        self.assertTrue(a.flags.c_contiguous)
        b = a.transpose(2, 0, 1)
        self.assertFalse(b.flags.c_contiguous)
        c = a[::-1]
        self.assertFalse(c.flags.c_contiguous)
        d = a[:, :, ::2]
        self.assertFalse(d.flags.c_contiguous)
