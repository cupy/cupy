import unittest

from cupy import testing


@testing.gpu
class TestArrayContiguity(unittest.TestCase):

    def test_is_contiguous(self):
        a = testing.shaped_arange((2, 3, 4))
        assert a.flags.c_contiguous is True
        b = a.transpose(2, 0, 1)
        assert b.flags.c_contiguous is False
        c = a[::-1]
        assert c.flags.c_contiguous is False
        d = a[:, :, ::2]
        assert d.flags.c_contiguous is False
