import unittest

from cupy import _core
from cupy import testing


@testing.gpu
class TestArrayOwndata(unittest.TestCase):

    def setUp(self):
        self.a = _core.ndarray(())

    def test_original_array(self):
        assert self.a.flags.owndata is True

    def test_view_array(self):
        v = self.a.view()
        assert v.flags.owndata is False

    def test_reshaped_array(self):
        r = self.a.reshape(())
        assert r.flags.owndata is False
