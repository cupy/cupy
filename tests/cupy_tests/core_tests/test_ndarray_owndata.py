import unittest

from cupy import core
from cupy import testing


@testing.gpu
class TestArrayOwndata(unittest.TestCase):

    def setUp(self):
        self.a = core.ndarray(())

    def test_original_array(self):
        assert self.a.flags.owndata

    def test_view_array(self):
        v = self.a.view()
        assert not v.flags.owndata

    def test_reshaped_array(self):
        r = self.a.reshape(())
        assert not r.flags.owndata
