import unittest

from cupy import core
from cupy import testing


@testing.gpu
class TestArrayOwndata(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.a = core.ndarray(())

    def test_original_array(self):
        self.assertTrue(self.a.flags.owndata)

    def test_view_array(self):
        v = self.a.view()
        self.assertFalse(v.flags.owndata)

    def test_reshaped_array(self):
        r = self.a.reshape(())
        self.assertFalse(r.flags.owndata)
