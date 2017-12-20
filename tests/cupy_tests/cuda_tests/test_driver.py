import unittest

import cupy
from cupy.cuda import driver


class TestDriver(unittest.TestCase):
    def test_ctxGetCurrent(self):
        # Make sure to create context.
        cupy.arange(1)
        self.assertNotEqual(0, driver.ctxGetCurrent())
