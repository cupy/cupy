import unittest

from cupy import cuda


@unittest.skipUnless(cuda.nvtx.available, 'nvtx is not installed')
class TestNVTX(unittest.TestCase):

    def test_Mark(self):
        cuda.nvtx.Mark('test:Mark', 0)

    def test_MarkC(self):
        cuda.nvtx.MarkC('test:MarkC', 0xFF000000)

    def test_RangePush(self):
        cuda.nvtx.RangePush('test:RangePush', 1)
        cuda.nvtx.RangePop()

    def test_RangePushC(self):
        cuda.nvtx.RangePushC('test:RangePushC', 0xFF000000)
        cuda.nvtx.RangePop()
