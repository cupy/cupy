import unittest

from cupy import cuda
from cupy.testing import attr


@unittest.skipUnless(cuda.nvtx_enabled, 'nvtx is not installed')
class TestNVTX(unittest.TestCase):

    @attr.gpu
    def test_Mark(self):
        cuda.nvtx.Mark('test:Mark', 0)

    @attr.gpu
    def test_MarkC(self):
        cuda.nvtx.MarkC('test:MarkC', 0xFF000000)

    @attr.gpu
    def test_RangePush(self):
        cuda.nvtx.RangePush('test:RangePush', 1)
        cuda.nvtx.RangePop()

    @attr.gpu
    def test_RangePushC(self):
        cuda.nvtx.RangePushC('test:RangePushC', 0xFF000000)
        cuda.nvtx.RangePop()
