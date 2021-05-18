import unittest

from cupy import cuda
from cupy.testing import _attr


@unittest.skipUnless(cuda.nvtx.available, 'nvtx is not installed')
class TestNVTX(unittest.TestCase):

    @_attr.gpu
    def test_Mark(self):
        cuda.nvtx.Mark('test:Mark', 0)

    @_attr.gpu
    def test_MarkC(self):
        cuda.nvtx.MarkC('test:MarkC', 0xFF000000)

    @_attr.gpu
    def test_RangePush(self):
        cuda.nvtx.RangePush('test:RangePush', 1)
        cuda.nvtx.RangePop()

    @_attr.gpu
    def test_RangePushC(self):
        cuda.nvtx.RangePushC('test:RangePushC', 0xFF000000)
        cuda.nvtx.RangePop()
