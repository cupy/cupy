from chainer.testing import attr
from cupy.cuda import nvtx
import unittest


class TestNVTX(unittest.TestCase):

    @attr.gpu
    def test_Mark(self):
        nvtx.Mark("test:Mark", 0)

    @attr.gpu
    def test_MarkC(self):
        nvtx.MarkC("test:MarkC", 0xFF000000)

    @attr.gpu
    def test_RangePush(self):
        nvtx.RangePush("test:RangePush", 1)
        nvtx.RangePop()

    @attr.gpu
    def test_RangePushC(self):
        nvtx.RangePushC("test:RangePushC", 0xFF000000)
        nvtx.RangePop()
