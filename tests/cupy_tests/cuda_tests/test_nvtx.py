import unittest
from cupy.cuda import nvtx


class TestNVTX(unittest.TestCase):
    def test_Mark(self):
        nvtx.Mark("test:Mark", 0)

    def test_MarkC(self):
        nvtx.MarkC("test:MarkC", 0xFF000000)

    def test_RangePush(self):
        nvtx.RangePush("test:RangePush", 1)
        nvtx.RangePop()

    def test_RangePushC(self):
        nvtx.RangePushC("test:RangePushC", 0xFF000000)
        nvtx.RangePop()
