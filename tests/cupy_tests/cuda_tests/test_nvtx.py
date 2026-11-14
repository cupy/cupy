from __future__ import annotations

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

    def test_RangeStartEnd(self):
        range_id = cuda.nvtx.RangeStart('test:RangeStartEnd', 0xFF000000)
        cuda.nvtx.RangeEnd(range_id)


@unittest.skipIf(cuda.nvtx.available, 'nvtx is installed')
class TestUnavailableNVTX(unittest.TestCase):

    def test_unavailable_operation(self):
        with self.assertRaisesRegex(RuntimeError, 'unavailable in this CuPy'):
            cuda.nvtx.RangePush('unavailable')


@unittest.skipUnless(
    cuda.runtime.is_hip and not cuda.nvtx.available,
    'Requires a HIP build without profiling support')
class TestUnavailableHIPProfiler(unittest.TestCase):

    def test_start_reports_unsupported(self):
        with self.assertRaisesRegex(
                cuda.runtime.CUDARuntimeError, 'hipErrorNotSupported'):
            cuda.runtime.profilerStart()

    def test_stop_reports_unsupported(self):
        with self.assertRaisesRegex(
                cuda.runtime.CUDARuntimeError, 'hipErrorNotSupported'):
            cuda.runtime.profilerStop()
