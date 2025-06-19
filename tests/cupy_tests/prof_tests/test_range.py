import unittest
from unittest import mock

from cupy import cuda
from cupy import prof


@unittest.skipUnless(cuda.nvtx.available, 'nvtx is required for time_range')
class TestTimeRange(unittest.TestCase):

    def test_time_range(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            with prof.time_range('test:time_range', color_id=-1):
                pass
            push.assert_called_once_with('test:time_range', -1)
            pop.assert_called_once_with()

    def test_time_range_with_ARGB(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePushC')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            with prof.time_range('test:time_range_with_ARGB',
                                 argb_color=0xFF00FF00):
                pass
            push.assert_called_once_with(
                'test:time_range_with_ARGB', 0xFF00FF00)
            pop.assert_called_once_with()

    def test_time_range_err(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            try:
                with prof.time_range('test:time_range_error', -1):
                    raise Exception()
            except Exception:
                pass
            push.assert_called_once_with('test:time_range_error', -1)
            pop.assert_called_once_with()

    def test_TimeRangeDecorator(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            @prof.TimeRangeDecorator()
            def f():
                pass
            f()
            # Default value of color id is -1
            push.assert_called_once_with('f', -1)
            pop.assert_called_once_with()

    def test_TimeRangeDecorator_with_ARGB(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePushC')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            @prof.TimeRangeDecorator(argb_color=0xFFFF0000)
            def f():
                pass
            f()
            push.assert_called_once_with('f', 0xFFFF0000)
            pop.assert_called_once_with()

    def test_TimeRangeDecorator_err(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            @prof.TimeRangeDecorator()
            def f():
                raise Exception()
            try:
                f()
            except Exception:
                pass
            # Default value of color id is -1
            push.assert_called_once_with('f', -1)
            pop.assert_called_once_with()


class TestTimeRangeNVTXUnavailable(unittest.TestCase):

    def setUp(self):
        self.nvtx_available = cuda.nvtx.available
        cuda.nvtx.available = False

    def tearDown(self):
        cuda.nvtx.available = self.nvtx_available

    def test_time_range(self):
        with self.assertRaises(RuntimeError):
            with prof.time_range(''):
                pass

    def test_time_range_decorator(self):
        with self.assertRaises(RuntimeError):
            prof.TimeRangeDecorator()
