import unittest

import mock

from cupy import prof


class TestTimeRange(unittest.TestCase):

    def test_time_range(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            with prof.time_range('test:range', -1):
                pass
            push.assert_called_once_with('test:range', -1)
            pop.assert_called_once_with()

    def test_time_range_err(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            try:
                with prof.time_range('test:range_error', -1):
                    raise Exception()
            except Exception:
                pass
            push.assert_called_once_with('test:range_error', -1)
            pop.assert_called_once_with()

    def test_time_rangeC(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePushC')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            with prof.time_rangeC('test:time_rangeC', 0):
                pass
            push.assert_called_once_with('test:time_rangeC', 0)
            pop.assert_called_once_with()

    def test_time_rangeC_err(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePushC')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            try:
                with prof.time_rangeC('test:time_rangeC_error', 0):
                    raise Exception()
            except Exception:
                pass
            push.assert_called_once_with('test:time_rangeC_error', 0)
            pop.assert_called_once_with()

    def test_TimeRangeDecorator(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            @prof.TimeRangeDecorator()
            def f():
                pass
            f()
            push.assert_called_once_with('f', 0)
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
            push.assert_called_once_with('f', 0)
            pop.assert_called_once_with()
