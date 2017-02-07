import unittest

import mock

from cupy import prof


class TestTimeRange(unittest.TestCase):

    def test_timerange(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            with prof.timerange('test:range', -1):
                pass
            push.assert_called_once_with('test:range', -1)
            pop.assert_called_once_with()

    def test_timerange_err(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            try:
                with prof.timerange('test:range_error', -1):
                    raise Exception()
            except Exception:
                pass
            push.assert_called_once_with('test:range_error', -1)
            pop.assert_called_once_with()

    def test_timerangeC(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePushC')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            with prof.timerangeC('test:timerangeC', 0):
                pass
            push.assert_called_once_with('test:timerangeC', 0)
            pop.assert_called_once_with()

    def test_timerangeC_err(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePushC')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            try:
                with prof.timerangeC('test:timerangeC_error', 0):
                    raise Exception()
            except Exception:
                pass
            push.assert_called_once_with('test:timerangeC_error', 0)
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
