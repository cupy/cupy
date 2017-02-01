import unittest

import mock

from cupy import cuda
from cupy.testing import attr


class TestProfile(unittest.TestCase):

    def test_profile(self):
        start_patch = mock.patch('cupy.cuda.profiler.start')
        stop_patch = mock.patch('cupy.cuda.profiler.stop')
        with start_patch as start, stop_patch as stop:
            with cuda.profile():
                pass
            start.assert_called_once_with()
            stop.assert_called_once_with()

    def test_err_case(self):
        start_patch = mock.patch('cupy.cuda.profiler.start')
        stop_patch = mock.patch('cupy.cuda.profiler.stop')
        with start_patch as start, stop_patch as stop:
            try:
                with cuda.profile():
                    raise Exception()
            except Exception:
                # ignore
                pass
            start.assert_called_once_with()
            stop.assert_called_once_with()

    @attr.gpu
    def test_timerange(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            with cuda.timerange('test:timerange', -1):
                pass
            push.assert_called_once_with('test:timerange', -1)
            pop.assert_called_once_with()

    @attr.gpu
    def test_timerange_err(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            try:
                with cuda.timerange('test:timerange_error', -1):
                    raise Exception()
            except Exception:
                pass
            push.assert_called_once_with('test:timerange_error', -1)
            pop.assert_called_once_with()

    @attr.gpu
    def test_timerangeC(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePushC')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            with cuda.timerangeC('test:timerangeC', 0):
                pass
            push.assert_called_once_with('test:timerangeC', 0)
            pop.assert_called_once_with()

    @attr.gpu
    def test_timerangeC_err(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePushC')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            try:
                with cuda.timerangeC('test:timerangeC_error', 0):
                    raise Exception()
            except Exception:
                pass
            push.assert_called_once_with('test:timerangeC_error', 0)
            pop.assert_called_once_with()

    @attr.gpu
    def test_TimeRangeDecorator(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            @cuda.TimeRangeDecorator()
            def f():
                pass
            f()
            push.assert_called_once_with('f', 0)
            pop.assert_called_once_with()

    @attr.gpu
    def test_TimeRangeDecorator_err(self):
        push_patch = mock.patch('cupy.cuda.nvtx.RangePush')
        pop_patch = mock.patch('cupy.cuda.nvtx.RangePop')
        with push_patch as push, pop_patch as pop:
            @cuda.TimeRangeDecorator()
            def f():
                raise Exception()
            try:
                f()
            except Exception:
                pass
            push.assert_called_once_with('f', 0)
            pop.assert_called_once_with()
