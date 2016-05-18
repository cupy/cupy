import unittest

import mock

from cupy import cuda


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
