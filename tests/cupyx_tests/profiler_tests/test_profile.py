from __future__ import annotations

import unittest
from unittest import mock

from cupyx import profiler


class TestProfile(unittest.TestCase):

    def test_profile(self):
        start_patch = mock.patch(
            'cupy_backends.cuda.api.runtime.profilerStart')
        stop_patch = mock.patch('cupy_backends.cuda.api.runtime.profilerStop')
        with start_patch as start, stop_patch as stop:
            with profiler.profile():
                pass
            start.assert_called_once_with()
            stop.assert_called_once_with()

    def test_err_case(self):
        start_patch = mock.patch(
            'cupy_backends.cuda.api.runtime.profilerStart')
        stop_patch = mock.patch('cupy_backends.cuda.api.runtime.profilerStop')
        with start_patch as start, stop_patch as stop:
            try:
                with profiler.profile():
                    raise Exception()
            except Exception:
                # ignore
                pass
            start.assert_called_once_with()
            stop.assert_called_once_with()
