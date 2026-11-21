from __future__ import annotations

import unittest
from unittest import mock

import cupy
import cupyx
import cupyx._runtime as runtime_module


def _get_error_func(error, *args, **kwargs):
    def raise_error(*_args, **_kwargs):
        raise error(*args, **kwargs)
    return raise_error


class TestRuntime(unittest.TestCase):
    def test_runtime(self):
        runtime = cupyx.get_runtime_info()
        assert cupy.__version__ == runtime.cupy_version
        assert cupy.__version__ in str(runtime)

    def test_error(self):
        runtime = cupyx.get_runtime_info()
        assert 'Error' not in str(runtime)

        with mock.patch(
                'cupy_backends.cuda.api.runtime.driverGetVersion',
                side_effect=_get_error_func(
                    cupy.cuda.runtime.CUDARuntimeError, 0)):
            runtime = cupyx.get_runtime_info()
            assert 'CUDARuntimeError' in str(runtime)

        with mock.patch(
                'cupy_backends.cuda.api.runtime.runtimeGetVersion',
                side_effect=_get_error_func(
                    cupy.cuda.runtime.CUDARuntimeError, 0)):
            runtime = cupyx.get_runtime_info()
            assert 'CUDARuntimeError' in str(runtime)

    def test_error_message_is_compact(self):
        with mock.patch(
                'cupy_backends.cuda.api.runtime.driverGetVersion',
                side_effect=RuntimeError(
                    'Failed to load library from many locations\n'
                    'candidate_path_1\ncandidate_path_2')):
            runtime = cupyx.get_runtime_info()
            output = str(runtime)
            assert (
                'CUDA Driver Version'
                '  : RuntimeError: Failed to load library from many locations'
            ) in output
            assert 'candidate_path_1' not in output
            assert 'candidate_path_2' not in output


class TestFormatError(unittest.TestCase):

    def test_single_line(self):
        err = RuntimeError('oops')
        assert runtime_module._format_error(err) == repr(err)

    def test_multi_line(self):
        err = RuntimeError('oops\ntraceback line')
        assert runtime_module._format_error(err) == 'RuntimeError: oops'
