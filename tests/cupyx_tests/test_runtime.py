from __future__ import annotations

import unittest
from unittest import mock

import numpy

import cupy
import cupyx


def _get_error_func(error, *args, **kwargs):
    def raise_error(*_args, **_kwargs):
        raise error(*args, **kwargs)
    return raise_error


class TestRuntime(unittest.TestCase):
    def test_runtime(self):
        runtime = cupyx.get_runtime_info()
        assert cupy.__version__ == runtime.cupy_version
        assert cupy.__version__ in str(runtime)

    def test_build_and_runtime_versions(self):
        runtime = cupyx.get_runtime_info()
        assert runtime.numpy_build_version is not None
        assert numpy.version.full_version == runtime.numpy_version
        output = str(runtime)
        for record in (
                'NumPy Build Version',
                'NumPy Runtime Version',
                'cuTENSOR Build Version',
                'cuTENSOR Runtime Version',
                'cuSPARSELt Build Version',
                'cuSPARSELt Runtime Version',
        ):
            assert record in output

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
