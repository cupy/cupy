import unittest
from unittest import mock

import cupy
import cupyx


from cupy.cuda import cudnn


def _get_error_func(error, *args, **kwargs):
    def raise_error(*_args, **_kwargs):
        raise error(*args, **kwargs)
    return raise_error


class TestRuntime(unittest.TestCase):
    def test_runtime(self):
        runtime = cupyx.get_runtime_info()
        assert cupy.__version__ == runtime.cupy_version
        assert cupy.__version__ in str(runtime)

    @unittest.skipUnless(cudnn.available, 'cuDNN is required')
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

        with mock.patch(
                'cupy_backends.cuda.libs.cudnn.getVersion',
                side_effect=_get_error_func(cudnn.CuDNNError, 0)):
            runtime = cupyx.get_runtime_info()
            assert 'CuDNNError' in str(runtime)
