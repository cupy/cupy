import unittest

import mock

import cupy
import cupyx


try:
    import cupy.cuda.cudnn as cudnn
except ImportError:
    cudnn = None


def _get_error_func(error, *args, **kwargs):
    def raise_error(*_args, **_kwargs):
        raise error(*args, **kwargs)
    return raise_error


class TestRuntime(unittest.TestCase):
    def test_runtime(self):
        runtime = cupyx.get_runtime_info()
        assert cupy.__version__ == runtime.cupy_version
        assert cupy.__version__ in str(runtime)

    @unittest.skipUnless(cudnn is not None, 'cuDNN is required')
    def test_error(self):
        runtime = cupyx.get_runtime_info()
        assert 'Error' not in str(runtime)

        with mock.patch(
                'cupy.cuda.runtime.driverGetVersion',
                side_effect=_get_error_func(
                    cupy.cuda.runtime.CUDARuntimeError, 0)):
            runtime = cupyx.get_runtime_info()
            assert 'CUDARuntimeError' in str(runtime)

        with mock.patch(
                'cupy.cuda.runtime.runtimeGetVersion',
                side_effect=_get_error_func(
                    cupy.cuda.runtime.CUDARuntimeError, 0)):
            runtime = cupyx.get_runtime_info()
            assert 'CUDARuntimeError' in str(runtime)

        with mock.patch(
                'cupy.cuda.cudnn.getVersion',
                side_effect=_get_error_func(cudnn.CuDNNError, 0)):
            runtime = cupyx.get_runtime_info()
            assert 'CuDNNError' in str(runtime)
