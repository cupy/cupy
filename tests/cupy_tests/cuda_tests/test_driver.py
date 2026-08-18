from __future__ import annotations

import pickle
import threading
import unittest

import cupy
from cupy import testing
from cupy.cuda import driver


@unittest.skipIf(cupy.cuda.runtime.is_hip, 'Context API is deprecated in HIP')
class TestDriver(unittest.TestCase):
    def test_ctxGetCurrent(self):
        # Make sure to create context.
        cupy.arange(1)
        assert 0 != driver.ctxGetCurrent()

    def test_ctxGetCurrent_thread(self):
        # Make sure to create context in main thread.
        cupy.arange(1)
        _result0 = None
        _result1 = None

        def f(self):
            nonlocal _result0, _result1
            _result0 = driver.ctxGetCurrent()
            cupy.cuda.Device().use()
            cupy.arange(1)
            _result1 = driver.ctxGetCurrent()

        t = threading.Thread(target=f, args=(self,))
        t.daemon = True
        t.start()
        t.join()

        # The returned context pointer must be NULL on sub thread
        # without valid context.
        assert 0 == _result0

        # After the context is created, it should return the valid
        # context pointer.
        assert 0 != _result1

    @testing.multi_gpu(2)
    def test_ctxGetDevice(self):
        with cupy.cuda.Device(1):
            dev = driver.ctxGetDevice()
            assert dev == 1
        with cupy.cuda.Device(0):
            dev = driver.ctxGetDevice()
            assert dev == 0

    def test_streamGetCtx(self):
        s = cupy.cuda.Stream()
        ctx = driver.streamGetCtx(s.ptr)
        ctx2 = driver.ctxGetCurrent()
        assert ctx == ctx2


class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = driver.CUDADriverError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)


class TestCheckStatusClearsStickyError(unittest.TestCase):
    """Regression test for the HIP 7+ driver-API sticky-error leak fixed
    in cupy_backends/cuda/api/driver.pyx::check_status."""

    def test_failed_get_function_does_not_poison_subsequent_launch(self):
        mod = cupy.RawModule(
            code='extern "C" __global__ void k(float *x) {}')
        with self.assertRaises(cupy.cuda.driver.CUDADriverError):
            mod.get_function('no_such_kernel')
        # cupy.random.* would raise CURAND_STATUS_LAUNCH_FAILURE pre-fix.
        result = cupy.random.uniform(-1, 1, 100).astype(cupy.float32)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, cupy.float32)
        # Same poisoning via the ufunc path.
        _ = (result * 2.0).sum()
