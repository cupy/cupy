import gc
import threading
import unittest

import pytest

import cupy
from cupy._creation import from_data
from cupy import cuda
from cupy import testing


@testing.parameterize(
    *testing.product({
        'stream_name': ['null', 'ptds'],
    }))
@testing.gpu
class TestStream(unittest.TestCase):

    def setUp(self):
        if cuda.runtime.is_hip and self.stream_name == 'ptds':
            self.skipTest('HIP does not support PTDS')

        self._prev_stream = cuda.get_current_stream()

        if self.stream_name == 'null':
            self.stream = cuda.Stream.null
        elif self.stream_name == 'ptds':
            self.stream = cuda.Stream.ptds
        self.stream.use()

    def tearDown(self):
        self._prev_stream.use()

    @unittest.skipIf(cuda.runtime.is_hip, 'This test is only for CUDA')
    def test_eq_cuda(self):
        null0 = self.stream
        if self.stream == cuda.Stream.null:
            null1 = cuda.Stream(True)
            null2 = cuda.Stream(True)
            null3 = cuda.Stream(ptds=True)
        else:
            null1 = cuda.Stream(ptds=True)
            null2 = cuda.Stream(ptds=True)
            null3 = cuda.Stream(True)
        null4 = cuda.Stream()

        assert null0 == null1
        assert null1 == null2
        assert null2 != null3
        assert null2 != null4

    @unittest.skipIf(not cuda.runtime.is_hip, 'This test is only for HIP')
    def test_eq_hip(self):
        null0 = self.stream
        null1 = cuda.Stream(True)
        null2 = cuda.Stream(True)
        null3 = cuda.Stream()

        assert null0 == null1
        assert null1 == null2
        assert null2 != null3

    def check_del(self, null, ptds):
        stream = cuda.Stream(null=null, ptds=ptds).use()
        assert stream is cuda.get_current_stream()
        stream_ptr = stream.ptr
        x = from_data.array([1, 2, 3])
        del stream
        assert stream_ptr == cuda.get_current_stream().ptr
        cuda.Stream.null.use()
        assert cuda.Stream.null is cuda.get_current_stream()
        # Want to test cudaStreamDestory is issued, but
        # runtime.streamQuery(stream_ptr) causes SEGV. We cannot test...
        del x

    def test_del_default(self):
        self.check_del(null=False, ptds=False)

    def test_del(self):
        null = self.stream == cuda.Stream.null
        if cuda.runtime.is_hip:
            ptds = False
        else:
            ptds = self.stream == cuda.Stream.ptds

        self.check_del(null=null, ptds=ptds)

    def test_get_and_add_callback(self):
        N = 100
        cupy_arrays = [testing.shaped_random((2, 3)) for _ in range(N)]

        if not cuda.runtime.is_hip:
            stream = self.stream
        else:
            # adding callbacks to the null stream in HIP would segfault...
            stream = cuda.Stream()

        out = []
        stream_list = []

        def _callback(s, _, t):
            out.append(t[0])
            stream_list.append(s.ptr)

        for i in range(N):
            numpy_array = cupy_arrays[i].get(stream=stream)
            stream.add_callback(
                _callback,
                (i, numpy_array))

        stream.synchronize()
        assert out == list(range(N))
        assert all(s == stream.ptr for s in stream_list)

    @unittest.skipIf(cuda.runtime.is_hip,
                     'HIP does not support launch_host_func')
    @unittest.skipIf(cuda.driver.get_build_version() < 10000,
                     'Only CUDA 10.0+ supports this')
    def test_launch_host_func(self):
        N = 100
        cupy_arrays = [testing.shaped_random((2, 3)) for _ in range(N)]

        stream = cuda.Stream.null

        out = []
        for i in range(N):
            numpy_array = cupy_arrays[i].get(stream=stream)
            stream.launch_host_func(
                lambda t: out.append(t[0]), (i, numpy_array))

        stream.synchronize()
        assert out == list(range(N))

    def test_with_statement(self):
        stream1 = cuda.Stream()
        stream2 = cuda.Stream()
        assert self.stream == cuda.get_current_stream()
        with stream1:
            assert stream1 == cuda.get_current_stream()
            with stream2:
                assert stream2 == cuda.get_current_stream()
            assert stream1 == cuda.get_current_stream()
        # self.stream is "forgotten"!
        assert cuda.Stream.null == cuda.get_current_stream()

    def test_use(self):
        stream1 = cuda.Stream().use()
        assert stream1 == cuda.get_current_stream()
        self.stream.use()
        assert self.stream == cuda.get_current_stream()

    @testing.multi_gpu(2)
    def test_per_device(self):
        with cuda.Device(0):
            stream0 = cuda.Stream()
            with stream0:
                assert stream0 == cuda.get_current_stream()
                with cuda.Device(1):
                    assert stream0 != cuda.get_current_stream()
                    assert cuda.Stream.null == cuda.get_current_stream()
                assert stream0 == cuda.get_current_stream()

    @testing.multi_gpu(2)
    def test_per_device_failure(self):
        with cuda.Device(0):
            stream0 = cuda.Stream()
        with cuda.Device(1):
            with pytest.raises(RuntimeError):
                with stream0:
                    pass
            with pytest.raises(RuntimeError):
                stream0.use()

    def test_mix_use_context(self):
        # See cupy/cupy#5143
        s1 = cuda.Stream()
        s2 = cuda.Stream()
        s3 = cuda.Stream()
        assert cuda.get_current_stream() == self.stream
        with s1:
            assert cuda.get_current_stream() == s1
            s2.use()
            assert cuda.get_current_stream() == s2
            with s3:
                assert cuda.get_current_stream() == s3
                del s2
            assert cuda.get_current_stream() == s1
        # self.stream is "forgotten"!
        assert cuda.get_current_stream() == cuda.Stream.null

    def test_stream_thread(self):
        s1 = None

        def f1(barrier, errors):
            global s1
            tid = barrier.wait()
            try:
                s1 = cuda.Stream()
                barrier.wait()  # until t2 starts
                s1.use()
                barrier.wait()  # until t2 uses the stream
                s1 = None
                gc.collect()
                barrier.wait()  # until t2 decrefs the stream
                assert cuda.get_current_stream() is not None
                cupy.arange(10)
                errors[tid] = False
            except Exception as e:
                print(f'error in {tid}: {e}')

        def f2(barrier, errors):
            global s1
            tid = barrier.wait()
            try:
                barrier.wait()  # until t1 creates the stream
                s1.use()
                barrier.wait()  # until t1 uses the stream
                s1 = None
                gc.collect()
                barrier.wait()  # until t1 decrefs the stream
                assert cuda.get_current_stream() is not None
                cupy.arange(10)
                errors[tid] = False
            except Exception as e:
                print(f'error in {tid}: {e}')

        barrier = threading.Barrier(2)
        errors = [True, True]
        threads = [
            threading.Thread(target=f1, args=(barrier, errors), daemon=True),
            threading.Thread(target=f2, args=(barrier, errors), daemon=True),
        ]
        del s1
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for err in errors:
            assert err is False


@testing.gpu
class TestExternalStream(unittest.TestCase):

    def setUp(self):
        self.stream_ptr = cuda.runtime.streamCreate()
        self.stream = cuda.ExternalStream(self.stream_ptr)

    def tearDown(self):
        cuda.runtime.streamDestroy(self.stream_ptr)

    def test_get_and_add_callback(self):
        N = 100
        cupy_arrays = [testing.shaped_random((2, 3)) for _ in range(N)]

        stream = self.stream

        out = []
        for i in range(N):
            numpy_array = cupy_arrays[i].get(stream=stream)
            stream.add_callback(
                lambda _, __, t: out.append(t[0]),
                (i, numpy_array))

        stream.synchronize()
        assert out == list(range(N))

    @unittest.skipIf(cuda.runtime.is_hip,
                     'HIP does not support launch_host_func')
    @unittest.skipIf(cuda.driver.get_build_version() < 10000,
                     'Only CUDA 10.0+ supports this')
    def test_launch_host_func(self):
        N = 100
        cupy_arrays = [testing.shaped_random((2, 3)) for _ in range(N)]

        stream = self.stream

        out = []
        for i in range(N):
            numpy_array = cupy_arrays[i].get(stream=stream)
            stream.launch_host_func(
                lambda t: out.append(t[0]), (i, numpy_array))

        stream.synchronize()
        assert out == list(range(N))
