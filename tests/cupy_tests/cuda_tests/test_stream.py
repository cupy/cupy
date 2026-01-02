from __future__ import annotations

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
class TestStream(unittest.TestCase):

    def setUp(self):
        self._prev_stream = cuda.get_current_stream()

        if self.stream_name == 'null':
            self.stream = cuda.Stream.null
        elif self.stream_name == 'ptds':
            self.stream = cuda.Stream.ptds
        self.stream.use()

    def tearDown(self):
        self._prev_stream.use()

    def test_eq(self):
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

    def test_hash(self):
        hash(self.stream)
        hash(cuda.Stream(True))
        hash(cuda.Stream(False))
        mapping = {cuda.Stream(): 1, cuda.Stream(): 2}  # noqa

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
        ptds = self.stream == cuda.Stream.ptds

        self.check_del(null=null, ptds=ptds)

    def test_get_and_add_callback(self):
        N = 100
        cupy_arrays = [testing.shaped_random((2, 3)) for _ in range(N)]

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
                    assert stream0 == cuda.get_current_stream(0)
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

    def test_create_with_flags(self):
        s1 = cuda.Stream()
        s2 = cuda.Stream(non_blocking=True)
        assert s1.is_non_blocking is False
        assert s2.is_non_blocking is True

    def test_create_with_priority(self):
        # parameterize wasn't used since priority gets
        # clamped when it isn't initialized within a specific
        # returned by `cudaDeviceGetStreamPriorityRange`.
        s1 = cuda.Stream(priority=0)
        s2 = cuda.Stream(priority=-1)
        s3 = cuda.Stream(priority=-3)
        assert s1.priority == 0
        assert s2.priority == -1
        # HIP clamps to -1
        assert s3.priority == -1 if cupy.cuda.runtime.is_hip else -3


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

    def test_deprecation_warning(self):
        # Test that ExternalStream raises a deprecation warning
        stream_ptr = cuda.runtime.streamCreate()
        try:
            with pytest.warns(
                DeprecationWarning,
                match='ExternalStream is deprecated'
            ):
                cuda.ExternalStream(stream_ptr)
        finally:
            cuda.runtime.streamDestroy(stream_ptr)


class TestCUDAStreamProtocol(unittest.TestCase):

    def test_cuda_stream_method(self):
        # Test that __cuda_stream__ returns the correct 2-tuple
        stream = cuda.Stream()
        result = stream.__cuda_stream__()

        # Check that it returns a 2-tuple of (version, stream_ptr)
        assert isinstance(result, tuple)
        assert len(result) == 2

        version, stream_ptr = result
        assert isinstance(version, int)
        assert isinstance(stream_ptr, int)
        assert version == 0  # Protocol version
        assert stream_ptr == stream.ptr

    def test_cuda_stream_method_null_stream(self):
        # Test __cuda_stream__ on null stream
        stream = cuda.Stream.null
        result = stream.__cuda_stream__()
        version, stream_ptr = result
        assert version == 0
        assert stream_ptr == stream.ptr

    def test_from_external_with_mock_stream(self):
        # Test Stream.from_external with a mock stream object
        class MockStream:
            def __init__(self, ptr):
                self._ptr = ptr

            def __cuda_stream__(self):
                return (0, self._ptr)

        # Create a real CUDA stream to get a valid pointer
        real_stream = cuda.Stream()
        mock_stream = MockStream(real_stream.ptr)

        # Create a CuPy stream from the mock stream
        cupy_stream = cuda.Stream.from_external(mock_stream)

        assert cupy_stream.ptr == real_stream.ptr
        # Device ID should be the current device
        assert cupy_stream.device_id >= 0
        # Verify that the foreign stream reference is kept
        assert hasattr(cupy_stream, '_foreign_stream_ref')
        assert cupy_stream._foreign_stream_ref is mock_stream

    def test_from_external_with_cupy_stream(self):
        # Test interoperability: CuPy stream -> external -> CuPy stream
        original_stream = cuda.Stream()

        # Use from_external to create a new stream from the original
        new_stream = cuda.Stream.from_external(original_stream)

        assert new_stream.ptr == original_stream.ptr
        # Device ID may differ since protocol doesn't provide it
        assert new_stream._foreign_stream_ref is original_stream

    def test_from_external_without_protocol(self):
        # Test that from_external raises AttributeError for objects
        # without __cuda_stream__
        obj = object()

        with pytest.raises(AttributeError, match='does not implement'):
            cuda.Stream.from_external(obj)

    def test_from_external_invalid_return_type(self):
        # Test that from_external raises TypeError for invalid return types
        class BadStream1:
            def __cuda_stream__(self):
                return 123  # Should return a 2-tuple

        with pytest.raises(TypeError, match='must return a 2-tuple'):
            cuda.Stream.from_external(BadStream1())

    def test_from_external_invalid_tuple_length(self):
        # Test invalid tuple length
        class BadStream2:
            def __cuda_stream__(self):
                return (0,)  # Should return a 2-tuple

        with pytest.raises(TypeError, match='must return a 2-tuple'):
            cuda.Stream.from_external(BadStream2())

    def test_from_external_invalid_element_types(self):
        # Test invalid element types in tuple
        class BadStream3:
            def __cuda_stream__(self):
                return ("not_an_int", 0)  # First element should be int

        with pytest.raises(TypeError, match=r'must return \(int, int\)'):
            cuda.Stream.from_external(BadStream3())

        class BadStream4:
            def __cuda_stream__(self):
                return (0, "not_an_int")  # Second element should be int

        with pytest.raises(TypeError, match=r'must return \(int, int\)'):
            cuda.Stream.from_external(BadStream4())

    def test_from_external_invalid_version(self):
        # Test unsupported protocol version
        class BadStream5:
            def __cuda_stream__(self):
                return (1, 12345)  # Version 1 is not supported

        with pytest.raises(ValueError, match='unsupported version'):
            cuda.Stream.from_external(BadStream5())

    def test_from_external_keeps_stream_alive(self):
        # Test that from_external keeps the foreign stream alive
        import gc
        import weakref

        class MockStream:
            def __init__(self, ptr):
                self._ptr = ptr

            def __cuda_stream__(self):
                return (0, self._ptr)

        real_stream = cuda.Stream()
        mock_stream = MockStream(real_stream.ptr)
        weak_ref = weakref.ref(mock_stream)

        # Create CuPy stream from mock stream
        cupy_stream = cuda.Stream.from_external(mock_stream)

        # Delete the mock stream reference
        del mock_stream
        gc.collect()

        # The mock stream should still be alive because cupy_stream holds
        # a reference
        assert weak_ref() is not None
        assert weak_ref() is cupy_stream._foreign_stream_ref

        # Delete cupy_stream
        del cupy_stream
        gc.collect()

        # Now the mock stream should be garbage collected
        assert weak_ref() is None

    def test_from_external_stream_usage(self):
        # Test that a stream created with from_external can be used normally
        class MockStream:
            def __init__(self, ptr):
                self._ptr = ptr

            def __cuda_stream__(self):
                return (0, self._ptr)

        real_stream = cuda.Stream()
        mock_stream = MockStream(real_stream.ptr)
        cupy_stream = cuda.Stream.from_external(mock_stream)

        # Test that we can use the stream
        with cupy_stream:
            cupy.arange(10)  # Create array on stream
            assert cupy.get_current_stream() == cupy_stream

        # Test that synchronize works
        cupy_stream.synchronize()

        # Test that we can get the stream's properties
        # Just check it doesn't error
        assert cupy_stream.done or not cupy_stream.done
