from __future__ import annotations

import gc
import threading

import pytest

import cupy
from cupy._creation import from_data
from cupy import cuda
from cupy import testing


class TestStream:

    def setup_method(self):
        self._prev_stream = cuda.get_current_stream()

    def teardown_method(self):
        self._prev_stream.use()

    def _get_stream(self, stream_name):
        if stream_name == 'null':
            return cuda.Stream.null
        elif stream_name == 'ptds':
            return cuda.Stream.ptds

    @pytest.mark.parametrize('stream_name', ['null', 'ptds'])
    def test_eq(self, stream_name):
        stream = self._get_stream(stream_name)
        stream.use()
        null0 = stream
        if stream == cuda.Stream.null:
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

    @pytest.mark.parametrize('stream_name', ['null', 'ptds'])
    def test_hash(self, stream_name):
        stream = self._get_stream(stream_name)
        stream.use()
        hash(stream)
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

    @pytest.mark.parametrize('stream_name', ['null', 'ptds'])
    def test_del(self, stream_name):
        stream = self._get_stream(stream_name)
        stream.use()
        null = stream == cuda.Stream.null
        ptds = stream == cuda.Stream.ptds

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

    @pytest.mark.parametrize('stream_name', ['null', 'ptds'])
    def test_with_statement(self, stream_name):
        stream = self._get_stream(stream_name)
        stream.use()
        stream1 = cuda.Stream()
        stream2 = cuda.Stream()
        assert stream == cuda.get_current_stream()
        with stream1:
            assert stream1 == cuda.get_current_stream()
            with stream2:
                assert stream2 == cuda.get_current_stream()
            assert stream1 == cuda.get_current_stream()
        # self.stream is "forgotten"!
        assert cuda.Stream.null == cuda.get_current_stream()

    @pytest.mark.parametrize('stream_name', ['null', 'ptds'])
    def test_use(self, stream_name):
        stream = self._get_stream(stream_name)
        stream.use()
        stream1 = cuda.Stream().use()
        assert stream1 == cuda.get_current_stream()
        stream.use()
        assert stream == cuda.get_current_stream()

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

    @pytest.mark.parametrize('stream_name', ['null', 'ptds'])
    def test_mix_use_context(self, stream_name):
        stream = self._get_stream(stream_name)
        stream.use()
        # See cupy/cupy#5143
        s1 = cuda.Stream()
        s2 = cuda.Stream()
        s3 = cuda.Stream()
        assert cuda.get_current_stream() == stream
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

    def test_mix_use_context_reset(self):
        # See cupy/cupy#8377
        s1 = cuda.Stream()
        s2 = cuda.Stream()
        s1.use()
        assert cuda.get_current_stream() == s1
        with s2:
            assert cuda.get_current_stream() == s2
        assert cuda.get_current_stream() == s1

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


class TestExternalStream:

    def setup_method(self):
        self.stream_ptr = cuda.runtime.streamCreate()
        # Test that ExternalStream raises a deprecation warning
        with pytest.warns(
            DeprecationWarning,
            match='ExternalStream is deprecated'
        ):
            self.stream = cuda.ExternalStream(self.stream_ptr)

    def teardown_method(self):
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


class TestCUDAStreamProtocol:

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
        # Device ID should be -1 (unknown) per protocol design
        assert cupy_stream.device_id == -1
        # Verify that the foreign stream reference is kept
        assert hasattr(cupy_stream, '_foreign_stream_ref')
        assert cupy_stream._foreign_stream_ref is mock_stream

    def test_from_external_with_cupy_stream(self):
        # Test interoperability: CuPy stream -> external -> CuPy stream
        original_stream = cuda.Stream()

        # Use from_external to create a new stream from the original
        new_stream = cuda.Stream.from_external(original_stream)

        assert new_stream.ptr == original_stream.ptr
        # Device ID is -1 since protocol doesn't provide it
        assert new_stream.device_id == -1
        assert new_stream._foreign_stream_ref is original_stream

    def test_from_external_without_protocol(self):
        # Test that from_external raises TypeError for objects
        # without __cuda_stream__
        obj = object()

        with pytest.raises(TypeError, match='does not implement'):
            cuda.Stream.from_external(obj)

    @pytest.mark.parametrize(
        'bad_return, expected_error, expected_match',
        [
            (123, TypeError, 'must return a 2-tuple'),
            ((0,), TypeError, 'must return a 2-tuple'),
            (('not_an_int', 0), TypeError, r'must return \(int, int\)'),
            ((0, 'not_an_int'), TypeError, r'must return \(int, int\)'),
            ((1, 12345), TypeError, 'unsupported version'),
            ((-1, 12345), TypeError, 'unsupported version'),
        ],
    )
    def test_from_external_invalid_returns(
            self, bad_return, expected_error, expected_match):
        class BadStream:
            def __cuda_stream__(self):
                return bad_return

        with pytest.raises(expected_error, match=expected_match):
            cuda.Stream.from_external(BadStream())

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
            assert cuda.get_current_stream() == cupy_stream

        # Test that synchronize works
        cupy_stream.synchronize()

        # Test that we can get the stream's properties
        # Just check it doesn't error
        assert cupy_stream.done or not cupy_stream.done
