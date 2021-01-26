import unittest

from cupy._creation import from_data
from cupy import cuda
from cupy import testing
from cupy.testing import attr


@testing.parameterize(
    *testing.product({
        'stream': [cuda.Stream.null, cuda.Stream.ptds],
    }))
class TestStream(unittest.TestCase):

    @attr.gpu
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

    def check_del(self, test_stream, null, ptds):
        stream = cuda.Stream(null=null, ptds=ptds).use()
        stream_ptr = stream.ptr
        x = from_data.array([1, 2, 3])
        del stream
        assert cuda.Stream.null == cuda.get_current_stream()
        # Want to test cudaStreamDestory is issued, but
        # runtime.streamQuery(stream_ptr) causes SEGV. We cannot test...
        del stream_ptr
        del x

    @attr.gpu
    def test_del_default(self):
        self.check_del(cuda.Stream.null, null=False, ptds=False)

    @attr.gpu
    def test_del(self):
        null = self.stream == cuda.Stream.null
        ptds = self.stream == cuda.Stream.ptds
        print(null, ptds)
        self.check_del(self.stream, null=null, ptds=ptds)

    @attr.gpu
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

    @attr.gpu
    @unittest.skipIf(cuda.runtime.is_hip, 'HIP does not support this')
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

    @attr.gpu
    def test_with_statement(self):
        stream1 = cuda.Stream()
        stream2 = cuda.Stream()
        assert self.stream == cuda.get_current_stream()
        with stream1:
            assert stream1 == cuda.get_current_stream()
            with stream2:
                assert stream2 == cuda.get_current_stream()
            assert stream1 == cuda.get_current_stream()
        assert self.stream == cuda.get_current_stream()

    @attr.gpu
    def test_use(self):
        stream1 = cuda.Stream().use()
        assert stream1 == cuda.get_current_stream()
        self.stream.use()
        assert self.stream == cuda.get_current_stream()


class TestExternalStream(unittest.TestCase):

    def setUp(self):
        self.stream_ptr = cuda.runtime.streamCreate()
        self.stream = cuda.ExternalStream(self.stream_ptr)

    def tearDown(self):
        cuda.runtime.streamDestroy(self.stream_ptr)

    @attr.gpu
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

    @attr.gpu
    @unittest.skipIf(cuda.runtime.is_hip, 'HIP does not support this')
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
