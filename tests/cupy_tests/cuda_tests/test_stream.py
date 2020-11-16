import unittest

from cupy._creation import from_data
from cupy import cuda
from cupy import testing
from cupy.testing import attr


class TestStream(unittest.TestCase):

    @attr.gpu
    def test_eq(self):
        null0 = cuda.Stream.null
        null1 = cuda.Stream(True)
        null2 = cuda.Stream(True)
        null3 = cuda.Stream()

        self.assertEqual(null0, null1)
        self.assertEqual(null1, null2)
        self.assertNotEqual(null2, null3)

    def check_del(self, null):
        stream = cuda.Stream(null=null).use()
        stream_ptr = stream.ptr
        x = from_data.array([1, 2, 3])
        del stream
        self.assertEqual(cuda.Stream.null, cuda.get_current_stream())
        # Want to test cudaStreamDestory is issued, but
        # runtime.streamQuery(stream_ptr) causes SEGV. We cannot test...
        del stream_ptr
        del x

    @attr.gpu
    def test_del(self):
        self.check_del(null=False)

    @attr.gpu
    def test_del_null(self):
        self.check_del(null=True)

    @attr.gpu
    def test_get_and_add_callback(self):
        N = 100
        cupy_arrays = [testing.shaped_random((2, 3)) for _ in range(N)]

        if not cuda.runtime.is_hip:
            stream = cuda.Stream.null
        else:
            # adding callbacks to the null stream in HIP would segfault...
            stream = cuda.Stream()

        out = []
        for i in range(N):
            numpy_array = cupy_arrays[i].get(stream=stream)
            stream.add_callback(
                lambda _, __, t: out.append(t[0]),
                (i, numpy_array))

        stream.synchronize()
        self.assertEqual(out, list(range(N)))

    @attr.gpu
    def test_with_statement(self):
        stream1 = cuda.Stream()
        stream2 = cuda.Stream()
        self.assertEqual(cuda.Stream.null, cuda.get_current_stream())
        with stream1:
            self.assertEqual(stream1, cuda.get_current_stream())
            with stream2:
                self.assertEqual(stream2, cuda.get_current_stream())
            self.assertEqual(stream1, cuda.get_current_stream())
        self.assertEqual(cuda.Stream.null, cuda.get_current_stream())

    @attr.gpu
    def test_use(self):
        stream1 = cuda.Stream().use()
        self.assertEqual(stream1, cuda.get_current_stream())
        cuda.Stream.null.use()
        self.assertEqual(cuda.Stream.null, cuda.get_current_stream())


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

        if not cuda.runtime.is_hip:
            stream = cuda.Stream.null
        else:
            # adding callbacks to the null stream in HIP would segfault...
            stream = cuda.Stream()

        out = []
        for i in range(N):
            numpy_array = cupy_arrays[i].get(stream=stream)
            stream.add_callback(
                lambda _, __, t: out.append(t[0]),
                (i, numpy_array))

        stream.synchronize()
        self.assertEqual(out, list(range(N)))
