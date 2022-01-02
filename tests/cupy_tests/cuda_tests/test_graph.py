import pytest

import cupy
from cupy import cuda
from cupy import testing
import cupyx


@pytest.mark.skipif(cuda.runtime.is_hip,
                    reason='HIP does not support this')
@pytest.mark.skipif(cuda.driver.get_build_version() < 10010,
                    reason='Only CUDA 10.1+ supports this')
class TestGraph:

    def _helper1(self, a):
        # this tests ufuncs involving simple arithmetic
        a = a + 3
        a = a * 7.6
        return a**2

    def _helper2(self, a):
        # this tests ufuncs involving math API calls
        a = 3 * cupy.sin(a)
        return cupy.sqrt(a)

    def _helper3(self, a):
        # this tests CUDA library calls
        a = a * cupy.fft.fft(a)
        return cupy.fft.ifft(a)

    def _helper4(self, a):
        # this tests a common pattern in CuPy internal in which the host
        # operation depends on intermediate outcome on GPU (and thus requires
        # synchronization)
        result = cupy.zeros((1,), dtype=cupy.int32)
        if a.sum() > 0:  # synchronize!
            result += 1
        if a[-1] >= 0:  # synchronize!
            result += 2
        return result

    @pytest.mark.parametrize('upload', (True, False))
    def test_capture_run_on_same_stream(self, upload):
        s = cupy.cuda.Stream(non_blocking=True)

        for n in range(3):
            func = getattr(self, '_helper{}'.format(n+1))
            a = cupy.random.random((100,))

            with s:
                s.begin_capture()
                out1 = func(a)
                g = s.end_capture()
                if upload and cuda.runtime.runtimeGetVersion() >= 11010:
                    g.upload()
                g.launch()
            s.synchronize()

            out2 = func(a)
            testing.assert_array_equal(out1, out2)

    @pytest.mark.parametrize('upload', (True, False))
    def test_capture_run_on_different_streams(self, upload):
        s1 = cupy.cuda.Stream(non_blocking=True)
        s2 = cupy.cuda.Stream(non_blocking=True)

        for n in range(3):
            func = getattr(self, '_helper{}'.format(n+1))
            a = cupy.random.random((100,))

            with s1:
                s1.begin_capture()
                out1 = func(a)
                g = s1.end_capture()
            with s2:
                if upload and cuda.runtime.runtimeGetVersion() >= 11010:
                    g.upload()
                g.launch()
            s2.synchronize()

            out2 = func(a)
            testing.assert_array_equal(out1, out2)

    @pytest.mark.parametrize('upload', (True, False))
    def test_stream_is_capturing(self, upload):
        s = cupy.cuda.Stream(non_blocking=True)
        a = cupy.random.random((100,))

        with s:
            s.begin_capture()
            assert s.is_capturing()
            assert not cuda.Stream.null.is_capturing()
            b = a * 3
            g = s.end_capture()
        assert not s.is_capturing()
        assert not cuda.Stream.null.is_capturing()

        # check the graph integrity
        if upload and cuda.runtime.runtimeGetVersion() >= 11010:
            g.upload()
        g.launch()
        s.synchronize()
        testing.assert_array_equal(b, 3 * a)

    @pytest.mark.parametrize('upload', (True, False))
    def test_stream_fork_join(self, upload):
        s1 = cupy.cuda.Stream(non_blocking=True)
        s2 = cupy.cuda.Stream(non_blocking=True)
        e1 = cupy.cuda.Event()
        e2 = cupy.cuda.Event()
        a = cupy.random.random((100,))

        def func(x):
            return 3 * x + 1

        with s1:
            s1.begin_capture()
            out1 = a * 100
            e1.record(s1)
            s2.wait_event(e1)
            with s2:
                out2 = func(out1)
                e2.record(s2)
            s1.wait_event(e2)
            g = s1.end_capture()

        # check integrity
        assert not s1.is_capturing()
        assert not s2.is_capturing()
        if upload and cuda.runtime.runtimeGetVersion() >= 11010:
            g.upload()
        g.launch()
        s1.synchronize()
        testing.assert_array_equal(out2, func(a * 100))

    @pytest.mark.parametrize('upload', (True, False))
    def test_null_stream_cannot_capture(self, upload):
        s = cupy.cuda.Stream(non_blocking=False)
        a = cupy.random.random((100,))

        with s:
            s.begin_capture()
            b = a + 4
            assert s.is_capturing()
            # cudaStreamLegacy is unhappy when a blocking stream is capturing
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                cuda.Stream.null.is_capturing()
            assert 'cudaErrorStreamCaptureImplicit' in str(e.value)
            g = s.end_capture()
        assert not s.is_capturing()
        assert not cuda.Stream.null.is_capturing()

        # check the graph integrity
        if upload and cuda.runtime.runtimeGetVersion() >= 11010:
            g.upload()
        g.launch()
        s.synchronize()
        testing.assert_array_equal(b, a + 4)

    def test_stream_capture_failure1(self):
        s = cupy.cuda.Stream(non_blocking=True)

        with s:
            s.begin_capture()
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                s.synchronize()
            assert 'cudaErrorStreamCaptureUnsupported' in str(e.value)
            # invalid operation causes the capture sequence to be invalidated
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                g = s.end_capture()  # noqa
            assert 'cudaErrorStreamCaptureInvalidated' in str(e.value)

        # check s left the capture mode and permits normal usage
        assert not s.is_capturing()
        s.synchronize()

    def test_stream_capture_failure2(self):
        s1 = cupy.cuda.Stream(non_blocking=True)
        s2 = cupy.cuda.Stream(non_blocking=True)
        e2 = cupy.cuda.Event()
        a = cupy.random.random((100,))

        with s1:
            s1.begin_capture()
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                g = s2.end_capture()
            assert 'cudaErrorIllegalState' in str(e.value)
            e2.record(s1)
            s2.wait_event(e2)
            with s2:
                b = a**3  # noqa
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                g = s2.end_capture()
            assert 'cudaErrorStreamCaptureUnmatched' in str(e.value)
            # invalid operation causes the capture sequence to be invalidated
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                g = s1.end_capture()  # noqa
            assert 'cudaErrorStreamCaptureInvalidated' in str(e.value)

        # check both s1 and s2 left the capture mode and permit normal usage
        assert not s1.is_capturing()
        assert not s2.is_capturing()
        s1.synchronize()
        s2.synchronize()

    def test_stream_capture_failure3(self):
        s1 = cupy.cuda.Stream(non_blocking=True)
        s2 = cupy.cuda.Stream(non_blocking=True)
        e2 = cupy.cuda.Event()
        a = cupy.random.random((100,))

        with s1:
            s1.begin_capture()
            e2.record(s1)
            s2.wait_event(e2)
            with s2:
                # internally the function requires synchronization, which is
                # incompatible with stream capturing and so we raise
                with pytest.raises(RuntimeError) as e:
                    b = cupy.where(a > 0.5)  # noqa
                assert 'is capturing' in str(e.value)
            # invalid operation causes the capture sequence to be invalidated
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                g = s1.end_capture()  # noqa
            assert 'cudaErrorStreamCaptureUnjoined' in str(e.value)

        # check both s1 and s2 left the capture mode and permit normal usage
        assert not s1.is_capturing()
        assert not s2.is_capturing()
        s1.synchronize()
        s2.synchronize()

    def test_stream_capture_failure4(self):
        s = cupy.cuda.Stream(non_blocking=True)

        with s:
            s.begin_capture()
            # query the stream status is illegal during capturing
            s.done
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                s.end_capture()
            assert 'cudaErrorStreamCaptureInvalidated' in str(e.value)

        # check s left the capture mode and permits normal usage
        assert not s.is_capturing()
        s.synchronize()

    def test_stream_capture_failure5(self):
        s = cupy.cuda.Stream(non_blocking=True)
        func = self._helper4
        a = cupy.random.random((100,))

        with s:
            s.begin_capture()
            # internally the function requires synchronization, which is
            # incompatible with stream capturing and so we raise
            with pytest.raises(RuntimeError) as e:
                func(a)
            assert 'is capturing' in str(e.value)
            s.end_capture()

        # check s left the capture mode and permits normal usage
        assert not s.is_capturing()
        s.synchronize()

    def test_stream_capture_failure6(self):
        s = cupy.cuda.Stream(non_blocking=True)

        with s:
            s.begin_capture()
            # synchronize the stream is illegal during capturing
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                s.synchronize()
            assert 'cudaErrorStreamCaptureUnsupported' in str(e.value)
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                s.end_capture()
            assert 'cudaErrorStreamCaptureInvalidated' in str(e.value)

        # check s left the capture mode and permits normal usage
        assert not s.is_capturing()
        s.synchronize()

    def test_stream_capture_failure_cublas(self):
        s = cupy.cuda.Stream(non_blocking=True)
        a = cupy.random.random((3, 4))
        b = cupy.random.random((4, 5))

        with s:
            s.begin_capture()
            with pytest.raises(NotImplementedError) as e:
                cupy.matmul(a, b)
            assert 'cuBLAS' in str(e.value)
            s.end_capture()

        # check s left the capture mode and permits normal usage
        assert not s.is_capturing()
        s.synchronize()

    def test_stream_capture_failure_cusolver(self):
        s = cupy.cuda.Stream(non_blocking=True)
        a = cupy.random.random((8, 8))
        a += a.T

        with s:
            s.begin_capture()
            with pytest.raises(NotImplementedError) as e:
                cupy.linalg.svd(a)
            assert 'cuSOLVER' in str(e.value)
            s.end_capture()

        # check s left the capture mode and permits normal usage
        assert not s.is_capturing()
        s.synchronize()

    def test_stream_capture_failure_curand(self):
        s = cupy.cuda.Stream(non_blocking=True)

        with s:
            s.begin_capture()
            with pytest.raises(NotImplementedError) as e:
                cupy.random.random(100)
            assert 'cuRAND' in str(e.value)
            s.end_capture()

        # check s left the capture mode and permits normal usage
        assert not s.is_capturing()
        s.synchronize()

    def test_stream_capture_failure_cusparse(self):
        s = cupy.cuda.Stream(non_blocking=True)
        a = cupy.zeros((3, 4))
        a[0] = 1
        a = cupyx.scipy.sparse.csr_matrix(a)
        a.has_canonical_format  # avoid launching custom kernels during capture

        with s:
            s.begin_capture()
            with pytest.raises(NotImplementedError) as e:
                a * a.T
            assert 'cuSPARSE' in str(e.value)
            s.end_capture()

        # check s left the capture mode and permits normal usage
        assert not s.is_capturing()
        s.synchronize()
