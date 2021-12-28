import unittest

import pytest

import cupy
from cupy import cuda
from cupy import testing


@testing.gpu
@pytest.mark.skipif(cuda.runtime.is_hip,
                    reason='HIP does not support this')
@pytest.mark.skipif(cuda.driver.get_build_version() < 10010,
                    reason='Only CUDA 10.1+ supports this')
class TestGraph(unittest.TestCase):

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
        # synchronization); here, we are protected by some (async) CUDA API
        # calls as well as launching of custom kernels
        result = cupy.zeros((1,), dtype=cupy.int32)
        if a.sum() > 0:  # synchronize!
            result += 1
        if a[-1] >= 0:  # synchronize!
            result += 2
        return result

    def test_capture_run_on_same_stream(self):
        s = cupy.cuda.Stream(non_blocking=True)

        for n in range(4):
            func = getattr(self, '_helper{}'.format(n+1))
            a = cupy.random.random((100,))

            with s:
                s.begin_capture()
                out1 = func(a)
                g = s.end_capture()
                g.launch()
            s.synchronize()

            out2 = func(a)
            testing.assert_array_equal(out1, out2)

    def test_capture_run_on_different_streams(self):
        s1 = cupy.cuda.Stream(non_blocking=True)
        s2 = cupy.cuda.Stream(non_blocking=True)

        for n in range(4):
            func = getattr(self, '_helper{}'.format(n+1))
            a = cupy.random.random((100,))

            with s1:
                s1.begin_capture()
                out1 = func(a)
                g = s1.end_capture()
            with s2:
                g.launch()
            s2.synchronize()

            out2 = func(a)
            testing.assert_array_equal(out1, out2)

    def test_stream_is_capturing(self):
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
        g.launch()
        s.synchronize()
        testing.assert_array_equal(b, 3 * a)

    def test_stream_fork_join(self):
        # TODO(leofang): this is problematic when using nonzero()
        s1 = cupy.cuda.Stream(non_blocking=True)
        s2 = cupy.cuda.Stream(non_blocking=True)
        e1 = cupy.cuda.Event()
        e2 = cupy.cuda.Event()
        a = cupy.random.random((100,))

        def func(x):
            #return cupy.nonzero(x)
            return x+1

        with s1:
            s1.begin_capture()
            e1.record(s1)
            s2.wait_event(e1)
            with s2:
                #out1 = cupy.where(a > 0.5)
                #out1 = a * 100 + 2
                #out1 = cupy.nonzero(a)
                out1 = func(a)
            e2.record(s2)
            s1.wait_event(e2)
            g = s1.end_capture()

        # check integrity
        assert not s1.is_capturing()
        assert not s2.is_capturing()
        g.launch()
        s1.synchronize()
        #out2 = cupy.where(a > 0.5)
        #out2 = a * 100 + 2
        #out2 = cupy.nonzero(a)
        out2 = func(a)
        #testing.assert_array_list_equal(out1, out2)
        testing.assert_array_equal(out1, out2)

    def test_null_stream_cannot_capture(self):
        s = cupy.cuda.Stream(non_blocking=False)
        a = cupy.random.random((100,))

        with s:
            s.begin_capture()
            b = a + 4
            assert s.is_capturing()
            # cudaStreamLegacy is unhappy when a blocking stream is capturing
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                assert cuda.Stream.null.is_capturing()
            assert 'cudaErrorStreamCaptureImplicit' in str(e.value)
            g = s.end_capture()
        assert not s.is_capturing()
        assert not cuda.Stream.null.is_capturing()

        # check the graph integrity
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
                b = cupy.where(a > 0.5)  # noqa
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
            status = s.done
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                g = s.end_capture()
            assert 'cudaErrorStreamCaptureInvalidated' in str(e.value)
