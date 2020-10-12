import unittest
from unittest import mock

import numpy

import cupy
from cupy import testing
import cupyx


class TestRepeat(unittest.TestCase):

    def test_cpu_routine(self):
        with mock.patch('time.perf_counter',
                        mock.Mock(side_effect=[2.4, 3.8, 3.8] * 10)):
            with mock.patch('cupy.cuda.get_elapsed_time',
                            mock.Mock(return_value=2500)):
                mock_func = mock.Mock()
                mock_func.__name__ = 'test_name_xxx'
                x = cupy.testing.shaped_random((2, 3), cupy, 'int32')
                y = cupy.testing.shaped_random((2, 3), cupy, 'int32')
                assert mock_func.call_count == 0

                perf = cupyx.time.repeat(
                    mock_func, (x, y), n_repeat=10, n_warmup=3)

                assert perf.name == 'test_name_xxx'
                assert mock_func.call_count == 13
                assert perf.cpu_times.shape == (10,)
                assert perf.gpu_times.shape == (1, 10,)
                assert (perf.cpu_times == 1.4).all()
                assert (perf.gpu_times == 2.5).all()

    @testing.multi_gpu(2)
    def test_multigpu_routine(self):
        with mock.patch('time.perf_counter',
                        mock.Mock(side_effect=[2.4, 3.8, 3.8] * 10)):
            with mock.patch('cupy.cuda.get_elapsed_time',
                            mock.Mock(return_value=2500)):
                mock_func = mock.Mock()
                mock_func.__name__ = 'test_name_xxx'
                x = cupy.testing.shaped_random((2, 3), cupy, 'int32')
                y = cupy.testing.shaped_random((2, 3), cupy, 'int32')
                assert mock_func.call_count == 0

                perf = cupyx.time.repeat(
                    mock_func, (x, y), n_repeat=10, n_warmup=3, devices=(0, 1))

                assert perf.name == 'test_name_xxx'
                assert mock_func.call_count == 13
                assert perf.cpu_times.shape == (10,)
                assert perf.gpu_times.shape == (2, 10,)
                assert (perf.cpu_times == 1.4).all()
                assert (perf.gpu_times == 2.5).all()

    def test_repeat_max_duration(self):
        with mock.patch('time.perf_counter',
                        mock.Mock(side_effect=[1., 2., 2.] * 6)):
            with mock.patch('cupy.cuda.get_elapsed_time',
                            mock.Mock(return_value=2500)):
                mock_func = mock.Mock()
                mock_func.__name__ = 'test_name_xxx'
                x = cupy.testing.shaped_random((2, 3), cupy, 'int32')
                y = cupy.testing.shaped_random((2, 3), cupy, 'int32')
                assert mock_func.call_count == 0

                perf = cupyx.time.repeat(
                    mock_func, (x, y), n_warmup=3, max_duration=2.5)

                assert perf.name == 'test_name_xxx'
                assert mock_func.call_count == 6
                assert perf.cpu_times.shape == (3,)
                assert perf.gpu_times.shape == (1, 3)
                assert (perf.cpu_times == 1.).all()
                assert (perf.gpu_times == 2.5).all()

    def test_repeat_kwargs(self):
        x = cupy.random.rand(5)
        cupyx.time.repeat(
            cupy.nonzero, kwargs={'a': x}, n_repeat=1, n_warmup=1)


class TestPerfCaseResult(unittest.TestCase):
    def test_show_gpu(self):
        times = numpy.array([
            [5.4, 7.1, 6.0, 5.4, 4.2],
            [6.4, 4.3, 8.9, 9.6, 3.8],
        ]) * 1e-6
        perf = cupyx.time._PerfCaseResult('test_name_xxx', times, (0,))
        expected = (
            'test_name_xxx       :'
            '    CPU:    5.620 us   +/- 0.943 '
            '(min:    4.200 / max:    7.100) us '
            '    GPU-0:    6.600 us   +/- 2.344 '
            '(min:    3.800 / max:    9.600) us'
        )
        assert str(perf) == expected

    def test_no_show_gpu(self):
        times = numpy.array([
            [5.4, 7.1, 6.0, 5.4, 4.2],
            [6.4, 4.3, 8.9, 9.6, 3.8],
        ]) * 1e-6
        perf = cupyx.time._PerfCaseResult('test_name_xxx', times, (0,))
        expected = (
            'test_name_xxx       :'
            '    CPU:    5.620 us   +/- 0.943 '
            '(min:    4.200 / max:    7.100) us'
        )
        assert perf.to_str() == expected
        # Checks if the result does not change.
        assert perf.to_str() == expected

    def test_single_show_gpu(self):
        times = numpy.array([[5.4], [6.4]]) * 1e-6
        perf = cupyx.time._PerfCaseResult('test_name_xxx', times, (0,))
        assert str(perf) == ('test_name_xxx       :    CPU:    5.400 us '
                             '    GPU-0:    6.400 us')

    def test_single_no_show_gpu(self):
        times = numpy.array([[5.4], [6.4]]) * 1e-6
        perf = cupyx.time._PerfCaseResult('test_name_xxx', times, (0,))
        assert perf.to_str() == 'test_name_xxx       :    CPU:    5.400 us'

    def test_show_multigpu(self):
        times = numpy.array([[5.4], [6.4], [7.0], [8.1]]) * 1e-6
        perf = cupyx.time._PerfCaseResult('test_name_xxx', times, (0, 1, 2))
        assert str(perf) == ('test_name_xxx       :    CPU:    5.400 us '
                             '    GPU-0:    6.400 us '
                             '    GPU-1:    7.000 us '
                             '    GPU-2:    8.100 us')
