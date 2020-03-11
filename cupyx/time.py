import time

import numpy

import cupy
from cupy import util


class _PerfCaseResult(object):
    def __init__(self, name, ts):
        assert ts.ndim == 2 and ts.shape[0] == 2 and ts.shape[1] > 0
        self.name = name
        self._ts = ts

    @property
    def cpu_times(self):
        return self._ts[0]

    @property
    def gpu_times(self):
        return self._ts[1]

    @staticmethod
    def _to_str_per_item(device_name, t):
        assert t.ndim == 1
        assert t.size > 0
        t *= 1e6

        s = '    {}:{:9.03f} us'.format(device_name, t.mean())
        if t.size > 1:
            s += '   +/-{:6.03f} (min:{:9.03f} / max:{:9.03f}) us'.format(
                t.std(), t.min(), t.max())
        return s

    def to_str(self, show_gpu=False):
        results = [self._to_str_per_item('CPU', self._ts[0])]
        if show_gpu:
            results.append(self._to_str_per_item('GPU', self._ts[1]))
        return '{:<20s}:{}'.format(self.name, ' '.join(results))

    def __str__(self):
        return self.to_str(show_gpu=True)


def repeat(func, args=(), kwargs={}, n=10000, *, name=None, n_warmup=10):
    util.experimental('cupyx.time.repeat')
    if name is None:
        name = func.__name__

    if not callable(func):
        raise ValueError('`func` should be a callable object.')
    if not isinstance(args, tuple):
        raise ValueError('`args` should be of tuple type.')
    if not isinstance(kwargs, dict):
        raise ValueError('`kwargs` should be of dict type.')
    if not isinstance(n, int):
        raise ValueError('`n` should be an integer.')
    if not isinstance(name, str):
        raise ValueError('`str` should be a string.')
    if not isinstance(n_warmup, int):
        raise ValueError('`n_warmup` should be an integer.')

    ts = numpy.empty((2, n,), dtype=numpy.float64)
    ev1 = cupy.cuda.stream.Event()
    ev2 = cupy.cuda.stream.Event()

    for i in range(n_warmup):
        func(*args, **kwargs)

    ev1.record()
    ev1.synchronize()

    for i in range(n):
        ev1.record()
        t1 = time.perf_counter()

        func(*args, **kwargs)

        t2 = time.perf_counter()
        ev2.record()
        ev2.synchronize()
        cpu_time = t2 - t1
        gpu_time = cupy.cuda.get_elapsed_time(ev1, ev2) * 1e-3
        ts[0, i] = cpu_time
        ts[1, i] = gpu_time

    return _PerfCaseResult(name, ts)
