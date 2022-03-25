import concurrent.futures
import itertools
import threading

import numpy

import cupy


_thread_local = threading.local()
_lock = threading.Lock()


class DataRaceError(RuntimeError):
    pass


class ThreadDivergenceError(RuntimeError):
    pass


class _Global:

    def __init__(self, array, read, write):
        self._array = array
        self._read = read
        self._write = write

    def __repr__(self, array):
        return repr(array)

    def __getitem__(self, item):
        with _lock:
            out = self._array[item]
            if out.size == 1:
                self._read[item] = True
                if self._write[item]:
                    raise DataRaceError('read/write data race.')
                return out
            return _Global(out, self._read[item], self._write[item])

    def __setitem__(self, item, value):
        with _lock:
            if self._array[item].size != 1:
                raise TypeError('only size-1 arrays can be set.')
            if self._read[item]:
                raise DataRaceError('read/write data race.')
            if self._write[item]:
                raise DataRaceError('write/write data race.')
            self._array[item] = True
            self._array[item] = value


def _run_func(func, args, grid_dim, block_idx, block_dim, thread_idx):
    _thread_local.gridDim = grid_dim
    _thread_local.blockIdx = block_idx
    _thread_local.blockDim = block_dim
    _thread_local.threadIdx = thread_idx
    func(*args)
    del _thread_local.gridDim
    del _thread_local.blockIdx
    del _thread_local.blockDim
    del _thread_local.threadIdx


def simulate(func, args, grid_dim, block_dim):
    new_args = []
    for x in args:
        if isinstance(x, (numpy.ndarray, cupy.ndarray)):
            read = numpy.zeros(x.shape, dtype=numpy.bool_)
            write = numpy.zeros(x.shape, dtype=numpy.bool_)
            x = _Global(x, read, write)
        new_args.append(x)
    n_workers = numpy.prod(grid_dim) * numpy.prod(block_dim)
    with concurrent.futures.ThreadPoolExecutor(n_workers) as executor:
        futures = []
        for block_idx in itertools.product(*map(range, grid_dim)):
            for thread_idx in itertools.product(*map(range, block_dim)):
                params = (
                    _run_func, func, new_args,
                    grid_dim, block_idx, block_dim, thread_idx,
                )
                futures.append(executor.submit(*params))
        for future in concurrent.futures.as_completed(futures):
            future.result()
