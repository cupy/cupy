import numpy

import cupy
from cupy import cuda


class NpzFile(object):

    def __init__(self, npz_file, allocator):
        self.npz_file = npz_file
        self.allocator = allocator

    def __enter__(self):
        self.npz_file.__enter__()
        return self

    def __exit__(self, typ, val, traceback):
        self.npz_file.__exit__(typ, val, traceback)

    def __getitem__(self, key):
        arr = self.npz_file[key]
        return cupy.array(arr, allocator=self.allocator)

    def close(self):
        self.npz_file.close()


def load(file, mmap_mode=None, allocator=cuda.alloc):
    obj = numpy.load(file, mmap_mode)
    if isinstance(obj, numpy.ndarray):
        return cupy.array(obj, allocator=allocator)
    else:
        return NpzFile(obj, allocator)


def save(file, arr):
    numpy.save(file, cupy.asnumpy(arr))


def savez(file, *args, **kwds):
    args = map(cupy.asnumpy, args)
    for key in kwds:
        kwds[key] = cupy.asnumpy(kwds[key])
    numpy.savez(file, *args, **kwds)


def savez_compressed(file, *args, **kwds):
    args = map(cupy.asnumpy, args)
    for key in kwds:
        kwds[key] = cupy.asnumpy(kwds[key])
    numpy.savez_compressed(file, *args, **kwds)
