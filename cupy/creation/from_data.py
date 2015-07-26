import numpy

import cupy
from cupy import cuda
from cupy import elementwise
from cupy import internal


def array(obj, dtype=None, copy=True, order=None, subok=False, ndmin=0,
          allocator=None):
    if isinstance(obj, cupy.ndarray):
        # TODO(beam2d): Support these options
        assert order is None
        assert not subok

        if dtype is not None:
            dtype = numpy.dtype(dtype)

        if dtype is None or dtype == obj.dtype:
            if copy:
                a = obj.copy(allocator=allocator)
            else:
                a = obj
        else:
            a = cupy.empty_like(obj, dtype=dtype, allocator=allocator)
            elementwise.copy(obj, a)

        ndim = a.ndim
        if ndmin > ndim:
            a.shape = (1,) * (ndmin - ndim) + a.shape
        return a
    else:
        assert copy
        if allocator is None:
            allocator = cuda.alloc
        a_cpu = numpy.array(obj, dtype=dtype, copy=False, order=order,
                            subok=subok, ndmin=ndmin)
        if a_cpu.ndim > 0:
            a_cpu = numpy.ascontiguousarray(a_cpu)
        a = cupy.ndarray(a_cpu.shape, dtype=a_cpu.dtype, allocator=allocator)
        a.data.copy_from_host(internal.get_ndarray_ptr(a_cpu), a.nbytes)
        if a_cpu.dtype == a.dtype:
            return a
        else:
            return a.view(dtype=a_cpu.dtype)


def asarray(a, dtype=None, allocator=None):
    return cupy.array(a, dtype=dtype, copy=False, allocator=allocator)


asanyarray = asarray


def ascontiguousarray(a):
    # TODO(beam2d): Support dtype option
    if a.flags.c_contiguous:
        return a
    else:
        newarray = cupy.empty_like(a)
        elementwise.copy(a, newarray)
        return newarray


def asmatrix(data, dtype=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def copy(a, allocator=None):
    # TODO(beam2d): Support ordering option
    if allocator is None:
        allocator = a.allocator

    newarray = cupy.empty_like(a, allocator=allocator)
    f = a.flags
    if f.c_contiguous:
        newarray.data.copy_from(a.data, a.nbytes)
    else:
        elementwise.copy(a, newarray)
    return newarray


def frombuffer(buffer, dtype=float, count=-1, offset=0, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def fromfile(file, dtype=float, count=-1, sep='', allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def fromfunction(function, shape, allocator=None, **kwargs):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def fromiter(iterable, dtype, count=-1, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def fromstring(string, dtype=float, count=-1, sep='', allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def loadtxt(fname, dtype=numpy.float64, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0,
            allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError
