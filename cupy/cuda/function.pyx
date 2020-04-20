# distutils: language = c++

import numpy

cimport cpython  # NOQA
from libc.stdint cimport int8_t
from libc.stdint cimport int16_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdint cimport intptr_t
from libc.stdint cimport uintmax_t
from libcpp cimport vector

from cupy.core cimport _carray
from cupy.core cimport _memory_range
from cupy.core cimport _scalar
from cupy.core cimport core
from cupy.cuda cimport driver
from cupy.cuda cimport runtime
from cupy.cuda cimport stream as stream_module
from cupy.cuda.texture cimport TextureObject, SurfaceObject


cdef class CPointer:
    def __init__(self, p=0):
        self.ptr = <void*>p


cdef class CInt8(CPointer):
    cdef:
        int8_t val

    def __init__(self, int8_t v):
        self.val = v
        self.ptr = <void*>&self.val


cdef class CInt16(CPointer):
    cdef:
        int16_t val

    def __init__(self, int16_t v):
        self.val = v
        self.ptr = <void*>&self.val


cdef class CInt32(CPointer):
    cdef:
        int32_t val

    def __init__(self, int32_t v):
        self.val = v
        self.ptr = <void*>&self.val


cdef class CInt64(CPointer):
    cdef:
        int64_t val

    def __init__(self, int64_t v):
        self.val = v
        self.ptr = <void*>&self.val


cdef class CInt128(CPointer):
    cdef:
        double complex val

    def __init__(self, double complex v):
        self.val = v
        self.ptr = <void*>&self.val


cdef class CUIntMax(CPointer):
    cdef:
        uintmax_t val

    def __init__(self, uintmax_t v):
        self.val = v
        self.ptr = <void*>&self.val


cdef set _pointer_numpy_types = {numpy.dtype(i).type
                                 for i in '?bhilqBHILQefdFD'}


# A shortcut of Arg.from_obj(x).get_pointer() for the performance.
cdef inline CPointer _pointer(x):
    cdef Py_ssize_t itemsize
    if x is None:
        return CPointer()
    if isinstance(x, core.ndarray):
        return (<core.ndarray>x).get_pointer()
    if isinstance(x, TextureObject):
        return CUIntMax(x.ptr)

    if isinstance(x, SurfaceObject):
        return CUIntMax(x.ptr)

    if type(x) not in _pointer_numpy_types:
        if isinstance(x, int):
            x = numpy.int64(x)
        elif isinstance(x, float):
            x = numpy.float64(x)
        elif isinstance(x, bool):
            x = numpy.bool_(x)
        else:
            raise TypeError('Unsupported type %s' % type(x))

    itemsize = x.itemsize
    if itemsize == 1:
        return CInt8(x.view(numpy.int8))
    if itemsize == 2:
        return CInt16(x.view(numpy.int16))
    if itemsize == 4:
        return CInt32(x.view(numpy.int32))
    if itemsize == 8:
        return CInt64(x.view(numpy.int64))
    if itemsize == 16:
        return CInt128(x.view(numpy.complex128))
    raise TypeError('Unsupported type %s. (size=%d)', type(x), itemsize)


cdef inline size_t _get_stream(stream) except *:
    if stream is None:
        return stream_module.get_current_stream_ptr()
    else:
        return stream.ptr


cdef _launch(intptr_t func, Py_ssize_t grid0, int grid1, int grid2,
             Py_ssize_t block0, int block1, int block2,
             args, Py_ssize_t shared_mem, size_t stream,
             bint enable_cooperative_groups=False):
    cdef list pargs = []
    cdef vector.vector[void*] kargs
    cdef CPointer cp
    kargs.reserve(len(args))
    for a in args:
        if isinstance(a, Arg):
            cp = (<Arg>a).get_pointer()
        else:
            cp = _pointer(a)
        pargs.append(cp)
        kargs.push_back(cp.ptr)

    runtime._ensure_context()

    if enable_cooperative_groups:
        driver.launchCooperativeKernel(
            func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
            <int>shared_mem, stream, <intptr_t>&(kargs[0]))
    else:
        driver.launchKernel(
            func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
            <int>shared_mem, stream, <intptr_t>&(kargs[0]), <intptr_t>0)


cdef class Function:

    """CUDA kernel function."""

    def __init__(self, Module module, str funcname):
        self.module = module  # to keep module loaded
        self.ptr = driver.moduleGetFunction(module.ptr, funcname)

    def __call__(self, tuple grid, tuple block, args, size_t shared_mem=0,
                 stream=None, enable_cooperative_groups=False):
        grid = (grid + (1, 1))[:3]
        block = (block + (1, 1))[:3]

        s = _get_stream(stream)
        _launch(
            self.ptr,
            max(1, grid[0]), max(1, grid[1]), max(1, grid[2]),
            max(1, block[0]), max(1, block[1]), max(1, block[2]),
            args,
            shared_mem, s, enable_cooperative_groups)

    cpdef linear_launch(self, size_t size, args, size_t shared_mem=0,
                        size_t block_max_size=128, stream=None):
        # TODO(beam2d): Tune it
        cdef size_t gridx = min(
            0x7fffffffUL, (size + block_max_size - 1) // block_max_size)
        cdef size_t blockx = min(block_max_size, size)
        s = _get_stream(stream)
        _launch(
            self.ptr,
            gridx, 1, 1, blockx, 1, 1,
            args,
            shared_mem, s)


cdef class Module:

    """CUDA kernel module."""

    def __init__(self):
        self.ptr = 0

    def __dealloc__(self):
        if self.ptr:
            driver.moduleUnload(self.ptr)
            self.ptr = 0

    cpdef load_file(self, filename):
        if isinstance(filename, bytes):
            filename = filename.decode()
        runtime._ensure_context()
        self.ptr = driver.moduleLoad(filename)

    cpdef load(self, bytes cubin):
        runtime._ensure_context()
        self.ptr = driver.moduleLoadData(cubin)

    cpdef get_global_var(self, name):
        if isinstance(name, bytes):
            name = name.decode()
        return driver.moduleGetGlobal(self.ptr, name)

    cpdef get_function(self, name):
        if isinstance(name, bytes):
            name = name.decode()
        return Function(self, name)

    cpdef get_texref(self, name):
        if isinstance(name, bytes):
            name = name.decode()
        return driver.moduleGetTexRef(self.ptr, name)


cdef class LinkState:

    """CUDA link state."""

    def __init__(self):
        runtime._ensure_context()
        self.ptr = driver.linkCreate()

    def __dealloc__(self):
        if self.ptr:
            driver.linkDestroy(self.ptr)
            self.ptr = 0

    cpdef add_ptr_data(self, unicode data, unicode name):
        cdef bytes data_byte = data.encode()
        driver.linkAddData(self.ptr, driver.CU_JIT_INPUT_PTX, data_byte, name)

    cpdef add_ptr_file(self, unicode path):
        driver.linkAddFile(self.ptr, driver.CU_JIT_INPUT_LIBRARY, path)

    cpdef bytes complete(self):
        cubin = driver.linkComplete(self.ptr)
        return cubin


cdef class Arg:

    """Base class of internal wrapper of an argument."""

    def __init__(
            self,
            object obj,
            _ArgKind arg_kind,
            object typ,
            object dtype,
            int ndim,
            bint c_contiguous):
        self.obj = obj
        self.arg_kind = arg_kind
        self.type = typ
        self.dtype = dtype
        self.ndim = ndim
        self.c_contiguous = c_contiguous

    cdef _init_fast_base(
            self,
            object obj,
            _ArgKind arg_kind,
            object typ,
            object dtype,
            int ndim,
            bint c_contiguous):
        self.obj = obj
        self.arg_kind = arg_kind
        self.type = typ
        self.dtype = dtype
        self.ndim = ndim
        self.c_contiguous = c_contiguous

    @staticmethod
    cdef Arg from_obj(object obj):
        cdef Arg arg
        # Determine the kind of the argument
        if isinstance(obj, core.ndarray):
            arg = NdarrayArg.__new__(NdarrayArg)
            (<NdarrayArg>arg)._init_fast(
                obj, obj.ndim, obj.flags.c_contiguous)
            return arg
        if hasattr(obj, '__cuda_array_interface__'):
            # __cuda_array_interface__
            obj = core._convert_object_with_cuda_array_interface(obj)
            arg = NdarrayArg.__new__(NdarrayArg)
            (<NdarrayArg>arg)._init_fast(
                obj, obj.ndim, obj.flags.c_contiguous)
            return arg
        if _scalar.is_scalar(obj):
            # scalar
            arg = ScalarArg.__new__(ScalarArg)
            (<ScalarArg>arg)._init_fast(obj)
            return arg
        if isinstance(obj, TextureObject):
            return PointerArg(obj.ptr)

        raise TypeError('Unsupported type %s' % type(obj))

    @staticmethod
    cdef NdarrayArg from_ndarray(core.ndarray arr):
        return NdarrayArg(arr, c_contiguous=arr.flags.c_contiguous)

    @staticmethod
    cdef IndexerArg from_indexer(shape):
        return IndexerArg(shape)

    def __repr__(self):
        data = []
        if self.type is not None:
            data.append('type={}'.format(self.type.__name__),)
        data.append('dtype={}'.format(str(self.dtype)))
        data.append('ndim={}'.format(self.ndim))
        data += self.get_repr_data()
        return '<{} {}>'.format(
            self.__class__.__name__,
            ' '.join(data))

    def get_repr_data(self):
        # Returns a list of strs for presented in __repr__.
        return []

    cdef tuple get_immutable_key(self):
        return (
            self.arg_kind,
            self.type,
            self.dtype,
            self.ndim,
            self.c_contiguous
        )

    cdef bint is_ndarray(self):
        return self.arg_kind == ARG_KIND_NDARRAY

    cdef bint is_scalar(self):
        return self.arg_kind == ARG_KIND_SCALAR

    cdef CPointer get_pointer(self):
        raise NotImplementedError()


cdef class IndexerArg(Arg):

    def __init__(self, shape):
        ndim = len(shape)
        super().__init__(None, ARG_KIND_INDEXER, None, None, ndim, True)
        self.shape = shape

    cdef _init_fast(self, tuple shape):
        ndim = len(shape)
        self._init_fast_base(None, ARG_KIND_INDEXER, None, None, ndim, True)
        self.shape = shape

    # override
    def get_repr_data(self):
        return ['shape={}'.format(self.shape)]

    # override
    cdef CPointer get_pointer(self):
        cdef tuple shape = self.shape
        cdef Py_ssize_t size = 1
        for s in shape:
            size *= s
        return _carray.CIndexer(size, shape)


cdef class NdarrayArg(Arg):

    def __init__(self, core.ndarray obj, *, int ndim=-1, bint c_contiguous):
        if ndim == -1:
            ndim = obj.ndim
        super().__init__(
            obj,
            ARG_KIND_NDARRAY,
            core.ndarray,
            obj.dtype,
            obj.ndim,
            obj.flags.c_contiguous)

    cdef _init_fast(self, core.ndarray obj, int ndim, bint c_contiguous):
        self._init_fast_base(
            obj,
            ARG_KIND_NDARRAY,
            core.ndarray,
            obj.dtype,
            obj.ndim,
            obj.flags.c_contiguous)

    # override
    def get_repr_data(self):
        arr = self.obj
        return ['shape={}'.format(arr.shape)]

    # override
    cdef CPointer get_pointer(self):
        return (<core.ndarray>self.obj).get_pointer()

    cdef copy_in_arg_if_needed(self, list out_args):
        # Copies in_args if their memory ranges are overlapping with out_args.
        # Items in in_args are updated in-place.
        cdef core.ndarray arr = self.obj
        cdef NdarrayArg out
        cdef bint any_overlap = any([
            arr is not out and _memory_range.may_share_bounds(arr, out.obj)
            for out in out_args])
        if any_overlap:
            self.obj = arr.copy()

    cdef NdarrayArg as_ndim(self, int ndim):
        # Returns an ndarray _ArgInfo with altered ndim.
        # If ndim is the same, self is returned untouched.
        if self.ndim == ndim:
            return self
        return NdarrayArg(self.obj, ndim=ndim, c_contiguous=self.c_contiguous)


cdef class ScalarArg(Arg):

    def __init__(self, object obj):
        if _scalar.is_numpy_scalar(obj):
            numpy_scalar = obj
            dtype = obj.dtype
        else:
            numpy_scalar = _scalar.python_scalar_to_numpy_scalar(obj)
            dtype = numpy_scalar.dtype
        super().__init__(obj, ARG_KIND_SCALAR, None, dtype, 0, True)
        self._numpy_scalar = numpy_scalar
        self._dtype_applied = False

    cdef _init_fast(self, object obj):
        if _scalar.is_numpy_scalar(obj):
            numpy_scalar = obj
            dtype = obj.dtype
        else:
            numpy_scalar = _scalar.python_scalar_to_numpy_scalar(obj)
            dtype = numpy_scalar.dtype
        self._init_fast_base(obj, ARG_KIND_SCALAR, None, dtype, 0, True)
        self._numpy_scalar = numpy_scalar
        self._dtype_applied = False

    cdef object get_min_scalar_type(self):
        return _scalar.get_min_scalar_type(self._numpy_scalar)

    cdef apply_dtype(self, object dtype):
        self.dtype = numpy.dtype(dtype)
        self._dtype_applied = True

    # override
    def get_repr_data(self):
        return [
            'obj={!r}'.format(self.obj),
            'numpy_scalar={!r}'.format(self._numpy_scalar),
        ]

    # override
    cdef CPointer get_pointer(self):
        numpy_scalar = self._numpy_scalar
        return _scalar.CScalar.from_numpy_scalar_with_dtype(
            numpy_scalar, self.dtype.type)


cdef class PointerArg(Arg):

    def __init__(self, intptr_t ptr):
        super().__init__(CUIntMax(ptr), ARG_KIND_POINTER, None, None, 0, True)

    cdef _init_fast(self, intptr_t ptr):
        self._init_fast_base(
            CUIntMax(ptr), ARG_KIND_POINTER, None, None, 0, True)

    # override
    def get_repr_data(self):
        return ['ptr={!r}'.format(self.obj)]

    # override
    cdef CPointer get_pointer(self):
        return self.obj
