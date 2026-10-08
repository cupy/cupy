# distutils: language = c++
"""Ascend (CANN) implementation of the kernel Module/Function API.

Compiled as the module ``cupy.xpu.function`` against the shared declarations
in ``cupy/xpu/function.pxd`` (the per-backend pyx source is selected in
``install/cupy_builder/features/{cuda,ascend}.py``; the CUDA counterpart is
the upstream file ``cupy/cuda/function.pyx``).

Maps the upstream ``cupy.cuda.function`` interface onto the aclrt binary
API (``$ASCEND_HOME_PATH/include/acl/acl_rt.h``):

    CUDA (driver API)                     Ascend (aclrt)
    ------------------------------------  ------------------------------------
    cuModuleLoadData / moduleLoad   <->   aclrtCreateBinary + aclrtBinaryLoad
                                          (file: aclrtBinaryLoadFromFile)
    cuModuleUnload                  <->   aclrtBinaryUnLoad [+ DestroyBinary]
    cuModuleGetFunction             <->   aclrtBinaryGetFunction
    cuLaunchKernel(grid, block,     <->   aclrtKernelArgsInit/Append/Finalize
                   void* args[])          + aclrtLaunchKernelWithConfig
                                          (one-dimensional numBlocks, args
                                          appended to an args handle)
    cuModuleGetGlobal               <->   (no equivalent)
    cuLink* (PTX JIT link)          <->   (no equivalent: kernels are compiled
                                          out-of-tree by bisheng, see
                                          docs/ascend/CustomKernel.md)

Semantic differences that shape this implementation:

* Launch granularity is a single ``uint32_t numBlocks``; CUDA's grid x block
  hierarchy does not exist.  ``Function.__call__`` maps the *grid* product to
  ``numBlocks`` and accepts ``block`` for API compatibility only (see
  ``docs/ascend/CustomKernel.md`` section 4.3 for the mapping discussion).
* Kernel arguments are value-copied into an ``aclrtArgsHandle`` built with
  ``aclrtKernelArgsInit/Append/Finalize`` (same convention as the in-tree
  custom-kernel launcher in ``cupy/backends/ascend/acl_custom_kernels.h``),
  so each ``CPointer`` wrapper additionally carries the argument size.
* ``shared_mem`` maps to the ``ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE``
  launch attribute (dynamic UB / unified buffer).
"""

import numpy

from libc.stdint cimport int8_t
from libc.stdint cimport int16_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdint cimport intptr_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uintmax_t

from cupy._core cimport _carray
from cupy._core.core cimport _ndarray_base
from cupy.backends.backend.api cimport runtime
from cupy.xpu cimport stream as stream_module
from cupy.xpu.memory cimport MemoryPointer


# ---------------------------------------------------------------------------
# aclrt binary / kernel-launch API (CANN >= 8.5; verified on 9.0.1)
# ---------------------------------------------------------------------------
cdef extern from "acl/acl.h" nogil:
    ctypedef void* aclrtBinary
    ctypedef void* aclrtBinHandle
    ctypedef void* aclrtFuncHandle
    ctypedef void* aclrtArgsHandle
    ctypedef void* aclrtParamHandle
    ctypedef void* aclrtStream
    ctypedef int aclError

    aclrtBinary aclrtCreateBinary(const void* data, size_t dataLen)
    aclError aclrtDestroyBinary(aclrtBinary binary)
    aclError aclrtBinaryLoad(const aclrtBinary binary,
                             aclrtBinHandle* binHandle)
    aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle)
    aclError aclrtBinaryLoadFromFile(const char* binPath, void* options,
                                     aclrtBinHandle* binHandle)
    aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle,
                                    const char* kernelName,
                                    aclrtFuncHandle* funcHandle)

    aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle,
                                 aclrtArgsHandle* argsHandle)
    aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void* param,
                                   size_t paramSize,
                                   aclrtParamHandle* paramHandle)
    aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
    aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle,
                                         uint32_t numBlocks,
                                         aclrtStream stream,
                                         aclrtLaunchKernelCfg* cfg,
                                         aclrtArgsHandle argsHandle,
                                         void* reserve)

    # aclrtLaunchKernelAttrId (acl_rt.h)
    int ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE

    ctypedef struct aclrtLaunchKernelAttrValue:
        uint32_t dynUBufSize

    ctypedef struct aclrtLaunchKernelAttr:
        int id
        aclrtLaunchKernelAttrValue value

    ctypedef struct aclrtLaunchKernelCfg:
        aclrtLaunchKernelAttr* attrs
        size_t numAttrs


cdef inline int _check(aclError ret, str what) except -1:
    if ret != 0:
        raise RuntimeError(
            '{} failed with ACL error code {}'.format(what, ret))
    return 0


# ---------------------------------------------------------------------------
# Kernel-argument wrappers
# ---------------------------------------------------------------------------
cdef class CPointer:
    def __init__(self, p=0):
        self.ptr = <void*>p
        self.size = 0


cdef class CInt8(CPointer):
    cdef:
        int8_t val

    def __init__(self, int8_t v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = sizeof(self.val)


cdef class CInt16(CPointer):
    cdef:
        int16_t val

    def __init__(self, int16_t v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = sizeof(self.val)


cdef class CInt32(CPointer):
    cdef:
        int32_t val

    def __init__(self, int32_t v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = sizeof(self.val)


cdef class CInt64(CPointer):
    cdef:
        int64_t val

    def __init__(self, int64_t v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = sizeof(self.val)


cdef class CInt128(CPointer):
    cdef:
        double complex val

    def __init__(self, double complex v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = sizeof(self.val)


cdef class CUIntMax(CPointer):
    cdef:
        uintmax_t val

    def __init__(self, uintmax_t v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = sizeof(self.val)


cdef class CIntptr(CPointer):
    cdef:
        intptr_t val

    def __init__(self, intptr_t v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = sizeof(self.val)


cdef class CNumpyArray(CPointer):
    cdef:
        object val

    def __init__(self, v):
        self.val = v
        self.ptr = <void*><size_t>v.__array_interface__['data'][0]
        self.size = v.itemsize


cdef set _pointer_numpy_types = {numpy.dtype(i).type
                                 for i in '?bhilqBHILQefdFD'}


cdef inline CPointer _pointer(x):
    cdef Py_ssize_t itemsize
    cdef MemoryPointer mp

    if x is None:
        # a NULL device pointer passed by value (8 zero bytes)
        return CIntptr(0)
    if isinstance(x, _ndarray_base):
        mp = (<_ndarray_base>x).get_pointer()
        return CIntptr(<intptr_t>mp.ptr)
    if isinstance(x, _carray.Indexer):
        mp = (<_carray.Indexer>x).get_pointer()
        return CIntptr(<intptr_t>mp.ptr)
    if isinstance(x, MemoryPointer):
        return CIntptr(<intptr_t>x.ptr)
    if isinstance(x, CPointer):
        return x
    # Note: texture/surface objects have no Ascend equivalent; the
    # `cupy.cuda.texture` stub raises on construction, so no instance of them
    # can ever reach this point.
    if isinstance(x, numpy.ndarray):
        # Size-one numpy arrays are passed by value (same rule as CUDA; see
        # the upstream implementation for the rationale).
        if (x.size == 1):
            return CNumpyArray(x)
        else:
            msg = ('You are trying to pass a numpy.ndarray of shape {} as a '
                   'kernel parameter. Only numpy.ndarrays of size one can be '
                   'passed by value. If you meant to pass a pointer to device'
                   ' memory, you need to pass a cupy.ndarray instead.')
            raise TypeError(msg.format(x.shape))

    if type(x) not in _pointer_numpy_types:
        if isinstance(x, int):
            x = numpy.int64(x)
        elif isinstance(x, float):
            x = numpy.float64(x)
        elif isinstance(x, bool):
            x = numpy.bool_(x)
        elif isinstance(x, complex):
            x = numpy.complex128(x)
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


cdef _launch(intptr_t func, size_t num_blocks,
             args, size_t shared_mem, size_t stream):
    """Append `args` to an aclrt args handle and launch the kernel.

    `func` is an aclrtFuncHandle, `num_blocks` the (already mapped) value for
    ``aclrtLaunchKernelWithConfig``'s ``numBlocks``.
    """
    cdef list pargs = []
    cdef CPointer cp
    cdef aclrtArgsHandle args_handle = NULL
    cdef aclrtParamHandle param
    cdef aclrtLaunchKernelAttr attr
    cdef aclrtLaunchKernelCfg cfg
    cdef aclError ret

    runtime._ensure_context()

    _check(aclrtKernelArgsInit(<aclrtFuncHandle>func, &args_handle),
           'aclrtKernelArgsInit')
    for a in args:
        cp = _pointer(a)
        pargs.append(cp)  # keep the CPointer objects alive until launch
        _check(aclrtKernelArgsAppend(args_handle, cp.ptr, cp.size, &param),
               'aclrtKernelArgsAppend')
    _check(aclrtKernelArgsFinalize(args_handle), 'aclrtKernelArgsFinalize')

    if shared_mem > 0:
        # dynamic UB (unified buffer) == CUDA dynamic shared memory
        attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE
        attr.value.dynUBufSize = <uint32_t>shared_mem
        cfg.attrs = &attr
        cfg.numAttrs = 1
        ret = aclrtLaunchKernelWithConfig(
            <aclrtFuncHandle>func, <uint32_t>num_blocks,
            <aclrtStream>stream, &cfg, args_handle, NULL)
    else:
        ret = aclrtLaunchKernelWithConfig(
            <aclrtFuncHandle>func, <uint32_t>num_blocks,
            <aclrtStream>stream, NULL, args_handle, NULL)
    _check(ret, 'aclrtLaunchKernelWithConfig')


cdef class Function:

    """AscendC kernel function (aclrtFuncHandle)."""

    def __init__(self, Module module, str funcname):
        cdef bytes name
        self.module = module  # to keep the binary loaded
        name = funcname.encode()
        _check(aclrtBinaryGetFunction(<aclrtBinHandle>module.ptr, name,
                                      <aclrtFuncHandle*>&self.ptr),
               'aclrtBinaryGetFunction')

    def __call__(self, tuple grid, tuple block, args, size_t shared_mem=0,
                stream=None, enable_cooperative_groups=False):
        # Ascend has no grid x block hierarchy: the grid product maps to
        # `numBlocks` and `block` is accepted for API compatibility only.
        # (Mapping discussion: docs/ascend/CustomKernel.md section 4.3.)
        grid = (grid + (1, 1))[:3]
        block = (block + (1, 1))[:3]
        num_blocks = (max(1, grid[0]) * max(1, grid[1]) * max(1, grid[2]))
        s = _get_stream(stream)
        _launch(self.ptr, num_blocks, args, shared_mem, s)

    cpdef linear_launch(self, size_t size, args, size_t shared_mem=0,
                        size_t block_max_size=128, stream=None,
                        bint enable_cooperative_groups=False):
        # `block_max_size` is treated as the workload per AI-core block,
        # mirroring the CUPY_CUSTOM_KERNEL_PER_BLOCK convention of
        # cupy/backends/ascend/acl_custom_kernels.h.
        cdef size_t num_blocks = (size + block_max_size - 1) // block_max_size
        s = _get_stream(stream)
        _launch(self.ptr, num_blocks, args, shared_mem, s)


cdef class Module:

    """Kernel binary module (an aclrtBinHandle)."""

    def __init__(self):
        self.ptr = 0
        self._binary = 0
        self.mapping = None

    def __dealloc__(self):
        cdef aclrtBinHandle handle
        if self.ptr:
            handle = <aclrtBinHandle>self.ptr
            self.ptr = 0
            aclrtBinaryUnLoad(handle)
        if self._binary:
            aclrtDestroyBinary(<aclrtBinary>self._binary)
            self._binary = 0

    cpdef load_file(self, filename):
        cdef bytes b
        if isinstance(filename, bytes):
            b = filename
        else:
            b = filename.encode()
        runtime._ensure_context()
        _check(aclrtBinaryLoadFromFile(b, NULL, <aclrtBinHandle*>&self.ptr),
               'aclrtBinaryLoadFromFile')

    cpdef load(self, bytes cubin):
        """Load an in-memory aicore fatbin (bisheng-compiled ``.o``)."""
        runtime._ensure_context()
        binary = aclrtCreateBinary(<const void*><char*>cubin, len(cubin))
        if binary == NULL:
            raise RuntimeError('aclrtCreateBinary failed')
        self._binary = <intptr_t>binary
        _check(aclrtBinaryLoad(binary, <aclrtBinHandle*>&self.ptr),
               'aclrtBinaryLoad')

    cpdef get_global_var(self, name):
        raise NotImplementedError(
            'module global variables are not supported on Ascend '
            '(no aclrt equivalent of cuModuleGetGlobal)')

    cpdef get_function(self, name):
        if isinstance(name, bytes):
            name = name.decode()
        return Function(self, name)

    cpdef _set_mapping(self, dict mapping):
        self.mapping = mapping


cdef class LinkState:

    """Not available on Ascend: there is no device-code linker/JIT.

    AscendC kernels are compiled out-of-tree by bisheng into an aicore
    fatbin (see docs/ascend/CustomKernel.md); there is nothing to link.
    """

    def __init__(self):
        runtime._ensure_context()
        self.ptr = 0

    def __dealloc__(self):
        self.ptr = 0

    cpdef add_ptr_data(self, bytes data, unicode name):
        raise NotImplementedError(
            'device code linking is not supported on Ascend '
            '(compile with bisheng instead, see docs/ascend/CustomKernel.md)')

    cpdef add_ptr_file(self, unicode path):
        raise NotImplementedError(
            'device code linking is not supported on Ascend '
            '(compile with bisheng instead, see docs/ascend/CustomKernel.md)')

    cpdef bytes complete(self):
        raise NotImplementedError(
            'device code linking is not supported on Ascend '
            '(compile with bisheng instead, see docs/ascend/CustomKernel.md)')
