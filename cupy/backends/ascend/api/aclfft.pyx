# distutils: language = c++
"""cuFFT-compatible Python binding for the CANN FFT library (ops-fft).

This module mirrors the public API of :mod:`cupy.cuda.cufft` (``Plan1d``,
``PlanNd``, the ``CUFFT_*`` constants, ``get_current_plan()``) so that
:mod:`cupy.fft._fft` can use either backend with minimal changes. The CANN
FFT library (ops-fft) deliberately borrows the cuFFT API shape: the
``aclfftType`` enum values are identical to cuFFT's, and the normalization
semantics (forward unscaled / inverse scaled by N) match cuFFT's default.

Known deviations from cuFFT (see ``docs/ascend/ascend_fft.md``):

* ``aclfftExec*`` take **host** pointers and are **synchronous**; this
  binding hides that behind ``Plan1d.fft(device_in, device_out, direction)``
  by staging through host buffers (``cupy.asnumpy`` / ``cupy.asarray``).
* Single precision only (FP32): double-precision plan types raise
  ``NotImplementedError``.
* 1-D plans only for general use (``PlanNd`` exists for API compatibility
  but raises on construction); 2-D C2C plans exist in ops-fft but are only
  supported for a very restricted set of shapes on Ascend 910B.
* No callbacks and no multi-GPU support.
"""
from __future__ import annotations

import threading

import numpy

import cupy

from cupy.backends.backend.api cimport runtime
from cupy.xpu cimport stream as stream_module
from libc.stdint cimport intptr_t


# ---------------------------------------------------------------------------
# Extern declarations (vendored header: cann_ops_fft.h, next to this file)
# ---------------------------------------------------------------------------
cdef extern from "cann_ops_fft.h" nogil:
    # The header defines `typedef enum aclfftResult_t {...} aclfftResult;` and
    # `typedef enum aclfftType_t {...} aclfftType;`. Declaring them as int
    # aliases makes Cython emit the (header-defined) typedef names verbatim,
    # so all generated code uses the real enum types; the enum member values
    # are numerically identical to the CUFFT_* constants below.
    ctypedef int aclfftResult
    ctypedef int aclfftType

    ctypedef struct aclfftHandle_t
    ctypedef aclfftHandle_t* aclfftHandle

    # opaque pointer types; only ever cast to/from integers, never dereferenced
    ctypedef struct aclfftComplex
    ctypedef float aclfftReal
    ctypedef struct aclrtStream_t
    ctypedef aclrtStream_t* aclrtStream

    aclfftResult aclfftPlan1d(aclfftHandle* plan, int nx, aclfftType type,
                              int batch, int dimType)
    aclfftResult aclfftPlan2d(aclfftHandle* plan, int batch, int nx, int ny,
                              aclfftType type)
    aclfftResult aclfftSetStream(aclfftHandle plan, aclrtStream stream)
    aclfftResult aclfftExecC2C(aclfftHandle plan, aclfftComplex* idata,
                               aclfftComplex* odata, int direction)
    aclfftResult aclfftExecR2C(aclfftHandle plan, aclfftReal* idata,
                               aclfftComplex* odata)
    aclfftResult aclfftExecC2R(aclfftHandle plan, aclfftComplex* idata,
                               aclfftReal* odata)
    aclfftResult aclfftDestroy(aclfftHandle plan)
    const char* aclfftGetErrorString(aclfftResult result)

    # ACLFFT_HORIZONTAL: FFT along each row of a C-contiguous (batch, nx)
    # buffer. Vertical transforms have too restrictive constraints to be
    # useful for cupy (power-of-two nx, batch a multiple of 128).
    int ACLFFT_HORIZONTAL


# ---------------------------------------------------------------------------
# Public constants (values are identical to cuFFT's, by design of ops-fft)
# ---------------------------------------------------------------------------
CUFFT_C2C = 0x29
CUFFT_R2C = 0x2a
CUFFT_C2R = 0x2c
CUFFT_Z2Z = 0x69
CUFFT_D2Z = 0x6a
CUFFT_Z2D = 0x6c
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

# ACLFFT_* aliases for direct users of this module
ACLFFT_FORWARD = CUFFT_FORWARD
ACLFFT_BACKWARD = CUFFT_INVERSE

# Capability flags read by cupy.fft._backend.get_cufft() (see its docstring)
supports_nd_plan = False   # no cufftMakePlanMany / strides equivalent
supports_callbacks = False # no cuFFT callback support


_RESULT = {
    0: 'ACLFFT_SUCCESS',
    1: 'ACLFFT_INVALID_PLAN',
    2: 'ACLFFT_ALLOC_FAILED',
    3: 'ACLFFT_INVALID_TYPE',
    4: 'ACLFFT_INVALID_VALUE',
    5: 'ACLFFT_INTERNAL_ERROR',
    6: 'ACLFFT_EXEC_FAILED',
    7: 'ACLFFT_SETUP_FAILED',
    8: 'ACLFFT_INVALID_SIZE',
    9: 'ACLFFT_UNALIGNED_DATA',
    10: 'ACLFFT_INCOMPLETE_PARAMETER_LIST',
    11: 'ACLFFT_INVALID_DEVICE',
    12: 'ACLFFT_PARSE_ERROR',
    13: 'ACLFFT_NO_WORKSPACE',
    14: 'ACLFFT_NOT_IMPLEMENTED',
    16: 'ACLFFT_NOT_SUPPORTED',
}


class CuFFTError(RuntimeError):
    def __init__(self, int result):
        self.result = result
        self.errstr = _RESULT.get(result, 'unknown')
        super().__init__(f'cuFFT error: {self.errstr} ({result})')


cdef inline int check_result(int result) except -1:
    if result != 0:
        raise CuFFTError(result)
    return 0


cdef int _to_acl_type(int fft_type) except -1:
    # aclfftType enum values are numerically identical to the cuFFT ones
    if fft_type not in (CUFFT_C2C, CUFFT_R2C, CUFFT_C2R):
        raise ValueError(f'unsupported fft type for aclfft: {fft_type:#x}')
    return fft_type


def check_result_py(int result):
    check_result(result)


def getVersion():
    """Return the ops-fft version.

    ``aclfftGetVersion`` is declared by ops-fft but not implemented yet, so
    this returns ``0`` (unknown) rather than failing.
    """
    return 0


# ---------------------------------------------------------------------------
# Thread-local current plan (context-manager support), mirroring cupy.cuda.cufft
# ---------------------------------------------------------------------------
cdef object _thread_local = threading.local()


def get_current_plan():
    return getattr(_thread_local, '_current_plan', None)


# ---------------------------------------------------------------------------
# Plan1d
# ---------------------------------------------------------------------------
cdef class Plan1d:
    """A 1-D FFT plan for the CANN FFT library.

    Signature-compatible with ``cupy.cuda.cufft.Plan1d``.
    """

    cdef aclfftHandle handle
    cdef readonly intptr_t handle_ptr
    cdef readonly int nx
    cdef readonly int fft_type
    cdef readonly int batch
    cdef readonly object gpus
    cdef readonly object work_area

    def __init__(self, int nx, int fft_type, int batch, devices=None,
                 out=None):
        cdef aclfftHandle plan = NULL
        cdef aclfftResult result
        cdef int acl_type

        if devices is not None:
            raise NotImplementedError(
                'multi-GPU FFT is not supported on the Ascend backend')

        self.handle = NULL
        self.work_area = None
        self.gpus = None
        self.nx = nx
        self.fft_type = fft_type
        self.batch = batch

        if batch == 0:
            # bookkeeping-only plan (mirrors cuFFT behaviour); fft() is a no-op
            return

        if fft_type in (CUFFT_Z2Z, CUFFT_D2Z, CUFFT_Z2D):
            raise NotImplementedError(
                'aclfft does not support double precision (FP64) FFT')

        acl_type = _to_acl_type(fft_type)
        # ops-fft manages workspace internally (workSize is always 0), so no
        # work-area handling is required here, unlike the cuFFT plan.
        with nogil:
            result = aclfftPlan1d(&plan, nx, <aclfftType>acl_type, batch,
                                  ACLFFT_HORIZONTAL)
        if result != 0:
            # aclfftPlan1d 失败时仍可能已经分配出 plan，报错前必须回收，
            # 否则句柄泄漏（它既不会存进 self.handle，也不会被 __dealloc__ 看到）。
            if plan != NULL:
                with nogil:
                    aclfftDestroy(plan)
                plan = NULL
            check_result(result)

        self.handle = plan
        self.handle_ptr = <intptr_t>plan

    def __dealloc__(self):
        cdef aclfftResult result
        if self.handle != NULL:
            with nogil:
                result = aclfftDestroy(self.handle)
            self.handle = NULL

    def __enter__(self):
        _thread_local._current_plan = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _thread_local._current_plan = None

    def _output_dtype_and_shape(self, a):
        shape = list(a.shape)
        if self.fft_type == CUFFT_C2C:
            dtype = numpy.complex64
        elif self.fft_type == CUFFT_R2C:
            shape[-1] = shape[-1] // 2 + 1
            dtype = numpy.complex64
        elif self.fft_type == CUFFT_C2R:
            shape[-1] = self.nx
            dtype = numpy.float32
        else:
            raise NotImplementedError(
                'aclfft only supports single precision (FP32) transforms')
        return tuple(shape), dtype

    def get_output_array(self, a):
        shape, dtype = self._output_dtype_and_shape(a)
        return cupy.empty(shape, dtype)

    def check_output_array(self, a, out):
        """Verify shape and dtype of the output array."""
        shape, dtype = self._output_dtype_and_shape(a)
        if out.shape != shape:
            raise ValueError(f'out must have shape {shape}.')
        if out.dtype != dtype:
            raise ValueError(
                'out dtype mismatch: found {}, expected {}'.format(
                    out.dtype, dtype))

    def fft(self, a, out, direction):
        cdef aclfftResult result
        cdef intptr_t s, p_in, p_out
        cdef int dir_int = direction

        if self.handle == NULL:
            return

        # Bind the plan to cupy's current stream. ops-fft Exec is synchronous
        # (it syncs internally), so this is mostly bookkeeping.
        s = stream_module.get_current_stream_ptr()
        with nogil:
            result = aclfftSetStream(self.handle, <aclrtStream>s)
        check_result(result)

        # aclfftExec* take host pointers and do H2D / kernel / D2H internally,
        # so stage the data through host buffers. The host-side staging copy
        # also protects against the ops-fft behaviour of modifying the input
        # for power-of-two sizes >= 32768.
        hin = numpy.ascontiguousarray(cupy.asnumpy(a))
        shape, dtype = self._output_dtype_and_shape(a)
        hout = numpy.empty(shape, dtype)

        p_in = hin.ctypes.data
        p_out = hout.ctypes.data

        if self.fft_type == CUFFT_C2C:
            with nogil:
                result = aclfftExecC2C(self.handle, <aclfftComplex*>p_in,
                                       <aclfftComplex*>p_out, dir_int)
        elif self.fft_type == CUFFT_R2C:
            with nogil:
                result = aclfftExecR2C(self.handle, <aclfftReal*>p_in,
                                       <aclfftComplex*>p_out)
        elif self.fft_type == CUFFT_C2R:
            with nogil:
                result = aclfftExecC2R(self.handle, <aclfftComplex*>p_in,
                                       <aclfftReal*>p_out)
        else:
            raise NotImplementedError(
                'aclfft only supports single precision (FP32) transforms')
        check_result(result)

        out[...] = cupy.asarray(hout)


# ---------------------------------------------------------------------------
# PlanNd (API-compatibility stub)
# ---------------------------------------------------------------------------
cdef class PlanNd:
    """N-D FFT plan stub.

    ops-fft has no equivalent of ``cufftMakePlanMany`` (advanced data layout
    with strides), so N-D plans are not supported. ``cupy.fft._fft`` degrades
    to repeated 1-D transforms via the ``supports_nd_plan`` capability flag
    in :mod:`cupy.fft._backend`; this class only exists so that
    ``isinstance(plan, cufft.PlanNd)`` checks in ``cupy.fft`` keep working and
    a user-supplied N-D plan fails with a clear message.
    """

    def __init__(self, shape, inembed, int istride, int idist, onembed,
                 int ostride, int odist, int fft_type, int batch, str order,
                 int last_axis, last_size):
        raise NotImplementedError(
            'aclfft does not support N-dimensional plans (no '
            'cufftMakePlanMany equivalent); use 1-D transforms instead')

    def fft(self, a, out, direction):
        raise NotImplementedError(
            'aclfft does not support N-dimensional plans')

    def get_output_array(self, a, order='C'):
        raise NotImplementedError(
            'aclfft does not support N-dimensional plans')

    def check_output_array(self, a, out):
        raise NotImplementedError(
            'aclfft does not support N-dimensional plans')
