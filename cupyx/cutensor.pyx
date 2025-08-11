import numpy as _numpy

import cupy as _cupy
from cupy import _util
from cupy.cuda import device as _device

cimport cython
from libcpp cimport vector
from libc.stdint cimport intptr_t, uint32_t, uint64_t, int64_t
from cupy._core._carray cimport shape_t
from cupy._core.core cimport _ndarray_base
from cupy._core cimport internal

from cupy_backends.cuda.api.runtime cimport getDeviceCount
from cupy_backends.cuda cimport stream as stream_module

from cupy._core cimport core
from cupy._core cimport _reduction
from cupy.cuda cimport device
from cupy.cuda.pinned_memory cimport alloc_pinned_memory, is_memory_pinned
from cupy_backends.cuda.libs cimport cutensor
from cupy.cuda import Device

cdef dict _handles = {}
cdef dict _tensor_descriptors = {}
cdef dict _plan_preferences = {}
cdef dict _plans = {}
cdef dict _modes = {}
cdef dict _scalars = {}
cdef dict _elementwise_binary_operators = {}
cdef dict _elementwise_trinary_operators = {}
cdef dict _reduction_operators = {}
cdef dict _contraction_operators = {}
cdef dict _mg_handles = {}
cdef dict _mg_tensor_descriptors = {}
cdef dict _mg_copy_descriptors = {}
cdef dict _mg_copy_plans = {}
cdef dict _mg_contraction_descriptors = {}
cdef dict _mg_contraction_finds = {}
cdef dict _mg_contraction_plans = {}

cdef dict _available_compute_capability = {
    'contraction': 60,
    'reduction': 60,
    'elementwise': 60,
    'copyMg': 60,
    'contractMg': 60,
}


@_util.memoize(for_each_device=True)
def check_availability(name):
    if name in _available_compute_capability:
        compute_capability = int(_device.get_compute_capability())
        if compute_capability < _available_compute_capability[name]:
            return False
    return True


###############################################################################
# Handle: This class encapsulates the opaque structure `cutensorHandle_t`
# holding cuTENSOR's library context.
###############################################################################

cdef class Handle:
    cdef intptr_t _ptr

    def __init__(self):
        self._ptr = cutensor.create()

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroy(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr


###############################################################################
# TensorDescriptor: This class encapsulates the opaque structure
# `cutensorTensorDescriptor_t` representing a tensor descriptor.
###############################################################################

cdef class TensorDescriptor:
    cdef intptr_t _ptr
    cdef int _cutensor_dtype

    def __init__(self, intptr_t handle, uint32_t num_modes, intptr_t extent,
                 intptr_t stride, int cutensor_dtype,
                 uint32_t alignment_req=256):
        self._ptr = cutensor.createTensorDescriptor(
            handle, num_modes, extent, stride, cutensor_dtype, alignment_req)
        self._cutensor_dtype = cutensor_dtype

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyTensorDescriptor(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr

    @property
    def cutensor_dtype(self):
        return self._cutensor_dtype


###############################################################################
# OperationDescriptor: This class encapsulates the opaque structure
# `cutensorOperationDescriptor_t` representing any type of problem descriptor.
###############################################################################

cdef class OperationDescriptor:
    cdef intptr_t _ptr

    def __init__(self):
        self._ptr = <intptr_t>NULL

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyOperationDescriptor(self._ptr)
        self._ptr = <intptr_t>NULL

    def create_elementwise_binary(
            self, intptr_t handle,
            intptr_t descA, intptr_t modeA, int opA,
            intptr_t descC, intptr_t modeC, int opC,
            intptr_t descD, intptr_t modeD,
            int opAC, int descCompute):
        self._ptr = cutensor.createElementwiseBinary(
            handle, descA, modeA, opA, descC, modeC, opC, descD, modeD,
            opAC, descCompute)

    def create_elementwise_trinary(
            self, intptr_t handle,
            intptr_t descA, intptr_t modeA, int opA,
            intptr_t descB, intptr_t modeB, int opB,
            intptr_t descC, intptr_t modeC, int opC,
            intptr_t descD, intptr_t modeD,
            int opAB, int opABC, int descCompute):
        self._ptr = cutensor.createElementwiseTrinary(
            handle, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC,
            descD, modeD, opAB, opABC, descCompute)

    def create_reduction(
            self, intptr_t handle,
            intptr_t descA, intptr_t modeA, int opA,
            intptr_t descC, intptr_t modeC, int opC,
            intptr_t descD, intptr_t modeD,
            int opReduce, int descCompute):
        self._ptr = cutensor.createReduction(
            handle, descA, modeA, opA, descC, modeC, opC,
            descD, modeD, opReduce, descCompute)

    def create_contraction(
            self, intptr_t handle,
            intptr_t descA, intptr_t modeA, int opA,
            intptr_t descB, intptr_t modeB, int opB,
            intptr_t descC, intptr_t modeC, int opC,
            intptr_t descD, intptr_t modeD,
            int descCompute):
        self._ptr = cutensor.createContraction(
            handle, descA, modeA, opA, descB, modeB, opB,
            descC, modeC, opC, descD, modeD, descCompute)

    @property
    def ptr(self):
        return self._ptr


###############################################################################
# PlanPreference: This class encapsulates the opaque structure
# `cutensorPlanPreference_t` that narrow down the space of applicable
# algorithms/variants/kernels.
###############################################################################

cdef class PlanPreference:
    cdef intptr_t _ptr

    def __init__(self, intptr_t handle, int algo, int jit_mode):
        self._ptr = cutensor.createPlanPreference(handle, algo, jit_mode)

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyPlanPreference(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr


###############################################################################
# Plan: This class encapsulates the opaque structure `cutensorPlan_t`
# representing a plan for ops like contraction, reduction, elementwise.
###############################################################################

cdef class Plan:
    cdef intptr_t _ptr

    def __init__(self, intptr_t handle, intptr_t desc, intptr_t pref,
                 uint64_t ws_limit):
        self._ptr = cutensor.createPlan(handle, desc, pref, ws_limit)

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyPlan(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr


cpdef Handle _get_handle():
    cdef int dev = device.get_device_id()
    if dev not in _handles:
        _handles[dev] = Handle()
    return _handles[dev]


cpdef int _get_cutensor_dtype(dtype) except -1:
    cdef str dtype_char
    try:
        dtype_char = dtype.char
    except AttributeError:
        dtype_char = dtype

    if dtype_char == 'e':
        return cutensor.R_16F
    elif dtype_char == 'f':
        return cutensor.R_32F
    elif dtype_char == 'd':
        return cutensor.R_64F
    elif dtype_char == 'E':
        # complex32, not supported in NumPy
        return cutensor.C_16F
    elif dtype_char == 'F':
        return cutensor.C_32F
    elif dtype_char == 'D':
        return cutensor.C_64F
    else:
        raise TypeError('dtype is not supported: {}'.format(dtype))


cpdef TensorDescriptor create_tensor_descriptor(_ndarray_base a):
    """Create a tensor descriptor

    Args:
        a (cupy.ndarray): tensor for which a descritpor are created.

    Returns:
        (TensorDescriptor): A instance of class TensorDescriptor.
    """
    handle = _get_handle()
    alignment_req = a.itemsize
    key = (handle.ptr, a.dtype, tuple(a.shape),
           tuple(a.strides), alignment_req)
    if a.data.ptr & (alignment_req - 1) != 0:
        raise ValueError("Missaligned array")
    if key not in _tensor_descriptors:
        num_modes = a.ndim
        extent = _numpy.array(a.shape, dtype=_numpy.int64)
        stride = _numpy.array(a.strides, dtype=_numpy.int64) // a.itemsize
        cutensor_dtype = _get_cutensor_dtype(a.dtype)
        _tensor_descriptors[key] = TensorDescriptor(
            handle.ptr, num_modes, extent.ctypes.data, stride.ctypes.data,
            cutensor_dtype, alignment_req=alignment_req)
    return _tensor_descriptors[key]


cpdef PlanPreference create_plan_preference(
        int algo=cutensor.ALGO_DEFAULT, int jit_mode=cutensor.JIT_MODE_NONE):
    """Create a plan preference

    Args:
        algo (cutensorAlgo_t): Specify the algorithm to be used for
            contraction. For other than contractoin, there is no need to
            specify this.
        jit_mode (cutensorJitMode_t): Specify whether and how to use JIT.

    Returns:
        (PlanPreference): A instance of class PlanPreference.
    """
    handle = _get_handle()
    key = (handle.ptr, algo, jit_mode)
    if key not in _plan_preferences:
        _plan_preferences[key] = PlanPreference(handle.ptr, algo, jit_mode)
    return _plan_preferences[key]


cpdef Plan create_plan(
        OperationDescriptor desc, PlanPreference pref, uint64_t ws_limit=0):
    """Create a plan

    Args:
        desc (OperationDescriptor):
        pref (PlanPreference):
        ws_limit (uint64_t):

    Returns:
        (Plan): A instance of class Plan.
    """
    handle = _get_handle()
    key = (handle.ptr, desc.ptr, pref.ptr, ws_limit)
    if key not in _plans:
        _plans[key] = Plan(handle.ptr, desc.ptr, pref.ptr, ws_limit)
    return _plans[key]


cdef class Mode:
    cdef:
        object _array
        readonly int ndim
        readonly intptr_t data

    def __init__(self, mode):
        self._array = _numpy.array(mode, dtype=_numpy.int32)
        assert self._array.ndim == 1
        self.ndim = self._array.size
        self.data = self._array.ctypes.data

    def __repr__(self):
        return 'mode(' + ', '.join([str(x) for x in self._array]) + ')'

    def __eq__(self, other):
        if not isinstance(other, Mode):
            return False
        return (self._array == (<Mode>other)._array).all()


def create_mode(*mode):
    """Create the tensor mode from the given integers or characters.

    Args:
        mode (tuple of int/str): A tuple that holds the labels of the modes
            of tensor A (e.g., if A_{x,y,z}, mode_A = {'x','y','z'})
    """
    integer_mode = []
    for x in mode:
        if isinstance(x, int):
            integer_mode.append(x)
        elif isinstance(x, str):
            integer_mode.append(ord(x))
        else:
            raise TypeError('Cannot create tensor mode: {}'.format(type(x)))
    return _create_mode_with_cache(tuple(integer_mode))


cdef inline Mode _create_mode_with_cache(axis_or_ndim):
    cdef Mode mode
    if axis_or_ndim in _modes:
        mode = _modes[axis_or_ndim]
    else:
        if type(axis_or_ndim) is int:
            mode = Mode(tuple(range(axis_or_ndim)))
        else:
            mode = Mode(axis_or_ndim)
        _modes[axis_or_ndim] = mode
    return mode


cdef inline Mode _auto_create_mode(_ndarray_base array, mode):
    if not isinstance(mode, Mode):
        mode = create_mode(*mode)
    if array.ndim != mode.ndim:
        raise ValueError(
            'ndim mismatch: {} != {}'.format(array.ndim, mode.ndim))
    return mode


cdef class _Scalar:
    cdef:
        object _array
        readonly intptr_t ptr

    def __init__(self, value, dtype):
        self._array = _numpy.asarray(value, dtype=dtype)
        self.ptr = self._array.ctypes.data

    def __repr__(self):
        return (
            'scalar(' + str(self._array.item()) +
            ', dtype=' + str(self._array.dtype) + ')')


cdef inline _Scalar _create_scalar(scale, dtype):
    cdef _Scalar scalar
    key = (scale, dtype)
    if key in _scalars:
        scalar = _scalars[key]
    else:
        scalar = _Scalar(scale, dtype)
        _scalars[key] = scalar
    return scalar


cdef dict _elementwise_binary_compute_descs = {
    # [cutensor_dtype_A][*_C]
    cutensor.R_16F: {cutensor.R_16F: cutensor.COMPUTE_DESC_16F,
                     cutensor.R_32F: cutensor.COMPUTE_DESC_32F},
    cutensor.R_32F: {cutensor.R_16F: cutensor.COMPUTE_DESC_32F,
                     cutensor.R_32F: cutensor.COMPUTE_DESC_32F,
                     cutensor.R_64F: cutensor.COMPUTE_DESC_64F},
    cutensor.R_64F: {cutensor.R_32F: cutensor.COMPUTE_DESC_64F,
                     cutensor.R_64F: cutensor.COMPUTE_DESC_64F},
    cutensor.C_32F: {cutensor.C_32F: cutensor.COMPUTE_DESC_32F,
                     cutensor.C_64F: cutensor.COMPUTE_DESC_64F},
    cutensor.C_64F: {cutensor.C_32F: cutensor.COMPUTE_DESC_64F,
                     cutensor.C_64F: cutensor.COMPUTE_DESC_64F},
}


cpdef OperationDescriptor create_elementwise_binary(
        TensorDescriptor desc_A, Mode mode_A, int op_A,
        TensorDescriptor desc_C, Mode mode_C, int op_C,
        TensorDescriptor desc_D, Mode mode_D, int op_AC,
        int compute_desc=0):
    """Create a operation descriprot for element-wise binary operation:

        D = op_AC(alpha * op_A(A), gamma * op_C(C))

    Args:
        desc_A (TensorDescriptor):
        mode_A (Mode):
        op_A (cutensorOperator_t):
        desc_C (TensorDescriptor):
        mode_C (Mode):
        op_C (cutensorOperator_t):
        desc_D (TensorDescriptor):
        mode_D (Mode):
        op_AC (cutensorOperator_t):
        compute_desc (cutensorComputeDescriptor_t):

    Returns:
        (Descriptor): A instance of class OperationDescriptor.
    """
    compute_descs = _elementwise_binary_compute_descs
    ct_dtype_A = desc_A.cutensor_dtype
    ct_dtype_C = desc_C.cutensor_dtype
    if not (ct_dtype_A in compute_descs and
            ct_dtype_C in compute_descs[ct_dtype_A]):
        raise ValueError(
            'This combination of cutensor dtype is not supported in '
            'elementwise_binary ({}, {})'.format(ct_dtype_A, ct_dtype_C))
    if compute_desc == 0:
        compute_desc = compute_descs[ct_dtype_A][ct_dtype_C]
    handle = _get_handle()
    key = (handle.ptr,
           desc_A.ptr, mode_A.data, op_A,
           desc_C.ptr, mode_C.data, op_C,
           desc_D.ptr, mode_D.data, op_AC, compute_desc)
    if key not in _elementwise_binary_operators:
        _elementwise_binary_operators[key] = OperationDescriptor()
        _elementwise_binary_operators[key].create_elementwise_binary(
            handle.ptr,
            desc_A.ptr, mode_A.data, op_A,
            desc_C.ptr, mode_C.data, op_C,
            desc_D.ptr, mode_D.data, op_AC, compute_desc)
    return _elementwise_binary_operators[key]


def elementwise_binary(
        alpha, _ndarray_base A, mode_A,
        gamma, _ndarray_base C, mode_C,
        _ndarray_base out=None,
        op_A=cutensor.OP_IDENTITY, op_C=cutensor.OP_IDENTITY,
        op_AC=cutensor.OP_ADD, compute_desc=0):
    """Element-wise tensor operation for two input tensors

    This function performs a element-wise tensor operation of the form:

        C = op_AC(alpha * op_A(A), gamma * op_C(C))

    Args:
        alpha (scalar): Scaling factor for tensor A.
        A (cupy.ndarray): Input tensor.
        mode_A (tuple of int/str or Mode): A mode for tensor A.
        gamma (scalar): Scaling factor for tensor C.
        C (cupy.ndarray): Input/output tensor.
        mode_C (tuple of int/str or Mode): A mode for tensor C.
        op_A (cutensorOperator_t): Element-wise unary operator for tensor A.
        op_C (cutensorOperator_t): Element-wise unary operator for tensor C.
        op_AC (cutensorOperator_t): Element-wise binary operator for A and C.
        compute_desc (cutensorComputeDescriptor_t): Compute type for the
            intermediate computation.

    Returns:
        out (cupy.ndarray): Output tensor.

    Examples:
        See examples/cutensor/elementwise_binary.py
    """
    if out is None:
        out = core._ndarray_init(
            _cupy.ndarray, C._shape, dtype=C.dtype, obj=None)
    elif C.dtype != out.dtype:
        raise ValueError('dtype mismatch: {} != {}'.format(C.dtype, out.dtype))
    elif not internal.vector_equal(C._shape, out._shape):
        raise ValueError('shape mismatch: {} != {}'.format(C.shape, out.shape))

    desc_A = create_tensor_descriptor(A)
    desc_C = create_tensor_descriptor(C)
    desc_out = create_tensor_descriptor(out)
    mode_A = _auto_create_mode(A, mode_A)
    mode_C = _auto_create_mode(C, mode_C)
    mode_out = mode_C
    operator = create_elementwise_binary(
        desc_A, mode_A, op_A, desc_C, mode_C, op_C,
        desc_out, mode_out, op_AC, compute_desc)
    plan_pref = create_plan_preference()
    plan = create_plan(operator, plan_pref)
    cutensor.elementwiseBinaryExecute(
        _get_handle().ptr, plan.ptr,
        _create_scalar(alpha, out.dtype).ptr, A.data.ptr,
        _create_scalar(gamma, out.dtype).ptr, C.data.ptr,
        out.data.ptr)
    return out


cdef dict _elementwise_trinary_compute_descs = {
    # [cutensor_dtype_A/B][*_C]
    cutensor.R_16F: {cutensor.R_16F: cutensor.COMPUTE_DESC_16F,
                     cutensor.R_32F: cutensor.COMPUTE_DESC_32F},
    cutensor.R_32F: {cutensor.R_16F: cutensor.COMPUTE_DESC_32F,
                     cutensor.R_32F: cutensor.COMPUTE_DESC_32F,
                     cutensor.R_64F: cutensor.COMPUTE_DESC_64F},
    cutensor.R_64F: {cutensor.R_32F: cutensor.COMPUTE_DESC_64F,
                     cutensor.R_64F: cutensor.COMPUTE_DESC_64F},
    cutensor.C_32F: {cutensor.C_32F: cutensor.COMPUTE_DESC_32F,
                     cutensor.C_64F: cutensor.COMPUTE_DESC_64F},
    cutensor.C_64F: {cutensor.C_32F: cutensor.COMPUTE_DESC_64F,
                     cutensor.C_64F: cutensor.COMPUTE_DESC_64F},
}


cpdef OperationDescriptor create_elementwise_trinary(
        TensorDescriptor desc_A, Mode mode_A, int op_A,
        TensorDescriptor desc_B, Mode mode_B, int op_B,
        TensorDescriptor desc_C, Mode mode_C, int op_C,
        TensorDescriptor desc_D, Mode mode_D,
        int op_AB, int op_ABC, int compute_desc=0):
    """Create a operation descriprot for element-wise trinary operatoin:

        D = op_ABC(op_AB(alpha * op_A(A), beta * op_B(B)), gamma * op_C(C))

    Args:
        desc_A (TensorDescriptor):
        mode_A (Mode):
        op_A (cutensorOperator_t):
        desc_B (TensorDescriptor):
        mode_B (Mode):
        op_B (cutensorOperator_t):
        desc_C (TensorDescriptor):
        mode_C (Mode):
        op_C (cutensorOperator_t):
        desc_D (TensorDescriptor):
        mode_D (Mode):
        op_AB (cutensorOperator_t):
        op_ABC (cutensorOperator_t):
        compute_desc (cutensorComputeDescriptor_t):

    Returns:
        (Descriptor): A instance of class OperationDescriptor.
    """
    compute_descs = _elementwise_trinary_compute_descs
    ct_dtype_A = desc_A.cutensor_dtype
    ct_dtype_B = desc_B.cutensor_dtype
    ct_dtype_C = desc_C.cutensor_dtype
    if not (ct_dtype_A == ct_dtype_B):
        raise ValueError(
            'cutensor dtype mismatch: {} != {}'.format(ct_dtype_A, ct_dtype_B))
    if not (ct_dtype_A in compute_descs and
            ct_dtype_C in compute_descs[ct_dtype_A]):
        raise ValueError(
            'This combination of cutensor dtype is not supported in '
            'elementwise_trinary ({}, {})'.format(ct_dtype_A, ct_dtype_C))
    if compute_desc == 0:
        compute_desc = compute_descs[ct_dtype_A][ct_dtype_C]
    handle = _get_handle()
    key = (handle.ptr,
           desc_A.ptr, mode_A.data, op_A,
           desc_B.ptr, mode_B.data, op_B,
           desc_C.ptr, mode_C.data, op_C,
           desc_D.ptr, mode_D.data, op_AB, op_ABC, compute_desc)
    if key not in _elementwise_trinary_operators:
        _elementwise_trinary_operators[key] = OperationDescriptor()
        _elementwise_trinary_operators[key].create_elementwise_trinary(
            handle.ptr,
            desc_A.ptr, mode_A.data, op_A,
            desc_B.ptr, mode_B.data, op_B,
            desc_C.ptr, mode_C.data, op_C,
            desc_D.ptr, mode_D.data, op_AB, op_ABC, compute_desc)
    return _elementwise_trinary_operators[key]


def elementwise_trinary(
        alpha, _ndarray_base A, mode_A,
        beta, _ndarray_base B, mode_B,
        gamma, _ndarray_base C, mode_C,
        _ndarray_base out=None,
        op_A=cutensor.OP_IDENTITY, op_B=cutensor.OP_IDENTITY,
        op_C=cutensor.OP_IDENTITY, op_AB=cutensor.OP_ADD,
        op_ABC=cutensor.OP_ADD, compute_desc=0):
    """Element-wise tensor operation for three input tensors

    This function performs a element-wise tensor operation of the form:

        C = op_ABC(op_AB(alpha * op_A(A), beta * op_B(B)), gamma * op_C(C))

    Args:
        alpha (scalar): Scaling factor for tensor A.
        A (cupy.ndarray): Input tensor.
        mode_A (tuple of int/str or Mode): A mode for tensor A.
        beta (scalar): Scaling factor for tensor B.
        B (cupy.ndarray): Input tensor.
        mode_B (tuple of int/str or Mode): A mode for tensor B.
        gamma (scalar): Scaling factor for tensor C.
        C (cupy.ndarray): Input/output tensor.
        mode_C (tuple of int/str or Mode): A mode for tensor C.
        op_A (cutensorOperator_t): Element-wise unary operator for tensor A.
        op_B (cutensorOperator_t): Element-wise unary operator for tensor B.
        op_C (cutensorOperator_t): Element-wise unary operator for tensor C.
        op_AB (cutensorOperator_t): Element-wise binary operator for A and B.
        op_ABC (cutensorOperator_t): Element-wise binary operator for AB and C.
        compute_desc (cutensorComputeDescriptor_t): Compute type for the
            intermediate computation.

    Returns:
        out (cupy.ndarray): Output tensor.

    Examples:
        See examples/cutensor/elementwise_trinary.py
    """
    if out is None:
        out = core._ndarray_init(
            _cupy.ndarray, C._shape, dtype=C.dtype, obj=None)
    elif C.dtype != out.dtype:
        raise ValueError('dtype mismatch: {} != {}'.format(C.dtype, out.dtype))
    elif not internal.vector_equal(C._shape, out._shape):
        raise ValueError('shape mismatch: {} != {}'.format(C.shape, out.shape))

    desc_A = create_tensor_descriptor(A)
    desc_B = create_tensor_descriptor(B)
    desc_C = create_tensor_descriptor(C)
    desc_out = create_tensor_descriptor(out)
    mode_A = _auto_create_mode(A, mode_A)
    mode_B = _auto_create_mode(B, mode_B)
    mode_C = _auto_create_mode(C, mode_C)
    mode_out = mode_C
    operator = create_elementwise_trinary(
        desc_A, mode_A, op_A, desc_B, mode_B, op_B,
        desc_C, mode_C, op_C, desc_out, mode_out,
        op_AB, op_ABC, compute_desc)
    plan_pref = create_plan_preference()
    plan = create_plan(operator, plan_pref)
    cutensor.elementwiseTrinaryExecute(
        _get_handle().ptr, plan.ptr,
        _create_scalar(alpha, out.dtype).ptr, A.data.ptr,
        _create_scalar(beta, out.dtype).ptr, B.data.ptr,
        _create_scalar(gamma, out.dtype).ptr, C.data.ptr,
        out.data.ptr)
    return out


cdef dict _contraction_compute_descs = {
    # [cutensor_dtype_A][*_B][*_C]
    cutensor.R_16F: {
        cutensor.R_16F: {cutensor.R_16F: cutensor.COMPUTE_DESC_32F}
    },
    cutensor.R_32F: {
        cutensor.R_32F: {cutensor.R_32F: cutensor.COMPUTE_DESC_32F}
    },
    cutensor.R_64F: {
        cutensor.R_64F: {cutensor.R_64F: cutensor.COMPUTE_DESC_64F},
        cutensor.C_64F: {cutensor.C_64F: cutensor.COMPUTE_DESC_64F}
    },
    cutensor.C_32F: {
        cutensor.C_32F: {cutensor.C_32F: cutensor.COMPUTE_DESC_32F}
    },
    cutensor.C_64F: {
        cutensor.R_64F: {cutensor.C_64F: cutensor.COMPUTE_DESC_64F},
        cutensor.C_64F: {cutensor.C_64F: cutensor.COMPUTE_DESC_64F}
    }
}


cpdef OperationDescriptor create_contraction(
        TensorDescriptor desc_A, Mode mode_A, int op_A,
        TensorDescriptor desc_B, Mode mode_B, int op_B,
        TensorDescriptor desc_C, Mode mode_C, int op_C,
        int compute_desc=0):
    """Create a operation descriprot for contraction:

        C = alpha * op_A(A) * op_B(B) + beta * op_C(C)

    Args:
        desc_A (TensorDescriptor):
        mode_A (Mode):
        op_A (cutensorOperator_t):
        desc_B (TensorDescriptor):
        mode_B (Mode):
        op_B (cutensorOperator_t):
        desc_C (TensorDescriptor):
        mode_C (Mode):
        op_C (cutensorOperator_t):
        compute_desc (cutensorComputeDescriptor_t):

    Returns:
        (Descriptor): A instance of class Descriptor which holds a pointer to
        tensor descriptor and its destructor.
    """
    compute_descs = _contraction_compute_descs
    ct_dtype_A = desc_A.cutensor_dtype
    ct_dtype_B = desc_B.cutensor_dtype
    ct_dtype_C = desc_C.cutensor_dtype
    if not (ct_dtype_A in compute_descs and
            ct_dtype_B in compute_descs[ct_dtype_A] and
            ct_dtype_C in compute_descs[ct_dtype_A][ct_dtype_B]):
        raise ValueError(
            'This cutensor dtype combination is not supported in contraction'
            ' ({}, {}, {})'.format(ct_dtype_A, ct_dtype_B, ct_dtype_C))
    if compute_desc == 0:
        compute_desc = compute_descs[ct_dtype_A][ct_dtype_B][ct_dtype_C]
    handle = _get_handle()
    key = (handle.ptr,
           desc_A.ptr, mode_A.data, op_A,
           desc_B.ptr, mode_B.data, op_B,
           desc_C.ptr, mode_C.data, op_C,
           compute_desc)
    if key not in _contraction_operators:
        _contraction_operators[key] = OperationDescriptor()
        _contraction_operators[key].create_contraction(
            handle.ptr,
            desc_A.ptr, mode_A.data, op_A,
            desc_B.ptr, mode_B.data, op_B,
            desc_C.ptr, mode_C.data, op_C,
            desc_C.ptr, mode_C.data, compute_desc)
    return _contraction_operators[key]


cdef _get_scalar_dtype(out_dtype):
    if out_dtype == _numpy.float16:
        return _numpy.float32
    else:
        return out_dtype


def contraction(
        alpha, _ndarray_base A, mode_A, _ndarray_base B, mode_B,
        beta, _ndarray_base C, mode_C,
        int op_A=cutensor.OP_IDENTITY, int op_B=cutensor.OP_IDENTITY,
        int op_C=cutensor.OP_IDENTITY, int algo=cutensor.ALGO_DEFAULT,
        int jit_mode=cutensor.JIT_MODE_NONE, int compute_desc=0,
        int ws_pref=cutensor.WORKSPACE_RECOMMENDED):
    """General tensor contraction

    This routine computes the tensor contraction:

        C = alpha * op_A(A) * op_B(B) + beta * op_C(C)

    Args:
        alpha (scalar): Scaling factor for A * B.
        A (cupy.ndarray): Input tensor.
        mode_A (tuple of int/str or Mode): A mode for tensor A.
        B (cupy.ndarray): Input tensor.
        mode_B (tuple of int/str or Mode): A mode for tensor B.
        beta (scalar): Scaling factor for C.
        C (cupy.ndarray): Input/output tensor.
        mode_C (tuple of int/str or Mode): A mode for tensor C.
        algo (cutensorAlgo_t): Allows users to select a specific algorithm.
            ALGO_DEFAULT lets the heuristic choose the algorithm.
            Any value >= 0 selects a specific GEMM-like algorithm and
            deactivates the heuristic. If a specified algorithm is not
            supported, STATUS_NOT_SUPPORTED is returned.
        jit_mode (cutensorJitMode_t): Specify whether and how to use JIT.
        compute_desc (cutensorComputeDescriptor_t): Compute type for the
            intermediate computation.
        ws_pref (cutensorWorksizePreference_t): User preference for the
            workspace of cuTensor.

    Returns:
        out (cupy.ndarray): Output tensor.

    Examples:
        See examples/cutensor/contraction.py
    """
    desc_A = create_tensor_descriptor(A)
    desc_B = create_tensor_descriptor(B)
    desc_C = create_tensor_descriptor(C)
    mode_A = _auto_create_mode(A, mode_A)
    mode_B = _auto_create_mode(B, mode_B)
    mode_C = _auto_create_mode(C, mode_C)
    operator = create_contraction(
        desc_A, mode_A, op_A, desc_B, mode_B, op_B, desc_C, mode_C, op_C,
        compute_desc)
    plan_pref = create_plan_preference(algo=algo, jit_mode=jit_mode)
    ws_size = cutensor.estimateWorkspaceSize(
        _get_handle().ptr, operator.ptr, plan_pref.ptr, ws_pref)
    plan = create_plan(operator, plan_pref, ws_limit=ws_size)
    ws = core._ndarray_init(
        _cupy.ndarray, shape_t(1, ws_size), dtype=_numpy.int8, obj=None)
    scalar_dtype = _get_scalar_dtype(C.dtype)
    out = C
    cutensor.contract(
        _get_handle().ptr, plan.ptr,
        _create_scalar(alpha, scalar_dtype).ptr, A.data.ptr, B.data.ptr,
        _create_scalar(beta, scalar_dtype).ptr, C.data.ptr, out.data.ptr,
        ws.data.ptr, ws_size)
    return out


cdef dict _reduction_compute_descs = {
    # [cutensor_dtype_A][*_C]
    cutensor.R_16F: {cutensor.R_16F: cutensor.COMPUTE_DESC_16F},
    cutensor.R_32F: {cutensor.R_32F: cutensor.COMPUTE_DESC_32F},
    cutensor.R_64F: {cutensor.R_64F: cutensor.COMPUTE_DESC_64F},
    cutensor.C_32F: {cutensor.C_32F: cutensor.COMPUTE_DESC_32F},
    cutensor.C_64F: {cutensor.C_64F: cutensor.COMPUTE_DESC_64F},
}


cpdef OperationDescriptor create_reduction(
        TensorDescriptor desc_A, Mode mode_A, int op_A,
        TensorDescriptor desc_C, Mode mode_C, int op_C,
        int op_reduce, int compute_desc=0):
    """Create a operation descriprot for reduce operation:

        C = alpha * reduce_op(op_A(A)) + beta * op_C(C))

    Args:
        desc_A (TensorDescriptor):
        mode_A (Mode):
        op_A (cutensorOperator_t):
        desc_C (TensorDescriptor):
        mode_C (Mode):
        op_C (cutensorOperator_t):
        op_reduce (cutensorOperator_t):

    Returns:
        (Descriptor): A instance of class OperationDescriptor.
    """
    compute_descs = _reduction_compute_descs
    ct_dtype_A = desc_A.cutensor_dtype
    ct_dtype_C = desc_C.cutensor_dtype
    if not (ct_dtype_A in compute_descs and
            ct_dtype_C in compute_descs[ct_dtype_A]):
        raise ValueError(
            'This combination of cutensor dtype is not supported in reduction '
            '({}, {})'.format(ct_dtype_A, ct_dtype_C))
    if compute_desc == 0:
        compute_desc = compute_descs[ct_dtype_A][ct_dtype_C]
    handle = _get_handle()
    key = (handle.ptr,
           desc_A.ptr, mode_A.data, op_A,
           desc_C.ptr, mode_C.data, op_C,
           op_reduce, compute_desc)
    if key not in _reduction_operators:
        _reduction_operators[key] = OperationDescriptor()
        _reduction_operators[key].create_reduction(
            handle.ptr,
            desc_A.ptr, mode_A.data, op_A,
            desc_C.ptr, mode_C.data, op_C,
            desc_C.ptr, mode_C.data,
            op_reduce, compute_desc)
    return _reduction_operators[key]


def reduction(
        alpha, _ndarray_base A, mode_A,
        beta, _ndarray_base C, mode_C,
        int op_A=cutensor.OP_IDENTITY, int op_C=cutensor.OP_IDENTITY,
        int op_reduce=cutensor.OP_ADD, compute_desc=0):
    """Tensor reduction

    This routine computes the tensor reduction:

        C = alpha * reduce_op(op_A(A)) + beta * op_C(C)

    Args:
        alpha (scalar): Scaling factor for A.
        A (cupy.ndarray): Input tensor.
        mode_A (tuple of int/str or Mode): A mode for tensor A.
        beta (scalar): Scaling factor for C.
        C (cupy.ndarray): Input/output tensor.
        mode_C (tuple of int/str or Mode): A mode for tensor C.
        op_A (cutensorOperator_t): Element-wise unary operator for tensor A.
        op_C (cutensorOperator_t): Element-wise unary operator for tensor C.
        op_reduce (cutensorOperator_t): Binary operator used to reduce A.
        compute_desc (cutensorComputeDescriptor_t): Compute type for the
            intermediate computation.

    Returns:
        out (cupy.ndarray): Output tensor.

    Examples:
        See examples/cutensor/reduction.py
    """
    desc_A = create_tensor_descriptor(A)
    desc_C = create_tensor_descriptor(C)
    mode_A = _auto_create_mode(A, mode_A)
    mode_C = _auto_create_mode(C, mode_C)
    out = C
    operator = create_reduction(
        desc_A, mode_A, op_A, desc_C, mode_C, op_C, op_reduce, compute_desc)
    plan_pref = create_plan_preference()
    ws_size = cutensor.estimateWorkspaceSize(
        _get_handle().ptr, operator.ptr, plan_pref.ptr,
        cutensor.WORKSPACE_RECOMMENDED)
    plan = create_plan(operator, plan_pref, ws_limit=ws_size)
    ws = core._ndarray_init(
        _cupy.ndarray, shape_t(1, ws_size), dtype=_numpy.int8, obj=None)
    cutensor.reduce(
        _get_handle().ptr, plan.ptr,
        _create_scalar(alpha, out.dtype).ptr, A.data.ptr,
        _create_scalar(beta, out.dtype).ptr, C.data.ptr, out.data.ptr,
        ws.data.ptr, ws_size)
    return out


_cutensor_dtypes = [
    # TODO(asi1024): Support float16
    # _numpy.float16,
    _numpy.float32,
    _numpy.float64,
    _numpy.complex64,
    _numpy.complex128,
]


def _try_reduction_routine(
        _ndarray_base x, axis, dtype, _ndarray_base out, keepdims, reduce_op,
        alpha, beta):
    cdef _ndarray_base in_arg, out_arg
    cdef shape_t out_shape
    cdef tuple reduce_axis, out_axis

    if not check_availability('reduction'):
        return None

    if dtype is None:
        dtype = x.dtype

    if dtype not in _cutensor_dtypes:
        return None
    if dtype != x.dtype:
        return None

    if x.ndim == 0:
        return None
    if x.size == 0:
        return None
    if not x._c_contiguous:
        # TODO(asi1024): Support also for F-contiguous array
        return None

    # if x.size == 1 and cutensor.get_version() == 10400:
    #     # WAR: element-1 reduction is buggy
    #     return None

    in_arg = x

    reduce_axis, out_axis = _reduction._get_axis(axis, x.ndim)
    if len(reduce_axis) == 0:
        return None
    out_shape = _reduction._get_out_shape(
        x._shape, reduce_axis, out_axis, keepdims)
    if out is None:
        out = core._ndarray_init(
            _cupy.ndarray, out_shape, dtype=dtype, obj=None)
    elif not internal.vector_equal(out._shape, out_shape):
        # TODO(asi1024): Support broadcast
        return None
    elif out.dtype != dtype:
        return None
    elif not out._c_contiguous:
        # TODO(asi1024): Support also for F-contiguous array
        return None

    if not keepdims:
        out_arg = out.reshape(
            _reduction._get_out_shape(x._shape, reduce_axis, out_axis, False))
    else:
        out_arg = out
    # TODO(kmaeahshi): need to zero out when beta != 0

    # TODO(asi1024): Remove temporary fix
    in_arg._set_contiguous_strides(in_arg.itemsize, True)
    out_arg._set_contiguous_strides(out_arg.itemsize, True)

    try:
        return reduction(
            alpha, in_arg, _create_mode_with_cache(in_arg._shape.size()),
            beta, out_arg, _create_mode_with_cache(out_axis),
            op_reduce=reduce_op)
    except ValueError:
        return None


@cython.profile(False)
cdef inline bint _all_positive(const vector.vector[Py_ssize_t]& args):
    # cuTENSOR requires each stride > 0.
    for i in range(<Py_ssize_t>args.size()):
        if args[i] <= 0:
            return False
    return True


def _try_elementwise_binary_routine(
        _ndarray_base a, _ndarray_base c, dtype, _ndarray_base out, op, alpha,
        gamma):
    if not check_availability('elementwise'):
        return None

    if dtype is None:
        dtype = a.dtype
    if dtype not in _cutensor_dtypes:
        return None

    if not (a.dtype == c.dtype == dtype):
        return None
    if not internal.vector_equal(a._shape, c._shape):
        return None
    if a.size == 0:
        return None
    if not (_all_positive(a._strides) and _all_positive(c._strides)):
        return None
    compute_dtype = a.dtype

    if compute_dtype.kind == 'c' and (
            op == cutensor.OP_MAX or op == cutensor.OP_MIN):
        return None

    if out is None:
        if c._c_contiguous:
            pass
        elif a._c_contiguous:
            a, c = c, a
            alpha, gamma = gamma, alpha
        elif c._f_contiguous:
            pass
        elif a._f_contiguous:
            a, c = c, a
            alpha, gamma = gamma, alpha
        else:
            return None

        # Determine a template object from which we initialize the output when
        # inputs have subclass instances
        def issubclass1(cls, classinfo):
            return issubclass(cls, classinfo) and cls is not classinfo
        subtype = _cupy.ndarray
        template = None
        a_type, c_type = type(a), type(c)
        if issubclass1(a_type, _cupy.ndarray):
            subtype = a_type
            template = a
        elif issubclass1(c_type, _cupy.ndarray):
            subtype = c_type
            template = c

        out = core._create_ndarray_from_shape_strides(
            subtype, c._shape, c._strides, compute_dtype, template)
    elif out.dtype != compute_dtype:
        return None
    elif not internal.vector_equal(c._shape, out._shape):
        return None
    elif not internal.vector_equal(c._strides, out._strides):
        return None
    elif not _all_positive(out._strides):
        return None

    try:
        return elementwise_binary(
            alpha, a, _create_mode_with_cache(a._shape.size()),
            gamma, c, _create_mode_with_cache(c._shape.size()),
            out=out, op_AC=op)
    except ValueError:
        return None

###############################################################################
# MgHandle: This class encapsulates the opaque structure `cutensorMgHandle_t`
# holding cuTENSORMG's library context.
###############################################################################


cdef class MgHandle:
    cdef intptr_t _ptr
    cdef object _devices

    def __init__(self, devices):
        num_devices = len(devices)
        self._ptr = cutensor.createMg(num_devices, devices.ctypes.data)
        self._devices = devices

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyMg(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr

    @property
    def num_devices(self):
        return len(self._devices)

    @property
    def devices(self):
        return self._devices


###############################################################################
# MgTensorDescriptor: This class encapsulates the opaque structure
# `cutensorMgTensorDescriptor_t` representing a multi-GPU tensor descriptor.
###############################################################################

cdef class MgTensorDescriptor:
    cdef intptr_t _ptr
    cdef int _cutensor_dtype
    # It seems that cuda_type is the same as cutensor_type,
    # so just reuse the cutensor_type here

    def __init__(self, intptr_t handle, uint32_t num_modes, intptr_t extent,
                 intptr_t element_stride, intptr_t block_size,
                 intptr_t block_stride, intptr_t device_count,
                 uint32_t num_device, intptr_t devices,
                 int cutensor_dtype):
        self._ptr = cutensor.createMgTensorDescriptor(
            handle, num_modes, extent, element_stride, block_size,
            block_stride, device_count, num_device, devices, cutensor_dtype)
        self._cutensor_dtype = cutensor_dtype

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyMgTensorDescriptor(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr

    @property
    def cutensor_dtype(self):
        return self._cutensor_dtype


###############################################################################
# MgCopyDescriptor: This class encapsulates the opaque structure
# `cutensorMgCopyDescriptor_t` representing a multi-GPU copy descriptor.
###############################################################################

cdef class MgCopyDescriptor:
    cdef intptr_t _ptr

    def __init__(self, intptr_t handle, intptr_t descDst,
                 intptr_t modeDst, intptr_t descSrc,
                 intptr_t modeSrc):
        self._ptr = cutensor.createMgCopyDescriptor(
            handle, descDst, modeDst,
            descSrc, modeSrc)

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyMgCopyDescriptor(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr


###############################################################################
# MgCopyPlan: This class encapsulates the opaque structure
# `cutensorMgCopyPlan_t` representing a plan for multi-GPU copy.
###############################################################################

cdef class MgCopyPlan:
    cdef intptr_t _ptr

    def __init__(self, intptr_t handle, intptr_t desc,
                 intptr_t workspaceDeviceSize,
                 uint64_t workspaceHostSize):
        self._ptr = cutensor.createMgCopyPlan(
            handle, desc, workspaceDeviceSize, workspaceHostSize)

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyMgCopyPlan(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr


###############################################################################
# MgContractionDescriptor: This class encapsulates the opaque structure
# `cutensorMgContractionDescriptor_t` representing a plan for multi-GPU
# contraction.
###############################################################################

cdef class MgContractionDescriptor:
    cdef intptr_t _ptr

    def __init__(self, intptr_t handle, intptr_t descA, intptr_t modeA,
                 intptr_t descB, intptr_t modeB, intptr_t descC,
                 intptr_t modeC, intptr_t descD, intptr_t modeD,
                 int compute):
        self._ptr = cutensor.createMgContractionDescriptor(
            handle, descA, modeA, descB, modeB, descC, modeC, descD,
            modeD, compute)

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyMgContractionDescriptor(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr


###############################################################################
# MgContractionFind: This class encapsulates the opaque structure
# `cutensorMgContractionFind_t` representing a find object to set
# preferences for algorithm selection.
###############################################################################

cdef class MgContractionFind:
    cdef intptr_t _ptr

    def __init__(self, intptr_t handle, int algo):
        self._ptr = cutensor.createMgContractionFind(handle, algo)

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyMgContractionFind(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr


###############################################################################
# MgContractionPlan: This class encapsulates the opaque structure
# `cutensorMgContractionPlan_t` representing a plan for multi-GPU contraction.
###############################################################################

cdef class MgContractionPlan:
    cdef intptr_t _ptr

    def __init__(self, intptr_t handle, intptr_t desc, intptr_t find,
                 intptr_t workspaceDeviceSize,
                 uint64_t workspaceHostSize):
        self._ptr = cutensor.createMgContractionPlan(
            handle, desc, find, workspaceDeviceSize, workspaceHostSize)

    def __dealloc__(self):
        if self._ptr is not 0:
            cutensor.destroyMgContractionPlan(self._ptr)
        self._ptr = <intptr_t>NULL

    @property
    def ptr(self):
        return self._ptr

cpdef MgHandle _get_mg_handle(devices=None):
    if devices is None:
        devices = _numpy.arange(getDeviceCount(), dtype=_numpy.int32)
    else:
        devices = _numpy.asarray(devices, dtype=_numpy.int32)
    key = tuple(devices)
    if key not in _mg_handles:
        _mg_handles[key] = MgHandle(devices)
    return _mg_handles[key]


class ndarray_mg:
    """A wrapper to catch all attributes for multi-GPU array.
    """
    def __init__(self, x, **kwgs):
        """
        Args:
            x (numpy.ndarray or cupy.ndarray or list of cupy.ndarray): Arrays
                to wrap. Non-distributed host(device) storage scheme is meaned
                if x is a numpy(cupy) ndarray. Both pinned memory and pageable
                host memory are allowed.
            **kwargs: Arbitrary keyword arguments.
                extent (Sequence): The size of the multi-GPU array. When x is
                    a non-distributed array, the default value is `x.shape`.
                element_stride (Sequence): The offset (in linear memory)
                    between two adjacent elements in each mode. When x is
                    a non-distributed array, the default value is
                    `x.shape // x.itemsize`.
                block_size (Sequence): The size of a block in each mode. If
                    this parameter is not set, means an unblocked tensor
                    (i.e., each mode only has a single block that is equal to
                    its extent).
                block_stride (Sequence): The offset (in linear memory) between
                    two adjacent blocks in each mode. If this parameter is not
                    set, means a dense block-interleaved layout.
                device_count (Sequence): The number of devices that each mode
                    is distributed across in a block-cyclic fashion.
        """
        _util.experimental("cupy.cutensor.ndarray_mg")
        self.x = x
        if isinstance(x, list):
            self.dtype = x[0].dtype
            self._extent = _numpy.asarray(kwgs['extent'], dtype=_numpy.int64)
            self._element_stride = _numpy.asarray(kwgs['element_stride'],
                                                  dtype=_numpy.int64)
            self.num_devices = len(x)
            self.data = _numpy.empty(self.num_devices, dtype=_numpy.int64)
            self._devices = _numpy.empty(self.num_devices, dtype=_numpy.int64)
            for i in range(self.num_devices):
                self._devices[i] = x[i].device.id
            for i in range(self.num_devices):
                assert isinstance(x[i], _ndarray_base), "Distributed array can only on GPU"  # NOQA
                self.data[i] = x[i].data.ptr
        elif isinstance(x, _numpy.ndarray) or isinstance(x, _ndarray_base):
            self.dtype = x.dtype
            self.num_devices = 1
            self._extent = _numpy.asarray(kwgs.get('extent', x.shape),
                                          dtype=_numpy.int64)
            if 'element_stride' in kwgs:
                self._element_stride = _numpy.asarray(
                    kwgs['element_stride'], dtype=_numpy.int64)
            else:
                self._element_stride = _numpy.asarray(
                    x.strides, dtype=_numpy.int64) // x.itemsize
            self.data = _numpy.empty(1, dtype=_numpy.uint64)
            if isinstance(x, _ndarray_base):
                self.data[0] = x.data.ptr
                self._devices = _numpy.asarray([x.device.id],
                                               dtype=_numpy.int32)
            else:
                self.data[0] = x.ctypes.data
                if is_memory_pinned(x.ctypes.data):
                    self._devices = _numpy.asarray(
                        [cutensor.CUTENSOR_MG_DEVICE_HOST_PINNED],
                        dtype=_numpy.int32)
                else:
                    self._devices = _numpy.asarray(
                        [cutensor.CUTENSOR_MG_DEVICE_HOST],
                        dtype=_numpy.int32)
        else:
            raise ValueError("Unexpected input type")
        self.key = [self.dtype, tuple(self._extent),
                    tuple(self._element_stride),
                    tuple(self._devices)]
        if 'block_size' in kwgs:
            self._block_size = _numpy.asarray(kwgs['block_size'],
                                              dtype=_numpy.int64)
            self.key.append(tuple(self._block_size))
        else:
            self._block_size = None
            self.key.append(None)
        if 'block_stride' in kwgs:
            self._block_stride = _numpy.asarray(kwgs['block_stride'],
                                                dtype=_numpy.int64)
            self.key.append(tuple(self._block_stride))
        else:
            self._block_stride = None
            self.key.append(None)
        if 'device_count' in kwgs:
            self._device_count = _numpy.asarray(kwgs['device_count'],
                                                dtype=_numpy.int32)
            self.key.append(tuple(self._device_count))
        else:
            self._device_count = None
            self.key.append(None)

    @property
    def extent(self):
        return self._extent.ctypes.data

    @property
    def element_stride(self):
        return self._element_stride.ctypes.data

    @property
    def block_size(self):
        if self._block_size is None:
            return 0
        else:
            return self._block_size.ctypes.data

    @property
    def block_stride(self):
        if self._block_stride is None:
            return 0
        else:
            return self._block_stride.ctypes.data

    @property
    def device_count(self):
        if self._device_count is None:
            return 0
        else:
            return self._device_count.ctypes.data

    @property
    def devices(self):
        return self._devices.ctypes.data

    @property
    def ptr(self):
        return self.data.ctypes.data

    @property
    def ndim(self):
        return len(self._extent)


def to_mg_ndarray(x, **kwgs):
    if isinstance(x, ndarray_mg):
        return x
    else:
        return ndarray_mg(x, **kwgs)


cpdef MgTensorDescriptor create_mg_tensor_descriptor(a, devices=None):
    """Create a multi-GPU tensor descriptor

    Args:
        a (cupy.ndarrayMg): The multi-GPU tensor for which a descritpor are
            created.
        devices: device list to determine the multi-GPU handle.

    Returns:
        (MgTensorDescriptor): A instance of class MgTensorDescriptor.
    """
    handle = _get_mg_handle(devices)
    key = (handle.ptr, *a.key)
    if key not in _mg_tensor_descriptors:
        # It seems that cuda_type is the same as cutensor_type,
        # so just reuse the cutensor_type here
        cutensor_dtype = _get_cutensor_dtype(a.dtype)
        _mg_tensor_descriptors[key] = MgTensorDescriptor(
            handle.ptr, a.ndim, a.extent,
            a.element_stride, a.block_size,
            a.block_stride, a.device_count,
            a.num_devices, a.devices, cutensor_dtype)
    return _mg_tensor_descriptors[key]


cpdef MgCopyDescriptor create_mg_copy_descriptor(
        MgTensorDescriptor descDst, Mode modeDst, MgTensorDescriptor descSrc,
        Mode modeSrc, devices=None):
    """Create a multi-GPU copy descriptor.

    Args:
        descDst (MgTensorDescriptor):
        Mode modeDst (Mode):
        descSrc (MgTensorDescriptor):
        Mode modeSrc (Mode):
        devices (Sequence): device list to determine the multi-GPU handle.

    Returns:
        (MgCopyDescriptor): A instance of class MgCopyDescriptor.
    """
    handle = _get_mg_handle(devices)
    key = (handle.ptr, descDst.ptr, modeDst.data, descSrc.ptr, modeSrc.data)
    if key not in _mg_copy_descriptors:
        _mg_copy_descriptors[key] = MgCopyDescriptor(
            handle.ptr, descDst.ptr, modeDst.data,
            descSrc.ptr, modeSrc.data)
    return _mg_copy_descriptors[key]


cpdef MgCopyPlan create_mg_copy_plan(
        MgCopyDescriptor desc, workspaceDeviceSize,
        uint64_t workspaceHostSize, devices=None):
    """Create a copy plan

    Args:
        desc (MgCopyDescriptor):
        workspaceDeviceSize (numpy.ndarray):
        workspaceHostSize (int):

    Returns:
        (MgCopyPlan): A instance of class MgCopyPlan.
    """
    handle = _get_mg_handle(devices)
    key = (handle.ptr, desc.ptr, tuple(workspaceDeviceSize), workspaceHostSize)
    if key not in _mg_copy_plans:
        _mg_copy_plans[key] = MgCopyPlan(
            handle.ptr, desc.ptr, workspaceDeviceSize.ctypes.data,
            workspaceHostSize)
    return _mg_copy_plans[key]


def copyMg(dst, mode_Dst, src, mode_Src, deviceBuf=None, hostBuf=None,
           streams=None, devices=None):
    """Copy according to the MgTensorDescriptors and modes.

    Args:
        dst (numpy.ndarray or cupy.ndarray or ndarray_mg):
        mode_Dst (tuple of int/str or Mode):
        src (numpy.ndarray or cupy.ndarray or ndarray_mg):
        mode_Src (tuple of int/str or Mode):
        deviceBuf (list of cupy.ndarray or None): `None` means allocated
            when needed.
        hostBuf (numpy.ndarray or None): Pinned memory. `None` means allocated
            when needed.
        streams (Sequence of Stream or None): The streams to perform the copy.
        devices (Sequence): device list to determine the multi-GPU handle.

    Returns:
        (None):
    """
    handle = _get_mg_handle(devices)
    dst = to_mg_ndarray(dst)
    src = to_mg_ndarray(src)
    descDst = create_mg_tensor_descriptor(dst, devices)
    descSrc = create_mg_tensor_descriptor(src, devices)
    mode_Dst = create_mode(*mode_Dst)
    mode_Src = create_mode(*mode_Src)
    desc = create_mg_copy_descriptor(descDst, mode_Dst, descSrc, mode_Src,
                                     devices)
    num_devices = handle.num_devices
    deviceBufSize = _numpy.empty(num_devices, dtype=_numpy.int64)
    if deviceBuf is None or hostBuf is None:
        hostBufSize = cutensor.getMgCopyWorkspace(
            handle.ptr, desc.ptr, deviceBufSize.ctypes.data
        )
    if hostBuf is None:
        mem = alloc_pinned_memory(hostBufSize)
        hostBuf = _numpy.ndarray((hostBufSize),
                                 dtype=_numpy.int8, buffer=mem)
    else:
        assert is_memory_pinned(hostBuf.ctypes.data), "Host buffer must be pinned memory"  # NOQA
        hostBufSize = hostBuf.nbytes
    if deviceBuf is None:
        deviceBuf = []
        for idevice, s in zip(handle.devices, deviceBufSize):
            with Device(idevice):
                deviceBuf.append(core._ndarray_init(
                    _cupy.ndarray, shape_t(1, s),
                    dtype=_numpy.int8, obj=None))
    else:
        for i in range(num_devices):
            assert deviceBuf[i].device.id == handle.devices[i], "Device buffer list must be on devices "+ str(handle.devices)   # NOQA
            deviceBufSize[i] = deviceBuf[i].nbytes
    hostBufptr = 0 if hostBufSize == 0 else hostBuf.ctypes.data
    deviceBufptr = _numpy.zeros(num_devices, dtype=_numpy.uint64)
    for i in range(num_devices):
        if deviceBufSize[i] != 0:
            deviceBufptr[i] = deviceBuf[i].data.ptr
        else:
            deviceBufptr[i] = 0
    plan = create_mg_copy_plan(desc, deviceBufSize, hostBufSize)
    for i in range(num_devices):
        deviceBufptr[i] = deviceBuf[i].data.ptr
    streamPtrs = _numpy.zeros(num_devices, dtype=_numpy.int64)
    if streams is None:
        for i in range(num_devices):
            streamPtrs[i] = stream_module.get_stream_ptr(handle.devices[i])
    else:
        for i in range(num_devices):
            streamPtrs[i] = streams[i].ptr
    cutensor._copyMg(handle.ptr, plan.ptr, dst.ptr, src.ptr,
                     deviceBufptr.ctypes.data, hostBufptr,
                     streamPtrs.ctypes.data)


def copyMgWorkspace(dst, mode_Dst, src, mode_Src, devices=None):
    """Getting the workspace needed to perform the multi-GPU copy.

    Args:
        dst (numpy.ndarray or cupy.ndarray or ndarray_mg):
        mode_Dst (tuple of int/str or Mode):
        src (numpy.ndarray or cupy.ndarray or ndarray_mg):
        mode_Src (tuple of int/str or Mode):
        devices (Sequence): device list to determine the multi-GPU handle.

    Returns:
        hostBufSize (Int): Host buffer needed in bytes.
        deviceBufSize (numpy.ndarray): Device buffer needed in bytes.
    """
    handle = _get_mg_handle(devices)
    dst = to_mg_ndarray(dst)
    src = to_mg_ndarray(src)
    descDst = create_mg_tensor_descriptor(dst, devices)
    descSrc = create_mg_tensor_descriptor(src, devices)
    mode_Dst = create_mode(*mode_Dst)
    mode_Src = create_mode(*mode_Src)
    desc = create_mg_copy_descriptor(descDst, mode_Dst, descSrc, mode_Src,
                                     devices)
    num_devices = handle.num_devices

    deviceBufSize = _numpy.empty(num_devices, dtype=_numpy.int64)
    hostBufSize = cutensor.getMgCopyWorkspace(
        handle.ptr, desc.ptr, deviceBufSize.ctypes.data
    )
    return hostBufSize, deviceBufSize


cpdef MgContractionDescriptor create_mg_contraction_descriptor(
        MgTensorDescriptor descA, Mode modeA,
        MgTensorDescriptor descB, Mode modeB,
        MgTensorDescriptor descC, Mode modeC,
        MgTensorDescriptor descD, Mode modeD,
        int compute_desc=0, devices=None):
    """Create a multi-GPU contraction descriptor.

    Args:
        descA (MgTensorDescriptor):
        ModeA (Mode):
        descB (MgTensorDescriptor):
        ModeB (Mode):
        descC (MgTensorDescriptor):
        ModeC (Mode):
        descD (MgTensorDescriptor):
        ModeD (Mode):
        compute_desc (cutensorComputeDescriptor_t): Datatype used in
            contraction. Default value is `0`, means determined by dtype
            of tensors. Other choices are:
                COMPUTE_DESC_16F = 1
                COMPUTE_DESC_16BF = 1024
                COMPUTE_DESC_TF32 = 4096
                COMPUTE_DESC_3xTF32 = 8192
                COMPUTE_DESC_32F = 4
                COMPUTE_DESC_64F = 16
        devices (Sequence): device list to determine the multi-GPU handle.

    Returns:
        (MgContractionDescriptor): A instance of class
            MgContractionDescriptor.
    """
    compute_descs = _contraction_compute_descs
    ct_dtype_A = descA.cutensor_dtype
    ct_dtype_B = descB.cutensor_dtype
    ct_dtype_C = descC.cutensor_dtype
    if not (ct_dtype_A in compute_descs and
            ct_dtype_B in compute_descs[ct_dtype_A] and
            ct_dtype_C in compute_descs[ct_dtype_A][ct_dtype_B]):
        raise ValueError(
            'This cutensor dtype combination is not supported in contraction'
            ' ({}, {}, {})'.format(ct_dtype_A, ct_dtype_B, ct_dtype_C))
    if compute_desc == 0:
        compute_desc = compute_descs[ct_dtype_A][ct_dtype_B][ct_dtype_C]
    handle = _get_mg_handle(devices)
    key = (handle.ptr,
           descA.ptr, modeA.data,
           descB.ptr, modeB.data,
           descC.ptr, modeC.data,
           descD.ptr, modeD.data,
           compute_desc)
    if key not in _mg_contraction_descriptors:
        _mg_contraction_descriptors[key]=MgContractionDescriptor(
            handle.ptr,
            descA.ptr, modeA.data,
            descB.ptr, modeB.data,
            descC.ptr, modeC.data,
            descD.ptr, modeD.data, compute_desc)
    return _mg_contraction_descriptors[key]

cpdef MgContractionFind create_mg_contraction_find(
        int algo=cutensor.CUTENSORMG_ALGO_DEFAULT, devices=None):
    """Create a multi-GPU contraction find descriptor.

    Args:
        algo (cutensorMgAlgo_t): Currently, only the default value
            `CUTENSORMG_ALGO_DEFAULT` is allowed.
        devices (Sequence): device list to determine the multi-GPU handle.

    Returns:
        (MgContractionFind): A instance of class MgContractionFind.
    """
    handle = _get_mg_handle(devices)
    key = (handle.ptr, algo)
    if key not in _mg_contraction_finds:
        _mg_contraction_finds[key]=MgContractionFind(
            handle.ptr, algo)
    return _mg_contraction_finds[key]

cpdef MgContractionPlan create_mg_contraction_plan(
        MgContractionDescriptor desc, MgContractionFind find,
        workspaceDeviceSize, int64_t workspaceHostSize, devices=None):
    """Create a multi-GPU contraction find descriptor.

    Args:
        desc (MgContractionDescriptor):
        find (MgContractionFind):
        workspaceDeviceSize (numpy.ndarray):
        workspaceHostSize (int):
        devices (Sequence): device list to determine the multi-GPU handle.

    Returns:
        (MgContractionPlan): A instance of class MgContractionPlan.
    """
    handle = _get_mg_handle(devices)
    key = (handle.ptr, desc.ptr, find.ptr, tuple(workspaceDeviceSize),
           workspaceHostSize)
    if key not in _mg_contraction_plans:
        _mg_contraction_plans[key]=MgContractionPlan(
            handle.ptr, desc.ptr, find.ptr, workspaceDeviceSize.ctypes.data,
            workspaceHostSize)
    return _mg_contraction_plans[key]


def contractionMg(alpha, A, modeA, B, modeB, beta, C, modeC,
                  D=None, modeD=None, int compute_desc=0,
                  int ws_pref=cutensor.WORKSPACE_RECOMMENDED,
                  deviceBuf=None, hostBuf=None, streams=None,
                  devices=None):
    """Multi-GPU contraction.

    Args:
        alpha (scalar): Scaling factor for A * B.
        A (numpy.ndarray or cupy.ndarray or ndarray_mg): Input tensor.
        modeA (tuple of int/str or Mode): A mode for tensor A.
        B (numpy.ndarray or cupy.ndarray or ndarray_mg): Input tensor.
        modeB (tuple of int/str or Mode): A mode for tensor B.
        beta (scalar): Scaling factor for C.
        C (numpy.ndarray or cupy.ndarray or ndarray_mg): Input tensor.
        modeC (tuple of int/str or Mode): A mode for tensor C.
        D (numpy.ndarray or cupy.ndarray or ndarray_mg or None): Output
            tensor. Default value `None` means using C for output.
        modeD (tuple of int/str or Mode or None): A mode for tensor D.
            Default value `None` means the same as modeC.
        compute_desc (cutensorComputeDescriptor_t): Datatype used in
            contraction. Default value is `0`, means determined by dtype
            of tensors. Other choices are:
                COMPUTE_DESC_16F = 1
                COMPUTE_DESC_16BF = 1024
                COMPUTE_DESC_TF32 = 4096
                COMPUTE_DESC_3xTF32 = 8192
                COMPUTE_DESC_32F = 4
                COMPUTE_DESC_64F = 16
        ws_pref (cutensorWorksizePreference_t): User preference for the
            workspace of cuTensor. Defaule value is
            'cutensor.WORKSPACE_RECOMMENDED'.
        deviceBuf (list of cupy.ndarray or None): `None` means allocated
            when needed.
        hostBuf (numpy.ndarray or None): Pinned memory. `None` means allocated
            when needed.
        streams (Sequence of Stream or None): The streams to perform the
            contraction.
        devices (Sequence): device list to determine the multi-GPU handle.

    Returns:
        (None):
    """
    handle = _get_mg_handle(devices)
    A = to_mg_ndarray(A)
    B = to_mg_ndarray(B)
    C = to_mg_ndarray(C)
    descA = create_mg_tensor_descriptor(A, devices)
    descB = create_mg_tensor_descriptor(B, devices)
    descC = create_mg_tensor_descriptor(C, devices)
    modeA = create_mode(*modeA)
    modeB = create_mode(*modeB)
    modeC = create_mode(*modeC)
    if D is None:
        D = C
        descD = descC
        modeD = modeC
    else:
        D = to_mg_ndarray(D)
        descD = create_mg_tensor_descriptor(D, devices)
        modeD = create_mode(*modeD)
    desc = create_mg_contraction_descriptor(
        descA, modeA, descB, modeB, descC, modeC, descD, modeD,
        compute_desc, devices)
    find = create_mg_contraction_find(devices=devices)
    num_devices = handle.num_devices
    deviceBufSize = _numpy.zeros((num_devices,), dtype=_numpy.int64)
    if deviceBuf is None or hostBuf is None:
        hostBufSize = cutensor.getMgContractionWorkspace(
            handle.ptr, desc.ptr, find.ptr, ws_pref,
            deviceBufSize.ctypes.data)
    if hostBuf is None:
        mem = alloc_pinned_memory(hostBufSize)
        hostBuf = _numpy.ndarray((hostBufSize), dtype=_numpy.int8, buffer=mem)
    else:
        assert is_memory_pinned(hostBuf.ctypes.data), "Host buffer must be pinned memory"  # NOQA
        hostBufSize = hostBuf.nbytes
    if deviceBuf is None:
        deviceBuf = []
        for idevice, s in zip(handle.devices, deviceBufSize):
            with Device(idevice):
                deviceBuf.append(core._ndarray_init(
                    _cupy.ndarray, shape_t(1, s),
                    dtype=_numpy.int8, obj=None))
    else:
        for i in range(num_devices):
            assert deviceBuf[i].device.id == handle.devices[i], "Device buffer list must be on devices "+ str(handle.devices)   # NOQA
            deviceBufSize[i] = deviceBuf[i].nbytes
    hostBufptr = 0 if hostBufSize == 0 else hostBuf.ctypes.data
    deviceBufptr = _numpy.zeros(num_devices, dtype=_numpy.uint64)
    for i in range(num_devices):
        if deviceBufSize[i] != 0:
            deviceBufptr[i] = deviceBuf[i].data.ptr
        else:
            deviceBufptr[i] = 0
    plan = create_mg_contraction_plan(
        desc, find, deviceBufSize, hostBufSize, devices=devices)
    streamPtrs = _numpy.zeros(num_devices, dtype=_numpy.int64)
    if streams is None:
        for i in range(num_devices):
            streamPtrs[i] = stream_module.get_stream_ptr(handle.devices[i])
    else:
        for i in range(num_devices):
            streamPtrs[i] = streams[i].ptr
    scalar_dtype = _get_scalar_dtype(C.dtype)
    cutensor._contractMg(
        handle.ptr, plan.ptr, _create_scalar(alpha, scalar_dtype).ptr, A.ptr,
        B.ptr, _create_scalar(beta, scalar_dtype).ptr, C.ptr, D.ptr,
        deviceBufptr.ctypes.data, hostBufptr, streamPtrs.ctypes.data)


def contractionMgWorkspace(
        alpha, A, modeA, B, modeB, beta, C, modeC, D=None, modeD=None,
        int compute_desc=0, int ws_pref=cutensor.WORKSPACE_RECOMMENDED,
        devices=None):
    """Getting the workspace needed to perform the multi-GPU contraction.

    Args:
        alpha (scalar): Scaling factor for A * B.
        A (numpy.ndarray or cupy.ndarray or ndarray_mg): Input tensor.
        modeA (tuple of int/str or Mode): A mode for tensor A.
        B (numpy.ndarray or cupy.ndarray or ndarray_mg): Input tensor.
        modeB (tuple of int/str or Mode): A mode for tensor B.
        beta (scalar): Scaling factor for C.
        C (numpy.ndarray or cupy.ndarray or ndarray_mg): Input tensor.
        modeC (tuple of int/str or Mode): A mode for tensor C.
        D (numpy.ndarray or cupy.ndarray or ndarray_mg or None): Output
            tensor. Default value `None` means using C for output.
        modeD (tuple of int/str or Mode or None): A mode for tensor D.
            Default value `None` means the same as modeC.
        compute_desc (cutensorComputeDescriptor_t): Datatype used in
            contraction. Default value is `0`, means determined by dtype
            of tensors. Other choices are:
                COMPUTE_DESC_16F = 1
                COMPUTE_DESC_16BF = 1024
                COMPUTE_DESC_TF32 = 4096
                COMPUTE_DESC_3xTF32 = 8192
                COMPUTE_DESC_32F = 4
                COMPUTE_DESC_64F = 16
        ws_pref (cutensorWorksizePreference_t): User preference for the
            workspace of cuTensor. Defaule value is
            'cutensor.WORKSPACE_RECOMMENDED'.
        devices (Sequence): device list to determine the multi-GPU handle.

    Returns:
        hostBufSize (Int): Host buffer needed in bytes.
        deviceBufSize (numpy.ndarray): Device buffer needed in bytes.
    """
    handle = _get_mg_handle(devices)
    A = to_mg_ndarray(A)
    B = to_mg_ndarray(B)
    C = to_mg_ndarray(C)
    descA = create_mg_tensor_descriptor(A, devices)
    descB = create_mg_tensor_descriptor(B, devices)
    descC = create_mg_tensor_descriptor(C, devices)
    modeA = create_mode(*modeA)
    modeB = create_mode(*modeB)
    modeC = create_mode(*modeC)
    if D is None:
        D = C
        descD = descC
        modeD = modeC
    else:
        D = to_mg_ndarray(D)
        descD = create_mg_tensor_descriptor(D, devices)
        modeD = create_mode(*modeD)
    desc = create_mg_contraction_descriptor(
        descA, modeA, descB, modeB, descC, modeC, descD, modeD,
        compute_desc, devices)
    find = create_mg_contraction_find(devices=devices)
    num_devices = handle.num_devices
    deviceBufSize = _numpy.zeros((num_devices,), dtype=_numpy.int64)
    hostBufSize = cutensor.getMgContractionWorkspace(
        handle.ptr, desc.ptr, find.ptr, ws_pref,
        deviceBufSize.ctypes.data)
    return hostBufSize, deviceBufSize
