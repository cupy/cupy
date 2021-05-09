import numpy as _numpy
import warnings as _warnings

import cupy as _cupy

from libc.stdint cimport intptr_t, uint32_t, uint64_t
from cupy._core._carray cimport shape_t
from cupy._core.core cimport ndarray
from cupy._core cimport internal
from cupy_backends.cuda.libs.cutensor cimport Handle
from cupy_backends.cuda.libs.cutensor cimport TensorDescriptor
from cupy_backends.cuda.libs.cutensor cimport ContractionDescriptor
from cupy_backends.cuda.libs.cutensor cimport ContractionFind
from cupy_backends.cuda.libs.cutensor cimport ContractionPlan

from cupy._core cimport core
from cupy._core cimport _dtype
from cupy._core cimport _routines_linalg as _linalg
from cupy._core cimport _reduction
from cupy.cuda cimport device
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda.libs cimport cutensor


cdef dict _handles = {}
cdef dict _tensor_descriptors = {}
cdef dict _contraction_descriptors = {}
cdef dict _contraction_finds = {}
cdef dict _contraction_plans = {}
cdef dict _modes = {}
cdef dict _scalars = {}
cdef dict _dict_contraction = {
    'eee': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.R_MIN_32F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.R_MIN_32F},
    'fff': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.R_MIN_32F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.R_MIN_32F,
            _linalg.COMPUTE_TYPE_FP16: cutensor.R_MIN_16F},
    'ddd': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.R_MIN_64F,
            _linalg.COMPUTE_TYPE_FP64: cutensor.R_MIN_64F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.R_MIN_32F},
    'FFF': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.C_MIN_32F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.C_MIN_32F},
    'DDD': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.C_MIN_64F,
            _linalg.COMPUTE_TYPE_FP64: cutensor.C_MIN_64F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.C_MIN_32F},
    'dDD': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.C_MIN_64F,
            _linalg.COMPUTE_TYPE_FP64: cutensor.C_MIN_64F},
    'DdD': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.C_MIN_64F,
            _linalg.COMPUTE_TYPE_FP64: cutensor.C_MIN_64F},
}
cdef dict _dict_contraction_v10200 = {
    'eee': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.COMPUTE_32F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.COMPUTE_32F,
            _linalg.COMPUTE_TYPE_FP16: cutensor.COMPUTE_16F},
    'fff': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.COMPUTE_32F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.COMPUTE_32F,
            _linalg.COMPUTE_TYPE_TF32: cutensor.COMPUTE_TF32,
            _linalg.COMPUTE_TYPE_FP16: cutensor.COMPUTE_16F},
    'ddd': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.COMPUTE_64F,
            _linalg.COMPUTE_TYPE_FP64: cutensor.COMPUTE_64F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.COMPUTE_32F},
    'FFF': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.COMPUTE_32F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.COMPUTE_32F,
            _linalg.COMPUTE_TYPE_TF32: cutensor.COMPUTE_TF32,
            _linalg.COMPUTE_TYPE_FP16: cutensor.COMPUTE_16F},
    'DDD': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.COMPUTE_64F,
            _linalg.COMPUTE_TYPE_FP64: cutensor.COMPUTE_64F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.COMPUTE_32F},
    'dDD': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.COMPUTE_64F,
            _linalg.COMPUTE_TYPE_FP64: cutensor.COMPUTE_64F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.COMPUTE_32F},
    'DdD': {_linalg.COMPUTE_TYPE_DEFAULT: cutensor.COMPUTE_64F,
            _linalg.COMPUTE_TYPE_FP64: cutensor.COMPUTE_64F,
            _linalg.COMPUTE_TYPE_FP32: cutensor.COMPUTE_32F},
}
cdef dict _dict_compute_type = {
    'e': cutensor.R_MIN_16F,
    'f': cutensor.R_MIN_32F,
    'd': cutensor.R_MIN_64F,
    'F': cutensor.C_MIN_32F,
    'D': cutensor.C_MIN_64F,
}
cdef dict _dict_compute_type_v10200 = {
    'e': cutensor.COMPUTE_16F,
    'f': cutensor.COMPUTE_32F,
    'd': cutensor.COMPUTE_64F,
    'F': cutensor.COMPUTE_32F,
    'D': cutensor.COMPUTE_64F,
}


cdef class Mode(object):

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
        return 'mode([' + ', '.join(self._array) + '])'


cdef class _Scalar(object):

    cdef:
        object _array
        readonly intptr_t ptr

    def __init__(self, value, dtype):
        self._array = _numpy.asarray(value, dtype=dtype)
        self.ptr = self._array.ctypes.data

    def __repr__(self):
        return self._array.item()


cdef Handle _get_handle():
    cdef Handle handle
    cdef int dev = device.get_device_id()
    if dev not in _handles:
        handle = Handle()
        cutensor.init(handle)
        _handles[dev] = handle
        return handle
    return _handles[dev]


cdef int _get_cutensor_compute_type(numpy_dtype) except -1:
    if cutensor.get_version() >= 10200:
        # version 1.2.0 or later
        dict_compute_type = _dict_compute_type_v10200
    else:
        dict_compute_type = _dict_compute_type
    key = _numpy.dtype(numpy_dtype).char
    if key not in dict_compute_type:
        raise TypeError('Dtype {} is not supported'.format(numpy_dtype))
    return dict_compute_type[key]


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
    return Mode(integer_mode)


cdef inline Mode _auto_create_mode(ndarray array, mode):
    if not isinstance(mode, Mode):
        mode = create_mode(*mode)
    if array.ndim != mode.ndim:
        raise ValueError(
            'ndim mismatch: {} != {}'.format(array.ndim, mode.ndim))
    return mode


cdef inline _set_compute_dtype(array_dtype, compute_dtype=None):
    if compute_dtype is None:
        if array_dtype == _numpy.float16:
            compute_dtype = _numpy.float32
        else:
            compute_dtype = array_dtype
    return compute_dtype


cdef inline _Scalar _create_scalar(scale, dtype):
    cdef _Scalar scalar
    key = (scale, dtype)
    if key in _scalars:
        scalar = _scalars[key]
    else:
        scalar = _Scalar(scale, dtype)
        _scalars[key] = scalar
    return scalar


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


cpdef TensorDescriptor create_tensor_descriptor(
        ndarray a, int uop=cutensor.OP_IDENTITY, Handle handle=None):
    """Create a tensor descriptor

    Args:
        a (cupy.ndarray): tensor for which a descritpor are created.
        uop (cutensorOperator_t): unary operator that will be applied to each
            element of the corresponding tensor in a lazy fashion (i.e., the
            algorithm uses this tensor as its operand only once). The
            original data of this tensor remains unchanged.

    Returns:
        (Descriptor): A instance of class Descriptor which holds a pointer to
            tensor descriptor and its destructor.
    """
    if handle is None:
        handle = _get_handle()
    key = (handle.ptr, a.dtype, tuple(a.shape), tuple(a.strides), uop)
    if key in _tensor_descriptors:
        desc = _tensor_descriptors[key]
        return desc
    num_modes = a.ndim
    extent = _numpy.array(a.shape, dtype=_numpy.int64)
    stride = _numpy.array(a.strides, dtype=_numpy.int64) // a.itemsize
    cuda_dtype = _dtype.to_cuda_dtype(a.dtype, is_half_allowed=True)
    desc = TensorDescriptor()
    cutensor.initTensorDescriptor(
        handle, desc, num_modes, extent.ctypes.data, stride.ctypes.data,
        cuda_dtype, uop)
    _tensor_descriptors[key] = desc
    return desc


def elementwise_trinary(
        alpha, ndarray A, TensorDescriptor desc_A, mode_A,
        beta, ndarray B, TensorDescriptor desc_B, mode_B,
        gamma, ndarray C, TensorDescriptor desc_C, mode_C,
        ndarray out=None,
        op_AB=cutensor.OP_ADD, op_ABC=cutensor.OP_ADD, compute_dtype=None):
    """Element-wise tensor operation for three input tensors

    This function performs a element-wise tensor operation of the form:

        D_{Pi^C(i_0,i_1,...,i_nc)} =
            op_ABC(op_AB(alpha * uop_A(A_{Pi^A(i_0,i_1,...,i_na)}),
                         beta  * uop_B(B_{Pi^B(i_0,i_1,...,i_nb)})),
                         gamma * uop_C(C_{Pi^C(i_0,i_1,...,i_nc)}))

    See cupy/cuda/cutensor.elementwiseTrinary() for details.

    Args:
        alpha (scalar): Scaling factor for tensor A.
        A (cupy.ndarray): Input tensor.
        desc_A (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor A.
        mode_A (cutensor.Mode): A mode object created by `create_mode`.
        beta (scalar): Scaling factor for tensor B.
        B (cupy.ndarray): Input tensor.
        desc_B (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor B.
        mode_B (cutensor.Mode): A mode object created by `create_mode`.
        gamma (scalar): Scaling factor for tensor C.
        C (cupy.ndarray): Input tensor.
        desc_C (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor C.
        mode_C (cutensor.Mode): A mode object created by `create_mode`.
        out (cupy.ndarray): Output tensor.
        op_AB (cutensorOperator_t): Element-wise binary operator.
        op_ABC (cutensorOperator_t): Element-wise binary operator.
        compute_dtype (numpy.dtype): Compute type for the intermediate
            computation.

    Returns:
        out (cupy.ndarray): Output tensor.

    Examples:
        See examples/cutensor/elementwise_trinary.py
    """
    if not (A.dtype == B.dtype == C.dtype):
        raise ValueError(
            'dtype mismatch: ({}, {}, {})'.format(A.dtype, B.dtype, C.dtype))
    if not (A._c_contiguous and B._c_contiguous and C._c_contiguous):
        raise ValueError('The inputs should be contiguous arrays.')

    if out is None:
        out = core._ndarray_init(C._shape, dtype=C.dtype)
    elif C.dtype != out.dtype:
        raise ValueError('dtype mismatch: {} != {}'.format(C.dtype, out.dtype))
    elif not internal.vector_equal(C._shape, out._shape):
        raise ValueError('shape mismatch: {} != {}'.format(C.shape, out.shape))
    elif not out._c_contiguous:
        raise ValueError('`out` should be a contiguous array.')

    if compute_dtype is None:
        compute_dtype = A.dtype

    return _elementwise_trinary_impl(
        _get_handle(),
        _create_scalar(alpha, compute_dtype),
        A, desc_A, _auto_create_mode(A, mode_A),
        _create_scalar(beta, compute_dtype),
        B, desc_B, _auto_create_mode(B, mode_B),
        _create_scalar(gamma, compute_dtype),
        C, desc_C, _auto_create_mode(C, mode_C),
        out, op_AB, op_ABC,
        _dtype.to_cuda_dtype(compute_dtype, is_half_allowed=True))


cdef inline ndarray _elementwise_trinary_impl(
        Handle handle,
        _Scalar alpha, ndarray A, TensorDescriptor desc_A, Mode mode_A,
        _Scalar beta, ndarray B, TensorDescriptor desc_B, Mode mode_B,
        _Scalar gamma, ndarray C, TensorDescriptor desc_C, Mode mode_C,
        ndarray out, int op_AB, int op_ABC, int compute_type):
    cutensor.elementwiseTrinary(
        handle,
        alpha.ptr, A.data.ptr, desc_A, mode_A.data,
        beta.ptr, B.data.ptr, desc_B, mode_B.data,
        gamma.ptr, C.data.ptr, desc_C, mode_C.data,
        out.data.ptr, desc_C, mode_C.data,
        op_AB, op_ABC, compute_type)
    return out


def elementwise_binary(
        alpha, ndarray A, TensorDescriptor desc_A, mode_A,
        gamma, ndarray C, TensorDescriptor desc_C, mode_C,
        ndarray out=None,
        op_AC=cutensor.OP_ADD, compute_dtype=None):
    """Element-wise tensor operation for two input tensors

    This function performs a element-wise tensor operation of the form:

        D_{Pi^C(i_0,i_1,...,i_n)} =
            op_AC(alpha * uop_A(A_{Pi^A(i_0,i_1,...,i_n)}),
                  gamma * uop_C(C_{Pi^C(i_0,i_1,...,i_n)}))

    See elementwise_trinary() for details.

    Examples:
        See examples/cutensor/elementwise_binary.py
    """
    if A.dtype != C.dtype:
        raise ValueError('dtype mismatch: {} != {}'.format(A.dtype, C.dtype))
    if not (A._c_contiguous and C._c_contiguous):
        raise ValueError('The inputs should be contiguous arrays.')

    if out is None:
        out = core._ndarray_init(C._shape, dtype=C.dtype)
    elif C.dtype != out.dtype:
        raise ValueError('dtype mismatch: {} != {}'.format(C.dtype, out.dtype))
    elif not internal.vector_equal(C._shape, out._shape):
        raise ValueError('shape mismatch: {} != {}'.format(C.shape, out.shape))
    elif not out._c_contiguous:
        raise ValueError('`out` should be a contiguous array.')

    if compute_dtype is None:
        compute_dtype = A.dtype

    return _elementwise_binary_impl(
        _get_handle(),
        _create_scalar(alpha, compute_dtype),
        A, desc_A, _auto_create_mode(A, mode_A),
        _create_scalar(gamma, compute_dtype),
        C, desc_C, _auto_create_mode(A, mode_C),
        out, op_AC, _dtype.to_cuda_dtype(compute_dtype, is_half_allowed=True))


cdef inline ndarray _elementwise_binary_impl(
        Handle handle,
        _Scalar alpha, ndarray A, TensorDescriptor desc_A, Mode mode_A,
        _Scalar gamma, ndarray C, TensorDescriptor desc_C, Mode mode_C,
        ndarray out, int op_AC, int compute_type):
    cutensor.elementwiseBinary(
        handle,
        alpha.ptr, A.data.ptr, desc_A, mode_A.data,
        gamma.ptr, C.data.ptr, desc_C, mode_C.data,
        out.data.ptr, desc_C, mode_C.data,
        op_AC, compute_type)
    return out


cdef inline ContractionDescriptor _create_contraction_descriptor(
        Handle handle,
        ndarray A, TensorDescriptor desc_A, Mode mode_A,
        ndarray B, TensorDescriptor desc_B, Mode mode_B,
        ndarray C, TensorDescriptor desc_C, Mode mode_C,
        int cutensor_compute_type):
    """Create a contraction descriptor"""
    cdef uint32_t alignment_req_A = cutensor.getAlignmentRequirement(
        handle, A.data.ptr, desc_A)
    cdef uint32_t alignment_req_B = cutensor.getAlignmentRequirement(
        handle, B.data.ptr, desc_B)
    cdef uint32_t alignment_req_C = cutensor.getAlignmentRequirement(
        handle, C.data.ptr, desc_C)
    cdef ContractionDescriptor desc

    key = (handle.ptr, cutensor_compute_type,
           desc_A.ptr, mode_A.data, alignment_req_A,
           desc_B.ptr, mode_B.data, alignment_req_B,
           desc_C.ptr, mode_C.data, alignment_req_C)
    if key in _contraction_descriptors:
        desc = _contraction_descriptors[key]
        return desc

    desc = ContractionDescriptor()
    cutensor.initContractionDescriptor(
        handle,
        desc,
        desc_A, mode_A.data, alignment_req_A,
        desc_B, mode_B.data, alignment_req_B,
        desc_C, mode_C.data, alignment_req_C,
        desc_C, mode_C.data, alignment_req_C,
        cutensor_compute_type)
    _contraction_descriptors[key] = desc
    return desc


cdef inline ContractionFind _create_contraction_find(Handle handle, int algo):
    """Create a contraction find"""
    cdef ContractionFind find

    key = (handle.ptr, algo)
    if key in _contraction_finds:
        find = _contraction_finds[key]
    else:
        find = ContractionFind()
        cutensor.initContractionFind(handle, find, algo)
        _contraction_finds[key] = find
    return find


cdef inline ContractionPlan _create_contraction_plan(
        Handle handle,
        ContractionDescriptor desc, ContractionFind find, uint64_t ws_size):
    """Create a contraction plan"""
    cdef ContractionPlan plan

    key = (handle.ptr, desc.ptr, find.ptr, ws_size)
    if key in _contraction_plans:
        plan = _contraction_plans[key]
    else:
        plan = ContractionPlan()
        cutensor.initContractionPlan(handle, plan, desc, find, ws_size)
        _contraction_plans[key] = plan
    return plan


cdef _get_contraction_compute_type(a_dtype, b_dtype, out_dtype, compute_dtype):
    key = a_dtype.char + b_dtype.char + out_dtype.char
    if cutensor.get_version() >= 10200:
        # version 1.2.0 or later
        dict_contraction = _dict_contraction_v10200
    else:
        dict_contraction = _dict_contraction
    if key not in dict_contraction:
        raise ValueError('Un-supported dtype combinations: ({}, {}, {})'.
                         format(a_dtype, b_dtype, out_dtype))
    compute_capability = int(device.get_compute_capability())
    if compute_capability < 70 and 'e' in key:
        raise ValueError('FP16 dtype is only supported on GPU with compute '
                         'capability 7.0 or higher.')
    if compute_dtype is None:
        compute_type = _linalg.get_compute_type(out_dtype)
    else:
        compute_dtype = _numpy.dtype(compute_dtype)
        if compute_dtype.char == 'e':
            compute_type = _linalg.COMPUTE_TYPE_FP16
        elif compute_dtype.char in 'fF':
            compute_type = _linalg.COMPUTE_TYPE_FP32
        elif compute_dtype.char in 'dD':
            compute_type = _linalg.COMPUTE_TYPE_FP64
        else:
            raise ValueError('Un-supported dtype: {}'.format(compute_dtype))
    if compute_type in dict_contraction[key]:
        cutensor_compute_type = dict_contraction[key][compute_type]
        if not (compute_capability < 70 and
                cutensor_compute_type in (cutensor.R_MIN_16F,
                                          cutensor.C_MIN_16F,
                                          cutensor.COMPUTE_16F)):
            return cutensor_compute_type
    _warnings.warn('Use of compute type ({}) for the dtype combination '
                   '({}, {}, {}) is not supported in cuTENSOR contraction on '
                   'GPU with compute capability ({}). Default compute type '
                   'will be used instead.'.
                   format(_linalg.compute_type_to_str(compute_type),
                          a_dtype, b_dtype, out_dtype, compute_capability))
    return dict_contraction[key][_linalg.COMPUTE_TYPE_DEFAULT]


cdef _get_scalar_dtype(out_dtype):
    if out_dtype == _numpy.float16:
        return _numpy.float32
    else:
        return out_dtype


def contraction(
        alpha, ndarray A, TensorDescriptor desc_A, mode_A,
        ndarray B, TensorDescriptor desc_B, mode_B,
        beta, ndarray C, TensorDescriptor desc_C, mode_C,
        compute_dtype=None,
        int algo=cutensor.ALGO_DEFAULT,
        int ws_pref=cutensor.WORKSPACE_RECOMMENDED):
    """General tensor contraction

    This routine computes the tensor contraction:

        C = alpha * uop_A(A) * uop_B(B) + beta * uop_C(C)

    See cupy/cuda/cutensor.contraction for details.

    Args:
        alpha (scalar): Scaling factor for A * B.
        A (cupy.ndarray): Input tensor.
        desc_A (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor A.
        mode_A (cutensor.Mode): A mode object created by `create_mode`.
        B (cupy.ndarray): Input tensor.
        desc_B (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor B.
        mode_B (cutensor.Mode): A mode object created by `create_mode`.
        beta (scalar): Scaling factor for C.
        C (cupy.ndarray): Input/output tensor.
        desc_C (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor C.
        mode_C (cutensor.Mode): A mode object created by `create_mode`.
        compute_dtype (numpy.dtype): Compute type for the intermediate
            computation.
        algo (cutensorAlgo_t): Allows users to select a specific algorithm.
            ALGO_DEFAULT lets the heuristic choose the algorithm.
            Any value >= 0 selects a specific GEMM-like algorithm and
            deactivates the heuristic. If a specified algorithm is not
            supported, STATUS_NOT_SUPPORTED is returned.
        ws_pref (cutensorWorksizePreference_t): User preference for the
            workspace of cuTensor.

    Returns:
        out (cupy.ndarray): Output tensor.

    Examples:
        See examples/cutensor/contraction.py
    """
    if not (A._c_contiguous and B._c_contiguous and C._c_contiguous):
        raise ValueError('The inputs should be contiguous arrays.')
    compute_type = _get_contraction_compute_type(A.dtype, B.dtype, C.dtype,
                                                 compute_dtype)
    scalar_dtype = _get_scalar_dtype(C.dtype)

    return _contraction_impl(
        _get_handle(),
        _create_scalar(alpha, scalar_dtype),
        A, desc_A, _auto_create_mode(A, mode_A),
        B, desc_B, _auto_create_mode(B, mode_B),
        _create_scalar(beta, scalar_dtype),
        C, desc_C, _auto_create_mode(C, mode_C),
        compute_type, algo, ws_pref)


cdef inline ndarray _contraction_impl(
        Handle handle,
        _Scalar alpha, ndarray A, TensorDescriptor desc_A, Mode mode_A,
        ndarray B, TensorDescriptor desc_B, Mode mode_B,
        _Scalar beta, ndarray C, TensorDescriptor desc_C, Mode mode_C,
        int cutensor_compute_type, int algo, int ws_pref):
    cdef ContractionDescriptor desc
    cdef ContractionFind find
    cdef ContractionPlan plan
    cdef uint64_t ws_size
    cdef ndarray out, ws

    out = C

    desc = _create_contraction_descriptor(
        handle,
        A, desc_A, mode_A,
        B, desc_B, mode_B,
        C, desc_C, mode_C,
        cutensor_compute_type)

    find = _create_contraction_find(handle, algo)

    # Allocate workspace
    ws_size = cutensor.contractionGetWorkspace(handle, desc, find, ws_pref)
    try:
        ws = core._ndarray_init(shape_t(1, ws_size), dtype=_numpy.int8)
    except Exception:
        _warnings.warn('cuTENSOR: failed to allocate memory of workspace '
                       'with preference ({}) and size ({}).'
                       ''.format(ws_pref, ws_size))
        ws_size = cutensor.contractionGetWorkspace(
            handle, desc, find, cutensor.WORKSPACE_MIN)
        ws = core._ndarray_init(shape_t(1, ws_size), dtype=_numpy.int8)

    plan = _create_contraction_plan(handle, desc, find, ws_size)

    cutensor.contraction(
        handle, plan,
        alpha.ptr, A.data.ptr, B.data.ptr,
        beta.ptr, C.data.ptr, out.data.ptr,
        ws.data.ptr, ws_size)
    return out


def contraction_max_algos():
    """Returns the maximum number of algorithms for cutensor()

    See cupy/cuda/cutensor.contractionMaxAlgos() for details.
    """
    return cutensor.contractionMaxAlgos()


def reduction(
        alpha, ndarray A, TensorDescriptor desc_A, mode_A,
        beta, ndarray C, TensorDescriptor desc_C, mode_C,
        int reduce_op=cutensor.OP_ADD, compute_dtype=None):
    """Tensor reduction

    This routine computes the tensor reduction:

        C = alpha * reduce_op(uop_A(A)) + beta * uop_C(C))

    See :func:`cupy.cuda.cutensor.reduction` for details.

    Args:
        alpha (scalar): Scaling factor for A.
        A (cupy.ndarray): Input tensor.
        desc_A (class Descriptor): A descriptor that holds the information
            about the data type, modes, strides and unary operator (uop_A) of
            tensor A.
        mode_A (cutensor.Mode): A mode object created by `create_mode`.
        beta (scalar): Scaling factor for C.
        C (cupy.ndarray): Input/output tensor.
        desc_C (class Descriptor): A descriptor that holds the information
            about the data type, modes, strides and unary operator (uop_C) of
            tensor C.
        mode_C (cutensor.Mode): A mode object created by `create_mode`.
        reduce_op (cutensorOperator_t): Binary operator used to reduce A.
        compute_dtype (numpy.dtype): Compute type for the intermediate
            computation.

    Returns:
        out (cupy.ndarray): Output tensor.

    Examples:
        See examples/cutensor/reduction.py
    """
    cdef Handle handle

    if A.dtype != C.dtype:
        raise ValueError('dtype mismatch: {} != {}'.format(A.dtype, C.dtype))
    if not (A._c_contiguous and C._c_contiguous):
        raise ValueError('The inputs should be contiguous arrays.')

    compute_dtype = _set_compute_dtype(A.dtype, compute_dtype)

    return _reduction_impl(
        _get_handle(),
        _create_scalar(alpha, compute_dtype),
        A, desc_A, _auto_create_mode(A, mode_A),
        _create_scalar(beta, compute_dtype),
        C, desc_C, _auto_create_mode(C, mode_C),
        reduce_op, _get_cutensor_compute_type(compute_dtype)
    )


cdef inline ndarray _reduction_impl(
        Handle handle,
        _Scalar alpha, ndarray A, TensorDescriptor desc_A, Mode mode_A,
        _Scalar beta, ndarray C, TensorDescriptor desc_C, Mode mode_C,
        int reduce_op, int cutensor_compute_type):
    cdef uint64_t ws_size
    cdef ndarray ws, out

    out = C
    ws_size = cutensor.reductionGetWorkspace(
        handle,
        A.data.ptr, desc_A, mode_A.data,
        C.data.ptr, desc_C, mode_C.data,
        out.data.ptr, desc_C, mode_C.data,
        reduce_op, cutensor_compute_type)
    try:
        ws = core._ndarray_init(shape_t(1, ws_size), dtype=_numpy.int8)
    except _cupy.cuda.memory.OutOfMemoryError:
        _warnings.warn('cuTENSOR: failed to allocate memory of workspace '
                       '(size: {}).'.format(ws_size))
        ws_size = 0
        ws = core._ndarray_init(shape_t(1, ws_size), dtype=_numpy.int8)

    cutensor.reduction(
        handle,
        alpha.ptr, A.data.ptr, desc_A, mode_A.data,
        beta.ptr, C.data.ptr, desc_C, mode_C.data,
        out.data.ptr, desc_C, mode_C.data,
        reduce_op, cutensor_compute_type, ws.data.ptr, ws_size)
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
        ndarray x, axis, dtype, ndarray out, keepdims, reduce_op, alpha, beta):
    cdef Handle handle
    cdef ndarray in_arg, out_arg
    cdef shape_t out_shape
    cdef tuple reduce_axis, out_axis
    cdef TensorDescriptor desc_in, desc_out

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

    in_arg = x

    reduce_axis, out_axis = _reduction._get_axis(axis, x.ndim)
    if len(reduce_axis) == 0:
        return None
    out_shape = _reduction._get_out_shape(
        x._shape, reduce_axis, out_axis, keepdims)
    if out is None:
        out = core._ndarray_init(out_shape, dtype=dtype)
    elif not internal.vector_equal(out._shape, out_shape):
        # TODO(asi1024): Support broadcast
        return None
    elif out.dtype != dtype:
        return None
    elif not out._c_contiguous:
        # TODO(asi1024): Support also for F-contiguous array
        return None

    if keepdims:
        out_arg = out.reshape(
            _reduction._get_out_shape(x._shape, reduce_axis, out_axis, False))
    else:
        out_arg = out

    # TODO(asi1024): Remove temporary fix
    in_arg._set_contiguous_strides(in_arg.itemsize, True)
    out_arg._set_contiguous_strides(out_arg.itemsize, True)

    handle = _get_handle()

    desc_in = create_tensor_descriptor(in_arg, handle=handle)
    desc_out = create_tensor_descriptor(out_arg, handle=handle)

    compute_dtype = _set_compute_dtype(in_arg.dtype, dtype)

    _reduction_impl(
        handle,
        _create_scalar(alpha, compute_dtype),
        in_arg,
        desc_in,
        _create_mode_with_cache(in_arg._shape.size()),
        _create_scalar(beta, compute_dtype),
        out_arg,
        desc_out,
        _create_mode_with_cache(out_axis),
        reduce_op, _get_cutensor_compute_type(compute_dtype))

    return out
