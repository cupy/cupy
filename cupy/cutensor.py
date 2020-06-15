import numpy
import warnings

import cupy
from cupy.cuda import cutensor
from cupy.cuda import device
from cupy.cuda import runtime
from cupy import util

_handles = {}
_tensor_descriptors = {}
_contraction_descriptors = {}
_contraction_finds = {}
_contraction_plans = {}


class Descriptor(object):

    def __init__(self, descriptor, destroyer=None):
        self.value = descriptor
        self.destroy = destroyer

    def __del__(self, is_shutting_down=util.is_shutting_down):
        if is_shutting_down():
            return
        if self.destroy is None:
            self.value = None
        elif self.value is not None:
            self.destroy(self.value)
            self.value = None


def get_handle():
    dev = device.get_device_id()
    if dev not in _handles:
        handle = cutensor.Handle()
        cutensor.init(handle)
        _handles[dev] = handle
    return _handles[dev]


def get_cuda_dtype(numpy_dtype):
    if numpy_dtype == numpy.float16:
        return runtime.CUDA_R_16F
    elif numpy_dtype == numpy.float32:
        return runtime.CUDA_R_32F
    elif numpy_dtype == numpy.float64:
        return runtime.CUDA_R_64F
    elif numpy_dtype == numpy.complex64:
        return runtime.CUDA_C_32F
    elif numpy_dtype == numpy.complex128:
        return runtime.CUDA_C_64F
    else:
        raise TypeError('Dtype {} is not supported'.format(numpy_dtype))


def get_cutensor_dtype(numpy_dtype):
    if numpy_dtype == numpy.float16:
        return cutensor.R_MIN_16F
    elif numpy_dtype == numpy.float32:
        return cutensor.R_MIN_32F
    elif numpy_dtype == numpy.float64:
        return cutensor.R_MIN_64F
    elif numpy_dtype == numpy.complex64:
        return cutensor.C_MIN_32F
    elif numpy_dtype == numpy.complex128:
        return cutensor.C_MIN_64F
    else:
        raise TypeError('Dtype {} is not supported'.format(numpy_dtype))


def _convert_mode(mode):
    return numpy.array([ord(x) if isinstance(x, str) else x for x in mode],
                       dtype=numpy.int32)


def _set_compute_dtype(array_dtype, compute_dtype=None):
    if compute_dtype is None:
        if array_dtype == numpy.float16:
            compute_dtype = numpy.float32
        else:
            compute_dtype = array_dtype
    return compute_dtype


def create_tensor_descriptor(a, uop=cutensor.OP_IDENTITY):
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
    handle = get_handle()
    key = (handle.ptr, a.dtype, tuple(a.shape), tuple(a.strides), uop)
    if key in _tensor_descriptors:
        desc = _tensor_descriptors[key]
        return desc
    num_modes = a.ndim
    extent = numpy.array(a.shape, dtype=numpy.int64)
    stride = numpy.array(a.strides, dtype=numpy.int64) // a.itemsize
    cuda_dtype = get_cuda_dtype(a.dtype)
    desc = cutensor.TensorDescriptor()
    cutensor.initTensorDescriptor(
        handle, desc, num_modes, extent.ctypes.data, stride.ctypes.data,
        cuda_dtype, uop)
    _tensor_descriptors[key] = desc
    return desc


def elementwise_trinary(alpha, A, desc_A, mode_A,
                        beta, B, desc_B, mode_B,
                        gamma, C, desc_C, mode_C,
                        out=None,
                        op_AB=cutensor.OP_ADD, op_ABC=cutensor.OP_ADD,
                        compute_dtype=None):
    """Element-wise tensor operation for three input tensors

    This function performs a element-wise tensor operation of the form:

        D_{Pi^C(i_0,i_1,...,i_nc)} =
            op_ABC(op_AB(alpha * uop_A(A_{Pi^A(i_0,i_1,...,i_na)}),
                         beta  * uop_B(B_{Pi^B(i_0,i_1,...,i_nb)})),
                         gamma * uop_C(C_{Pi^C(i_0,i_1,...,i_nc)}))

    See cupy/cuda/cutensor.elementwiseTrinary() for details.

    Args:
        alpha: Scaling factor for tensor A.
        A (cupy.ndarray): Input tensor.
        desc_A (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor A.
        mode_A (tuple of int/str): A tuple that holds the labels of the modes
            of tensor A (e.g., if A_{x,y,z}, mode_A = {'x','y','z'})
        beta: Scaling factor for tensor B.
        B (cupy.ndarray): Input tensor.
        desc_B (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor B.
        mode_B (tuple of int/str): A tuple that holds the labels of the modes
            of tensor B.
        gamma: Scaling factor for tensor C.
        C (cupy.ndarray): Input tensor.
        desc_C (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor C.
        mode_C (tuple of int/str): A tuple that holds the labels of the modes
            of tensor C.
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
    if not (A.flags.c_contiguous
            and B.flags.c_contiguous
            and C.flags.c_contiguous):
        raise ValueError('The inputs should be contiguous arrays.')

    if out is None:
        out = cupy.ndarray(C.shape, dtype=C.dtype)
    elif C.dtype != out.dtype:
        raise ValueError('dtype mismatch: {} != {}'.format(C.dtype, out.dtype))
    elif C.shape != out.shape:
        raise ValueError('shape mismatch: {} != {}'.format(C.shape, out.shape))
    elif not out.flags.c_contiguous:
        raise ValueError('`out` should be a contiguous array.')

    if A.ndim != len(mode_A):
        raise ValueError('ndim mismatch: {} != {}'.format(A.ndim, len(mode_A)))
    if B.ndim != len(mode_B):
        raise ValueError('ndim mismatch: {} != {}'.format(B.ndim, len(mode_B)))
    if C.ndim != len(mode_C):
        raise ValueError('ndim mismatch: {} != {}'.format(C.ndim, len(mode_C)))

    mode_A = _convert_mode(mode_A)
    mode_B = _convert_mode(mode_B)
    mode_C = _convert_mode(mode_C)

    if compute_dtype is None:
        compute_dtype = A.dtype
    alpha = numpy.array(alpha, compute_dtype)
    beta = numpy.array(beta, compute_dtype)
    gamma = numpy.array(gamma, compute_dtype)
    handle = get_handle()
    cuda_dtype = get_cuda_dtype(compute_dtype)
    cutensor.elementwiseTrinary(
        handle,
        alpha.ctypes.data,
        A.data.ptr, desc_A, mode_A.ctypes.data,
        beta.ctypes.data,
        B.data.ptr, desc_B, mode_B.ctypes.data,
        gamma.ctypes.data,
        C.data.ptr, desc_C, mode_C.ctypes.data,
        out.data.ptr, desc_C, mode_C.ctypes.data,
        op_AB, op_ABC, cuda_dtype)
    return out


def elementwise_binary(alpha, A, desc_A, mode_A,
                       gamma, C, desc_C, mode_C,
                       out=None,
                       op_AC=cutensor.OP_ADD,
                       compute_dtype=None):
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
    if not (A.flags.c_contiguous and C.flags.c_contiguous):
        raise ValueError('The inputs should be contiguous arrays.')

    if out is None:
        out = cupy.ndarray(C.shape, dtype=C.dtype)
    elif C.dtype != out.dtype:
        raise ValueError('dtype mismatch: {} != {}'.format(C.dtype, out.dtype))
    elif C.shape != out.shape:
        raise ValueError('shape mismatch: {} != {}'.format(C.shape, out.shape))
    elif not out.flags.c_contiguous:
        raise ValueError('`out` should be a contiguous array.')

    if A.ndim != len(mode_A):
        raise ValueError('ndim mismatch: {} != {}'.format(A.ndim, len(mode_A)))
    if C.ndim != len(mode_C):
        raise ValueError('ndim mismatch: {} != {}'.format(C.ndim, len(mode_C)))

    mode_A = _convert_mode(mode_A)
    mode_C = _convert_mode(mode_C)

    if compute_dtype is None:
        compute_dtype = A.dtype
    alpha = numpy.array(alpha, compute_dtype)
    gamma = numpy.array(gamma, compute_dtype)
    handle = get_handle()
    cuda_dtype = get_cuda_dtype(compute_dtype)
    cutensor.elementwiseBinary(
        handle,
        alpha.ctypes.data,
        A.data.ptr, desc_A, mode_A.ctypes.data,
        gamma.ctypes.data,
        C.data.ptr, desc_C, mode_C.ctypes.data,
        out.data.ptr, desc_C, mode_C.ctypes.data,
        op_AC, cuda_dtype)
    return out


def _create_contraction_descriptor(A, desc_A, mode_A, B, desc_B, mode_B,
                                   C, desc_C, mode_C, compute_dtype=None):
    """Create a contraction descriptor"""
    assert A.dtype == B.dtype == C.dtype
    assert A.ndim == len(mode_A)
    assert B.ndim == len(mode_B)
    assert C.ndim == len(mode_C)
    compute_dtype = _set_compute_dtype(A.dtype, compute_dtype)
    handle = get_handle()
    alignment_req_A = cutensor.getAlignmentRequirement(
        handle, A.data.ptr, desc_A)
    alignment_req_B = cutensor.getAlignmentRequirement(
        handle, B.data.ptr, desc_B)
    alignment_req_C = cutensor.getAlignmentRequirement(
        handle, C.data.ptr, desc_C)
    key = (handle.ptr, compute_dtype,
           desc_A.ptr, tuple(mode_A), alignment_req_A,
           desc_B.ptr, tuple(mode_B), alignment_req_B,
           desc_C.ptr, tuple(mode_C), alignment_req_C)
    if key in _contraction_descriptors:
        desc = _contraction_descriptors[key]
        return desc
    mode_A = _convert_mode(mode_A)
    mode_B = _convert_mode(mode_B)
    mode_C = _convert_mode(mode_C)
    cutensor_dtype = get_cutensor_dtype(compute_dtype)
    desc = cutensor.ContractionDescriptor()
    cutensor.initContractionDescriptor(
        handle,
        desc,
        desc_A, mode_A.ctypes.data, alignment_req_A,
        desc_B, mode_B.ctypes.data, alignment_req_B,
        desc_C, mode_C.ctypes.data, alignment_req_C,
        desc_C, mode_C.ctypes.data, alignment_req_C,
        cutensor_dtype)
    _contraction_descriptors[key] = desc
    return desc


def _create_contraction_plan(desc, algo, ws_pref):
    """Create a contraction plan"""
    handle = get_handle()
    key = (handle.ptr, algo)
    if key in _contraction_finds:
        find = _contraction_finds[key]
    else:
        find = cutensor.ContractionFind()
        cutensor.initContractionFind(handle, find, algo)
        _contraction_finds[key] = find

    ws_allocation_success = False
    for pref in (ws_pref, cutensor.WORKSPACE_MIN):
        ws_size = cutensor.contractionGetWorkspace(handle, desc, find, pref)
        try:
            ws = cupy.ndarray((ws_size,), dtype=numpy.int8)
            ws_allocation_success = True
        except Exception:
            warnings.warn('cuTENSOR: failed to allocate memory of workspace '
                          'with preference ({}) and size ({}).'
                          ''.format(pref, ws_size))
        if ws_allocation_success:
            break
    if not ws_allocation_success:
        raise RuntimeError('cuTENSOR: failed to allocate memory of workspace.')

    key = (handle.ptr, desc.ptr, find.ptr, ws_size)
    if key in _contraction_plans:
        plan = _contraction_plans[key]
    else:
        plan = cutensor.ContractionPlan()
        cutensor.initContractionPlan(handle, plan, desc, find, ws_size)
        _contraction_plans[key] = plan

    return plan, ws, ws_size


def contraction(alpha, A, desc_A, mode_A, B, desc_B, mode_B,
                beta, C, desc_C, mode_C, compute_dtype=None,
                algo=cutensor.ALGO_DEFAULT,
                ws_pref=cutensor.WORKSPACE_RECOMMENDED):
    """General tensor contraction

    This routine computes the tensor contraction:

        C = alpha * uop_A(A) * uop_B(B) + beta * uop_C(C)

    See cupy/cuda/cutensor.contraction for details.

    Args:
        alpha: Scaling factor for A * B.
        A (cupy.ndarray): Input tensor.
        desc_A (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor A.
        mode_A (tuple of int/str): A tuple that holds the labels of the modes
            of tensor A (e.g., if A_{x,y,z}, mode_A = {'x','y','z'})
        B (cupy.ndarray): Input tensor.
        desc_B (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor B.
        mode_B (tuple of int/str): A tuple that holds the labels of the modes
            of tensor B.
        beta: Scaling factor for C.
        C (cupy.ndarray): Input/output tensor.
        desc_C (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor C.
        mode_C (tuple of int/str): A tuple that holds the labels of the modes
            of tensor C.
        compute_dtype (numpy.dtype): Compute type for the intermediate
            computation.
        algo (cutenorAlgo_t): Allows users to select a specific algorithm.
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
    if not (A.dtype == B.dtype == C.dtype):
        raise ValueError(
            'dtype mismatch: ({}, {}, {})'.format(A.dtype, B.dtype, C.dtype))
    if not (A.flags.c_contiguous
            and B.flags.c_contiguous
            and C.flags.c_contiguous):
        raise ValueError('The inputs should be contiguous arrays.')

    if A.ndim != len(mode_A):
        raise ValueError('ndim mismatch: {} != {}'.format(A.ndim, len(mode_A)))
    if B.ndim != len(mode_B):
        raise ValueError('ndim mismatch: {} != {}'.format(B.ndim, len(mode_B)))
    if C.ndim != len(mode_C):
        raise ValueError('ndim mismatch: {} != {}'.format(C.ndim, len(mode_C)))

    out = C
    compute_dtype = _set_compute_dtype(A.dtype, compute_dtype)
    handle = get_handle()
    alpha = numpy.array(alpha, compute_dtype)
    beta = numpy.array(beta, compute_dtype)
    desc = _create_contraction_descriptor(A, desc_A, mode_A,
                                          B, desc_B, mode_B,
                                          C, desc_C, mode_C,
                                          compute_dtype=compute_dtype)
    plan, ws, ws_size = _create_contraction_plan(desc, algo, ws_pref)
    cutensor.contraction(handle, plan,
                         alpha.ctypes.data, A.data.ptr, B.data.ptr,
                         beta.ctypes.data, C.data.ptr, out.data.ptr,
                         ws.data.ptr, ws_size)
    return out


def contraction_max_algos():
    """Returns the maximum number of algorithms for cutensor()

    See cupy/cuda/cutensor.contractionMaxAlgos() for details.
    """
    return cutensor.contractionMaxAlgos()


def reduction(alpha, A, desc_A, mode_A, beta, C, desc_C, mode_C,
              reduce_op=cutensor.OP_ADD, compute_dtype=None):
    """Tensor reduction

    This routine computes the tensor reduction:

        C = alpha * reduce_op(uop_A(A)) + beta * uop_C(C))

    See :func:`cupy.cuda.cutensor.reduction` for details.

    Args:
        alpha: Scaling factor for A.
        A (cupy.ndarray): Input tensor.
        desc_A (class Descriptor): A descriptor that holds the information
            about the data type, modes, strides and unary operator (uop_A) of
            tensor A.
        mode_A (tuple of int/str): A tuple that holds the labels of the modes
            of tensor A (e.g., if A_{x,y,z}, mode_A = {'x','y','z'})
        beta: Scaling factor for C.
        C (cupy.ndarray): Input/output tensor.
        desc_C (class Descriptor): A descriptor that holds the information
            about the data type, modes, strides and unary operator (uop_C) of
            tensor C.
        mode_C (tuple of int/str): A tuple that holds the labels of the modes
            of tensor C.
        reduce_op (cutensorOperator_t): Binary operator used to reduce A.
        compute_dtype (numpy.dtype): Compute type for the intermediate
            computation.

    Returns:
        out (cupy.ndarray): Output tensor.

    Examples:
        See examples/cutensor/reduction.py
    """
    if A.dtype != C.dtype:
        raise ValueError('dtype mismatch: {} != {}'.format(A.dtype, C.dtype))
    if not (A.flags.c_contiguous and C.flags.c_contiguous):
        raise ValueError('The inputs should be contiguous arrays.')

    if A.ndim != len(mode_A):
        raise ValueError('ndim mismatch: {} != {}'.format(A.ndim, len(mode_A)))
    if C.ndim != len(mode_C):
        raise ValueError('ndim mismatch: {} != {}'.format(C.ndim, len(mode_C)))

    mode_A = _convert_mode(mode_A)
    mode_C = _convert_mode(mode_C)
    out = C
    compute_dtype = _set_compute_dtype(A.dtype, compute_dtype)
    alpha = numpy.array(alpha, compute_dtype)
    beta = numpy.array(beta, compute_dtype)
    handle = get_handle()
    cutensor_dtype = get_cutensor_dtype(compute_dtype)
    ws_size = cutensor.reductionGetWorkspace(
        handle,
        A.data.ptr, desc_A, mode_A.ctypes.data,
        C.data.ptr, desc_C, mode_C.ctypes.data,
        out.data.ptr, desc_C, mode_C.ctypes.data,
        reduce_op, cutensor_dtype)
    try:
        ws = cupy.ndarray((ws_size,), dtype=numpy.int8)
    except cupy.cuda.memory.OutOfMemoryError:
        warnings.warn('cuTENSOR: failed to allocate memory of workspace '
                      '(size: {}).'.format(ws_size))
        ws_size = 0
        ws = cupy.ndarray((ws_size,), dtype=numpy.int8)
    cutensor.reduction(handle,
                       alpha.ctypes.data,
                       A.data.ptr, desc_A, mode_A.ctypes.data,
                       beta.ctypes.data,
                       C.data.ptr, desc_C, mode_C.ctypes.data,
                       out.data.ptr, desc_C, mode_C.ctypes.data,
                       reduce_op, cutensor_dtype, ws.data.ptr, ws_size)
    return out
