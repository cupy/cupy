import numpy
import warnings

import cupy
from cupy.cuda import cutensor
from cupy.cuda import device
from cupy.cuda import runtime

_handles = {}


def get_handle():
    dev = device.get_device_id()
    if dev not in _handles:
        _handles[dev] = cutensor.create()
    return _handles[dev]


class Descriptor:

    def __init__(self, descriptor, destroyer):
        self.value = descriptor
        self.destroy = destroyer

    def __dealloc__(self):
        if self.value is not None:
            self.destroy(self.value)
            self.value = None


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


def create_tensor_descriptor(a, uop=cutensor.OP_IDENTITY,
                             vector_width=1, vector_mode_index=0):
    """Create a tensor descriptor

    Args:
        a (cupy.ndarray): tensor for which a descritpor are created.
        uop (cutensorOperator_t): unary operator that will be applied to each
            element of the corresponding tensor in a lazy fashion (i.e., the
            algorithm uses this tensor as its operand only once). The
            original data of this tensor remains unchanged.
        vectorWidth (integer): The vectorization-width of the vectorized mode
            (i.e., the number of consecutive elements in that mode). Set this
            value to 1 if no vectorization is desired. Allowed values are
            limited to 1 (this should likely be your default choice), 2, 4, 8,
            16, and 32.
        vectorModeIndex (integer): The position of the mode that is vectorized
            (from left to right, 0-indexed). For instance, vectorModeIndex == i
            means that the mode corresponding to extent[i] and stride[i] is
            vectorized. This value is ignored if the vectorWidth is set to 1.

    Returns:
        (Descriptor): A instance of class Descriptor which holds a pointer to
            tensor descriptor and its destructor.
    """
    num_modes = a.ndim
    extent = numpy.array(a.shape, dtype=numpy.int64)
    stride = numpy.array(a.strides, dtype=numpy.int64) // a.itemsize
    data_type = get_cuda_dtype(a.dtype)
    desc = cutensor.createTensorDescriptor(
        num_modes, extent.ctypes.data, stride.ctypes.data, data_type, uop,
        vector_width, vector_mode_index)
    return Descriptor(desc, cutensor.destroyTensorDescriptor)


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
            of tensor A (e.g., if A_{x,y,z} => mode_A = {'x','y','z'})
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
    """
    assert A.dtype == B.dtype == C.dtype
    assert A.ndim == len(mode_A)
    assert B.ndim == len(mode_B)
    assert C.ndim == len(mode_C)
    mode_A = numpy.array([ord(x) if isinstance(x, str) else x for x in mode_A],
                         dtype=numpy.int32)
    mode_B = numpy.array([ord(x) if isinstance(x, str) else x for x in mode_B],
                         dtype=numpy.int32)
    mode_C = numpy.array([ord(x) if isinstance(x, str) else x for x in mode_C],
                         dtype=numpy.int32)
    if out is None:
        out = cupy.ndarray(C.shape, dtype=C.dtype)
    else:
        assert C.dtype == out.dtype
        assert C.ndim == out.ndim
        for i in range(C.ndim):
            assert C.shape[i] == out.shape[i]
    if compute_dtype is None:
        compute_dtype = A.dtype
    alpha = numpy.array(alpha, compute_dtype)
    beta = numpy.array(beta, compute_dtype)
    gamma = numpy.array(gamma, compute_dtype)
    handle = get_handle()
    compute_dtype = get_cuda_dtype(compute_dtype)
    cutensor.elementwiseTrinary(
        handle,
        alpha.ctypes.data,
        A.data.ptr, desc_A.value, mode_A.ctypes.data,
        beta.ctypes.data,
        B.data.ptr, desc_B.value, mode_B.ctypes.data,
        gamma.ctypes.data,
        C.data.ptr, desc_C.value, mode_C.ctypes.data,
        out.data.ptr, desc_C.value, mode_C.ctypes.data,
        op_AB, op_ABC, compute_dtype)
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
    """
    assert A.dtype == C.dtype
    assert A.ndim == len(mode_A)
    assert C.ndim == len(mode_C)
    mode_A = numpy.array([ord(x) if isinstance(x, str) else x for x in mode_A],
                         dtype=numpy.int32)
    mode_C = numpy.array([ord(x) if isinstance(x, str) else x for x in mode_C],
                         dtype=numpy.int32)
    if out is None:
        out = cupy.ndarray(C.shape, dtype=C.dtype)
    else:
        assert C.dtype == out.dtype
        assert C.ndim == out.ndim
        for i in range(C.ndim):
            assert C.shape[i] == out.shape[i]
    if compute_dtype is None:
        compute_dtype = A.dtype
    alpha = numpy.array(alpha, compute_dtype)
    gamma = numpy.array(gamma, compute_dtype)
    handle = get_handle()
    compute_dtype = get_cuda_dtype(compute_dtype)
    cutensor.elementwiseBinary(
        handle,
        alpha.ctypes.data,
        A.data.ptr, desc_A.value, mode_A.ctypes.data,
        gamma.ctypes.data,
        C.data.ptr, desc_C.value, mode_C.ctypes.data,
        out.data.ptr, desc_C.value, mode_C.ctypes.data,
        op_AC, compute_dtype)
    return out


def contraction(alpha, A, desc_A, mode_A, B, desc_B, mode_B,
                beta, C, desc_C, mode_C,
                uop=cutensor.OP_IDENTITY, compute_dtype=None,
                algo=cutensor.ALGO_DEFAULT,
                ws_pref=cutensor.WORKSPACE_RECOMMENDED):
    """General tensor contraction

    This routine computes the tensor contraction:

        C = uop(alpha * uop_A(A) * uop_B(B) + beta * uop_C(C))

    See cupy/cuda/cutensor.contraction for details.

    Args:
        alpha: Scaling factor for A * B.
        A (cupy.ndarray): Input tensor.
        desc_A (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor A.
        mode_A (tuple of int/str): A tuple that holds the labels of the modes
            of tensor A (e.g., if A_{x,y,z} => mode_A = {'x','y','z'})
        B (cupy.ndarray): Input tensor.
        desc_B (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor B.
        mode_B (tuple of int/str): A tuple that holds the labels of the modes
            of tensor B.
        beta: Scaling factor for C.
        C (cupy.ndarray): Input tensor.
        desc_C (class Descriptor): A descriptor that holds the information
            about the data type, modes, and strides of tensor C.
        mode_C (tuple of int/str): A tuple that holds the labels of the modes
            of tensor C.
        uop (cutensorOperator_t): The element-wise unary operator.
        compute_dtype (numpy.dtype): Compute type for the intermediate
            computation.
        algo (cutenorAlgo_t): Allows users to select a specific algorithm.
            ALGO_DEFAULT lets the heuristic choose the algorithm.
            Any value >= 0 selects a specific GEMM-like algorithm and
            deactivates the heuristic. If a specified algorithm is not
            supported, STATUS_NOT_SUPPORTED is returned.
        ws_perf (cutensorWorksizePreference_t): User preference for the
            workspace of cuTensor.

    Returns:
        out (cupy.ndarray): Output tensor.
    """
    assert A.dtype == B.dtype == C.dtype
    assert A.ndim == len(mode_A)
    assert B.ndim == len(mode_B)
    assert C.ndim == len(mode_C)
    mode_A = numpy.array([ord(x) if isinstance(x, str) else x for x in mode_A],
                         dtype=numpy.int32)
    mode_B = numpy.array([ord(x) if isinstance(x, str) else x for x in mode_B],
                         dtype=numpy.int32)
    mode_C = numpy.array([ord(x) if isinstance(x, str) else x for x in mode_C],
                         dtype=numpy.int32)
    out = C
    if compute_dtype is None:
        if A.dtype == numpy.float16:
            compute_dtype = numpy.float32
        else:
            compute_dtype = A.dtype
    alpha = numpy.array(alpha, compute_dtype)
    beta = numpy.array(beta, compute_dtype)
    handle = get_handle()
    compute_dtype = get_cuda_dtype(compute_dtype)
    ws_allocation_success = False
    for pref in (ws_pref, cutensor.WORKSPACE_MIN):
        ws_size = cutensor.contractionGetWorkspace(
            handle,
            A.data.ptr, desc_A.value, mode_A.ctypes.data,
            B.data.ptr, desc_B.value, mode_B.ctypes.data,
            C.data.ptr, desc_C.value, mode_C.ctypes.data,
            out.data.ptr, desc_C.value, mode_C.ctypes.data,
            uop, compute_dtype, algo, pref)
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
    cutensor.contraction(handle,
                         alpha.ctypes.data,
                         A.data.ptr, desc_A.value, mode_A.ctypes.data,
                         B.data.ptr, desc_B.value, mode_B.ctypes.data,
                         beta.ctypes.data,
                         C.data.ptr, desc_C.value, mode_C.ctypes.data,
                         out.data.ptr, desc_C.value, mode_C.ctypes.data,
                         uop, compute_dtype, algo, ws.data.ptr, ws_size)
    return out


def contraction_max_algos():
    """Returns the maximum number of algorithms for cutensor()

    See cupy/cuda/cutensor.contractionMaxAlgos() for details.
    """
    return cutensor.contractionMaxAlgos()
