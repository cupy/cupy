from libcpp cimport vector

import atexit
import threading
import warnings

import numpy

from cupy.core cimport core
from cupy.cuda cimport cudnn
from cupy.cuda cimport device
from cupy.core cimport internal
from cupy.cuda cimport memory

from cupy.cuda import cudnn as py_cudnn


cdef int _cudnn_version = cudnn.getVersion()
cdef _thread_local = threading.local()

cdef vector.vector[size_t] _handles


cpdef size_t get_handle() except? 0:
    cdef int dev
    dev = device.get_device_id()
    if _handles.size() <= dev:
        _handles.resize(dev + 1, 0)
    ret = _handles[dev]
    if ret != 0:
        return ret
    ret = cudnn.create()
    _handles[dev] = ret
    return ret


@atexit.register
def reset_handles():
    for handle in _handles:
        if handle:
            cudnn.destroy(handle)
    _handles.clear()


cpdef dict _get_nd_tensor_cache():
    if not hasattr(_thread_local, 'cudnn_nd_tensor_cache'):
        _thread_local.cudnn_nd_tensor_cache = {}
    return _thread_local.cudnn_nd_tensor_cache


cdef size_t _max_workspace_size = 8 * 1024 * 1024


cpdef size_t get_max_workspace_size():
    """Gets the workspace size for cuDNN.

    Check "cuDNN Library User Guide" for detail.

    Returns:
        int: The workspace size for cuDNN.

    """
    return _max_workspace_size


cpdef set_max_workspace_size(size):
    """Sets the workspace size for cuDNN.

    Check "cuDNN Library User Guide" for detail.

    Args:
        size: The workspace size for cuDNN.

    """
    global _max_workspace_size
    _max_workspace_size = size


cdef class Descriptor:

    cdef public size_t value
    cdef object destroy

    def __init__(self, descriptor, destroyer):
        self.value = descriptor
        self.destroy = destroyer

    def __del__(self):
        if self.value:
            self.destroy(self.value)
            self.value = 0


cpdef int get_data_type(dtype) except? -1:
    cdef char t = ord(dtype.char)
    if t == 'f':
        return cudnn.CUDNN_DATA_FLOAT
    elif t == 'd':
        return cudnn.CUDNN_DATA_DOUBLE
    elif t == 'e':
        return cudnn.CUDNN_DATA_HALF
    else:
        raise TypeError('Dtype {} is not supported in cuDNN'.format(dtype))


cpdef int _get_byte_size(int data_type) except -1:
    if data_type == cudnn.CUDNN_DATA_HALF:
        return 2
    elif data_type == cudnn.CUDNN_DATA_FLOAT:
        return 4
    elif data_type == cudnn.CUDNN_DATA_DOUBLE:
        return 8
    else:
        raise TypeError('Invalid cuDNN data type: {}'.format(data_type))


cpdef _create_tensor_nd_descriptor(
        size_t desc, core.ndarray arr, int data_type=-1):
    cdef vector.vector[int] c_shape, c_strides
    cdef Py_ssize_t itemsize, s
    if data_type == -1:  # `-1` is used instead of `None`
        data_type = get_data_type(arr.dtype)
    itemsize = arr.itemsize
    for s in arr._strides:
        c_strides.push_back(s // itemsize)
    for s in arr._shape:
        c_shape.push_back(s)
    cudnn.setTensorNdDescriptor(
        desc, data_type, arr._shape.size(), <size_t>&c_shape[0],
        <size_t>&c_strides[0])


cpdef _create_tensor_descriptor(size_t desc, core.ndarray arr,
                                int format):
    if not arr._c_contiguous:
        raise ValueError('cupy.cudnn supports c-contiguous arrays only')
    if arr._shape.size() == 4:
        data_type = get_data_type(arr.dtype)
        cudnn.setTensor4dDescriptor(desc, format, data_type,
                                    arr._shape[0], arr._shape[1],
                                    arr._shape[2], arr._shape[3])
    else:
        _create_tensor_nd_descriptor(desc, arr)


cpdef _create_tensor_descriptor_as4darray(size_t desc,
                                          core.ndarray arr):
    cdef Py_ssize_t dim1, dim2
    assert arr._c_contiguous
    data_type = get_data_type(arr.dtype)
    dim1 = 1
    if arr._shape.size() > 0:
        dim1 = arr._shape[0]
    dim2 = arr.size // dim1
    cudnn.setTensor4dDescriptor(desc, cudnn.CUDNN_TENSOR_NCHW, data_type,
                                dim1, dim2, 1, 1)


cpdef _create_filter_descriptor(
        size_t desc, core.ndarray arr, int format=cudnn.CUDNN_TENSOR_NCHW):
    cdef vector.vector[int] c_shape
    cdef Py_ssize_t s, ndim = arr._shape.size()
    data_type = get_data_type(arr.dtype)
    if ndim == 4:
        cudnn.setFilter4dDescriptor_v4(
            desc, data_type, format,
            arr._shape[0], arr._shape[1], arr._shape[2], arr._shape[3])
    else:
        for s in arr._shape:
            c_shape.push_back(s)
        cudnn.setFilterNdDescriptor_v4(
            desc, data_type, format, ndim, <size_t>&c_shape[0])


cpdef _create_convolution_descriptor(
        size_t desc, tuple pad, tuple stride, tuple dilation, int groups,
        object dtype, int mode, bint use_tensor_core):
    cdef int d0, d1, p0, p1, s0, s1
    cdef vector.vector[int] c_pad, c_stride, c_dilation
    ndim = len(pad)
    if ndim != len(stride):
        raise ValueError('pad and stride must be of same length')

    compute_type = get_data_type(dtype)
    # TODO(takagi) Temporarily use computing precision of FP32 for
    #     storing precision of FP16.
    if compute_type == cudnn.CUDNN_DATA_HALF:
        compute_type = cudnn.CUDNN_DATA_FLOAT

    if ndim != 2:
        c_pad = pad
        c_stride = stride
        if dilation is None:
            c_dilation.assign(ndim, 1)
        else:
            c_dilation = dilation
            if _cudnn_version < 6000:
                for i in c_dilation:
                    if i != 1:
                        raise ValueError(
                            'dilation must be one when cuDNN < 6.0')
        cudnn.setConvolutionNdDescriptor_v3(
            desc, ndim, <size_t>&c_pad[0], <size_t>&c_stride[0],
            <size_t>&c_dilation[0], mode, compute_type)
    else:
        if dilation is None:
            d0 = d1 = 1
        else:
            d0, d1 = dilation
            if _cudnn_version < 6000 and (d0 != 1 or d1 != 1):
                raise ValueError('dilation must be one when cuDNN < 6.0')
        p0, p1 = pad
        s0, s1 = stride
        cudnn.setConvolution2dDescriptor_v5(
            desc, p0, p1, s0, s1, d0, d1, mode, compute_type)
    if _cudnn_version >= 7000:
        if use_tensor_core:
            math_type = cudnn.CUDNN_TENSOR_OP_MATH
            cudnn.setConvolutionMathType(desc, math_type)
        if groups > 1:
            cudnn.setConvolutionGroupCount(desc, groups)
    elif groups > 1:
        raise ValueError('groups must be one when cuDNN < 7.0')


def create_tensor_descriptor(arr, format=cudnn.CUDNN_TENSOR_NCHW):
    desc = Descriptor(cudnn.createTensorDescriptor(),
                      py_cudnn.destroyTensorDescriptor)
    _create_tensor_descriptor(desc.value, arr, format)
    return desc


def create_uninitialized_tensor_descriptor():
    """Create uninitialized tensor descriptor.

    Create a cudnnCreateTensorDescriptor_t that is not yet initialized.
    This is used by the batch normalization functions.
    """
    return Descriptor(cudnn.createTensorDescriptor(),
                      py_cudnn.destroyTensorDescriptor)


def create_tensor_nd_descriptor(core.ndarray arr):
    cdef dict cache
    if arr.size == 0:
        return Descriptor(0, None)
    if not arr.flags.c_contiguous:
        raise ValueError('cupy.cudnn supports c-contiguous arrays only')
    data_type = get_data_type(arr.dtype)
    key = (data_type, tuple(arr._shape))
    cache = _get_nd_tensor_cache()
    if key in cache:
        return cache[key]

    # numpy's stride is defined in bytes, but cudnn's stride is defined in
    # size of element
    desc = Descriptor(cudnn.createTensorDescriptor(),
                      py_cudnn.destroyTensorDescriptor)
    _create_tensor_nd_descriptor(desc.value, arr, data_type)
    cache[key] = desc
    return desc


def create_filter_descriptor(arr, format=cudnn.CUDNN_TENSOR_NCHW):
    desc = Descriptor(cudnn.createFilterDescriptor(),
                      py_cudnn.destroyFilterDescriptor)
    _create_filter_descriptor(desc.value, arr, format)
    return desc


def create_convolution_descriptor(pad, stride, dtype,
                                  mode=cudnn.CUDNN_CROSS_CORRELATION,
                                  dilation=None,
                                  use_tensor_core=False,
                                  groups=1):
    desc = Descriptor(cudnn.createConvolutionDescriptor(),
                      py_cudnn.destroyConvolutionDescriptor)
    _create_convolution_descriptor(
        desc.value, pad, stride, dilation, groups,
        dtype, mode, use_tensor_core)
    return desc


cdef _create_pooling_descriptor(
        size_t desc, tuple ksize, tuple stride, tuple pad, int mode):
    cdef vector.vector[int] c_ksize, c_pad, c_stride
    cdef int ndim = len(ksize)
    if ndim != len(stride) or ndim != len(pad):
        raise ValueError('ksize, stride, and pad must be of same length')
    if ndim == 2:
        cudnn.setPooling2dDescriptor_v4(
            desc, mode, cudnn.CUDNN_NOT_PROPAGATE_NAN, ksize[0],
            ksize[1], pad[0], pad[1], stride[0], stride[1])
    else:
        c_ksize = ksize
        c_pad = pad
        c_stride = stride
        cudnn.setPoolingNdDescriptor_v4(
            desc, mode, cudnn.CUDNN_NOT_PROPAGATE_NAN, ndim,
            <size_t>&c_ksize[0], <size_t>&c_pad[0], <size_t>&c_stride[0])

    return desc


def create_pooling_descriptor(ksize, stride, pad, int mode):
    desc = Descriptor(cudnn.createPoolingDescriptor(),
                      py_cudnn.destroyPoolingDescriptor)
    _create_pooling_descriptor(desc.value, ksize, stride, pad, mode)
    return desc


def activation_forward(core.ndarray x, int mode, double coef=0.0):
    cdef float float_zero = 0, float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero, one
    cdef core.ndarray y
    if x.dtype == 'd':
        zero = <size_t>&double_zero
        one = <size_t>&double_one
    else:
        zero = <size_t>&float_zero
        one = <size_t>&float_one

    x = core.ascontiguousarray(x)
    y = core.ndarray(x._shape, x.dtype)

    handle = get_handle()
    desc = cudnn.createTensorDescriptor()
    act_desc = cudnn.createActivationDescriptor()
    try:
        _create_tensor_descriptor_as4darray(desc, x)
        cudnn.setActivationDescriptor(
            act_desc, mode, cudnn.CUDNN_NOT_PROPAGATE_NAN, coef)
        cudnn.activationForward_v4(
            handle, act_desc, one, desc, x.data.ptr,
            zero, desc, y.data.ptr)
    finally:
        cudnn.destroyActivationDescriptor(act_desc)
        cudnn.destroyTensorDescriptor(desc)
    return y


def activation_backward(core.ndarray x, core.ndarray y, core.ndarray gy,
                        int mode, float coef=0.0):
    cdef float float_zero = 0, float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero, one
    cdef core.ndarray gx
    if x.dtype == 'd':
        zero = <size_t>&double_zero
        one = <size_t>&double_one
    else:
        zero = <size_t>&float_zero
        one = <size_t>&float_one

    gx = core.ndarray(x._shape, x.dtype)
    x = core.ascontiguousarray(x)
    y = core.ascontiguousarray(y)
    gy = core.ascontiguousarray(gy)

    handle = get_handle()
    desc = cudnn.createTensorDescriptor()
    act_desc = cudnn.createActivationDescriptor()
    try:
        _create_tensor_descriptor_as4darray(desc, y)
        cudnn.setActivationDescriptor(
            act_desc, mode, cudnn.CUDNN_NOT_PROPAGATE_NAN, coef)
        cudnn.activationBackward_v4(
            handle, act_desc, one, desc, y.data.ptr,
            desc, gy.data.ptr, desc, x.data.ptr,
            zero, desc, gx.data.ptr)
    finally:
        cudnn.destroyActivationDescriptor(act_desc)
        cudnn.destroyTensorDescriptor(desc)
    return gx


cdef int _create_tensor_descriptor_for_softmax(
        size_t desc, core.ndarray arr, int axis) except?-1:
    cdef Py_ssize_t left, center, right
    assert arr._c_contiguous
    data_type = get_data_type(arr.dtype)
    if axis < 0:
        axis += arr._shape.size()
    left = 1
    for i in range(0, axis):
        left *= arr._shape[i]
    center = arr._shape[axis]
    right = 1
    for i in range(axis + 1, arr._shape.size()):
        right *= arr._shape[i]
    cudnn.setTensor4dDescriptor(desc, cudnn.CUDNN_TENSOR_NCHW, data_type,
                                left, center, right, 1)
    if center == 1 and right == 1:
        return cudnn.CUDNN_SOFTMAX_MODE_INSTANCE
    else:
        return cudnn.CUDNN_SOFTMAX_MODE_CHANNEL


def softmax_forward(core.ndarray x, int axis, int algorithm):
    cdef float float_zero = 0, float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero, one
    cdef core.ndarray y
    if x.dtype == 'd':
        zero = <size_t>&double_zero
        one = <size_t>&double_one
    else:
        zero = <size_t>&float_zero
        one = <size_t>&float_one

    x = core.ascontiguousarray(x)
    y = core.ndarray(x._shape, x.dtype)

    handle = get_handle()
    desc = cudnn.createTensorDescriptor()
    try:
        cudnn_mode = _create_tensor_descriptor_for_softmax(desc, x, axis)
        cudnn.softmaxForward(
            handle, algorithm, cudnn_mode,
            one, desc, x.data.ptr, zero, desc, y.data.ptr)
    finally:
        cudnn.destroyTensorDescriptor(desc)
    return y


def softmax_backward(core.ndarray y, core.ndarray gy, int axis, int algorithm):
    cdef float float_zero = 0, float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero, one
    cdef core.ndarray gx
    if y.dtype == 'd':
        zero = <size_t>&double_zero
        one = <size_t>&double_one
    else:
        zero = <size_t>&float_zero
        one = <size_t>&float_one

    gx = core.ndarray(y._shape, y.dtype)
    y = core.ascontiguousarray(y)
    gy = core.ascontiguousarray(gy)

    handle = get_handle()
    desc = cudnn.createTensorDescriptor()
    try:
        cudnn_mode = _create_tensor_descriptor_for_softmax(desc, y, axis)
        cudnn.softmaxBackward(
            handle, algorithm, cudnn_mode,
            one, desc, y.data.ptr, desc, gy.data.ptr, zero, desc, gx.data.ptr)
    finally:
        cudnn.destroyTensorDescriptor(desc)
    return gx


def create_dropout_descriptor(
        handle, dropout, states, state_size_in_bytes, seed):
    desc = Descriptor(cudnn.createDropoutDescriptor(),
                      py_cudnn.destroyDropoutDescriptor)
    cudnn.setDropoutDescriptor(desc.value, handle, dropout,
                               states, state_size_in_bytes, seed)
    return desc


def set_dropout_descriptor(desc, handle, dropout):
    # When the fourth argument is NULL, random state is not updated.
    cudnn.setDropoutDescriptor(desc.value, handle, dropout, 0, 0, 0)


def create_rnn_descriptor(hidden_size, num_layers, dropout_desc,
                          input_mode, direction, mode, data_type, algo=None):
    desc = Descriptor(cudnn.createRNNDescriptor(),
                      py_cudnn.destroyRNNDescriptor)
    if _cudnn_version >= 6000:
        _handle = get_handle()
        if algo is None:
            algo = cudnn.CUDNN_RNN_ALGO_STANDARD
        cudnn.setRNNDescriptor_v6(
            _handle, desc.value, hidden_size, num_layers, dropout_desc.value,
            input_mode, direction, mode, algo, data_type)
    else:
        cudnn.setRNNDescriptor_v5(
            desc.value, hidden_size, num_layers, dropout_desc.value,
            input_mode, direction, mode, data_type)
    return desc


def get_rnn_lin_layer_matrix_params(
        handle, rnn_desc, layer, x_desc, w_desc, core.ndarray w, lin_layer_id):
    cdef size_t ptr = 0
    w_data_ptr = w.data.ptr
    mat_desc = cudnn.createFilterDescriptor()
    try:
        cudnn.getRNNLinLayerMatrixParams(
            handle, rnn_desc.value, layer, x_desc.value, w_desc.value,
            w.data.ptr, lin_layer_id, mat_desc, <size_t>&ptr)
        data_type, _, _, dim = cudnn.getFilterNdDescriptor(mat_desc, 3)
    finally:
        cudnn.destroyFilterDescriptor(mat_desc)
    byte_size = _get_byte_size(data_type)
    offset = (ptr - w.data.ptr) // byte_size
    size = internal.prod(dim)
    mat = w[offset: offset + size]
    return mat


def get_rnn_lin_layer_bias_params(
        handle, rnn_desc, layer, x_desc, w_desc, core.ndarray w, lin_layer_id):
    cdef size_t ptr = 0
    bias_desc = cudnn.createFilterDescriptor()
    try:
        cudnn.getRNNLinLayerBiasParams(
            handle, rnn_desc.value, layer, x_desc.value, w_desc.value,
            w.data.ptr, lin_layer_id, bias_desc, <size_t>&ptr)
        data_type, _, _, dim = cudnn.getFilterNdDescriptor(bias_desc, 3)
    finally:
        cudnn.destroyFilterDescriptor(bias_desc)
    byte_size = _get_byte_size(data_type)
    offset = (ptr - w.data.ptr) // byte_size
    size = internal.prod(dim)
    bias = w[offset: offset + size]
    return bias


cdef class _DescriptorArray:

    cdef:
        vector.vector[size_t] _value
        object _destroy

    def __init__(self, destroyer):
        self._destroy = destroyer

    def __del__(self):
        for desc in self._value:
            self._destroy(desc)

    def append(self, desc):
        self._value.push_back(desc)

    @property
    def data(self):
        return <size_t>&self._value[0]


cdef _DescriptorArray _make_tensor_descriptor_array(xs, lengths):
    """Make an array of pointers denoting pointers of tensor descriptors.

    """
    cdef _DescriptorArray descs = _DescriptorArray(
        py_cudnn.destroyTensorDescriptor)
    cdef size_t desc
    cdef int data_type = get_data_type(xs.dtype)
    cdef vector.vector[int] c_shape, c_strides
    cdef Py_ssize_t itemsize = xs.itemsize
    cdef Py_ssize_t s
    cdef int length

    # RNN APIs assumes ndim == 3.
    for s in xs._strides:
        c_strides.push_back(s // itemsize)
    for _ in range(3 - len(xs._strides)):
        c_strides.push_back(1)
    for s in xs._shape:
        c_shape.push_back(s)
    for _ in range(3 - len(xs._strides)):
        c_shape.push_back(1)

    for length in lengths:
        c_shape[0] = length
        desc = cudnn.createTensorDescriptor()
        descs.append(desc)
        cudnn.setTensorNdDescriptor(
            desc, data_type, 3, <size_t>&c_shape[0], <size_t>&c_strides[0])

    return descs


cdef memory.MemoryPointer _make_rnn_workspace(
        Descriptor rnn_desc, int length, _DescriptorArray descs):
    cdef size_t handle = get_handle()
    cdef size_t work_size = cudnn.getRNNWorkspaceSize(
        handle, rnn_desc.value, length, descs.data)
    return memory.alloc(work_size)


cdef Py_ssize_t _get_n_layers(int direction_mode, core.ndarray hx):
    if direction_mode == cudnn.CUDNN_BIDIRECTIONAL:
        return hx._shape[0] // 2
    else:  # cudnn.CUDNN_UNIDIRECTIONAL
        return hx._shape[0]


def rnn_forward_inference(
        DropoutStates states, int direction_mode, int rnn_mode,
        core.ndarray hx, core.ndarray cx, core.ndarray w, core.ndarray xs,
        lengths):
    hx = core.ascontiguousarray(hx)
    cx = core.ascontiguousarray(cx)
    w = core.ascontiguousarray(w)
    xs = core.ascontiguousarray(xs)

    cdef int length = len(lengths)
    cdef int n_layers = _get_n_layers(direction_mode, hx)
    cdef int n_units = hx.shape[2]
    cdef int input_units
    if direction_mode == cudnn.CUDNN_BIDIRECTIONAL:
        input_units = n_units * 2
    else:  # cudnn.CUDNN_UNIDIRECTIONAL
        input_units = n_units

    cdef core.ndarray ys = core.ndarray((len(xs), input_units), dtype=xs.dtype)
    cdef size_t handle = get_handle()

    cdef Descriptor rnn_desc = create_rnn_descriptor(
        n_units, n_layers, states._desc,
        cudnn.CUDNN_LINEAR_INPUT, direction_mode,
        rnn_mode, get_data_type(xs.dtype))

    if cx is None:
        cx = core.ndarray(0, dtype=xs.dtype)
    cdef core.ndarray cy = core.ndarray(cx.shape, cx.dtype)

    cdef _DescriptorArray xs_descs = _make_tensor_descriptor_array(xs, lengths)
    cdef Descriptor hx_desc = create_tensor_nd_descriptor(hx)
    cdef Descriptor w_desc = create_filter_descriptor(w)

    cdef _DescriptorArray ys_descs = _make_tensor_descriptor_array(ys, lengths)
    cdef core.ndarray hy = core.ndarray(hx.shape, hx.dtype)
    cdef Descriptor hy_desc = create_tensor_nd_descriptor(hy)
    cdef Descriptor cx_desc = create_tensor_nd_descriptor(cx)
    cdef Descriptor cy_desc = create_tensor_nd_descriptor(cy)

    cdef memory.MemoryPointer workspace = _make_rnn_workspace(
        rnn_desc, length, xs_descs)

    cudnn.RNNForwardInference(
        handle, rnn_desc.value, length,
        xs_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
        cx_desc.value, cx.data.ptr, w_desc.value, w.data.ptr,
        ys_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
        cy_desc.value, cy.data.ptr, workspace.ptr, workspace.mem.size)

    return hy, cy, ys


def rnn_forward_training(
        DropoutStates states, int direction_mode, int rnn_mode,
        core.ndarray hx, core.ndarray cx, core.ndarray w, core.ndarray xs,
        lengths):
    hx = core.ascontiguousarray(hx)
    cx = core.ascontiguousarray(cx)
    w = core.ascontiguousarray(w)
    xs = core.ascontiguousarray(xs)

    cdef int length = len(lengths)
    cdef int n_layers = _get_n_layers(direction_mode, hx)
    cdef int n_units = hx.shape[2]
    cdef int input_units

    if direction_mode == cudnn.CUDNN_BIDIRECTIONAL:
        input_units = n_units * 2
    else:  # cudnn.CUDNN_UNIDIRECTIONAL
        input_units = n_units

    cdef core.ndarray ys = core.ndarray((len(xs), input_units), dtype=xs.dtype)
    cdef size_t handle = get_handle()

    cdef Descriptor rnn_desc = create_rnn_descriptor(
        n_units, n_layers, states._desc,
        cudnn.CUDNN_LINEAR_INPUT, direction_mode,
        rnn_mode, get_data_type(xs.dtype))

    if cx is None:
        cx = core.ndarray(0, dtype=xs.dtype)
    cdef core.ndarray cy = core.ndarray(cx.shape, cx.dtype)

    cdef _DescriptorArray xs_descs = _make_tensor_descriptor_array(xs, lengths)
    cdef Descriptor hx_desc = create_tensor_nd_descriptor(hx)
    cdef Descriptor w_desc = create_filter_descriptor(w)

    cdef _DescriptorArray ys_descs = _make_tensor_descriptor_array(ys, lengths)
    cdef core.ndarray hy = core.ndarray(hx.shape, hx.dtype)
    cdef Descriptor hy_desc = create_tensor_nd_descriptor(hy)
    cdef Descriptor cx_desc = create_tensor_nd_descriptor(cx)
    cdef Descriptor cy_desc = create_tensor_nd_descriptor(cy)

    cdef memory.MemoryPointer workspace = _make_rnn_workspace(
        rnn_desc, length, xs_descs)

    cdef size_t reserve_size = cudnn.getRNNTrainingReserveSize(
        handle, rnn_desc.value, length, xs_descs.data)
    cdef memory.MemoryPointer reserve_space = memory.alloc(reserve_size)
    cudnn.RNNForwardTraining(
        handle, rnn_desc.value, length,
        xs_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
        cx_desc.value, cx.data.ptr, w_desc.value, w.data.ptr,
        ys_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
        cy_desc.value, cy.data.ptr, workspace.ptr, workspace.mem.size,
        reserve_space.ptr, reserve_space.mem.size)

    return reserve_space, hy, cy, ys


def rnn_backward_data(
        DropoutStates states, int direction_mode, int rnn_mode,
        core.ndarray hx, core.ndarray cx, core.ndarray w, core.ndarray xs,
        core.ndarray ys, memory.MemoryPointer reserve_space,
        core.ndarray dhy, core.ndarray dcy, core.ndarray dys,
        lengths):
    hx = core.ascontiguousarray(hx)
    cx = core.ascontiguousarray(cx)
    w = core.ascontiguousarray(w)
    xs = core.ascontiguousarray(xs)
    ys = core.ascontiguousarray(ys)
    dhy = core.ascontiguousarray(dhy)
    dcy = core.ascontiguousarray(dcy)
    dys = core.ascontiguousarray(dys)

    cdef int length = len(lengths)
    cdef int n_layers = _get_n_layers(direction_mode, hx)
    cdef int n_units = hx.shape[2]

    cdef size_t handle = get_handle()
    cdef Descriptor rnn_desc = create_rnn_descriptor(
        n_units, n_layers, states._desc,
        cudnn.CUDNN_LINEAR_INPUT, direction_mode,
        rnn_mode, get_data_type(xs.dtype))

    cdef _DescriptorArray xs_descs = _make_tensor_descriptor_array(xs, lengths)
    cdef _DescriptorArray ys_descs = _make_tensor_descriptor_array(ys, lengths)
    cdef _DescriptorArray dys_descs = _make_tensor_descriptor_array(
        dys, lengths)

    cdef memory.MemoryPointer workspace = _make_rnn_workspace(
        rnn_desc, length, xs_descs)

    if cx is None:
        cx = dcy = core.ndarray(0, dtype=xs.dtype)
    cdef core.ndarray dcx = core.ndarray(cx.shape, cx.dtype)

    cdef Descriptor dhy_desc = create_tensor_nd_descriptor(dhy)
    cdef Descriptor hx_desc = create_tensor_nd_descriptor(hx)
    cdef Descriptor w_desc = create_filter_descriptor(w)

    cdef core.ndarray dxs = core.ndarray(xs.shape, xs.dtype)
    cdef _DescriptorArray dxs_descs = _make_tensor_descriptor_array(
        dxs, lengths)
    cdef core.ndarray dhx = core.ndarray(hx.shape, hx.dtype)
    cdef Descriptor dhx_desc = create_tensor_nd_descriptor(dhx)
    cdef Descriptor cx_desc = create_tensor_nd_descriptor(cx)
    cdef Descriptor dcx_desc = create_tensor_nd_descriptor(dcx)
    cdef Descriptor dcy_desc = create_tensor_nd_descriptor(dcy)

    cudnn.RNNBackwardData(
        handle, rnn_desc.value, length,
        ys_descs.data, ys.data.ptr,
        dys_descs.data, dys.data.ptr, dhy_desc.value, dhy.data.ptr,
        dcy_desc.value, dcy.data.ptr, w_desc.value, w.data.ptr,
        hx_desc.value, hx.data.ptr, cx_desc.value, cx.data.ptr,
        dxs_descs.data, dxs.data.ptr, dhx_desc.value, dhx.data.ptr,
        dcx_desc.value, dcx.data.ptr, workspace.ptr, workspace.mem.size,
        reserve_space.ptr, reserve_space.mem.size)

    return dhx, dcx, dxs


def rnn_backward_weights(
        DropoutStates states, int direction_mode, int rnn_mode,
        core.ndarray xs, core.ndarray hx, core.ndarray ys,
        core.ndarray w,
        memory.MemoryPointer reserve_space, lengths):
    xs = core.ascontiguousarray(xs)
    hx = core.ascontiguousarray(hx)
    ys = core.ascontiguousarray(ys)
    w = core.ascontiguousarray(w)

    cdef int length = len(lengths)
    cdef int n_layers = _get_n_layers(direction_mode, hx)
    cdef int n_units = hx.shape[2]

    cdef size_t handle = get_handle()
    cdef Descriptor rnn_desc = create_rnn_descriptor(
        n_units, n_layers, states._desc,
        cudnn.CUDNN_LINEAR_INPUT, direction_mode,
        rnn_mode, get_data_type(xs.dtype))

    cdef _DescriptorArray xs_descs = _make_tensor_descriptor_array(xs, lengths)
    cdef _DescriptorArray ys_descs = _make_tensor_descriptor_array(ys, lengths)
    cdef Descriptor hx_desc = create_tensor_nd_descriptor(hx)

    cdef memory.MemoryPointer workspace = _make_rnn_workspace(
        rnn_desc, length, xs_descs)

    cdef core.ndarray dw = core.ndarray(w.shape, w.dtype)
    dw[...] = 0
    cdef Descriptor dw_desc = create_filter_descriptor(dw)

    cudnn.RNNBackwardWeights(
        handle, rnn_desc.value, length,
        xs_descs.data, xs.data.ptr,
        hx_desc.value, hx.data.ptr, ys_descs.data, ys.data.ptr,
        workspace.ptr, workspace.mem.size, dw_desc.value, dw.data.ptr,
        reserve_space.ptr, reserve_space.mem.size)
    return dw


def create_dropout_states(handle):
    warnings.warn('create_dropout_states is deprecated.'
                  'Please use DropoutStates class instead.',
                  DeprecationWarning)
    state_size = cudnn.dropoutGetStatesSize(handle)
    return core.ndarray((state_size,), 'b')


def create_spatial_transformer_descriptor(sampler_type, dtype, nb_dims, dim_A):
    desc = Descriptor(cudnn.createSpatialTransformerDescriptor(),
                      py_cudnn.destroySpatialTransformerDescriptor)
    data_type = get_data_type(dtype)

    cudnn.setSpatialTransformerDescriptor(
        desc.value, sampler_type, data_type, nb_dims, dim_A)
    return desc


def add_tensor(handle, alpha, biasDesc, biasData, beta, srcDestDesc,
               srcDestData):
    cudnn.addTensor_v3(handle, alpha, biasDesc,
                       biasData, beta, srcDestDesc, srcDestData)


def create_op_tensor_descriptor(op_type, dtype):
    desc = Descriptor(cudnn.createOpTensorDescriptor(),
                      py_cudnn.destroyOpTensorDescriptor)
    data_type = get_data_type(dtype)

    cudnn.setOpTensorDescriptor(desc.value, op_type, data_type,
                                cudnn.CUDNN_NOT_PROPAGATE_NAN)
    return desc


def create_reduce_tensor_descriptor(reduce_type, dtype):
    desc = Descriptor(cudnn.createReduceTensorDescriptor(),
                      py_cudnn.destroyReduceTensorDescriptor)
    data_type = get_data_type(dtype)
    if reduce_type in (cudnn.CUDNN_REDUCE_TENSOR_MIN,
                       cudnn.CUDNN_REDUCE_TENSOR_MAX):
        indices = cudnn.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES
    else:
        indices = cudnn.CUDNN_REDUCE_TENSOR_NO_INDICES

    cudnn.setReduceTensorDescriptor(desc.value, reduce_type, data_type,
                                    cudnn.CUDNN_NOT_PROPAGATE_NAN,
                                    indices,
                                    cudnn.CUDNN_32BIT_INDICES)
    return desc


cpdef bint is_tensor_core_available(dtype) except *:
    return (_cudnn_version >= 7000 and
            (<str>dtype.char) == 'e' and
            int(device.get_compute_capability()) == 70)


cdef class DropoutStates:

    cdef public:
        # TODO(unno): Make these attributes private. This is for backward
        # compatibility.
        memory.MemoryPointer _states
        Descriptor _desc

    def __init__(self, handle, seed):
        cdef size_t cudnn_handle
        if handle is None:
            cudnn_handle = get_handle()
        else:
            cudnn_handle = handle
        state_size = cudnn.dropoutGetStatesSize(cudnn_handle)
        self._states = memory.alloc(state_size)
        self._desc = create_dropout_descriptor(
            cudnn_handle, 0., self._states.ptr,
            state_size, seed)

    def set_dropout_ratio(self, dropout_ratio):
        cudnn_handle = get_handle()
        set_dropout_descriptor(self._desc, cudnn_handle, dropout_ratio)

    def forward(self, handle, core.ndarray x, dropout_ratio):
        cdef core.ndarray y, reserve_space
        cdef size_t cudnn_handle
        # This is for backward compatibility.
        if handle is None:
            cudnn_handle = get_handle()
        else:
            cudnn_handle = handle
        set_dropout_descriptor(self._desc, cudnn_handle, dropout_ratio)

        x = core.ascontiguousarray(x)
        y = core.ndarray(x._shape, x.dtype)

        x_desc = cudnn.createTensorDescriptor()
        try:
            _create_tensor_descriptor_as4darray(x_desc, x)
            reserve_size = cudnn.getDropoutReserveSpaceSize(x_desc)
            reserve_space = core.ndarray((reserve_size,), 'b')

            cudnn.dropoutForward(cudnn_handle, self._desc.value,
                                 x_desc, x.data.ptr, x_desc, y.data.ptr,
                                 reserve_space.data.ptr, reserve_size)
        finally:
            cudnn.destroyTensorDescriptor(x_desc)
        return reserve_space, y

    def backward(self, handle, core.ndarray dy, dropout_ratio,
                 core.ndarray reserve_space):
        cdef core.ndarray dx
        cdef size_t cudnn_handle
        # This is for backward compatibility.
        if handle is None:
            cudnn_handle = get_handle()
        else:
            cudnn_handle = handle
        set_dropout_descriptor(self._desc, cudnn_handle, dropout_ratio)

        dy = core.ascontiguousarray(dy)
        dx = core.ndarray(dy._shape, dy.dtype)

        dy_desc = cudnn.createTensorDescriptor()
        try:
            _create_tensor_descriptor_as4darray(dy_desc, dy)
            cudnn.dropoutBackward(cudnn_handle, self._desc.value,
                                  dy_desc, dy.data.ptr,
                                  dy_desc, dx.data.ptr,
                                  reserve_space.data.ptr,
                                  reserve_space.size)
        finally:
            cudnn.destroyTensorDescriptor(dy_desc)
        return dx


cdef class _Algorithm:
    cdef:
        int algo
        int mathType
        size_t memory


cdef _Algorithm _get_algorithm(int algo, size_t memory, int mathType=0):
    cdef _Algorithm ret = _Algorithm.__new__(_Algorithm)
    ret.algo = algo
    ret.mathType = mathType
    ret.memory = memory
    return ret


cdef dict _algorithm_fwd_cache = {}
cdef dict _algorithm_bwd_filter_cache = {}
cdef dict _algorithm_bwd_data_cache = {}


cpdef _warn_algorithm_fwd(
        core.ndarray x, core.ndarray W, core.ndarray y, tuple conv_param):
    warnings.warn(
        'Tensor Core mode is set but the selected convolution forward '
        'algorithm is not a Tensor Core enabled algorithm. '
        'This might be due to lack of workspace memory. '
        'x.shape:{}, W.shape:{}, y.shape:{}, pad:{}, stride:{}'
        .format(x.shape, W.shape, y.shape, conv_param[0], conv_param[1]),
        RuntimeWarning)


cpdef _Algorithm _find_algorithm_fwd(
        core.ndarray x, core.ndarray W, core.ndarray y, tuple conv_param,
        size_t handle, size_t x_desc, size_t filter_desc, size_t conv_desc,
        size_t y_desc, size_t max_workspace_size, bint use_tensor_core):
    cdef _Algorithm algo
    key = (x.data.device.id, x.shape, W.shape, y.shape, conv_param,
           max_workspace_size)
    algo = _algorithm_fwd_cache.get(key, None)
    if algo is not None:
        return algo
    workspace = memory.alloc(max_workspace_size)
    if _cudnn_version >= 7000:
        perf = cudnn.findConvolutionForwardAlgorithmEx_v7(
            handle, x_desc, x.data.ptr, filter_desc, W.data.ptr, conv_desc,
            y_desc, y.data.ptr, 1, workspace.ptr, max_workspace_size)[0]
        if use_tensor_core and perf.mathType != cudnn.CUDNN_TENSOR_OP_MATH:
            _warn_algorithm_fwd(x, W, y, conv_param)
        algo = _get_algorithm(perf.algo, perf.memory, perf.mathType)
    else:
        perf_old = cudnn.findConvolutionForwardAlgorithmEx(
            handle, x_desc, x.data.ptr, filter_desc, W.data.ptr, conv_desc,
            y_desc, y.data.ptr, 1, workspace.ptr, max_workspace_size)[0]
        algo = _get_algorithm(
            perf_old['algo'], perf_old['memory'], cudnn.CUDNN_DEFAULT_MATH)
    _algorithm_fwd_cache[key] = algo
    return algo


cpdef _Algorithm _get_algorithm_fwd(
        core.ndarray x, core.ndarray W, core.ndarray y, tuple conv_param,
        size_t handle, size_t x_desc, size_t filter_desc, size_t conv_desc,
        size_t y_desc, size_t max_workspace_size, bint use_tensor_core):
    cdef list ret
    if use_tensor_core and _cudnn_version >= 7000:
        ret = cudnn.getConvolutionForwardAlgorithm_v7(
            handle, x_desc, filter_desc, conv_desc, y_desc, 10)
        for i, perf in enumerate(ret):
            if perf.memory <= max_workspace_size:
                break
        else:
            raise RuntimeError('No conv fwd algo available with workspace size'
                               ' less equal {}'.format(max_workspace_size))
        if i != 0:
            warnings.warn(
                'The best algo of conv fwd might not be selected due to '
                'lack of workspace size ({})'.format(max_workspace_size))
        algo = perf.algo
        workspace_size = perf.memory
        math_type = perf.mathType
        if math_type != cudnn.CUDNN_TENSOR_OP_MATH:
            _warn_algorithm_fwd(x, W, y, conv_param)
    else:
        algo = cudnn.getConvolutionForwardAlgorithm_v6(
            handle, x_desc, filter_desc, conv_desc, y_desc,
            cudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            max_workspace_size)
        workspace_size = cudnn.getConvolutionForwardWorkspaceSize(
            handle, x_desc, filter_desc, conv_desc, y_desc, algo)
        math_type = cudnn.CUDNN_DEFAULT_MATH
    return _get_algorithm(algo, workspace_size, math_type)


cpdef _warn_algorithm_bwd_filter(
        core.ndarray x, core.ndarray dy, core.ndarray dW, tuple conv_param):
    warnings.warn(
        'Tensor Core mode is set but the selected convolution backward '
        'filter algorithm is not a Tensor Core enabled algorithm. '
        'This might be due to lack of workspace memory. '
        'x.shape:{}, dy.shape:{}, dW.shape:{}, pad:{}, stride:{}'
        .format(x.shape, dy.shape, dW.shape, conv_param[0], conv_param[1]),
        RuntimeWarning)


cpdef _Algorithm _find_algorithm_bwd_filter(
        core.ndarray x, core.ndarray dy, core.ndarray dW, tuple conv_param,
        size_t handle, size_t x_desc, size_t dy_desc, size_t conv_desc,
        size_t filter_desc, size_t max_workspace_size, bint use_tensor_core):
    cdef _Algorithm algo
    key = (x.data.device.id, x.shape, dW.shape, dy.shape, conv_param,
           max_workspace_size)
    algo = _algorithm_bwd_filter_cache.get(key, None)
    if algo is not None:
        return algo
    workspace = memory.alloc(max_workspace_size)
    if _cudnn_version >= 7000:
        perf = cudnn.findConvolutionBackwardFilterAlgorithmEx_v7(
            handle, x_desc, x.data.ptr, dy_desc, dy.data.ptr, conv_desc,
            filter_desc, dW.data.ptr, 1, workspace.ptr, max_workspace_size)[0]
        algo = _get_algorithm(perf.algo, perf.memory, perf.mathType)
        if use_tensor_core and perf.mathType != cudnn.CUDNN_TENSOR_OP_MATH:
            _warn_algorithm_bwd_filter(x, dy, dW, conv_param)
    else:
        perf_old = cudnn.findConvolutionBackwardFilterAlgorithmEx(
            handle, x_desc, x.data.ptr, dy_desc, dy.data.ptr, conv_desc,
            filter_desc, dW.data.ptr, 1, workspace.ptr, max_workspace_size)[0]
        algo = _get_algorithm(
            perf_old['algo'], perf_old['memory'], cudnn.CUDNN_DEFAULT_MATH)
    _algorithm_bwd_filter_cache[key] = algo
    return algo


cpdef _Algorithm _get_algorithm_bwd_filter(
        core.ndarray x, core.ndarray dy, core.ndarray dW, tuple conv_param,
        size_t handle, size_t x_desc, size_t gy_desc, size_t conv_desc,
        size_t filter_desc, size_t max_workspace_size, bint use_tensor_core):
    cdef list ret
    if use_tensor_core and _cudnn_version >= 7000:
        ret = cudnn.getConvolutionBackwardFilterAlgorithm_v7(
            handle, x_desc, gy_desc, conv_desc, filter_desc, 10)
        for i, perf in enumerate(ret):
            if perf.memory <= max_workspace_size:
                break
        else:
            raise RuntimeError(
                'No conv bwd filter algo available with workspace size less '
                'equal {}'.format(max_workspace_size))
        if i != 0:
            warnings.warn(
                'The best algo of conv bwd filter might not not selected due '
                'to lack of workspace size ({})'.format(max_workspace_size))
        algo = perf.algo
        workspace_size = perf.memory
        math_type = perf.mathType
        if math_type != cudnn.CUDNN_TENSOR_OP_MATH:
            _warn_algorithm_bwd_filter(x, dy, dW, conv_param)
    else:
        algo = cudnn.getConvolutionBackwardFilterAlgorithm_v6(
            handle, x_desc, gy_desc, conv_desc, filter_desc,
            cudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            max_workspace_size)
        workspace_size = cudnn.getConvolutionBackwardFilterWorkspaceSize(
            handle, x_desc, gy_desc, conv_desc, filter_desc, algo)
        math_type = cudnn.CUDNN_DEFAULT_MATH
    return _get_algorithm(algo, workspace_size, math_type)


cpdef _warn_algorithm_bwd_data(
        core.ndarray W, core.ndarray x, core.ndarray y, tuple conv_param):
    warnings.warn(
        'Tensor Core mode is set but the selected convolution backward '
        'filter algorithm is not a Tensor Core enabled algorithm. '
        'This might be due to lack of workspace memory. '
        'W.shape:{}, x.shape:{}, y.shape:{}, pad:{}, stride:{}'
        .format(W.shape, x.shape, y.shape, conv_param[0], conv_param[1]),
        RuntimeWarning)


cpdef _Algorithm _find_algorithm_bwd_data(
        core.ndarray W, core.ndarray x, core.ndarray y, tuple conv_param,
        size_t handle, size_t filter_desc, size_t x_desc, size_t conv_desc,
        size_t y_desc, size_t max_workspace_size, bint use_tensor_core):
    cdef _Algorithm algo
    key = (x.data.device.id, W.shape, x.shape, y.shape, conv_param,
           max_workspace_size)
    algo = _algorithm_bwd_data_cache.get(key, None)
    if algo is not None:
        return algo
    workspace = memory.alloc(max_workspace_size)
    if _cudnn_version >= 7000:
        perf = cudnn.findConvolutionBackwardDataAlgorithmEx_v7(
            handle, filter_desc, W.data.ptr, x_desc, x.data.ptr, conv_desc,
            y_desc, y.data.ptr, 1, workspace.ptr, max_workspace_size)[0]
        if use_tensor_core:
            if perf.mathType != cudnn.CUDNN_TENSOR_OP_MATH:
                _warn_algorithm_bwd_data(W, x, y, conv_param)
        algo = _get_algorithm(perf.algo, perf.memory, perf.mathType)
    else:
        perf_old = cudnn.findConvolutionBackwardDataAlgorithmEx(
            handle, filter_desc, W.data.ptr, x_desc, x.data.ptr, conv_desc,
            y_desc, y.data.ptr, 1, workspace.ptr, max_workspace_size)[0]
        algo = _get_algorithm(
            perf_old['algo'], perf_old['memory'], cudnn.CUDNN_DEFAULT_MATH)
    _algorithm_bwd_data_cache[key] = algo
    return algo


cpdef _Algorithm _get_algorithm_bwd_data(
        core.ndarray W, core.ndarray x, core.ndarray y, tuple conv_param,
        size_t handle, size_t filter_desc, size_t x_desc, size_t conv_desc,
        size_t y_desc, size_t max_workspace_size, bint use_tensor_core):
    cdef list ret
    if use_tensor_core and _cudnn_version >= 7000:
        ret = cudnn.getConvolutionBackwardDataAlgorithm_v7(
            handle, filter_desc, x_desc, conv_desc, y_desc, 10)
        for i, perf in enumerate(ret):
            if perf.memory <= max_workspace_size:
                break
        else:
            raise RuntimeError(
                'No conv bwd data algo available with workspace size less '
                'equal {}'.format(max_workspace_size))
        if i != 0:
            warnings.warn(
                'The best algo of conv bwd data might not not selected due '
                'to lack of workspace size ({})'.format(max_workspace_size))
        algo = perf.algo
        workspace_size = perf.memory
        math_type = perf.mathType
        if math_type != cudnn.CUDNN_TENSOR_OP_MATH:
            _warn_algorithm_bwd_data(W, x, y, conv_param)
    else:
        algo = cudnn.getConvolutionBackwardDataAlgorithm_v6(
            handle, filter_desc, x_desc, conv_desc, y_desc,
            cudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            max_workspace_size)
        workspace_size = cudnn.getConvolutionBackwardDataWorkspaceSize(
            handle, filter_desc, x_desc, conv_desc, y_desc, algo)
        math_type = cudnn.CUDNN_DEFAULT_MATH
    return _get_algorithm(algo, workspace_size, math_type)


cpdef bint _should_use_tensor_core(
        str tensor_core_mode, object dtype) except *:
    if tensor_core_mode == 'auto':
        return is_tensor_core_available(dtype)
    elif tensor_core_mode == 'always':
        # TODO(oktua): more strict condition
        return is_tensor_core_available(dtype)
    elif tensor_core_mode == 'never':
        return False
    else:
        raise ValueError(
            'tensor_code_mode must be either of "always", "auto", or "never".')


def convolution_forward(
        core.ndarray x, core.ndarray W, core.ndarray b, core.ndarray y,
        tuple pad, tuple stride, tuple dilation, int groups, *,
        bint auto_tune, str tensor_core):
    cdef int dev_id = x.data.device.id
    assert dev_id == W.data.device.id
    assert dev_id == y.data.device.id

    cdef float float_zero = 0, float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero, one
    if x.dtype == 'd':
        zero = <size_t>&double_zero
        one = <size_t>&double_one
    else:
        zero = <size_t>&float_zero
        one = <size_t>&float_one

    cdef bint use_tensor_core = _should_use_tensor_core(tensor_core, x.dtype)
    cdef tuple conv_param = (pad, stride, x.dtype, use_tensor_core)

    # cuDNN 7 supports dilation only in *_FWD_ALGO_IMPLICIT_GEMM, but
    # it supports Tensor Cores only in *_FWD_ALGO_IMPLICIT_PRECOMP_GEMM.
    if use_tensor_core:
        for i in dilation:
            if i > 1:
                use_tensor_core = False
                break

    handle = get_handle()
    x = core.ascontiguousarray(x)
    W = core.ascontiguousarray(W)

    # TODO(okuta) check performance
    cdef size_t x_desc = cudnn.createTensorDescriptor()
    cdef size_t y_desc = cudnn.createTensorDescriptor()
    cdef size_t b_desc = cudnn.createTensorDescriptor()
    cdef size_t filter_desc = cudnn.createFilterDescriptor()
    cdef size_t conv_desc = cudnn.createConvolutionDescriptor()

    cdef size_t max_workspace_size = get_max_workspace_size()
    cdef vector.vector[Py_ssize_t] b_shape
    cdef _Algorithm perf
    try:
        _create_tensor_nd_descriptor(x_desc, x, -1)
        _create_tensor_nd_descriptor(y_desc, y, -1)
        _create_filter_descriptor(filter_desc, W, cudnn.CUDNN_TENSOR_NCHW)
        _create_convolution_descriptor(
            conv_desc, pad, stride, dilation, groups, x.dtype,
            cudnn.CUDNN_CROSS_CORRELATION, use_tensor_core)

        if auto_tune:
            perf = _find_algorithm_fwd(
                x, W, y, conv_param, handle, x_desc, filter_desc,
                conv_desc, y_desc, max_workspace_size, use_tensor_core)
        else:
            perf = _get_algorithm_fwd(
                x, W, y, conv_param, handle, x_desc, filter_desc,
                conv_desc, y_desc, max_workspace_size, use_tensor_core)

        if _cudnn_version >= 7000:
            cudnn.setConvolutionMathType(conv_desc, perf.mathType)

        workspace = memory.alloc(perf.memory)

        cudnn.convolutionForward(
            handle, one, x_desc, x.data.ptr, filter_desc, W.data.ptr,
            conv_desc, perf.algo, workspace.ptr, perf.memory, zero, y_desc,
            y.data.ptr)
        del workspace, x, W

        if b is not None:
            assert dev_id == b.data.device.id
            b_shape.assign(y._shape.size(), 1)
            b_shape[1] = -1
            b = core.ascontiguousarray(b)._reshape(b_shape)
            _create_tensor_nd_descriptor(b_desc, b, -1)
            cudnn.addTensor_v3(handle, one, b_desc,
                               b.data.ptr, one, y_desc, y.data.ptr)
    finally:
        cudnn.destroyTensorDescriptor(x_desc)
        cudnn.destroyTensorDescriptor(y_desc)
        cudnn.destroyTensorDescriptor(b_desc)
        cudnn.destroyFilterDescriptor(filter_desc)
        cudnn.destroyConvolutionDescriptor(conv_desc)


def convolution_backward_filter(
        core.ndarray x, core.ndarray gy, core.ndarray gW,
        tuple pad, tuple stride, tuple dilation, int groups, *,
        bint deterministic, bint auto_tune, str tensor_core):
    cdef int dev_id = x.data.device.id
    assert dev_id == gy.data.device.id
    assert dev_id == gW.data.device.id

    cdef float float_zero = 0, float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero, one
    if x.dtype == 'd':
        zero = <size_t>&double_zero
        one = <size_t>&double_one
    else:
        zero = <size_t>&float_zero
        one = <size_t>&float_one

    cdef bint use_tensor_core = _should_use_tensor_core(tensor_core, x.dtype)
    cdef tuple conv_param = (pad, stride, x.dtype, use_tensor_core)

    handle = get_handle()
    x = core.ascontiguousarray(x)
    gy = core.ascontiguousarray(gy)

    # TODO(okuta) check performance
    cdef size_t x_desc = cudnn.createTensorDescriptor()
    cdef size_t gy_desc = cudnn.createTensorDescriptor()
    cdef size_t filter_desc = cudnn.createFilterDescriptor()
    cdef size_t conv_desc = cudnn.createConvolutionDescriptor()

    cdef _Algorithm perf
    cdef int algo
    cdef size_t max_workspace_size = get_max_workspace_size()
    cdef size_t workspace_size = 0
    try:
        _create_tensor_nd_descriptor(x_desc, x, -1)
        _create_tensor_nd_descriptor(gy_desc, gy, -1)
        _create_filter_descriptor(filter_desc, gW, cudnn.CUDNN_TENSOR_NCHW)
        _create_convolution_descriptor(
            conv_desc, pad, stride, dilation, groups, x.dtype,
            cudnn.CUDNN_CROSS_CORRELATION, use_tensor_core)

        if deterministic:
            algo = cudnn.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
            workspace_size = cudnn.getConvolutionBackwardFilterWorkspaceSize(
                handle, x_desc, gy_desc, conv_desc, filter_desc, algo)
            math_type = cudnn.CUDNN_DEFAULT_MATH
            # TODO(okuta): check workspace size
        else:
            if auto_tune:
                perf = _find_algorithm_bwd_filter(
                    x, gy, gW, conv_param, handle, x_desc, gy_desc, conv_desc,
                    filter_desc, max_workspace_size, use_tensor_core)
            else:
                perf = _get_algorithm_bwd_filter(
                    x, gy, gW, conv_param, handle, x_desc, gy_desc, conv_desc,
                    filter_desc, max_workspace_size, use_tensor_core)
            algo = perf.algo
            workspace_size = perf.memory
            math_type = perf.mathType

        if _cudnn_version >= 7000:
            cudnn.setConvolutionMathType(conv_desc, math_type)

        workspace = memory.alloc(workspace_size)

        cudnn.convolutionBackwardFilter_v3(
            handle, one, x_desc, x.data.ptr, gy_desc,
            gy.data.ptr, conv_desc, algo, workspace.ptr,
            workspace_size, zero, filter_desc, gW.data.ptr)
    finally:
        cudnn.destroyTensorDescriptor(x_desc)
        cudnn.destroyTensorDescriptor(gy_desc)
        cudnn.destroyFilterDescriptor(filter_desc)
        cudnn.destroyConvolutionDescriptor(conv_desc)


def convolution_backward_data(
        core.ndarray W, core.ndarray x, core.ndarray b, core.ndarray y,
        tuple pad, tuple stride, tuple dilation, int groups, *,
        bint deterministic, bint auto_tune, str tensor_core):
    cdef int dev_id = W.data.device.id
    assert dev_id == x.data.device.id
    assert dev_id == y.data.device.id

    cdef float float_zero = 0, float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero, one
    if x.dtype == 'd':
        zero = <size_t>&double_zero
        one = <size_t>&double_one
    else:
        zero = <size_t>&float_zero
        one = <size_t>&float_one

    cdef bint use_tensor_core = _should_use_tensor_core(tensor_core, x.dtype)
    cdef tuple conv_param = (pad, stride, x.dtype, use_tensor_core)

    # cuDNN 7 supports dilation only in *_FWD_ALGO_IMPLICIT_GEMM, but
    # it supports Tensor Cores only in *_FWD_ALGO_IMPLICIT_PRECOMP_GEMM.
    if use_tensor_core:
        for i in dilation:
            if i > 1:
                use_tensor_core = False
                break

    handle = get_handle()
    x = core.ascontiguousarray(x)
    W = core.ascontiguousarray(W)

    # TODO(okuta) check performance
    cdef size_t x_desc = cudnn.createTensorDescriptor()
    cdef size_t y_desc = cudnn.createTensorDescriptor()
    cdef size_t b_desc = cudnn.createTensorDescriptor()
    cdef size_t filter_desc = cudnn.createFilterDescriptor()
    cdef size_t conv_desc = cudnn.createConvolutionDescriptor()

    cdef _Algorithm perf
    cdef int algo
    cdef size_t max_workspace_size = get_max_workspace_size()
    cdef size_t workspace_size = 0
    cdef vector.vector[Py_ssize_t] b_shape
    try:
        _create_tensor_nd_descriptor(x_desc, x, -1)
        _create_tensor_nd_descriptor(y_desc, y, -1)
        _create_filter_descriptor(filter_desc, W, cudnn.CUDNN_TENSOR_NCHW)
        _create_convolution_descriptor(
            conv_desc, pad, stride, dilation, groups, x.dtype,
            cudnn.CUDNN_CROSS_CORRELATION, use_tensor_core)

        if deterministic:
            algo = cudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
            workspace_size = cudnn.getConvolutionBackwardDataWorkspaceSize(
                handle, filter_desc, x_desc, conv_desc, y_desc, algo)
            math_type = cudnn.CUDNN_DEFAULT_MATH
            # TODO(okuta): check workspace size
        else:
            if auto_tune:
                perf = _find_algorithm_bwd_data(
                    W, x, y, conv_param, handle, filter_desc, x_desc,
                    conv_desc, y_desc, max_workspace_size, use_tensor_core)
            else:
                perf = _get_algorithm_bwd_data(
                    W, x, y, conv_param, handle, filter_desc, x_desc,
                    conv_desc, y_desc, max_workspace_size, use_tensor_core)
            algo = perf.algo
            workspace_size = perf.memory
            math_type = perf.mathType

        if _cudnn_version >= 7000:
            cudnn.setConvolutionMathType(conv_desc, math_type)

        workspace = memory.alloc(workspace_size)

        cudnn.convolutionBackwardData_v3(
            handle, one, filter_desc, W.data.ptr, x_desc, x.data.ptr,
            conv_desc, algo, workspace.ptr, workspace_size, zero, y_desc,
            y.data.ptr)

        del workspace, x, W

        if b is not None:
            assert dev_id == b.data.device.id
            b_shape.assign(y._shape.size(), 1)
            b_shape[1] = -1
            b = core.ascontiguousarray(b)._reshape(b_shape)
            _create_tensor_nd_descriptor(b_desc, b, -1)
            cudnn.addTensor_v3(handle, one, b_desc, b.data.ptr, one, y_desc,
                               y.data.ptr)
    finally:
        cudnn.destroyTensorDescriptor(x_desc)
        cudnn.destroyTensorDescriptor(y_desc)
        cudnn.destroyTensorDescriptor(b_desc)
        cudnn.destroyFilterDescriptor(filter_desc)
        cudnn.destroyConvolutionDescriptor(conv_desc)


def pooling_forward(
        core.ndarray x, core.ndarray y,
        tuple ksize, tuple stride, tuple pad, int mode):
    cdef float float_zero = 0, float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero, one
    if x.dtype == 'd':
        zero = <size_t>&double_zero
        one = <size_t>&double_one
    else:
        zero = <size_t>&float_zero
        one = <size_t>&float_one
    x = core.ascontiguousarray(x)
    handle = get_handle()
    x_desc = cudnn.createTensorDescriptor()
    y_desc = cudnn.createTensorDescriptor()
    pool_desc = cudnn.createPoolingDescriptor()
    try:
        _create_tensor_nd_descriptor(x_desc, x)
        _create_tensor_nd_descriptor(y_desc, y)
        _create_pooling_descriptor(pool_desc, ksize, stride, pad, mode)
        cudnn.poolingForward(
            handle, pool_desc, one, x_desc,
            x.data.ptr, zero, y_desc, y.data.ptr)
    finally:
        cudnn.destroyTensorDescriptor(x_desc)
        cudnn.destroyTensorDescriptor(y_desc)
        cudnn.destroyPoolingDescriptor(pool_desc)
    return y


def pooling_backward(
        core.ndarray x, core.ndarray y, core.ndarray gy,
        tuple ksize, tuple stride, tuple pad, int mode):
    cdef float float_zero = 0, float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero, one
    cdef core.ndarray gx
    if x.dtype == 'd':
        zero = <size_t>&double_zero
        one = <size_t>&double_one
    else:
        zero = <size_t>&float_zero
        one = <size_t>&float_one

    gx = core.ndarray(x._shape, x.dtype)
    x = core.ascontiguousarray(x)
    gy = core.ascontiguousarray(gy)

    handle = get_handle()
    x_desc = cudnn.createTensorDescriptor()
    y_desc = cudnn.createTensorDescriptor()
    pool_desc = cudnn.createPoolingDescriptor()
    try:
        _create_tensor_nd_descriptor(x_desc, x)
        _create_tensor_nd_descriptor(y_desc, y)
        _create_pooling_descriptor(pool_desc, ksize, stride, pad, mode)
        cudnn.poolingBackward(
            handle, pool_desc,
            one, y_desc, y.data.ptr, y_desc, gy.data.ptr,
            x_desc, x.data.ptr, zero, x_desc, gx.data.ptr)
    finally:
        cudnn.destroyTensorDescriptor(x_desc)
        cudnn.destroyTensorDescriptor(y_desc)
        cudnn.destroyPoolingDescriptor(pool_desc)
    return gx


cdef _create_tensor_descriptor_for_bn(
        size_t desc, core.ndarray arr, bint is_for_conv2d):
    assert arr._c_contiguous
    data_type = get_data_type(arr.dtype)
    if is_for_conv2d:
        _create_tensor_nd_descriptor(desc, arr, data_type)
        return
    cdef Py_ssize_t dim1, dim2
    cdef int ndim = arr._shape.size()
    dim2 = 1
    if ndim > 0:
        dim2 = arr._shape[ndim - 1]
    dim1 = arr.size // dim2
    cudnn.setTensor4dDescriptor(desc, cudnn.CUDNN_TENSOR_NCHW, data_type,
                                dim1, dim2, 1, 1)


cdef _get_dtype_of_tensor_descriptor(size_t desc):
    cudnn_dtype, _, _, _, _, _, _, _, _ = cudnn.getTensor4dDescriptor(desc)
    if cudnn_dtype == cudnn.CUDNN_DATA_DOUBLE:
        return numpy.dtype(numpy.float64)
    elif cudnn_dtype == cudnn.CUDNN_DATA_FLOAT:
        return numpy.dtype(numpy.float32)
    elif cudnn_dtype == cudnn.CUDNN_DATA_HALF:
        return numpy.dtype(numpy.float16)
    else:
        raise RuntimeError('Unknown cudnn data type {} '.format(cudnn_dtype))


def batch_normalization_forward_training(
        core.ndarray x, core.ndarray gamma, core.ndarray beta,
        core.ndarray running_mean, core.ndarray running_var,
        core.ndarray mean, core.ndarray inv_std,
        double eps, double decay,
        bint is_for_conv2d, int cudnn_mode, bint debug):
    x = core.ascontiguousarray(x)
    dtype = x.dtype
    y = core.ndarray(x._shape, dtype)

    cdef float float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero = <size_t>&double_zero, one
    if x.dtype == 'd':
        one = <size_t>&double_one
    else:
        one = <size_t>&float_one

    handle = get_handle()
    cdef size_t x_desc = cudnn.createTensorDescriptor()
    cdef size_t derivedBnDesc = cudnn.createTensorDescriptor()
    try:
        _create_tensor_descriptor_for_bn(x_desc, x, is_for_conv2d)
        cudnn.deriveBNTensorDescriptor(derivedBnDesc, x_desc, cudnn_mode)
        dtype_param = _get_dtype_of_tensor_descriptor(derivedBnDesc)
        if dtype_param != dtype:
            gamma = gamma.astype(dtype_param)
            beta = beta.astype(dtype_param)
            running_mean_tmp = running_mean.astype(dtype_param)
            running_var_tmp = running_var.astype(dtype_param)
        else:
            running_mean_tmp = running_mean
            running_var_tmp = running_var
            gamma = core.ascontiguousarray(gamma)
            beta = core.ascontiguousarray(beta)

        # Factor used in the moving average
        factor = 1.0 - decay

        # Note: cuDNN computes the mini-batch mean and variance
        # internally. We can simply (optionally) pass
        # it the running-average mean and variance arrays.
        # Note: This API seems to set the inverse of the standard deviation
        # (instead of variance) to resultSaveInvVariance argument. The
        # current implementation of our BN depends on this behavior so that
        # we can reduce the number of reduction kernels.
        cudnn.batchNormalizationForwardTraining(
            handle, cudnn_mode, one, zero,
            x_desc, x.data.ptr, x_desc, y.data.ptr,
            derivedBnDesc, gamma.data.ptr,
            beta.data.ptr, factor, running_mean_tmp.data.ptr,
            running_var_tmp.data.ptr, eps,
            mean.data.ptr, inv_std.data.ptr)

        # Note: When the CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode is used,
        # there is a possibility of numerical overflow. You can use
        # queryRuntimeError() to make sure whether the overflow actually
        # occured or not during the batch normalization.
        if debug and cudnn_mode == cudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT:
            query_mode = cudnn.CUDNN_ERRQUERY_BLOCKING
            rstatus = cudnn.queryRuntimeError(handle, query_mode)
            if rstatus != cudnn.CUDNN_STATUS_SUCCESS:
                warnings.warn(
                    'A numerical overflow might have happend in cuDNN'
                    'batch normalization (status:{})'.format(rstatus))
    finally:
        cudnn.destroyTensorDescriptor(x_desc)
        cudnn.destroyTensorDescriptor(derivedBnDesc)
    if running_mean is not running_mean_tmp:
        running_mean[...] = running_mean_tmp
        running_var[...] = running_var_tmp
    return y


def batch_normalization_forward_inference(
        core.ndarray x, core.ndarray gamma, core.ndarray beta,
        core.ndarray mean, core.ndarray var,
        double eps, bint is_for_conv2d, int cudnn_mode):
    x = core.ascontiguousarray(x)
    dtype = x.dtype
    y = core.ndarray(x._shape, dtype)

    cdef float float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero = <size_t>&double_zero, one
    if x.dtype == 'd':
        one = <size_t>&double_one
    else:
        one = <size_t>&float_one

    handle = get_handle()
    cdef size_t x_desc = cudnn.createTensorDescriptor()
    cdef size_t derivedBnDesc = cudnn.createTensorDescriptor()
    try:
        _create_tensor_descriptor_for_bn(x_desc, x, is_for_conv2d)
        cudnn.deriveBNTensorDescriptor(derivedBnDesc, x_desc, cudnn_mode)
        dtype_param = _get_dtype_of_tensor_descriptor(derivedBnDesc)
        if dtype_param != dtype:
            gamma = gamma.astype(dtype_param)
            beta = beta.astype(dtype_param)
            mean = mean.astype(dtype_param)
            var = var.astype(dtype_param)
        else:
            gamma = core.ascontiguousarray(gamma)
            beta = core.ascontiguousarray(beta)

        cudnn.batchNormalizationForwardInference(
            handle, cudnn_mode, one, zero,
            x_desc, x.data.ptr, x_desc, y.data.ptr,
            derivedBnDesc, gamma.data.ptr, beta.data.ptr,
            mean.data.ptr, var.data.ptr, eps)
    finally:
        cudnn.destroyTensorDescriptor(x_desc)
        cudnn.destroyTensorDescriptor(derivedBnDesc)
    return y


def batch_normalization_backward(
        core.ndarray x, core.ndarray gamma, core.ndarray gy,
        core.ndarray mean, core.ndarray inv_std,
        double eps, bint is_for_conv2d, int cudnn_mode, bint debug):
    x = core.ascontiguousarray(x)
    dtype = x.dtype
    gx = core.ndarray(x._shape, dtype)
    ggamma = core.ndarray(gamma._shape, gamma.dtype)
    gbeta = core.ndarray(gamma._shape, gamma.dtype)

    cdef float float_one = 1
    cdef double double_zero = 0, double_one = 1
    cdef size_t zero = <size_t>&double_zero, one
    if x.dtype == 'd':
        one = <size_t>&double_one
    else:
        one = <size_t>&float_one

    handle = get_handle()
    cdef size_t x_desc = cudnn.createTensorDescriptor()
    cdef size_t derivedBnDesc = cudnn.createTensorDescriptor()
    try:
        _create_tensor_descriptor_for_bn(x_desc, x, is_for_conv2d)
        cudnn.deriveBNTensorDescriptor(derivedBnDesc, x_desc, cudnn_mode)
        dtype_param = _get_dtype_of_tensor_descriptor(derivedBnDesc)
        if dtype_param != dtype:
            gamma = gamma.astype(dtype_param)
        else:
            gamma = core.ascontiguousarray(gamma)
            gy = core.ascontiguousarray(gy)

        cudnn.batchNormalizationBackward(
            handle, cudnn_mode, one, zero, one, zero,
            x_desc, x.data.ptr,
            x_desc, gy.data.ptr, x_desc, gx.data.ptr,
            derivedBnDesc, gamma.data.ptr, ggamma.data.ptr, gbeta.data.ptr,
            eps, mean.data.ptr, inv_std.data.ptr)

        # Note: When the CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode is used,
        # there is a possibility of numerical overflow. You can use
        # queryRuntimeError() to make sure whether the overflow actually
        # occured or not during the batch normalization.
        if debug and cudnn_mode == cudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT:
            query_mode = cudnn.CUDNN_ERRQUERY_BLOCKING
            rstatus = cudnn.queryRuntimeError(handle, query_mode)
            if rstatus != cudnn.CUDNN_STATUS_SUCCESS:
                warnings.warn(
                    'A numerical overflow might have happend in cuDNN'
                    'batch normalization (status:{})'.format(rstatus))
    finally:
        cudnn.destroyTensorDescriptor(x_desc)
        cudnn.destroyTensorDescriptor(derivedBnDesc)

    if dtype_param is not dtype:
        ggamma = ggamma.astype(dtype)
        gbeta = gbeta.astype(dtype)
    return gx, ggamma, gbeta
