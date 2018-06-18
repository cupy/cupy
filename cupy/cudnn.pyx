from libcpp cimport vector

import atexit
import threading
import warnings

import numpy

from cupy.core cimport core
from cupy.cuda cimport cudnn
from cupy.cuda cimport device
from cupy.cuda cimport memory

import cupy
from cupy.core import internal
from cupy.cuda import cudnn as py_cudnn


cdef int _cudnn_version = cudnn.getVersion()
cdef _thread_local = threading.local()

cdef vector.vector[size_t] _handles


cpdef size_t get_handle() except *:
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


class Descriptor(object):

    def __init__(self, descriptor, destroyer):
        self.value = descriptor
        self.destroy = destroyer

    def __del__(self):
        if self.value:
            self.destroy(self.value)
            self.value = None


cpdef int get_data_type(dtype) except? -1:
    t = dtype.type
    if t is numpy.float32:
        return cudnn.CUDNN_DATA_FLOAT
    elif t is numpy.float64:
        return cudnn.CUDNN_DATA_DOUBLE
    elif t is numpy.float16:
        return cudnn.CUDNN_DATA_HALF
    else:
        raise TypeError('Dtype {} is not supported in cuDNN'.format(dtype))


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
        desc, data_type, arr.ndim, <size_t>&c_shape[0], <size_t>&c_strides[0])


cpdef _create_tensor_descriptor(size_t desc, core.ndarray arr, int format):
    if not arr.flags.c_contiguous:
        raise ValueError('cupy.cudnn supports c-contiguous arrays only')
    data_type = get_data_type(arr.dtype)
    if arr._shape.size() == 4:
        n, c, h, w = arr.shape
        data_type = get_data_type(arr.dtype)
        cudnn.setTensor4dDescriptor(desc, format, data_type, n, c, h, w)
    else:
        _create_tensor_nd_descriptor(desc, arr)


cpdef _create_filter_descriptor(
        size_t desc, core.ndarray arr, int format=cudnn.CUDNN_TENSOR_NCHW):
    cdef vector.vector[int] c_shape
    cdef Py_ssize_t s
    data_type = get_data_type(arr.dtype)
    if arr._shape.size() == 4:
        n, c, h, w = arr.shape
        cudnn.setFilter4dDescriptor_v4(
            desc, data_type, format, n, c, h, w)
    else:
        for s in arr._shape:
            c_shape.push_back(s)
        cudnn.setFilterNdDescriptor_v4(
            desc, data_type, format, arr.ndim, <size_t>&c_shape[0])


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
        p0, p1 = pad
        s0, s1 = stride
        if _cudnn_version < 6000 and (d0 != 1 or d1 != 1):
            raise ValueError('dilation must be one when cuDNN < 6.0')
        if _cudnn_version >= 5000:
            cudnn.setConvolution2dDescriptor_v5(
                desc, p0, p1, s0, s1, d0, d1, mode, compute_type)
        else:
            cudnn.setConvolution2dDescriptor_v4(
                desc, p0, p1, s0, s1, 1, 1, mode)
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
    if not arr.flags.c_contiguous:
        raise ValueError('cupy.cudnn supports c-contiguous arrays only')
    data_type = get_data_type(arr.dtype)
    shape = arr.shape
    key = (data_type, shape)
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


def create_pooling_descriptor(ksize, stride, pad, mode):
    cdef vector.vector[int] c_ksize, c_pad, c_stride
    ndim = len(ksize)
    if ndim != len(stride) or ndim != len(pad):
        raise ValueError('ksize, stride, and pad must be of same length')
    desc = Descriptor(cudnn.createPoolingDescriptor(),
                      py_cudnn.destroyPoolingDescriptor)
    if ndim == 2:
        cudnn.setPooling2dDescriptor_v4(
            desc.value, mode, cudnn.CUDNN_NOT_PROPAGATE_NAN, ksize[0],
            ksize[1], pad[0], pad[1], stride[0], stride[1])
    else:
        c_ksize = ksize
        c_pad = pad
        c_stride = stride
        cudnn.setPoolingNdDescriptor_v4(
            desc.value, mode, cudnn.CUDNN_NOT_PROPAGATE_NAN, ndim,
            <size_t>&c_ksize[0], <size_t>&c_pad[0], <size_t>&c_stride[0])

    return desc


cpdef core.ndarray _as4darray(core.ndarray arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    return arr.reshape(arr.shape[0], -1, 1, 1)


def activation_forward(core.ndarray x, int mode, double coef=0.0):
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
    y = cupy.empty_like(x)
    x = _as4darray(x)

    handle = get_handle()
    desc = cudnn.createTensorDescriptor()
    act_desc = cudnn.createActivationDescriptor()
    try:
        _create_tensor_descriptor(desc, x, cudnn.CUDNN_TENSOR_NCHW)
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
    if x.dtype == 'd':
        zero = <size_t>&double_zero
        one = <size_t>&double_one
    else:
        zero = <size_t>&float_zero
        one = <size_t>&float_one

    gx = cupy.empty_like(x)
    x = core.ascontiguousarray(x)
    gy = core.ascontiguousarray(gy)
    y_mat = _as4darray(y)

    handle = get_handle()
    desc = cudnn.createTensorDescriptor()
    act_desc = cudnn.createActivationDescriptor()
    try:
        _create_tensor_descriptor(desc, y_mat, cudnn.CUDNN_TENSOR_NCHW)
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
        _, _, _, dim = cudnn.getFilterNdDescriptor(mat_desc, 3)
    finally:
        cudnn.destroyFilterDescriptor(mat_desc)
    offset = (ptr - w.data.ptr) // 4
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
        _, _, _, dim = cudnn.getFilterNdDescriptor(bias_desc, 3)
    finally:
        cudnn.destroyFilterDescriptor(bias_desc)
    offset = (ptr - w.data.ptr) // 4
    size = internal.prod(dim)
    bias = w[offset: offset + size]
    return bias


def create_dropout_states(handle):
    warnings.warn('create_dropout_states is deprecated.'
                  'Please use DropoutStates class instead.',
                  DeprecationWarning)
    state_size = cudnn.dropoutGetStatesSize(handle)
    return cupy.empty((state_size,), dtype='b')


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
            dtype == numpy.float16 and
            int(device.get_compute_capability()) == 70)


class DropoutStates(object):

    def __init__(self, handle, seed):
        state_size = cudnn.dropoutGetStatesSize(handle)
        self._states = memory.alloc(state_size)
        self._desc = create_dropout_descriptor(
            handle, 0., self._states.ptr,
            state_size, seed)

    def forward(self, handle, x, dropout_ratio):
        if not isinstance(x, cupy.ndarray):
            raise TypeError('argument x must be an cupy.ndarray')

        set_dropout_descriptor(self._desc, handle, dropout_ratio)

        x = cupy.ascontiguousarray(x)
        y = cupy.empty_like(x)

        x_mat = _as4darray(x)
        x_desc = create_tensor_descriptor(x_mat)

        reserve_size = cudnn.getDropoutReserveSpaceSize(x_desc.value)
        reserve_space = cupy.empty((reserve_size,), dtype='b')

        cudnn.dropoutForward(handle, self._desc.value,
                             x_desc.value, x_mat.data.ptr,
                             x_desc.value, y.data.ptr,
                             reserve_space.data.ptr, reserve_size)
        return (reserve_space, y)

    def backward(self, handle, dy, dropout_ratio, reserve_space):
        if not isinstance(dy, cupy.ndarray):
            raise TypeError('argument dy must be an cupy.ndarray')

        set_dropout_descriptor(self._desc, handle, dropout_ratio)

        dy = cupy.ascontiguousarray(dy)
        dx = cupy.empty_like(dy)

        dy_mat = _as4darray(dy)
        dy_desc = create_tensor_descriptor(dy_mat)

        cudnn.dropoutBackward(handle, self._desc.value,
                              dy_desc.value, dy_mat.data.ptr,
                              dy_desc.value, dx.data.ptr,
                              reserve_space.data.ptr,
                              reserve_space.size)
        return dx


cdef dict _algorithm_fwd = {}
cdef dict _algorithm_bwd_filter = {}
cdef dict _algorithm_bwd_data = {}


cpdef _warn_algorithm_fwd(
        core.ndarray x, core.ndarray W, core.ndarray y, tuple conv_param):
    msg = 'Tensor Core mode is set but the selected convolution forward '\
          'algorithm is not a Tensor Core enabled algorithm. '\
          'This might be due to lack of workspace memory. '\
          'x.shape:{}, W.shape:{}, y.shape:{}, pad:{}, stride:{}'\
          .format(x.shape, W.shape, y.shape, conv_param[0], conv_param[1])
    warnings.warn(msg, RuntimeWarning)


cpdef tuple _find_algorithm_fwd(
        core.ndarray x, core.ndarray W, core.ndarray y, tuple conv_param,
        size_t handle, size_t x_desc, size_t filter_desc, size_t conv_desc,
        size_t y_desc, size_t max_workspace_size, bint use_tensor_core):
    key = (x.data.device.id, x.shape, W.shape, y.shape, conv_param,
           max_workspace_size)
    if key in _algorithm_fwd:
        return _algorithm_fwd[key]
    workspace = memory.alloc(max_workspace_size)
    if _cudnn_version >= 7000:
        ret = cudnn.findConvolutionForwardAlgorithmEx_v7(
            handle, x_desc, x.data.ptr, filter_desc, W.data.ptr, conv_desc,
            y_desc, y.data.ptr, 1, workspace.ptr, max_workspace_size)
        algo = (ret[0].algo, ret[0].memory)
        if use_tensor_core:
            if ret[0].mathType != cudnn.CUDNN_TENSOR_OP_MATH:
                _warn_algorithm_fwd(x, W, y, conv_param)
    else:
        ret = cudnn.findConvolutionForwardAlgorithmEx(
            handle, x_desc, x.data.ptr, filter_desc, W.data.ptr, conv_desc,
            y_desc, y.data.ptr, 1, workspace.ptr, max_workspace_size)
        algo = (ret[0]['algo'], ret[0]['memory'])
    _algorithm_fwd[key] = algo
    return algo


cpdef tuple _get_algorithm_fwd(
        core.ndarray x, core.ndarray W, core.ndarray y, tuple conv_param,
        size_t handle, size_t x_desc, size_t filter_desc, size_t conv_desc,
        size_t y_desc, size_t max_workspace_size, bint use_tensor_core):
    cdef int algo
    cdef workspace_size
    if use_tensor_core and _cudnn_version >= 7000:
        ret = cudnn.getConvolutionForwardAlgorithm_v7(
            handle, x_desc, filter_desc, conv_desc, y_desc, 10)
        for i in range(len(ret)):
            if ret[i].memory <= max_workspace_size:
                break
        else:
            raise RuntimeError('No conv fwd algo available with workspace size'
                               ' less equal {}'.format(max_workspace_size))
        if i != 0:
            msg = 'The best algo of conv fwd might not be selected due to '\
                  'lack of workspace size ({})'.format(max_workspace_size)
            warnings.warn(msg)
        algo = ret[i].algo
        workspace_size = ret[i].memory
        if ret[i].mathType != cudnn.CUDNN_TENSOR_OP_MATH:
            _warn_algorithm_fwd(x, W, y, conv_param)
    else:
        algo = cudnn.getConvolutionForwardAlgorithm_v6(
            handle, x_desc, filter_desc, conv_desc, y_desc,
            cudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            max_workspace_size)
        workspace_size = cudnn.getConvolutionForwardWorkspaceSize(
            handle, x_desc, filter_desc, conv_desc, y_desc, algo)
    return algo, workspace_size


cpdef _warn_algorithm_bwd_filter(
        core.ndarray x, core.ndarray dy, core.ndarray dW, tuple conv_param):
    msg = 'Tensor Core mode is set but the selected convolution backward '\
          'filter algorithm is not a Tensor Core enabled algorithm. '\
          'This might be due to lack of workspace memory. '\
          'x.shape:{}, dy.shape:{}, dW.shape:{}, pad:{}, stride:{}'\
          .format(x.shape, dy.shape, dW.shape, conv_param[0], conv_param[1])
    warnings.warn(msg, RuntimeWarning)


cpdef tuple _find_algorithm_bwd_filter(
        core.ndarray x, core.ndarray dy, core.ndarray dW, tuple conv_param,
        size_t handle, size_t x_desc, size_t dy_desc, size_t conv_desc,
        size_t filter_desc, size_t max_workspace_size, bint use_tensor_core):
    key = (x.data.device.id, x.shape, dW.shape, dy.shape, conv_param,
           max_workspace_size)
    if key in _algorithm_bwd_filter:
        return _algorithm_bwd_filter[key]
    workspace = memory.alloc(max_workspace_size)
    if _cudnn_version >= 7000:
        ret = cudnn.findConvolutionBackwardFilterAlgorithmEx_v7(
            handle, x_desc, x.data.ptr, dy_desc, dy.data.ptr, conv_desc,
            filter_desc, dW.data.ptr, 1, workspace.ptr, max_workspace_size)
        algo = (ret[0].algo, ret[0].memory)
        if use_tensor_core:
            if ret[0].mathType != cudnn.CUDNN_TENSOR_OP_MATH:
                _warn_algorithm_bwd_filter(x, dy, dW, conv_param)
    else:
        ret = cudnn.findConvolutionBackwardFilterAlgorithmEx(
            handle, x_desc, x.data.ptr, dy_desc, dy.data.ptr, conv_desc,
            filter_desc, dW.data.ptr, 1, workspace.ptr, max_workspace_size)
        algo = (ret[0]['algo'], ret[0]['memory'])
    _algorithm_bwd_filter[key] = algo
    return algo


cpdef tuple _get_algorithm_bwd_filter(
        core.ndarray x, core.ndarray dy, core.ndarray dW, tuple conv_param,
        size_t handle, size_t x_desc, size_t gy_desc, size_t conv_desc,
        size_t filter_desc, size_t max_workspace_size, bint use_tensor_core):
    cdef int algo
    cdef workspace_size
    if use_tensor_core and _cudnn_version >= 7000:
        ret = cudnn.getConvolutionBackwardFilterAlgorithm_v7(
            handle, x_desc, gy_desc, conv_desc, filter_desc, 10)
        for i in range(len(ret)):
            if ret[i].memory <= max_workspace_size:
                break
        else:
            msg = 'No conv bwd filter algo available with workspace size less '\
                  'equal {}'.format(max_workspace_size)
            raise RuntimeError(msg)
        if i != 0:
            msg = 'The best algo of conv bwd filter might not not selected '\
                  'due to lack of workspace size ({})'\
                  .format(max_workspace_size)
            warnings.warn(msg)
        algo = ret[i].algo
        workspace_size = ret[i].memory
        if ret[i].mathType != cudnn.CUDNN_TENSOR_OP_MATH:
            _warn_algorithm_bwd_filter(x, dy, dW, conv_param)
    else:
        algo = cudnn.getConvolutionBackwardFilterAlgorithm_v6(
            handle, x_desc, gy_desc, conv_desc, filter_desc,
            cudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            max_workspace_size)
        workspace_size = cudnn.getConvolutionBackwardFilterWorkspaceSize(
            handle, x_desc, gy_desc, conv_desc, filter_desc, algo)
    return algo, workspace_size


cpdef _warn_algorithm_bwd_data(
        core.ndarray W, core.ndarray x, core.ndarray y, tuple conv_param):
    msg = 'Tensor Core mode is set but the selected convolution backward '\
          'filter algorithm is not a Tensor Core enabled algorithm. '\
          'This might be due to lack of workspace memory. '\
          'W.shape:{}, x.shape:{}, y.shape:{}, pad:{}, stride:{}'\
          .format(W.shape, x.shape, y.shape, conv_param[0], conv_param[1])
    warnings.warn(msg, RuntimeWarning)


cpdef tuple _find_algorithm_bwd_data(
        core.ndarray W, core.ndarray x, core.ndarray y, tuple conv_param,
        size_t handle, size_t filter_desc, size_t x_desc, size_t conv_desc,
        size_t y_desc, size_t max_workspace_size, bint use_tensor_core):
    key = (x.data.device.id, W.shape, x.shape, y.shape, conv_param,
           max_workspace_size)
    if key in _algorithm_bwd_data:
        return _algorithm_bwd_data[key]
    workspace = memory.alloc(max_workspace_size)
    if _cudnn_version >= 7000:
        ret = cudnn.findConvolutionBackwardDataAlgorithmEx_v7(
            handle, filter_desc, W.data.ptr, x_desc, x.data.ptr, conv_desc,
            y_desc, y.data.ptr, 1, workspace.ptr, max_workspace_size)
        algo = (ret[0].algo, ret[0].memory)
        if use_tensor_core:
            if ret[0].mathType != cudnn.CUDNN_TENSOR_OP_MATH:
                _warn_algorithm_bwd_data(W, x, y, conv_param)
    else:
        ret = cudnn.findConvolutionBackwardDataAlgorithmEx(
            handle, filter_desc, W.data.ptr, x_desc, x.data.ptr, conv_desc,
            y_desc, y.data.ptr, 1, workspace.ptr, max_workspace_size)
        algo = (ret[0]['algo'], ret[0]['memory'])
    _algorithm_bwd_data[key] = algo
    return algo


cpdef tuple _get_algorithm_bwd_data(
        core.ndarray W, core.ndarray x, core.ndarray y, tuple conv_param,
        size_t handle, size_t filter_desc, size_t x_desc, size_t conv_desc,
        size_t y_desc, size_t max_workspace_size, bint use_tensor_core):
    cdef int algo
    cdef workspace_size
    if use_tensor_core and _cudnn_version >= 7000:
        ret = cudnn.getConvolutionBackwardDataAlgorithm_v7(
            handle, filter_desc, x_desc, conv_desc, y_desc, 10)
        for i in range(len(ret)):
            if ret[i].memory <= max_workspace_size:
                break
        else:
            msg = 'No conv bwd data algo available with workspace size less '\
                  'equal {}'.format(max_workspace_size)
            raise RuntimeError(msg)
        if i != 0:
            msg = 'The best algo of conv bwd data might not not selected due '\
                  'to lack of workspace size ({})'.format(max_workspace_size)
            warnings.warn(msg)
        algo = ret[i].algo
        workspace_size = ret[i].memory
        if ret[i].mathType != cudnn.CUDNN_TENSOR_OP_MATH:
            _warn_algorithm_bwd_data(W, x, y, conv_param)
    else:
        algo = cudnn.getConvolutionBackwardDataAlgorithm_v6(
            handle, filter_desc, x_desc, conv_desc, y_desc,
            cudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            max_workspace_size)
        workspace_size = cudnn.getConvolutionBackwardDataWorkspaceSize(
            handle, filter_desc, x_desc, conv_desc, y_desc, algo)
    return algo, workspace_size


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
    cdef tuple conv_param = (pad, stride, x.dtype)

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

    cdef int algo
    cdef size_t max_workspace_size = get_max_workspace_size()
    cdef size_t workspace_size = 0
    try:
        _create_tensor_nd_descriptor(x_desc, x, -1)
        _create_tensor_nd_descriptor(y_desc, y, -1)
        _create_filter_descriptor(filter_desc, W, cudnn.CUDNN_TENSOR_NCHW)
        _create_convolution_descriptor(
            conv_desc, pad, stride, dilation, groups, x.dtype,
            cudnn.CUDNN_CROSS_CORRELATION, use_tensor_core)

        if auto_tune and _cudnn_version >= 5000:
            algo, workspace_size = _find_algorithm_fwd(
                x, W, y, conv_param, handle, x_desc, filter_desc,
                conv_desc, y_desc, max_workspace_size, use_tensor_core)
        else:
            algo, workspace_size = _get_algorithm_fwd(
                x, W, y, conv_param, handle, x_desc, filter_desc,
                conv_desc, y_desc, max_workspace_size, use_tensor_core)

        workspace = memory.alloc(workspace_size)

        cudnn.convolutionForward(
            handle, one, x_desc, x.data.ptr, filter_desc, W.data.ptr,
            conv_desc, algo, workspace.ptr, workspace_size, zero, y_desc,
            y.data.ptr)
        del workspace, x, W

        if b is not None:
            assert dev_id == b.data.device.id
            ndim = y.ndim - 2
            b = core.ascontiguousarray(b).reshape((1, -1) + (1,) * ndim)
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
    cdef tuple conv_param = (pad, stride, x.dtype)

    handle = get_handle()
    x = core.ascontiguousarray(x)
    gy = core.ascontiguousarray(gy)

    # TODO(okuta) check performance
    cdef size_t x_desc = cudnn.createTensorDescriptor()
    cdef size_t gy_desc = cudnn.createTensorDescriptor()
    cdef size_t filter_desc = cudnn.createFilterDescriptor()
    cdef size_t conv_desc = cudnn.createConvolutionDescriptor()

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
            # TODO(okuta): check workspace size
        elif auto_tune and _cudnn_version >= 5000:
            algo, workspace_size = _find_algorithm_bwd_filter(
                x, gy, gW, conv_param, handle, x_desc, gy_desc, conv_desc,
                filter_desc, max_workspace_size, use_tensor_core)
        else:
            algo, workspace_size = _get_algorithm_bwd_filter(
                x, gy, gW, conv_param, handle, x_desc, gy_desc, conv_desc,
                filter_desc, max_workspace_size, use_tensor_core)

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
    cdef tuple conv_param = (pad, stride, x.dtype)

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

    cdef int algo
    cdef size_t max_workspace_size = get_max_workspace_size()
    cdef size_t workspace_size = 0
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
            # TODO(okuta): check workspace size
        elif auto_tune and _cudnn_version >= 5000:
            algo, workspace_size = _find_algorithm_bwd_data(
                W, x, y, conv_param, handle, filter_desc, x_desc,
                conv_desc, y_desc, max_workspace_size, use_tensor_core)
        else:
            algo, workspace_size = _get_algorithm_bwd_data(
                W, x, y, conv_param, handle, filter_desc, x_desc,
                conv_desc, y_desc, max_workspace_size, use_tensor_core)

        workspace = memory.alloc(workspace_size)

        cudnn.convolutionBackwardData_v3(
            handle, one, filter_desc, W.data.ptr, x_desc, x.data.ptr,
            conv_desc, algo, workspace.ptr, workspace_size, zero, y_desc,
            y.data.ptr)

        del workspace, x, W

        if b is not None:
            assert dev_id == b.data.device.id
            ndim = y.ndim - 2
            b = core.ascontiguousarray(b).reshape((1, -1) + (1,) * ndim)
            _create_tensor_nd_descriptor(b_desc, b, -1)
            cudnn.addTensor_v3(handle, one, b_desc, b.data.ptr, one, y_desc,
                               y.data.ptr)
    finally:
        cudnn.destroyTensorDescriptor(x_desc)
        cudnn.destroyTensorDescriptor(y_desc)
        cudnn.destroyTensorDescriptor(b_desc)
        cudnn.destroyFilterDescriptor(filter_desc)
        cudnn.destroyConvolutionDescriptor(conv_desc)
