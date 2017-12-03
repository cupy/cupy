from libcpp cimport vector

import atexit
import threading

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


class Descriptor(object):

    def __init__(self, descriptor, destroyer):
        self.value = descriptor
        self.destroy = destroyer

    def __del__(self):
        if self.value:
            self.destroy(self.value)
            self.value = None


cpdef get_data_type(dtype):
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
        size_t desc, core.ndarray arr, int data_type):
    cdef vector.vector[int] c_shape, c_strides
    cdef Py_ssize_t itemsize, s
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
        cudnn.setTensor4dDescriptor(desc, format, data_type, n, c, h, w)
    else:
        _create_tensor_nd_descriptor(desc, arr, data_type)


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
        desc, pad, stride, dtype, mode, dilation, int group,
        bint use_tensor_core):
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
        c_dilation.assign(ndim, 1)
        cudnn.setConvolutionNdDescriptor_v3(
            desc, ndim, <size_t>&c_pad[0], <size_t>&c_stride[0],
            <size_t>&c_dilation[0], mode, compute_type)
        return

    d0, d1 = dilation
    p0, p1 = pad
    s0, s1 = stride
    if _cudnn_version < 6000 and (d0 != 1 or d1 != 1):
        raise ValueError('dilation must be one when cudnn < 6.0')
    if _cudnn_version >= 5000:
        cudnn.setConvolution2dDescriptor_v5(
            desc, p0, p1, s0, s1, d0, d1, mode, compute_type)
        if _cudnn_version >= 7000 and use_tensor_core:
            math_type = cudnn.CUDNN_TENSOR_OP_MATH
            cudnn.setConvolutionMathType(desc, math_type)
            if group > 1:
                cudnn.setConvolutionGroupCount(desc.value, group)
    else:
        cudnn.setConvolution2dDescriptor_v4(desc, p0, p1, s0, s1, 1, 1, mode)


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
                                  dilation=(1, 1),
                                  use_tensor_core=False,
                                  group=1):
    desc = Descriptor(cudnn.createConvolutionDescriptor(),
                      py_cudnn.destroyConvolutionDescriptor)
    _create_convolution_descriptor(
        desc.value, pad, stride, dtype, mode, dilation, group, use_tensor_core)
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


def activation_forward(core.ndarray x, int mode):
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
            act_desc, mode, cudnn.CUDNN_NOT_PROPAGATE_NAN, 0.0)
        cudnn.activationForward_v4(
            handle, act_desc, one, desc, x.data.ptr,
            zero, desc, y.data.ptr)
    finally:
        cudnn.destroyActivationDescriptor(act_desc)
        cudnn.destroyTensorDescriptor(desc)
    return y


def activation_backward(core.ndarray x, core.ndarray y, core.ndarray gy,
                        int mode):
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
            act_desc, mode, cudnn.CUDNN_NOT_PROPAGATE_NAN, 0.0)
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
                          input_mode, direction, mode, data_type):
    desc = Descriptor(cudnn.createRNNDescriptor(),
                      py_cudnn.destroyRNNDescriptor)
    if _cudnn_version >= 7000:
        _handle = get_handle()
        _algo = cudnn.CUDNN_RNN_ALGO_STANDARD
        cudnn.setRNNDescriptor_v6(
            _handle, desc.value, hidden_size, num_layers, dropout_desc.value,
            input_mode, direction, mode, _algo, data_type)
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
