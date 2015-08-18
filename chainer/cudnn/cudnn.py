"""Common routines to use CuDNN."""

import atexit
import ctypes
import os

import numpy

from chainer import cuda
import libcudnn

enabled = int(os.environ.get('CHAINER_CUDNN', '1')) != 0
available = True


def get_ptr(x):
    return ctypes.c_void_p(x.ptr)


class Auto(object):

    """Object to be destroyed automatically."""

    def __init__(self, value, destroyer):
        self.value = value
        self.destroyer = destroyer

    def __del__(self):
        try:
            self.destroyer(self.value)
        except Exception:
            pass

_handles = {}
_pid = None


def get_default_handle():
    """Get the default handle of CuDNN."""

    global _handles, _pid

    pid = os.getpid()
    if _pid != pid:  # not initialized yet
        _handles = {}
        atexit.register(shutdown)
        _pid = pid

    device = cuda.Context.get_device()
    if device in _handles:
        return _handles[device]

    handle = libcudnn.cudnnCreate()
    _handles[device] = handle

    return handle


def shutdown():
    global _handles, _pid

    pid = os.getpid()
    if _pid != pid:  # not initialized
        return

    for handle in _handles.itervalues():
        libcudnn.cudnnDestroy(handle)

    _handles = {}
    _pid = None  # mark as uninitialized

_dtypes = {numpy.dtype('float32'): libcudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
           numpy.dtype('float64'): libcudnn.cudnnDataType['CUDNN_DATA_DOUBLE']}


def get_tensor_desc(x, h, w, form='CUDNN_TENSOR_NCHW'):
    """Create a tensor descriptor for given settings."""
    n = x.shape[0] if len(x.shape) >= 1 else 1
    c = x.size // (n * h * w)
    desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(
        desc, libcudnn.cudnnTensorFormat[form], _dtypes[x.dtype], n, c, h, w)
    return Auto(desc, libcudnn.cudnnDestroyTensorDescriptor)


def get_conv_bias_desc(x):
    """Create a bias tensor descriptor."""
    desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(
        desc, libcudnn.cudnnTensorFormat[
            'CUDNN_TENSOR_NCHW'], _dtypes[x.dtype],
        1, x.size, 1, 1)
    return Auto(desc, libcudnn.cudnnDestroyTensorDescriptor)

_default_conv_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']


def get_filter4d_desc(x, mode=_default_conv_mode):
    """Create a 2d convolution filter descriptor."""
    k, c, h, w = x.shape
    desc = libcudnn.cudnnCreateFilterDescriptor()
    libcudnn.cudnnSetFilter4dDescriptor(desc, _dtypes[x.dtype], k, c, h, w)
    return Auto(desc, libcudnn.cudnnDestroyFilterDescriptor)


def get_conv2d_desc(pad, stride, mode=_default_conv_mode):
    """Create a 2d convolution descriptor."""
    desc = libcudnn.cudnnCreateConvolutionDescriptor()
    libcudnn.cudnnSetConvolution2dDescriptor(
        desc, pad[0], pad[1], stride[0], stride[1], 1, 1, mode)
    return Auto(desc, libcudnn.cudnnDestroyConvolutionDescriptor)

_pool_mode = {
    'MAX': libcudnn.cudnnPoolingMode['CUDNN_POOLING_MAX'],
    'AVE': libcudnn.cudnnPoolingMode[
        'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING']
}


def get_pool2d_desc(ksize, stride, pad, mode):
    """Create a 2d pooling descriptor."""
    desc = libcudnn.cudnnCreatePoolingDescriptor()
    libcudnn.cudnnSetPooling2dDescriptor(
        desc, libcudnn.cudnnPoolingMode[mode], ksize[0], ksize[1],
        pad[0], pad[1], stride[0], stride[1])
    return Auto(desc, libcudnn.cudnnDestroyPoolingDescriptor)
