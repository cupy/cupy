import atexit
import ctypes

import numpy
import six

from cupy import cuda
from cupy.cuda import cudnn


_handles = {}


def get_handle():
    global _handles
    device = cuda.Device()
    handle = _handles.get(device.id, None)
    if handle is None:
        handle = cudnn.create()
        _handles[device.id] = handle
    return handle


@atexit.register
def reset_handles():
    global _handles
    handles = _handles
    _handles = {}

    for handle in six.itervalues(handles):
        cudnn.destroy(handle)


class Descriptor(object):

    def __init__(self, descriptor, destroyer):
        self.value = descriptor
        self.destroy = destroyer

    def __del__(self):
        if self.value:
            self.destroy(self.value)
            self.value = None


def get_data_type(dtype):
    if dtype.type == numpy.float32:
        return cudnn.CUDNN_DATA_FLOAT
    elif dtype.type == numpy.float64:
        return cudnn.CUDNN_DATA_DOUBLE
    else:
        raise TypeError('Dtype {} is not supported in CuDNN v2'.format(dtype))


def _get_strides(arr):
    return tuple(map(lambda s: s // arr.itemsize, arr.strides))


def _to_ctypes_array(tup, typ=ctypes.c_int):
    array_type = typ * len(tup)
    return array_type(*tup)


def create_tensor_descriptor(arr, format=cudnn.CUDNN_TENSOR_NCHW):
    desc = Descriptor(cudnn.createTensorDescriptor(),
                      cudnn.destroyTensorDescriptor)
    if arr.ndim != 4:
        raise ValueError('Supports 4-dimensional array only')
    if not arr.flags.c_contiguous:
        raise ValueError('Supoorts c-contigous array only')
    data_type = get_data_type(arr.dtype)
    cudnn.setTensor4dDescriptor(desc.value, format, data_type,
                                *arr.shape)

    return desc


def create_filter_descriptor(arr, mode=cudnn.CUDNN_CROSS_CORRELATION):
    desc = Descriptor(cudnn.createFilterDescriptor(),
                      cudnn.destroyFilterDescriptor)
    data_type = get_data_type(arr.dtype)
    if arr.ndim == 4:
        cudnn.setFilter4dDescriptor(desc.value, data_type, *arr.shape)
    else:
        cudnn.setFilterNdDescriptor(desc.value, data_type, arr.ndim,
                                    _to_ctypes_array(arr.shape))

    return desc


def create_convolution_descriptor(pad, stride,
                                  mode=cudnn.CUDNN_CROSS_CORRELATION):
    desc = Descriptor(cudnn.createConvolutionDescriptor(),
                      cudnn.destroyConvolutionDescriptor)
    ndim = len(pad)
    if ndim != len(stride):
        raise ValueError('pad and stride must be of same length')

    if ndim == 2:
        cudnn.setConvolution2dDescriptor(
            desc.value, pad[0], pad[1], stride[0], stride[1], 1, 1, mode)
    else:
        upscale = (1,) * ndim
        cudnn.setConvolutionNdDescriptor(
            desc.value, ndim, _to_ctypes_array(pad), _to_ctypes_array(stride),
            _to_ctypes_array(upscale), mode)

    return desc


def create_pooling_descriptor(ksize, stride, pad, mode):
    desc = Descriptor(cudnn.createPoolingDescriptor(),
                      cudnn.destroyPoolingDescriptor)
    ndim = len(ksize)
    if ndim != len(stride) or ndim != len(pad):
        raise ValueError('ksize, stride, and pad must be of same length')

    if ndim == 2:
        cudnn.setPooling2dDescriptor(
            desc.value, mode, ksize[0], ksize[1], pad[0], pad[1],
            stride[0], stride[1])
    else:
        cudnn.setPoolingNdDescriptor(
            desc.value, mode, ndim, _to_ctypes_array(ksize),
            _to_ctypes_array(pad), _to_ctypes_array(stride))

    return desc
