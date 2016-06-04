import atexit

import numpy
import six

from cupy import cuda
from cupy.cuda import cudnn


_cudnn_version = cudnn.getVersion()

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
    elif dtype.type == numpy.float16:
        return cudnn.CUDNN_DATA_HALF
    else:
        raise TypeError('Dtype {} is not supported in cuDNN'.format(dtype))


def _to_ctypes_array(tup, dtype=numpy.intc):
    return numpy.array(tup, dtype=dtype).ctypes


def create_tensor_descriptor(arr, format=cudnn.CUDNN_TENSOR_NCHW):
    desc = Descriptor(cudnn.createTensorDescriptor(),
                      cudnn.destroyTensorDescriptor)
    if arr.ndim != 4:
        raise ValueError('cupy.cudnn supports 4-dimensional arrays only')
    if not arr.flags.c_contiguous:
        raise ValueError('cupy.cudnn supports c-contiguous arrays only')
    data_type = get_data_type(arr.dtype)
    cudnn.setTensor4dDescriptor(desc.value, format, data_type,
                                *arr.shape)

    return desc


def create_filter_descriptor(arr, mode=cudnn.CUDNN_CROSS_CORRELATION):
    desc = Descriptor(cudnn.createFilterDescriptor(),
                      cudnn.destroyFilterDescriptor)
    data_type = get_data_type(arr.dtype)
    if arr.ndim == 4:
        cudnn.setFilter4dDescriptor_v3(desc.value, data_type, *arr.shape)
    else:
        c_shape = _to_ctypes_array(arr.shape)
        cudnn.setFilterNdDescriptor_v3(desc.value, data_type, arr.ndim,
                                       c_shape.data)

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
        c_pad = _to_ctypes_array(pad)
        c_stride = _to_ctypes_array(stride)
        c_upscale = _to_ctypes_array((1,) * ndim)
        cudnn.setConvolutionNdDescriptor_v2(
            desc.value, ndim, c_pad.data, c_stride.data, c_upscale.data, mode)

    return desc


def create_pooling_descriptor(ksize, stride, pad, mode):
    desc = Descriptor(cudnn.createPoolingDescriptor(),
                      cudnn.destroyPoolingDescriptor)
    ndim = len(ksize)
    if ndim != len(stride) or ndim != len(pad):
        raise ValueError('ksize, stride, and pad must be of same length')

    if ndim == 2:
        cudnn.setPooling2dDescriptor_v3(
            desc.value, mode, ksize[0], ksize[1], pad[0], pad[1],
            stride[0], stride[1])
    else:
        c_ksize = _to_ctypes_array(ksize)
        c_pad = _to_ctypes_array(pad)
        c_stride = _to_ctypes_array(stride)
        cudnn.setPoolingNdDescriptor_v3(
            desc.value, mode, ndim, c_ksize.data, c_pad.data, c_stride.data)

    return desc


if _cudnn_version >= 3000:
    def add_tensor(handle, alpha, biasDesc, biasData, beta, srcDestDesc,
                   srcDestData):
        cudnn.addTensor_v3(handle, alpha, biasDesc,
                           biasData, beta, srcDestDesc, srcDestData)
else:
    def add_tensor(handle, alpha, biasDesc, biasData, beta, srcDestDesc,
                   srcDestData):
        cudnn.addTensor_v2(handle, cudnn.CUDNN_ADD_SAME_C, alpha, biasDesc,
                           biasData, beta, srcDestDesc, srcDestData)
