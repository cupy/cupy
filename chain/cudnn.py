"""Common routines to use CuDNN."""

import ctypes
import libcudnn
import numpy
import pycuda.gpuarray as gpuarray

def get_ptr(x):
    return ctypes.c_void_p(int(x.gpudata))

_default_handle = None

def get_default_handle():
    """Get the default handle of CuDNN."""
    global _default_handle
    if _default_handle is None:
        _default_handle = libcudnn.cudnnCreate()
    return _default_handle

_dtypes = {numpy.dtype('float32'): libcudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
           numpy.dtype('float64'): libcudnn.cudnnDataType['CUDNN_DATA_DOUBLE']}

def get_tensor_desc(x, h, w, form='CUDNN_TENSOR_NCHW'):
    """Create a tensor description for given settings."""
    n = x.shape[0]
    c = x.size / (n * h * w)
    desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(
        desc, libcudnn.cudnnTensorFormat[form], _dtypes[x.dtype], n, c, h, w)
    return desc
