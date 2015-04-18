"""Common routines to use CuDNN."""

import ctypes
import libcudnn
import numpy

def get_ptr(x):
    return ctypes.c_void_p(int(x.gpudata))

class Auto(object):
    """Object to be destoryed automatically."""
    def __init__(self, value, destroyer):
        self.value = value
        self.destroyer = destroyer

    def __del__(self):
        try:
            self.destroyer(self.value)
        except:
            pass

_default_handle = None

def get_default_handle():
    """Get the default handle of CuDNN."""
    global _default_handle
    if _default_handle is None:
        _default_handle = Auto(libcudnn.cudnnCreate(), libcudnn.cudnnDestroy)
    return _default_handle.value

_dtypes = {numpy.dtype('float32'): libcudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
           numpy.dtype('float64'): libcudnn.cudnnDataType['CUDNN_DATA_DOUBLE']}

def get_tensor_desc(x, h, w, form='CUDNN_TENSOR_NCHW'):
    """Create a tensor description for given settings."""
    n = x.shape[0]
    c = x.size / (n * h * w)
    desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(
        desc, libcudnn.cudnnTensorFormat[form], _dtypes[x.dtype], n, c, h, w)
    return Auto(desc, libcudnn.cudnnDestroyTensorDescriptor)
