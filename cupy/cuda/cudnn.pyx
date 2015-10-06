"""Thin wrapper of CuDNN."""
# NOTE: This wrapper does not cover all APIs of CuDNN v2.
import ctypes
import sys

from cupy.cuda import internal
from cupy.cuda import runtime

if 'win32' == sys.platform:
    _cudnn = internal.load_library(
        internal.get_windows_cuda_library_names('cudnn'))
else:
    _cudnn = internal.load_library('cudnn')

_I = ctypes.c_int
_S = ctypes.c_size_t
_P = ctypes.c_void_p
_IP = ctypes.POINTER(_I)
_SP = ctypes.POINTER(_S)

Handle = _P
TensorDescriptor = _P
ConvolutionDescriptor = _P
PoolingDescriptor = _P
FilterDescriptor = _P

CUDNN_DATA_FLOAT = 0
CUDNN_DATA_DOUBLE = 1

CUDNN_TENSOR_NCHW = 0
CUDNN_TENSOR_NHWC = 1

CUDNN_ADD_IMAGE = 0
CUDNN_ADD_SAME_HW = 0
CUDNN_ADD_FEATURE_MAP = 1
CUDNN_ADD_SAME_CHW = 1
CUDNN_ADD_SAME_C = 2
CUDNN_ADD_FULL_TENSOR = 3

CUDNN_CONVOLUTION = 0
CUDNN_CROSS_CORRELATION = 1

CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2

CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1
CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2
CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3

CUDNN_SOFTMAX_FAST = 0
CUDNN_SOFTMAX_ACCURATE = 1

CUDNN_SOFTMAX_MODE_INSTANCE = 0
CUDNN_SOFTMAX_MODE_CHANNEL = 1

CUDNN_POOLING_MAX = 0
CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2

CUDNN_ACTIVATION_SIGMOID = 0
CUDNN_ACTIVATION_RELU = 1
CUDNN_ACTIVATION_TANH = 2


###############################################################################
# Error handling
###############################################################################

STATUS = {
    0: 'CUDNN_STATUS_SUCCESS',
    1: 'CUDNN_STATUS_NOT_INITIALIZED',
    2: 'CUDNN_STATUS_ALLOC_FAILED',
    3: 'CUDNN_STATUS_BAD_PARAM',
    4: 'CUDNN_STATUS_INTERNAL_ERROR',
    5: 'CUDNN_STATUS_INVALID_VALUE',
    6: 'CUDNN_STATUS_ARCH_MISMATCH',
    7: 'CUDNN_STATUS_MAPPING_ERROR',
    8: 'CUDNN_STATUS_EXECUTION_FAILED',
    9: 'CUDNN_STATUS_NOT_SUPPORTED',
    10: 'CUDNN_STATUS_LICENSE_ERROR',
}

_cudnn.cudnnGetErrorString.restype = ctypes.c_char_p
_cudnn.cudnnGetErrorString.argtypes = (_I,)


class CuDNNError(RuntimeError):

    def __init__(self, status):
        self.status = status
        msg = _cudnn.cudnnGetErrorString(status)
        super(CuDNNError, self).__init__('%s: %s' % (STATUS[status], msg))


def check_status(status):
    if status != 0:
        raise CuDNNError(status)


###############################################################################
# Initialization and CUDA cooperation
###############################################################################

_cudnn.cudnnCreate.argtypes = (_P,)


def create():
    handle = Handle()
    status = _cudnn.cudnnCreate(ctypes.byref(handle))
    check_status(status)
    return handle


_cudnn.cudnnDestroy.argtypes = (Handle,)


def destroy(handle):
    status = _cudnn.cudnnDestroy(handle)
    check_status(status)


_cudnn.cudnnSetStream.argtypes = (Handle, runtime.Stream)


def setStream(handle, stream):
    status = _cudnn.cudnnSetStream(handle, stream)
    check_status(status)


_cudnn.cudnnGetStream.argtypes = (Handle, _P)


def getStream(handle):
    stream = runtime.Stream()
    status = _cudnn.cudnnGetStream(handle, ctypes.byref(stream))
    check_status(status)
    return stream


###############################################################################
# Tensor manipulation
###############################################################################

_cudnn.cudnnCreateTensorDescriptor.argtypes = (_P,)


def createTensorDescriptor():
    descriptor = TensorDescriptor()
    status = _cudnn.cudnnCreateTensorDescriptor(ctypes.byref(descriptor))
    check_status(status)
    return descriptor


_cudnn.cudnnSetTensor4dDescriptor.argtypes = (TensorDescriptor, _I, _I,
                                              _I, _I, _I, _I)


def setTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w):
    status = _cudnn.cudnnSetTensor4dDescriptor(tensorDesc, format, dataType,
                                               n, c, h, w)
    check_status(status)


_cudnn.cudnnSetTensor4dDescriptorEx.argtypes = (
    TensorDescriptor, _I, _I, _I, _I, _I, _I, _I, _I, _I)


def setTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w,
                            nStride, cStride, hStride, wStride):
    status = _cudnn.cudnnSetTensor4dDescriptorEx(
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
    check_status(status)


_cudnn.cudnnSetTensorNdDescriptor.argtypes = (TensorDescriptor, _I, _I, _IP,
                                              _IP)


def setTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA):
    status = _cudnn.cudnnSetTensorNdDescriptor(
        tensorDesc, dataType, nbDims, dimA, strideA)
    check_status(status)


_cudnn.cudnnDestroyTensorDescriptor.argtypes = (TensorDescriptor,)


def destroyTensorDescriptor(tensorDesc):
    status = _cudnn.cudnnDestroyTensorDescriptor(tensorDesc)
    check_status(status)


_cudnn.cudnnAddTensor.argtypes = (Handle, _I, _P, TensorDescriptor, _P,
                                  _P, TensorDescriptor, _P)


def addTensor(handle, mode, alpha, biasDesc, biasData, beta, srcDestDesc,
              srcDestData):
    status = _cudnn.cudnnAddTensor(
        handle, mode, ctypes.byref(alpha), biasDesc, biasData,
        ctypes.byref(beta), srcDestDesc, srcDestData)
    check_status(status)


###############################################################################
# Filter manipulation
###############################################################################

_cudnn.cudnnCreateFilterDescriptor.argtypes = (_P,)


def createFilterDescriptor():
    desc = FilterDescriptor()
    status = _cudnn.cudnnCreateFilterDescriptor(ctypes.byref(desc))
    check_status(status)
    return desc


_cudnn.cudnnSetFilter4dDescriptor.argtypes = (FilterDescriptor, _I,
                                              _I, _I, _I, _I)


def setFilter4dDescriptor(filterDesc, dataType, k, c, h, w):
    status = _cudnn.cudnnSetFilter4dDescriptor(
        filterDesc, dataType, k, c, h, w)
    check_status(status)


_cudnn.cudnnSetFilterNdDescriptor.argtypes = (FilterDescriptor, _I, _IP)


def setFilterNdDescriptor(filterDesc, nbDims, filterDimA):
    status = _cudnn.cudnnSetFilterNdDescriptor(filterDesc, nbDims, filterDimA)
    check_status(status)


_cudnn.cudnnDestroyFilterDescriptor.argtypes = (FilterDescriptor,)


def destroyFilterDescriptor(filterDesc):
    status = _cudnn.cudnnDestroyFilterDescriptor(filterDesc)
    check_status(status)


###############################################################################
# Convolution
###############################################################################

_cudnn.cudnnCreateConvolutionDescriptor.argtypes = (_P,)


def createConvolutionDescriptor():
    desc = ConvolutionDescriptor()
    status = _cudnn.cudnnCreateConvolutionDescriptor(ctypes.byref(desc))
    check_status(status)
    return desc


_cudnn.cudnnSetConvolution2dDescriptor.argtypes = (
    ConvolutionDescriptor, _I, _I, _I, _I, _I, _I, _I)


def setConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, upscalex,
                               upscaley, mode):
    status = _cudnn.cudnnSetConvolution2dDescriptor(
        convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode)
    check_status(status)


_cudnn.cudnnSetConvolutionNdDescriptor.argtypes = (
    ConvolutionDescriptor, _I, _IP, _IP, _IP, _I)


def setConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA,
                               upscaleA, mode):
    status = _cudnn.cudnnSetConvolutionNdDescriptor(
        convDesc, arrayLength, padA, filterStrideA, upscaleA, mode)
    check_status(status)


_cudnn.cudnnDestroyConvolutionDescriptor.argtypes = (ConvolutionDescriptor,)


def destroyConvolutionDescriptor(convDesc):
    status = _cudnn.cudnnDestroyConvolutionDescriptor(convDesc)
    check_status(status)


_cudnn.cudnnGetConvolutionForwardAlgorithm.argtypes = (
    Handle, TensorDescriptor, FilterDescriptor, ConvolutionDescriptor,
    TensorDescriptor, _I, _S, _IP)


def getConvolutionForwardAlgorithm(handle, srcDesc, filterDesc, convDesc,
                                   destDesc, preference, memoryLimitInbytes):
    algo = _I()
    status = _cudnn.cudnnGetConvolutionForwardAlgorithm(
        handle, srcDesc, filterDesc, convDesc, destDesc, preference,
        memoryLimitInbytes, ctypes.byref(algo))
    check_status(status)
    return algo.value


_cudnn.cudnnGetConvolutionForwardWorkspaceSize.argtypes = (
    Handle, TensorDescriptor, FilterDescriptor, ConvolutionDescriptor,
    TensorDescriptor, _I, _SP)


def getConvolutionForwardWorkspaceSize(handle, srcDesc, filterDesc, convDesc,
                                       destDesc, algo):
    sizeInBytes = _S()
    status = _cudnn.cudnnGetConvolutionForwardWorkspaceSize(
        handle, srcDesc, filterDesc, convDesc, destDesc, algo,
        ctypes.byref(sizeInBytes))
    check_status(status)
    return sizeInBytes.value


_cudnn.cudnnConvolutionForward.argtypes = (
    Handle, _P, TensorDescriptor, _P, FilterDescriptor, _P,
    ConvolutionDescriptor, _I, _P, _S, _P, TensorDescriptor, _P)


def convolutionForward(handle, alpha, srcDesc, srcData, filterDesc, filterData,
                       convDesc, algo, workSpace, workSpaceSizeInBytes, beta,
                       destDesc, destData):
    status = _cudnn.cudnnConvolutionForward(
        handle, ctypes.byref(alpha), srcDesc, srcData, filterDesc, filterData,
        convDesc, algo, workSpace, workSpaceSizeInBytes, ctypes.byref(beta),
        destDesc, destData)
    check_status(status)


_cudnn.cudnnConvolutionBackwardBias.argtypes = (
    Handle, _P, TensorDescriptor, _P, _P, TensorDescriptor, _P)


def convolutionBackwardBias(handle, alpha, srcDesc, srcData, beta, destDesc,
                            destData):
    status = _cudnn.cudnnConvolutionBackwardBias(
        handle, ctypes.byref(alpha), srcDesc, srcData, ctypes.byref(beta),
        destDesc, destData)
    check_status(status)


_cudnn.cudnnConvolutionBackwardFilter.argtypes = (
    Handle, _P, TensorDescriptor, _P, TensorDescriptor, _P,
    ConvolutionDescriptor, _P, FilterDescriptor, _P)


def convolutionBackwardFilter(handle, alpha, srcDesc, srcData, diffDesc,
                              diffData, convDesc, beta, gradDesc, gradData):
    status = _cudnn.cudnnConvolutionBackwardFilter(
        handle, ctypes.byref(alpha), srcDesc, srcData, diffDesc, diffData,
        convDesc, ctypes.byref(beta), gradDesc, gradData)
    check_status(status)


_cudnn.cudnnConvolutionBackwardData.argtypes = (
    Handle, _P, FilterDescriptor, _P, TensorDescriptor, _P,
    ConvolutionDescriptor, _P, TensorDescriptor, _P)


def convolutionBackwardData(handle, alpha, filterDesc, filterData, diffDesc,
                            diffData, convDesc, beta, gradDesc, gradData):
    status = _cudnn.cudnnConvolutionBackwardData(
        handle, ctypes.byref(alpha), filterDesc, filterData, diffDesc,
        diffData, convDesc, ctypes.byref(beta), gradDesc, gradData)
    check_status(status)


###############################################################################
# Pooling
###############################################################################

_cudnn.cudnnCreatePoolingDescriptor.argtypes = (_P,)


def createPoolingDescriptor():
    desc = PoolingDescriptor()
    status = _cudnn.cudnnCreatePoolingDescriptor(ctypes.byref(desc))
    check_status(status)
    return desc


_cudnn.cudnnSetPooling2dDescriptor.argtypes = (
    PoolingDescriptor, _I, _I, _I, _I, _I, _I, _I)


def setPooling2dDescriptor(poolingDesc, mode, windowHeight, windowWidth,
                           verticalPadding, horizontalPadding, verticalStride,
                           horizontalStride):
    status = _cudnn.cudnnSetPooling2dDescriptor(
        poolingDesc, mode, windowHeight, windowWidth, verticalPadding,
        horizontalPadding, verticalStride, horizontalStride)
    check_status(status)


_cudnn.cudnnSetPoolingNdDescriptor.argtypes = (
    PoolingDescriptor, _I, _I, _IP, _IP, _IP)


def setPoolingNdDescriptor(poolingDesc, mode, nbDims, windowDimA, paddingA,
                           strideA):
    status = _cudnn.cudnnSetPoolingNdDescriptor(
        poolingDesc, mode, nbDims, windowDimA, paddingA, strideA)
    check_status(status)


_cudnn.cudnnDestroyPoolingDescriptor.argtypes = (PoolingDescriptor,)


def destroyPoolingDescriptor(poolingDesc):
    status = _cudnn.cudnnDestroyPoolingDescriptor(poolingDesc)
    check_status(status)


_cudnn.cudnnPoolingForward.argtypes = (
    Handle, PoolingDescriptor, _P, TensorDescriptor, _P, _P,
    TensorDescriptor, _P)


def poolingForward(handle, poolingDesc, alpha, srcDesc, srcData, beta,
                   destDesc, destData):
    status = _cudnn.cudnnPoolingForward(
        handle, poolingDesc, ctypes.byref(alpha), srcDesc, srcData,
        ctypes.byref(beta), destDesc, destData)
    check_status(status)


_cudnn.cudnnPoolingBackward.argtypes = (
    Handle, PoolingDescriptor, _P, TensorDescriptor, _P, TensorDescriptor, _P,
    TensorDescriptor, _P, _P, TensorDescriptor, _P)


def poolingBackward(handle, poolingDesc, alpha, srcDesc, srcData, srcDiffDesc,
                    srcDiffData, destDesc, destData, beta, destDiffDesc,
                    destDiffData):
    status = _cudnn.cudnnPoolingBackward(
        handle, poolingDesc, ctypes.byref(alpha), srcDesc, srcData,
        srcDiffDesc, srcDiffData, destDesc, destData, ctypes.byref(beta),
        destDiffDesc, destDiffData)
    check_status(status)


###############################################################################
# Activation
###############################################################################

_cudnn.cudnnSoftmaxForward.argtypes = (
    Handle, _I, _I, _P, TensorDescriptor, _P, _P, TensorDescriptor, _P)


def softmaxForward(handle, algorithm, mode, alpha, srcDesc, srcData, beta,
                   destDesc, destData):
    status = _cudnn.cudnnSoftmaxForward(
        handle, algorithm, mode, ctypes.byref(alpha), srcDesc, srcData,
        ctypes.byref(beta), destDesc, destData)
    check_status(status)


_cudnn.cudnnSoftmaxBackward.argtypes = (
    Handle, _I, _I, _P, TensorDescriptor, _P, TensorDescriptor, _P, _P,
    TensorDescriptor, _P)


def softmaxBackward(handle, algorithm, mode, alpha, srcDesc, srcData,
                    srcDiffDesc, srcDiffData, beta, destDiffDesc,
                    destDiffData):
    status = _cudnn.cudnnSoftmaxBackward(
        handle, algorithm, mode, ctypes.byref(alpha), srcDesc, srcData,
        srcDiffDesc, srcDiffData, ctypes.byref(beta), destDiffDesc,
        destDiffData)
    check_status(status)


_cudnn.cudnnActivationForward.argtypes = (
    Handle, _I, _P, TensorDescriptor, _P, _P, TensorDescriptor, _P)


def activationForward(handle, mode, alpha, srcDesc, srcData, beta, dstDesc,
                      dstData):
    status = _cudnn.cudnnActivationForward(
        handle, mode, ctypes.byref(alpha), srcDesc, srcData,
        ctypes.byref(beta), dstDesc, dstData)
    check_status(status)


_cudnn.cudnnActivationBackward.argtypes = (
    Handle, _I, _P, TensorDescriptor, _P, TensorDescriptor, _P,
    TensorDescriptor, _P, _P, TensorDescriptor, _P)


def activationBackward(handle, mode, alpha, srcDesc, srcData, srcDiffDesc,
                       srcDiffData, destDesc, destData, beta, destDiffDesc,
                       destDiffData):
    status = _cudnn.cudnnActivationBackward(
        handle, mode, ctypes.byref(alpha), srcDesc, srcData, srcDiffDesc,
        srcDiffData, destDesc, destData, ctypes.byref(beta), destDiffDesc,
        destDiffData)
    check_status(status)
