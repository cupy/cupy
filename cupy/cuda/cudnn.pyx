"""Thin wrapper of CuDNN."""
# NOTE: This wrapper does not cover all APIs of CuDNN v2.

###############################################################################
# Types
###############################################################################

from cupy.cuda.driver cimport Stream
ctypedef void* Handle
ctypedef void* TensorDescriptor
ctypedef void* ConvolutionDescriptor
ctypedef void* PoolingDescriptor
ctypedef void* FilterDescriptor

###############################################################################
# Extern
###############################################################################

cdef extern from "cublas.h":
    # Error handling
    const char* cudnnGetErrorString(int status)

    # Initialization and CUDA cooperation
    int cudnnCreate(Handle* handle)
    int cudnnDestroy(Handle handle)
    int cudnnSetStream(Handle handle, Stream stream)
    int cudnnGetStream(Handle handle, Stream* stream)

    # Tensor manipulation
    int cudnnCreateTensorDescriptor(TensorDescriptor* descriptor)
    int cudnnSetTensor4dDescriptor(
        TensorDescriptor tensorDesc, int format, int dataType,
        int n, int c, int h, int w)
    int cudnnSetTensor4dDescriptorEx(
        TensorDescriptor tensorDesc, int dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride)
    int cudnnSetTensorNdDescriptor(
        TensorDescriptor tensorDesc, int dataType, int nbDims,
        int* dimA, int* strideA)
    int cudnnDestroyTensorDescriptor(TensorDescriptor tensorDesc)
    int cudnnAddTensor(
        Handle handle, int mode, void* alpha, TensorDescriptor biasDesc,
        void* biasData, void* beta, TensorDescriptor srcDestDesc,
        void* srcDestData)

    # Filter manipulation
    int cudnnCreateFilterDescriptor(FilterDescriptor* filterDesc)
    int cudnnSetFilter4dDescriptor(
        FilterDescriptor filterDesc, int dataType, int n, int c, int h, int w)
    int cudnnSetFilterNdDescriptor(
        FilterDescriptor filterDesc, int dataType, int nbDims, int* filterDimA)
    int cudnnDestroyFilterDescriptor(FilterDescriptor filterDesc)

    # Convolution
    int cudnnCreateConvolutionDescriptor(ConvolutionDescriptor* convDesc)
    int cudnnSetConvolution2dDescriptor(
        ConvolutionDescriptor convDesc, int pad_h, int pad_w, int u, int v,
        int upscalex, int upscaley, int mode)
    int cudnnSetConvolutionNdDescriptor(
        ConvolutionDescriptor convDesc, int arrayLength, int* padA,
        int* filterStrideA, int* upscaleA, int mode)
    int cudnnDestroyConvolutionDescriptor(ConvolutionDescriptor conDesc)
    int cudnnGetConvolutionForwardAlgorithm(
        Handle handle, TensorDescriptor srcDesc, FilterDescriptor,
        ConvolutionDescriptor convDesc, TensorDescriptor destDesc,
        int preference, size_t memoryLimitInbytes, int* algo)
    int cudnnGetConvolutionForwardWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc, FilterDescriptor filterDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor destDesc, int algo,
        size_t* sizeInBytes)
    int cudnnConvolutionForward(
        Handle handle, void* alpha, TensorDescriptor srcDesc, void* srcData,
        FilterDescriptor filterDesc, void* filterData,
        ConvolutionDescriptor convDesc, int algo, void* workSpace,
        size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor destDesc, void* destData)
    int cudnnConvolutionBackwardBias(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor destDesc, void* destData)
    int cudnnConvolutionBackwardFilter(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, void* beta,
        FilterDescriptor gradDesc, void* gradData)
    int cudnnConvolutionBackwardData(
        Handle handle, void* alpha,
        FilterDescriptor filterDesc, void* filterData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, void* beta,
        TensorDescriptor gradDesc, void* gradData)

    # Pooling
    int cudnnCreatePoolingDescriptor(PoolingDescriptor* desc)
    int cudnnSetPooling2dDescriptor(
        PoolingDescriptor poolingDesc, int mode,
        int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding,
        int verticalStride, int horizontalStride)
    int cudnnSetPoolingNdDescriptor(
        PoolingDescriptor poolingDesc, int mode, int nbDims,
        int* windowDimA, int* paddingA, int* strideA)
    int cudnnDestroyPoolingDescriptor(PoolingDescriptor poolingDesc)
    int cudnnPoolingForward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData)
    int cudnnPoolingBackward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)

    # Activation
    int cudnnSoftmaxForward(
        Handle handle, int algorithm, int mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData)
    int cudnnSoftmaxBackward(
        Handle handle, int algorithm, int mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)
    int cudnnActivationForward(
        Handle handle, int mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData)
    int cudnnActivationBackward(
        Handle handle, int mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)

###############################################################################
# Enum
###############################################################################

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


class CuDNNError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        msg = cudnnGetErrorString(status)
        super(CuDNNError, self).__init__('%s: %s' % (STATUS[status], msg))


cpdef check_status(int status):
    if status != 0:
        raise CuDNNError(status)


###############################################################################
# Initialization and CUDA cooperation
###############################################################################

cpdef size_t create():
    cdef Handle handle
    status = cudnnCreate(&handle)
    check_status(status)
    return <size_t>handle


cpdef destroy(size_t handle):
    status = cudnnDestroy(<Handle>handle)
    check_status(status)


cpdef setStream(size_t handle, size_t stream):
    status = cudnnSetStream(<Handle>handle, <Stream>stream)
    check_status(status)


cpdef size_t getStream(size_t handle):
    cdef Stream stream
    status = cudnnGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


###############################################################################
# Tensor manipulation
###############################################################################

cpdef size_t createTensorDescriptor():
    cdef TensorDescriptor descriptor
    status = cudnnCreateTensorDescriptor(&descriptor)
    check_status(status)
    return <size_t>descriptor


cpdef setTensor4dDescriptor(size_t tensorDesc, int format, int dataType,
                            int n, int c, int h, int w):
    status = cudnnSetTensor4dDescriptor(
        <TensorDescriptor>tensorDesc, format, dataType, n, c, h, w)
    check_status(status)


cpdef setTensor4dDescriptorEx(size_t tensorDesc, int dataType,
                              int n, int c, int h, int w, int nStride,
                              int cStride, int hStride, int wStride):
    status = cudnnSetTensor4dDescriptorEx(
        <TensorDescriptor>tensorDesc, dataType, n, c, h, w,
        nStride, cStride, hStride, wStride)
    check_status(status)


cpdef setTensorNdDescriptor(size_t tensorDesc, int dataType, int nbDims,
                            size_t dimA, size_t strideA):
    status = cudnnSetTensorNdDescriptor(
        <TensorDescriptor>tensorDesc, dataType, nbDims, <int*>dimA,
        <int*>strideA)
    check_status(status)


cpdef destroyTensorDescriptor(size_t tensorDesc):
    status = cudnnDestroyTensorDescriptor(<TensorDescriptor>tensorDesc)
    check_status(status)


cpdef addTensor(handle, mode, size_t alpha,
                size_t biasDesc, size_t biasData, size_t beta,
                size_t srcDestDesc, size_t srcDestData):
    status = cudnnAddTensor(
        <Handle>handle, mode, <void*>alpha,
        <TensorDescriptor>biasDesc, <void*>biasData, <void*>beta,
        <TensorDescriptor>srcDestDesc, <void*>srcDestData)
    check_status(status)


###############################################################################
# Filter manipulation
###############################################################################

cpdef size_t createFilterDescriptor():
    cdef FilterDescriptor desc
    status = cudnnCreateFilterDescriptor(&desc)
    check_status(status)
    return <size_t>desc


cpdef setFilter4dDescriptor(size_t filterDesc,
                          int dataType, int k, int c, int h, int w):
    status = cudnnSetFilter4dDescriptor(
        <FilterDescriptor>filterDesc, dataType, k, c, h, w)
    check_status(status)


cpdef setFilterNdDescriptor(size_t filterDesc, int dataType, int nbDims,
                            size_t filterDimA):
    status = cudnnSetFilterNdDescriptor(
        <FilterDescriptor>filterDesc, dataType, nbDims, <int*>filterDimA)
    check_status(status)


cpdef destroyFilterDescriptor(size_t filterDesc):
    status = cudnnDestroyFilterDescriptor(<FilterDescriptor>filterDesc)
    check_status(status)


###############################################################################
# Convolution
###############################################################################

cpdef size_t createConvolutionDescriptor():
    cdef ConvolutionDescriptor desc
    status = cudnnCreateConvolutionDescriptor(&desc)
    check_status(status)
    return <size_t>desc


cpdef setConvolution2dDescriptor(size_t convDesc, int pad_h, int pad_w, int u,
                                 int v, int upscalex, int upscaley, int mode):
    status = cudnnSetConvolution2dDescriptor(
        <ConvolutionDescriptor>convDesc, pad_h, pad_w, u, v, upscalex,
        upscaley, mode)
    check_status(status)


cpdef setConvolutionNdDescriptor(size_t convDesc, int arrayLength, size_t padA,
                                 size_t filterStrideA, size_t upscaleA,
                                 int mode):
    status = cudnnSetConvolutionNdDescriptor(
        <ConvolutionDescriptor>convDesc, arrayLength, <int*>padA,
        <int*>filterStrideA, <int*>upscaleA, mode)
    check_status(status)


cpdef destroyConvolutionDescriptor(size_t convDesc):
    status = cudnnDestroyConvolutionDescriptor(<ConvolutionDescriptor>convDesc)
    check_status(status)


cpdef int getConvolutionForwardAlgorithm(
        size_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, int preference, size_t memoryLimitInbytes):
    cdef int algo
    status = cudnnGetConvolutionForwardAlgorithm(
        <Handle>handle, <TensorDescriptor>srcDesc,
        <FilterDescriptor>filterDesc, <ConvolutionDescriptor>convDesc,
        <TensorDescriptor>destDesc, preference, memoryLimitInbytes, &algo)
    check_status(status)
    return algo


cpdef size_t getConvolutionForwardWorkspaceSize(
        size_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, int algo):
    cdef size_t sizeInBytes
    status = cudnnGetConvolutionForwardWorkspaceSize(
        <Handle>handle, <TensorDescriptor>srcDesc,
        <FilterDescriptor>filterDesc, <ConvolutionDescriptor> convDesc,
        <TensorDescriptor>destDesc, algo, &sizeInBytes)
    check_status(status)
    return sizeInBytes


cpdef convolutionForward(size_t handle, size_t alpha,
                         size_t srcDesc, size_t srcData,
                         size_t filterDesc, size_t filterData,
                         size_t convDesc, int algo, size_t workSpace,
                         size_t workSpaceSizeInBytes, size_t beta,
                         size_t destDesc, size_t destData):
    status = cudnnConvolutionForward(
        <Handle>handle, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData,
        <FilterDescriptor>filterDesc, <void*>filterData,
        <ConvolutionDescriptor>convDesc, algo, <void*>workSpace,
        workSpaceSizeInBytes, <void*>beta,
        <TensorDescriptor>destDesc, <void*>destData)
    check_status(status)


cpdef convolutionBackwardBias(size_t handle, size_t alpha,
                              size_t srcDesc, size_t srcData, size_t beta,
                              size_t destDesc, size_t destData):
    status = cudnnConvolutionBackwardBias(
        <Handle>handle, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
        <TensorDescriptor>destDesc, <void*>destData)
    check_status(status)


cpdef convolutionBackwardFilter(size_t handle, size_t alpha,
                                size_t srcDesc, size_t srcData,
                                size_t diffDesc, size_t diffData,
                                size_t convDesc, size_t beta,
                                size_t gradDesc, size_t gradData):
    status = cudnnConvolutionBackwardFilter(
        <Handle>handle, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData,
        <TensorDescriptor>diffDesc, <void*>diffData,
        <ConvolutionDescriptor>convDesc, <void*>beta,
        <FilterDescriptor>gradDesc, <void*>gradData)
    check_status(status)


cpdef convolutionBackwardData(size_t handle, size_t alpha,
                              size_t filterDesc, size_t filterData,
                              size_t diffDesc, size_t diffData,
                              size_t convDesc, size_t beta,
                              size_t gradDesc, size_t gradData):
    status = cudnnConvolutionBackwardData(
        <Handle>handle, <void*>alpha,
        <FilterDescriptor>filterDesc, <void*>filterData,
        <TensorDescriptor>diffDesc, <void*>diffData,
        <ConvolutionDescriptor>convDesc, <void*>beta,
        <TensorDescriptor>gradDesc, <void*>gradData)
    check_status(status)


###############################################################################
# Pooling
###############################################################################

cpdef size_t createPoolingDescriptor():
    cdef PoolingDescriptor desc
    status = cudnnCreatePoolingDescriptor(&desc)
    check_status(status)
    return <size_t>desc


cpdef setPooling2dDescriptor(size_t poolingDesc, int mode,
                             int windowHeight, int windowWidth,
                             int verticalPadding, int horizontalPadding,
                             int verticalStride, int horizontalStride):
    status = cudnnSetPooling2dDescriptor(
        <PoolingDescriptor>poolingDesc, mode, windowHeight, windowWidth,
        verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    check_status(status)


cpdef setPoolingNdDescriptor(size_t poolingDesc, int mode, int nbDims,
                             size_t windowDimA, size_t paddingA,
                             size_t strideA):
    status = cudnnSetPoolingNdDescriptor(
        <PoolingDescriptor>poolingDesc, mode, nbDims,
        <int*>windowDimA, <int*>paddingA, <int*>strideA)
    check_status(status)


cpdef destroyPoolingDescriptor(size_t poolingDesc):
    status = cudnnDestroyPoolingDescriptor(<PoolingDescriptor>poolingDesc)
    check_status(status)


cpdef poolingForward(size_t handle, size_t poolingDesc, size_t alpha,
                     size_t srcDesc, size_t srcData, size_t beta,
                     size_t dstDesc, size_t dstData):
    status = cudnnPoolingForward(
        <Handle>handle, <PoolingDescriptor>poolingDesc, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
        <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef poolingBackward(size_t handle, size_t poolingDesc, size_t alpha,
                      size_t srcDesc, size_t srcData,
                      size_t srcDiffDesc, size_t srcDiffData, 
                      size_t destDesc, size_t destData, size_t beta,
                      size_t destDiffDesc, size_t destDiffData):
    status = cudnnPoolingBackward(
        <Handle>handle, <PoolingDescriptor>poolingDesc, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData,
        <TensorDescriptor>srcDiffDesc, <void*>srcDiffData,
        <TensorDescriptor>destDesc, <void*>destData, <void*>beta,
        <TensorDescriptor>destDiffDesc, <void*>destDiffData)
    check_status(status)


###############################################################################
# Activation
###############################################################################

cpdef softmaxForward(size_t handle, int algorithm, int mode, size_t alpha,
                     size_t srcDesc, size_t srcData, size_t beta,
                     size_t dstDesc, size_t dstData):
    status = cudnnSoftmaxForward(
        <Handle>handle, algorithm, mode, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
        <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef softmaxBackward(size_t handle, int algorithm, int mode, size_t alpha,
                      size_t srcDesc, size_t srcData,
                      size_t srcDiffDesc, size_t srcDiffData, size_t beta,
                      size_t destDiffDesc, size_t destDiffData):
    status = cudnnSoftmaxBackward(
        <Handle>handle, algorithm, mode, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData,
        <TensorDescriptor>srcDiffDesc, <void*>srcDiffData, <void*>beta,
        <TensorDescriptor>destDiffDesc, <void*>destDiffData)
    check_status(status)


cpdef activationForward(size_t handle, int mode, size_t alpha,
                        size_t srcDesc, size_t srcData, size_t beta,
                        size_t dstDesc, size_t dstData):
    status = cudnnActivationForward(
        <Handle>handle, mode, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
        <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef activationBackward(size_t handle, int mode, size_t alpha,
                         size_t srcDesc, size_t srcData,
                         size_t srcDiffDesc, size_t srcDiffData,
                         size_t destDesc, size_t destData, size_t beta,
                         size_t destDiffDesc, size_t destDiffData):
    status = cudnnActivationBackward(
        <Handle>handle, mode, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData,
        <TensorDescriptor>srcDiffDesc, <void*>srcDiffData,
        <TensorDescriptor>destDesc, <void*>destData, <void*>beta,
        <TensorDescriptor>destDiffDesc, <void*>destDiffData)
    check_status(status)
