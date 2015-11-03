###############################################################################
# Types
###############################################################################

from cupy.cuda.driver cimport Stream

cdef extern from *:
    ctypedef int ActivationMode 'cudnnActivationMode_t'
    ctypedef int AddMode 'cudnnAddMode_t'
    ctypedef int ConvolutionFwdAlgo 'cudnnConvolutionFwdAlgo_t'
    ctypedef int ConvolutionFwdPreference 'cudnnConvolutionFwdPreference_t'
    ctypedef int ConvolutionMode 'cudnnConvolutionMode_t'
    ctypedef int DataType 'cudnnDataType_t'
    ctypedef int PoolingMode 'cudnnPoolingMode_t'
    ctypedef int SoftmaxAlgorithm 'cudnnSoftmaxAlgorithm_t'
    ctypedef int SoftmaxMode 'cudnnSoftmaxMode_t'
    ctypedef int Status 'cudnnStatus_t'
    ctypedef int TensorFormat 'cudnnTensorFormat_t'

    ctypedef void* ConvolutionDescriptor 'cudnnConvolutionDescriptor_t'
    ctypedef void* FilterDescriptor 'cudnnFilterDescriptor_t'
    ctypedef void* Handle 'cudnnHandle_t'
    ctypedef void* PoolingDescriptor 'cudnnPoolingDescriptor_t'
    ctypedef void* TensorDescriptor 'cudnnTensorDescriptor_t'


###############################################################################
# Enum
###############################################################################

cpdef enum:
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
# Initialization and CUDA cooperation
###############################################################################

cpdef size_t create()
cpdef destroy(size_t handle)
cpdef setStream(size_t handle, size_t stream)
cpdef size_t getStream(size_t handle)


###############################################################################
# Tensor manipulation
###############################################################################

cpdef size_t createTensorDescriptor()
cpdef setTensor4dDescriptor(size_t tensorDesc, int format, int dataType,
                            int n, int c, int h, int w)
cpdef setTensor4dDescriptorEx(size_t tensorDesc, int dataType,
                              int n, int c, int h, int w, int nStride,
                              int cStride, int hStride, int wStride)
cpdef setTensorNdDescriptor(size_t tensorDesc, int dataType, int nbDims,
                            size_t dimA, size_t strideA)
cpdef destroyTensorDescriptor(size_t tensorDesc)
cpdef addTensor(
        size_t handle, int mode, size_t alpha, size_t biasDesc,
        size_t biasData, size_t beta, size_t srcDestDesc, size_t srcDestData)


###############################################################################
# Filter manipulation
###############################################################################

cpdef size_t createFilterDescriptor()
cpdef setFilter4dDescriptor(
        size_t filterDesc, int dataType, int k, int c, int h, int w)
cpdef setFilterNdDescriptor(
        size_t filterDesc, int dataType, int nbDims, size_t filterDimA)
cpdef destroyFilterDescriptor(size_t filterDesc)


###############################################################################
# Convolution
###############################################################################

cpdef size_t createConvolutionDescriptor()
cpdef setConvolution2dDescriptor(
        size_t convDesc, int pad_h, int pad_w, int u, int v, int upscalex,
        int upscaley, int mode)
cpdef setConvolutionNdDescriptor(
        size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
        size_t upscaleA, int mode)
cpdef destroyConvolutionDescriptor(size_t convDesc)
cpdef int getConvolutionForwardAlgorithm(
        size_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, int preference, size_t memoryLimitInbytes)
cpdef size_t getConvolutionForwardWorkspaceSize(
        size_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, int algo)
cpdef convolutionForward(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t filterDesc, size_t filterData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t destDesc, size_t destData)
cpdef convolutionBackwardBias(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t beta, size_t destDesc, size_t destData)
cpdef convolutionBackwardFilter(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t diffDesc, size_t diffData, size_t convDesc, size_t beta,
        size_t gradDesc, size_t gradData)
cpdef convolutionBackwardData(
        size_t handle, size_t alpha, size_t filterDesc, size_t filterData,
        size_t diffDesc, size_t diffData, size_t convDesc, size_t beta,
        size_t gradDesc, size_t gradData)


###############################################################################
# Pooling
###############################################################################

cpdef size_t createPoolingDescriptor()
cpdef setPooling2dDescriptor(
        size_t poolingDesc, int mode, int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding, int verticalStride,
        int horizontalStride)
cpdef setPoolingNdDescriptor(
        size_t poolingDesc, int mode, int nbDims, size_t windowDimA, 
        size_t paddingA, size_t strideA)
cpdef destroyPoolingDescriptor(size_t poolingDesc)
cpdef poolingForward(
        size_t handle, size_t poolingDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData)
cpdef poolingBackward(
        size_t handle, size_t poolingDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData,
        size_t destDesc, size_t destData, size_t beta, size_t destDiffDesc,
        size_t destDiffData)


###############################################################################
# Activation
###############################################################################

cpdef softmaxForward(
        size_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData)
cpdef softmaxBackward(
        size_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData, size_t beta,
        size_t destDiffDesc, size_t destDiffData)
cpdef activationForward(
        size_t handle, int mode, size_t alpha, size_t srcDesc, size_t srcData,
        size_t beta, size_t dstDesc, size_t dstData)
cpdef activationBackward(
        size_t handle, int mode, size_t alpha, size_t srcDesc, size_t srcData,
        size_t srcDiffDesc, size_t srcDiffData, size_t destDesc,
        size_t destData, size_t beta, size_t destDiffDesc,
        size_t destDiffData)
