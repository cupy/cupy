###############################################################################
# Types
###############################################################################

from cupy.cuda.driver cimport Stream

cdef extern from *:
    ctypedef int ActivationMode 'cudnnActivationMode_t'
    ctypedef int AddMode 'cudnnAddMode_t'
    ctypedef int ConvolutionBwdDataAlgo 'cudnnConvolutionBwdDataAlgo_t'
    ctypedef int ConvolutionBwdDataPreference 'cudnnConvolutionBwdDataPreference_t'
    ctypedef int ConvolutionBwdFilterAlgo 'cudnnConvolutionBwdFilterAlgo_t'
    ctypedef int ConvolutionBwdFilterPreference 'cudnnConvolutionBwdFilterPreference_t'
    ctypedef int ConvolutionFwdAlgo 'cudnnConvolutionFwdAlgo_t'
    ctypedef int ConvolutionFwdPreference 'cudnnConvolutionFwdPreference_t'
    ctypedef int ConvolutionMode 'cudnnConvolutionMode_t'
    ctypedef int DataType 'cudnnDataType_t'
    ctypedef int NanPropagation 'cudnnNanPropagation_t'
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
    CUDNN_DATA_HALF = 2

    CUDNN_NOT_PROPAGATE_NAN = 0
    CUDNN_PROPAGATE_NAN = 1

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
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5

    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2

    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3

    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2

    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3

    CUDNN_SOFTMAX_FAST = 0
    CUDNN_SOFTMAX_ACCURATE = 1
    CUDNN_SOFTMAX_LOG = 2

    CUDNN_SOFTMAX_MODE_INSTANCE = 0
    CUDNN_SOFTMAX_MODE_CHANNEL = 1

    CUDNN_POOLING_MAX = 0
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2

    CUDNN_ACTIVATION_SIGMOID = 0
    CUDNN_ACTIVATION_RELU = 1
    CUDNN_ACTIVATION_TANH = 2
    CUDNN_ACTIVATION_CLIPPED_RELU = 3

    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0

    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0

    CUDNN_BATCHNORM_PER_ACTIVATION = 0
    CUDNN_BATCHNORM_SPATIAL = 1


###############################################################################
# Initialization and CUDA cooperation
###############################################################################

cpdef size_t create() except *
cpdef destroy(size_t handle)
cpdef setStream(size_t handle, size_t stream)
cpdef size_t getStream(size_t handle) except *


###############################################################################
# Tensor manipulation
###############################################################################

cpdef size_t createTensorDescriptor() except *
cpdef setTensor4dDescriptor(size_t tensorDesc, int format, int dataType,
                            int n, int c, int h, int w)
cpdef setTensor4dDescriptorEx(size_t tensorDesc, int dataType,
                              int n, int c, int h, int w, int nStride,
                              int cStride, int hStride, int wStride)
cpdef setTensorNdDescriptor(size_t tensorDesc, int dataType, int nbDims,
                            size_t dimA, size_t strideA)
cpdef destroyTensorDescriptor(size_t tensorDesc)
cpdef addTensor_v2(
        size_t handle, int mode, size_t alpha, size_t biasDesc,
        size_t biasData, size_t beta, size_t srcDestDesc, size_t srcDestData)
cpdef addTensor_v3(size_t handle, size_t alpha, size_t bDesc,
                   size_t b, size_t beta, size_t yDesc, size_t y)


###############################################################################
# Filter manipulation
###############################################################################

cpdef size_t createFilterDescriptor() except *
cpdef setFilter4dDescriptor_v3(
        size_t filterDesc, int dataType, int k, int c, int h, int w)
cpdef setFilterNdDescriptor_v3(
        size_t filterDesc, int dataType, int nbDims, size_t filterDimA)
cpdef destroyFilterDescriptor(size_t filterDesc)


###############################################################################
# Convolution
###############################################################################

cpdef size_t createConvolutionDescriptor() except *
cpdef setConvolution2dDescriptor(
        size_t convDesc, int pad_h, int pad_w, int u, int v, int upscalex,
        int upscaley, int mode)
cpdef setConvolutionNdDescriptor_v2(
        size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
        size_t upscaleA, int mode)
cpdef setConvolutionNdDescriptor_v3(
        size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
        size_t upscaleA, int mode, int dataType)
cpdef destroyConvolutionDescriptor(size_t convDesc)
cpdef int getConvolutionForwardAlgorithm(
        size_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, ConvolutionFwdPreference preference,
        size_t memoryLimitInbytes) except *
cpdef size_t getConvolutionForwardWorkspaceSize(
        size_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, int algo) except *
cpdef convolutionForward(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t filterDesc, size_t filterData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t destDesc, size_t destData)
cpdef convolutionBackwardBias(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t beta, size_t destDesc, size_t destData)
cpdef int getConvolutionBackwardFilterAlgorithm(
        size_t handle, size_t srcDesc, size_t diffDesc, size_t convDesc,
        size_t filterDesc, ConvolutionBwdFilterPreference preference,
        size_t memoryLimitInbytes) except *
cpdef size_t getConvolutionBackwardFilterWorkspaceSize(
        size_t handle, size_t srcDesc, size_t diffDesc, size_t convDesc,
        size_t filterDesc, int algo) except *
cpdef convolutionBackwardFilter_v2(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t diffDesc, size_t diffData, size_t convDesc, size_t beta,
        size_t gradDesc, size_t gradData)
cpdef convolutionBackwardFilter_v3(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t diffDesc, size_t diffData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t gradDesc, size_t gradData)
cpdef int getConvolutionBackwardDataAlgorithm(
        size_t handle, size_t filterDesc, size_t diffDesc, size_t convDesc,
        size_t gradDesc, size_t preference,
        size_t memoryLimitInbytes) except *
cpdef size_t getConvolutionBackwardDataWorkspaceSize(
        size_t handle, size_t filterDesc, size_t diffDesc, size_t convDesc,
        size_t gradDesc, int algo) except *
cpdef convolutionBackwardData_v2(
        size_t handle, size_t alpha, size_t filterDesc, size_t filterData,
        size_t diffDesc, size_t diffData, size_t convDesc, size_t beta,
        size_t gradDesc, size_t gradData)
cpdef convolutionBackwardData_v3(
        size_t handle, size_t alpha, size_t filterDesc, size_t filterData,
        size_t diffDesc, size_t diffData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t gradDesc, size_t gradData)


###############################################################################
# Pooling
###############################################################################

cpdef size_t createPoolingDescriptor() except *
cpdef setPooling2dDescriptor_v3(
        size_t poolingDesc, int mode, int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding, int verticalStride,
        int horizontalStride)
cpdef setPoolingNdDescriptor_v3(
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
cpdef activationForward_v3(
        size_t handle, int mode, size_t alpha, size_t srcDesc, size_t srcData,
        size_t beta, size_t dstDesc, size_t dstData)
cpdef activationBackward_v3(
        size_t handle, int mode, size_t alpha, size_t srcDesc, size_t srcData,
        size_t srcDiffDesc, size_t srcDiffData, size_t destDesc,
        size_t destData, size_t beta, size_t destDiffDesc,
        size_t destDiffData)
