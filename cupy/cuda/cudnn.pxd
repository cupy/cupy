###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef int ActivationMode 'cudnnActivationMode_t'
    ctypedef int AddMode 'cudnnAddMode_t'
    ctypedef int BatchNormMode 'cudnnBatchNormMode_t'
    ctypedef int ConvolutionBwdDataAlgo 'cudnnConvolutionBwdDataAlgo_t'
    ctypedef int ConvolutionBwdDataPreference \
        'cudnnConvolutionBwdDataPreference_t'
    ctypedef struct ConvolutionBwdDataAlgoPerf \
        'cudnnConvolutionBwdDataAlgoPerf_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
    ctypedef int ConvolutionBwdFilterAlgo 'cudnnConvolutionBwdFilterAlgo_t'
    ctypedef int ConvolutionBwdFilterPreference \
        'cudnnConvolutionBwdFilterPreference_t'
    ctypedef struct ConvolutionBwdFilterAlgoPerf \
        'cudnnConvolutionBwdFilterAlgoPerf_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
    ctypedef int ConvolutionFwdAlgo 'cudnnConvolutionFwdAlgo_t'
    ctypedef int ConvolutionFwdPreference 'cudnnConvolutionFwdPreference_t'
    ctypedef struct ConvolutionFwdAlgoPerf 'cudnnConvolutionFwdAlgoPerf_t':
        int algo
        int status
        float time
        size_t memory
    ctypedef int ConvolutionMode 'cudnnConvolutionMode_t'
    ctypedef int DataType 'cudnnDataType_t'
    ctypedef int DirectionMode 'cudnnDirectionMode_t'
    ctypedef int NanPropagation 'cudnnNanPropagation_t'
    ctypedef int PoolingMode 'cudnnPoolingMode_t'
    ctypedef int RNNInputMode 'cudnnRNNInputMode_t'
    ctypedef int RNNMode 'cudnnRNNMode_t'
    ctypedef int SoftmaxAlgorithm 'cudnnSoftmaxAlgorithm_t'
    ctypedef int SoftmaxMode 'cudnnSoftmaxMode_t'
    ctypedef int Status 'cudnnStatus_t'
    ctypedef int TensorFormat 'cudnnTensorFormat_t'

    ctypedef void* ActivationDescriptor 'cudnnActivationDescriptor_t'
    ctypedef void* ConvolutionDescriptor 'cudnnConvolutionDescriptor_t'
    ctypedef void* DropoutDescriptor 'cudnnDropoutDescriptor_t'
    ctypedef void* FilterDescriptor 'cudnnFilterDescriptor_t'
    ctypedef void* Handle 'cudnnHandle_t'
    ctypedef void* PoolingDescriptor 'cudnnPoolingDescriptor_t'
    ctypedef void* RNNDescriptor 'cudnnRNNDescriptor_t'
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
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7

    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2

    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5

    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2

    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5

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

    CUDNN_RNN_RELU = 0
    CUDNN_RNN_TANH = 1
    CUDNN_LSTM = 2
    CUDNN_GRU = 3

    CUDNN_UNIDIRECTIONAL = 0
    CUDNN_BIDIRECTIONAL = 1

    CUDNN_LINEAR_INPUT = 0
    CUDNN_SKIP_INPUT = 1


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
cpdef setFilter4dDescriptor_v4(
    size_t filterDesc, int dataType, int format, int k, int c, int h, int w)
cpdef setFilterNdDescriptor_v4(
    size_t filterDesc, int dataType, int format, int nbDims, size_t filterDimA)
cpdef destroyFilterDescriptor(size_t filterDesc)


###############################################################################
# Convolution
###############################################################################

cpdef size_t createConvolutionDescriptor() except *
cpdef setConvolution2dDescriptor_v4(
    size_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h,
    int dilation_w, int mode)
cpdef setConvolution2dDescriptor_v5(
    size_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h,
    int dilation_w, int mode, size_t computeType)
cpdef setConvolutionNdDescriptor_v2(
    size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
    size_t dilationA, int mode)
cpdef setConvolutionNdDescriptor_v3(
    size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
    size_t dilationA, int mode, int dataType)
cpdef destroyConvolutionDescriptor(size_t convDesc)
cpdef findConvolutionForwardAlgorithm(
    size_t handle, size_t xDesc, size_t wDesc, size_t convDesc, size_t yDesc,
    int requestedAlgoCount)
cpdef findConvolutionForwardAlgorithmEx(
    size_t handle, size_t xDesc, size_t x, size_t wDesc, size_t w,
    size_t convDesc, size_t yDesc, size_t y, int requestedAlgoCount,
    size_t workSpace, size_t workSpaceSizeInBytes)
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
cpdef findConvolutionBackwardFilterAlgorithm(
    size_t handle, size_t xDesc, size_t dyDesc, size_t convDesc, size_t dwDesc,
    int requestedAlgoCount)
cpdef findConvolutionBackwardFilterAlgorithmEx(
    size_t handle, size_t xDesc, size_t x, size_t dyDesc, size_t dy,
    size_t convDesc, size_t dwDesc, size_t dw, int requestedAlgoCount,
    size_t workSpace, size_t workSpaceSizeInBytes)
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
cpdef findConvolutionBackwardDataAlgorithm(
    size_t handle, size_t wDesc, size_t dyDesc, size_t convDesc, size_t dxDesc,
    int requestedAlgoCount)
cpdef findConvolutionBackwardDataAlgorithmEx(
    size_t handle, size_t wDesc, size_t w, size_t dyDesc, size_t dy,
    size_t convDesc, size_t dxDesc, size_t dx,
    int requestedAlgoCount, size_t workSpace, size_t workSpaceSizeInBytes)
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
# Batch Normalization
###############################################################################

cpdef deriveBNTensorDescriptor(
    size_t derivedBnDesc, size_t xDesc, int mode)

cpdef batchNormalizationForwardTraining(
    size_t handle, int mode,
    size_t alpha, size_t beta, size_t xDesc,
    size_t x, size_t yDesc, size_t y,
    size_t bnScaleBiasMeanVarDesc, size_t bnScale,
    size_t bnBias, double exponentialAverageFactor,
    size_t resultRunningMean, size_t resultRunningVariance,
    double epsilon, size_t resultSaveMean, size_t resultSaveInvVariance)

cpdef batchNormalizationForwardInference(
    size_t handle, int mode,
    size_t alpha, size_t beta, size_t xDesc,
    size_t x, size_t yDesc, size_t y,
    size_t bnScaleBiasMeanVarDesc, size_t bnScale,
    size_t bnBias, size_t estimatedMean, size_t estimatedVariance,
    double epsilon)

cpdef batchNormalizationBackward(
    size_t handle, int mode,
    size_t alphaDataDiff, size_t betaDataDiff,
    size_t alphaParamDiff, size_t betaParamDiff,
    size_t xDesc, size_t x, size_t dyDesc,
    size_t dy, size_t dxDesc, size_t dx,
    size_t dBnScaleBiasDesc, size_t bnScale,
    size_t dBnScaleResult, size_t dBnBiasResult,
    double epsilon, size_t savedMean, size_t savedInvVariance)


###############################################################################
# Activation
###############################################################################

cpdef size_t createActivationDescriptor() except *
cpdef setActivationDescriptor(
    size_t activationDesc, int mode, int reluNanOpt, double reluCeiling)
cpdef destroyActivationDescriptor(size_t activationDesc)
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

cpdef size_t createDropoutDescriptor() except *
cpdef destroyDropoutDescriptor(size_t dropoutDesc)
cpdef size_t dropoutGetStatesSize(size_t handle) except *
