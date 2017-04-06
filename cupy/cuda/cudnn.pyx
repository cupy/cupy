# distutils: language = c++

"""Thin wrapper of cuDNN."""
# NOTE: This wrapper does not cover all APIs of cuDNN v4.
cimport cython
from libcpp cimport vector
import numpy

from cupy.cuda cimport driver


###############################################################################
# Extern
###############################################################################

cdef extern from "cupy_cudnn.h":
    # Error handling
    const char* cudnnGetErrorString(Status status) nogil

    # Version
    size_t cudnnGetVersion() nogil

    # Initialization and CUDA cooperation
    int cudnnCreate(Handle* handle) nogil
    int cudnnDestroy(Handle handle) nogil
    int cudnnSetStream(Handle handle, driver.Stream stream) nogil
    int cudnnGetStream(Handle handle, driver.Stream* stream) nogil

    # Tensor manipulation
    int cudnnCreateTensorDescriptor(TensorDescriptor* descriptor) nogil
    int cudnnSetTensor4dDescriptor(
        TensorDescriptor tensorDesc, TensorFormat format,
        DataType dataType, int n, int c, int h, int w) nogil
    int cudnnSetTensor4dDescriptorEx(
        TensorDescriptor tensorDesc, DataType dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride) nogil
    int cudnnSetTensorNdDescriptor(
        TensorDescriptor tensorDesc, DataType dataType, int nbDims,
        int* dimA, int* strideA) nogil
    int cudnnDestroyTensorDescriptor(TensorDescriptor tensorDesc) nogil
    int cudnnAddTensor_v2(
        Handle handle, AddMode mode, void* alpha,
        TensorDescriptor biasDesc, void* biasData, void* beta,
        TensorDescriptor srcDestDesc, void* srcDestData) nogil
    int cudnnAddTensor_v3(
        Handle handle, void* alpha, TensorDescriptor bDesc,
        void* b, void* beta, TensorDescriptor yDesc, void* y) nogil

    # Filter manipulation
    int cudnnCreateFilterDescriptor(FilterDescriptor* filterDesc) nogil
    int cudnnSetFilter4dDescriptor_v3(
        FilterDescriptor filterDesc, DataType dataType,
        int n, int c, int h, int w) nogil
    int cudnnSetFilterNdDescriptor_v3(
        FilterDescriptor filterDesc, DataType dataType, int nbDims,
        int* filterDimA) nogil
    int cudnnSetFilter4dDescriptor_v4(
        FilterDescriptor filterDesc, DataType dataType,
        TensorFormat format, int k, int c, int h, int w) nogil
    int cudnnSetFilterNdDescriptor_v4(
        FilterDescriptor filterDesc, DataType dataType,
        TensorFormat format, int nbDims, const int filterDimA[]) nogil
    int cudnnGetFilterNdDescriptor_v4(
        FilterDescriptor wDesc, int nbDimsRequested, DataType* dataType,
        TensorFormat* format, int* nbDims, int filterDimA[]) nogil
    int cudnnDestroyFilterDescriptor(FilterDescriptor filterDesc) nogil

    # Convolution
    int cudnnCreateConvolutionDescriptor(ConvolutionDescriptor* convDesc) nogil
    int cudnnSetConvolution2dDescriptor_v4(
        ConvolutionDescriptor convDesc, int pad_h, int pad_w, int u,
        int v, int dilation_h, int dilation_w, ConvolutionMode mode) nogil
    int cudnnSetConvolution2dDescriptor_v5(
        ConvolutionDescriptor convDesc, int pad_h, int pad_w, int u,
        int v, int dilation_h, int dilation_w, ConvolutionMode mode,
        DataType computeType) nogil
    int cudnnSetConvolutionNdDescriptor_v2(
        ConvolutionDescriptor convDesc, int arrayLength, int* padA,
        int* filterStrideA, int* dilationA, ConvolutionMode mode) nogil
    int cudnnSetConvolutionNdDescriptor_v3(
        ConvolutionDescriptor convDesc, int arrayLength, int* padA,
        int* filterStrideA, int* dilationA, ConvolutionMode mode,
        DataType dataType) nogil
    int cudnnDestroyConvolutionDescriptor(ConvolutionDescriptor conDesc) nogil
    int cudnnFindConvolutionForwardAlgorithm(
        Handle handle, TensorDescriptor xDesc, FilterDescriptor wDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor yDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionFwdAlgoPerf* perfResults) nogil
    int cudnnFindConvolutionForwardAlgorithmEx(
        Handle handle, TensorDescriptor xDesc, void* x,
        FilterDescriptor wDesc, void* w, ConvolutionDescriptor convDesc,
        TensorDescriptor yDesc, void* y, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionFwdAlgoPerf* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes) nogil
    int cudnnGetConvolutionForwardAlgorithm(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionFwdPreference preference,
        size_t memoryLimitInbytes, ConvolutionFwdAlgo* algo) nogil
    int cudnnGetConvolutionForwardWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionFwdAlgo algo,
        size_t* sizeInBytes) nogil
    int cudnnConvolutionForward(
        Handle handle, void* alpha, TensorDescriptor srcDesc,
        void* srcData, FilterDescriptor filterDesc, void* filterData,
        ConvolutionDescriptor convDesc, ConvolutionFwdAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor destDesc, void* destData) nogil
    int cudnnConvolutionBackwardBias(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor destDesc, void* destData) nogil
    int cudnnFindConvolutionBackwardFilterAlgorithm(
        Handle handle, TensorDescriptor xDesc, TensorDescriptor dyDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor dwDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionBwdFilterAlgoPerf* perfResults) nogil
    int cudnnFindConvolutionBackwardFilterAlgorithmEx(
        Handle handle, TensorDescriptor xDesc, void* x,
        TensorDescriptor dyDesc, void* dy, ConvolutionDescriptor convDesc,
        FilterDescriptor dwDesc, void* dw, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionBwdFilterAlgoPerf* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes) nogil
    int cudnnGetConvolutionBackwardFilterAlgorithm(
        Handle handle, TensorDescriptor srcDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor filterDesc,
        ConvolutionBwdFilterPreference preference,
        size_t memoryLimitInbytes, ConvolutionBwdFilterAlgo* algo) nogil
    int cudnnGetConvolutionBackwardFilterWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor filterDesc,
        ConvolutionBwdFilterAlgo algo, size_t* sizeInBytes) nogil
    int cudnnConvolutionBackwardFilter_v2(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, void* beta,
        FilterDescriptor gradDesc, void* gradData) nogil
    int cudnnConvolutionBackwardFilter_v3(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, ConvolutionBwdFilterAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        FilterDescriptor gradDesc, void* gradData) nogil
    int cudnnGetConvolutionBackwardDataAlgorithm(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor gradDesc,
        ConvolutionBwdDataPreference preference,
        size_t memoryLimitInbytes, ConvolutionBwdDataAlgo* algo) nogil
    int cudnnFindConvolutionBackwardDataAlgorithm(
        Handle handle, TensorDescriptor wDesc, TensorDescriptor dyDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor dxDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionBwdDataAlgoPerf* perfResults) nogil
    int cudnnFindConvolutionBackwardDataAlgorithmEx(
        Handle handle, FilterDescriptor wDesc, void* w,
        TensorDescriptor dyDesc, void* dy, ConvolutionDescriptor convDesc,
        TensorDescriptor dxDesc, void* dx, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionBwdDataAlgoPerf* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes) nogil
    int cudnnGetConvolutionBackwardDataWorkspaceSize(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor gradDesc,
        ConvolutionBwdDataAlgo algo, size_t* sizeInBytes) nogil
    int cudnnConvolutionBackwardData_v2(
        Handle handle, void* alpha,
        FilterDescriptor filterDesc, void* filterData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, void* beta,
        TensorDescriptor gradDesc, void* gradData) nogil
    int cudnnConvolutionBackwardData_v3(
        Handle handle, void* alpha,
        FilterDescriptor filterDesc, void* filterData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, ConvolutionBwdDataAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor gradDesc, void* gradData) nogil

    # Pooling
    int cudnnCreatePoolingDescriptor(PoolingDescriptor* desc) nogil
    int cudnnSetPooling2dDescriptor_v3(
        PoolingDescriptor poolingDesc, PoolingMode mode,
        int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding,
        int verticalStride, int horizontalStride) nogil
    int cudnnSetPooling2dDescriptor_v4(
        PoolingDescriptor poolingDesc, PoolingMode mode,
        NanPropagation maxpoolingNanOpt, int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding, int verticalStride,
        int horizontalStride) nogil
    int cudnnSetPoolingNdDescriptor_v3(
        PoolingDescriptor poolingDesc, PoolingMode mode, int nbDims,
        int* windowDimA, int* paddingA, int* strideA) nogil
    int cudnnSetPoolingNdDescriptor_v4(
        PoolingDescriptor poolingDesc, PoolingMode mode,
        NanPropagation maxpoolingNanOpt, int nbDims,
        int* windowDimA, int* paddingA, int* strideA) nogil
    int cudnnDestroyPoolingDescriptor(PoolingDescriptor poolingDesc) nogil
    int cudnnPoolingForward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData) nogil
    int cudnnPoolingBackward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData) nogil

    # Batch Normalization
    int cudnnDeriveBNTensorDescriptor(
        TensorDescriptor derivedBnDesc, TensorDescriptor xDesc,
        BatchNormMode mode) nogil
    int cudnnBatchNormalizationForwardTraining(
        Handle handle, BatchNormMode mode,
        void* alpha, void* beta, TensorDescriptor xDesc,
        void* x, TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
        void* bnBias, double exponentialAverageFactor,
        void* resultRunningMean, void* resultRunningVariance,
        double epsilon, void* resultSaveMean,
        void* resultSaveInvVariance) nogil
    int cudnnBatchNormalizationForwardInference(
        Handle handle, BatchNormMode mode,
        void* alpha, void* beta, TensorDescriptor xDesc,
        void* x, TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
        void* bnBias, void* estimatedMean, void* estimatedVariance,
        double epsilon) nogil
    int cudnnBatchNormalizationBackward(
        Handle handle, BatchNormMode mode,
        void* alphaDataDiff, void* betaDataDiff,
        void* alphaParamDiff, void* betaParamDiff,
        TensorDescriptor xDesc, void* x,
        TensorDescriptor dyDesc, void* dy,
        TensorDescriptor dxDesc, void* dx,
        TensorDescriptor dBnScaleBiasDesc, void* bnScale,
        void* dBnScaleResult, void* dBnBiasResult,
        double epsilon, void* savedMean, void* savedInvVariance) nogil

    # Activation
    int cudnnCreateActivationDescriptor(
        ActivationDescriptor* activationDesc) nogil
    int cudnnSetActivationDescriptor(
        ActivationDescriptor activationDesc, ActivationMode mode,
        NanPropagation reluNanOpt, double reluCeiling) nogil
    int cudnnDestroyActivationDescriptor(
        ActivationDescriptor activationDesc) nogil
    int cudnnSoftmaxForward(
        Handle handle, SoftmaxAlgorithm algorithm, SoftmaxMode mode,
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        void* beta, TensorDescriptor dstDesc, void* dstData) nogil
    int cudnnSoftmaxBackward(
        Handle handle, SoftmaxAlgorithm algorithm, SoftmaxMode mode,
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData) nogil
    int cudnnActivationForward_v3(
        Handle handle, ActivationMode mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData) nogil
    int cudnnActivationForward_v4(
        Handle handle, ActivationDescriptor activationDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData) nogil
    int cudnnActivationBackward_v3(
        Handle handle, ActivationMode mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData) nogil
    int cudnnActivationBackward_v4(
        Handle handle, ActivationDescriptor activationDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData) nogil

    # Dropout
    int cudnnCreateDropoutDescriptor(DropoutDescriptor* desc) nogil
    int cudnnDestroyDropoutDescriptor(DropoutDescriptor dropoutDesc) nogil
    int cudnnDropoutGetStatesSize(Handle handle, size_t* sizeInBytes) nogil
    int cudnnDropoutGetReserveSpaceSize(
        TensorDescriptor xDesc, size_t* sizeInBytes) nogil
    int cudnnSetDropoutDescriptor(
        DropoutDescriptor dropoutDesc, Handle handle, float dropout,
        void* states, size_t stateSizeInBytes, unsigned long long seed) nogil
    int cudnnDropoutBackward(
        Handle handle, DropoutDescriptor dropoutDesc,
        TensorDescriptor dydesc, void* dy, TensorDescriptor dxdesc,
        void* dx, void* reserveSpace, size_t reserveSpaceSizeInBytes) nogil

    # RNN
    int cudnnCreateRNNDescriptor(RNNDescriptor* rnnDesc) nogil
    int cudnnDestroyRNNDescriptor(RNNDescriptor rnnDesc) nogil
    int cudnnSetRNNDescriptor(
        RNNDescriptor rnnDesc, int hiddenSize,
        int numLayers, DropoutDescriptor dropoutDesc, RNNInputMode inputMode,
        DirectionMode direction, RNNMode mode, DataType dataType) nogil
    int cudnnGetRNNWorkspaceSize(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, size_t* sizeInBytes) nogil
    int cudnnGetRNNTrainingReserveSize(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, size_t* sizeInBytes) nogil
    int cudnnGetRNNParamsSize(
        Handle handle, RNNDescriptor rnnDesc, TensorDescriptor xDesc,
        size_t* sizeInBytes, DataType dataType) nogil
    int cudnnGetRNNLinLayerMatrixParams(
        Handle handle, RNNDescriptor rnnDesc, int layer,
        TensorDescriptor xDesc, FilterDescriptor wDesc, void* w,
        int linLayerID, FilterDescriptor linLayerMatDesc,
        void** linLayerMat) nogil
    int cudnnGetRNNLinLayerBiasParams(
        Handle handle, RNNDescriptor rnnDesc, int layer,
        TensorDescriptor xDesc, FilterDescriptor wDesc, void* w,
        int linLayerID, FilterDescriptor linLayerBiasDesc,
        void** linLayerBias) nogil
    int cudnnRNNForwardInference(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc,
        void* x, TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc,
        void* cx, FilterDescriptor wDesc, void* w, TensorDescriptor* yDesc,
        void* y, TensorDescriptor hyDesc, void* hy, TensorDescriptor cyDesc,
        void* cy, void* workspace, size_t workSpaceSizeInBytes) nogil
    int cudnnRNNForwardTraining(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, void* x,
        TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc, void* cx,
        FilterDescriptor wDesc, void* w, TensorDescriptor* yDesc, void* y,
        TensorDescriptor hyDesc, void* hy, TensorDescriptor cyDesc, void* cy,
        void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace,
        size_t reserveSpaceSizeInBytes) nogil
    int cudnnRNNBackwardData(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* yDesc, void* y,
        TensorDescriptor* dyDesc, void* dy,
        TensorDescriptor dhyDesc, void* dhy,
        TensorDescriptor dcyDesc, void* dcy,
        FilterDescriptor wDesc, void* w,
        TensorDescriptor hxDesc, void* hx,
        TensorDescriptor cxDesc, void* cx,
        TensorDescriptor* dxDesc, void* dx,
        TensorDescriptor dhxDesc, void* dhx,
        TensorDescriptor dcxDesc, void* dcx, void* workspace,
        size_t workSpaceSizeInBytes, void* reserveSpace,
        size_t reserveSpaceSizeInBytes) nogil
    int cudnnRNNBackwardWeights(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, void* x, TensorDescriptor hxDesc, void* hx,
        TensorDescriptor* yDesc, void* y,
        void* workspace, size_t workSpaceSizeInBytes, FilterDescriptor dwDesc,
        void* dw, void* reserveSpace, size_t reserveSpaceSizeInBytes) nogil

###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
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
        msg = cudnnGetErrorString(<Status>status)
        super(CuDNNError, self).__init__('%s: %s' % (STATUS[status], msg))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CuDNNError(status)


###############################################################################
# Version
###############################################################################

cpdef size_t getVersion() except *:
    return cudnnGetVersion()


###############################################################################
# Initialization and CUDA cooperation
###############################################################################

cpdef size_t create() except *:
    cdef Handle handle
    with nogil:
        status = cudnnCreate(&handle)
    check_status(status)
    return <size_t>handle


cpdef destroy(size_t handle):
    with nogil:
        status = cudnnDestroy(<Handle>handle)
    check_status(status)


cpdef setStream(size_t handle, size_t stream):
    status = cudnnSetStream(<Handle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t getStream(size_t handle) except *:
    cdef driver.Stream stream
    status = cudnnGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


###############################################################################
# Tensor manipulation
###############################################################################

cpdef size_t createTensorDescriptor() except *:
    cdef TensorDescriptor descriptor
    status = cudnnCreateTensorDescriptor(&descriptor)
    check_status(status)
    return <size_t>descriptor


cpdef setTensor4dDescriptor(size_t tensorDesc, int format, int dataType,
                            int n, int c, int h, int w):
    status = cudnnSetTensor4dDescriptor(
        <TensorDescriptor>tensorDesc, <TensorFormat>format,
        <DataType>dataType, n, c, h, w)
    check_status(status)


cpdef setTensor4dDescriptorEx(size_t tensorDesc, int dataType,
                              int n, int c, int h, int w, int nStride,
                              int cStride, int hStride, int wStride):
    status = cudnnSetTensor4dDescriptorEx(
        <TensorDescriptor>tensorDesc, <DataType>dataType, n, c, h, w,
        nStride, cStride, hStride, wStride)
    check_status(status)


cpdef setTensorNdDescriptor(size_t tensorDesc, int dataType, int nbDims,
                            size_t dimA, size_t strideA):
    status = cudnnSetTensorNdDescriptor(
        <TensorDescriptor>tensorDesc, <DataType>dataType, nbDims,
        <int*>dimA, <int*>strideA)
    check_status(status)


cpdef destroyTensorDescriptor(size_t tensorDesc):
    status = cudnnDestroyTensorDescriptor(<TensorDescriptor>tensorDesc)
    check_status(status)


cpdef addTensor_v2(
        size_t handle, int mode, size_t alpha, size_t biasDesc,
        size_t biasData, size_t beta, size_t srcDestDesc, size_t srcDestData):
    with nogil:
        status = cudnnAddTensor_v2(
            <Handle>handle, <AddMode>mode, <void*>alpha,
            <TensorDescriptor>biasDesc, <void*>biasData, <void*>beta,
            <TensorDescriptor>srcDestDesc, <void*>srcDestData)
    check_status(status)


cpdef addTensor_v3(size_t handle, size_t alpha, size_t bDesc,
                   size_t b, size_t beta, size_t yDesc, size_t y):
    with nogil:
        status = cudnnAddTensor_v3(
            <Handle>handle, <void*>alpha, <TensorDescriptor>bDesc,
            <void*>b, <void*>beta, <TensorDescriptor>yDesc, <void*>y)
    check_status(status)


###############################################################################
# Filter manipulation
###############################################################################

cpdef size_t createFilterDescriptor() except *:
    cdef FilterDescriptor desc
    status = cudnnCreateFilterDescriptor(&desc)
    check_status(status)
    return <size_t>desc


cpdef setFilter4dDescriptor_v3(
        size_t filterDesc, int dataType, int k, int c, int h, int w):
    status = cudnnSetFilter4dDescriptor_v3(
        <FilterDescriptor>filterDesc, <DataType>dataType, k, c, h, w)
    check_status(status)


cpdef setFilterNdDescriptor_v3(
        size_t filterDesc, int dataType, int nbDims, size_t filterDimA):
    status = cudnnSetFilterNdDescriptor_v3(
        <FilterDescriptor>filterDesc, <DataType>dataType, nbDims,
        <int*>filterDimA)
    check_status(status)


cpdef setFilter4dDescriptor_v4(
        size_t filterDesc, int dataType,
        int format, int k, int c, int h, int w):
    status = cudnnSetFilter4dDescriptor_v4(
        <FilterDescriptor>filterDesc, <DataType> dataType,
        <TensorFormat> format, k, c, h, w)
    check_status(status)


cpdef setFilterNdDescriptor_v4(
        size_t filterDesc, int dataType,
        int format, int nbDims, size_t filterDimA):
    status = cudnnSetFilterNdDescriptor_v4(
        <FilterDescriptor>filterDesc, <DataType>dataType,
        <TensorFormat>format, nbDims, <int*>filterDimA)
    check_status(status)


cpdef getFilterNdDescriptor(size_t wDesc, int nbDimsRequested):
    cdef DataType dataType
    cdef TensorFormat format
    cdef int nbDims
    cdef vector.vector[int] filterDimA
    filterDimA.resize(nbDimsRequested)

    status = cudnnGetFilterNdDescriptor_v4(
        <FilterDescriptor>wDesc, nbDimsRequested, &dataType,
        &format, &nbDims, &filterDimA[0])
    check_status(status)
    return (dataType, format, nbDims, tuple(filterDimA))


cpdef destroyFilterDescriptor(size_t filterDesc):
    status = cudnnDestroyFilterDescriptor(<FilterDescriptor>filterDesc)
    check_status(status)


###############################################################################
# Convolution
###############################################################################

cpdef size_t createConvolutionDescriptor() except *:
    cdef ConvolutionDescriptor desc
    status = cudnnCreateConvolutionDescriptor(&desc)
    check_status(status)
    return <size_t>desc


cpdef setConvolution2dDescriptor_v4(
        size_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h,
        int dilation_w, int mode):
    status = cudnnSetConvolution2dDescriptor_v4(
        <ConvolutionDescriptor>convDesc, pad_h, pad_w, u, v, dilation_h,
        dilation_w, <ConvolutionMode>mode)
    check_status(status)


cpdef setConvolution2dDescriptor_v5(
        size_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h,
        int dilation_w, int mode, size_t computeType):
    status = cudnnSetConvolution2dDescriptor_v5(
        <ConvolutionDescriptor>convDesc, pad_h, pad_w, u, v, dilation_h,
        dilation_w, <ConvolutionMode>mode, <DataType>computeType)
    check_status(status)


cpdef setConvolutionNdDescriptor_v2(
        size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
        size_t dilationA, int mode):
    status = cudnnSetConvolutionNdDescriptor_v2(
        <ConvolutionDescriptor>convDesc, arrayLength, <int*>padA,
        <int*>filterStrideA, <int*>dilationA, <ConvolutionMode>mode)
    check_status(status)


cpdef setConvolutionNdDescriptor_v3(
        size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
        size_t dilationA, int mode, int dataType):
    status = cudnnSetConvolutionNdDescriptor_v3(
        <ConvolutionDescriptor>convDesc, arrayLength, <int*>padA,
        <int*>filterStrideA, <int*>dilationA, <ConvolutionMode>mode,
        <DataType>dataType)
    check_status(status)


cpdef destroyConvolutionDescriptor(size_t convDesc):
    status = cudnnDestroyConvolutionDescriptor(
        <ConvolutionDescriptor>convDesc)
    check_status(status)

cpdef findConvolutionForwardAlgorithm(
        size_t handle, size_t xDesc, size_t wDesc, size_t convDesc,
        size_t yDesc, int requestedAlgoCount):
    cdef vector.vector[ConvolutionFwdAlgoPerf] perfResults
    cdef vector.vector[int] returnedAlgoCount
    perfResults.resize(requestedAlgoCount)
    returnedAlgoCount.resize(1)
    status = cudnnFindConvolutionForwardAlgorithm(
        <Handle> handle, <TensorDescriptor>xDesc, <FilterDescriptor>wDesc,
        <ConvolutionDescriptor>convDesc, <TensorDescriptor>yDesc,
        requestedAlgoCount, &returnedAlgoCount[0], &perfResults[0])
    check_status(status)
    return returnedAlgoCount[0], perfResults

cpdef findConvolutionForwardAlgorithmEx(
        size_t handle, size_t xDesc, size_t x, size_t wDesc, size_t w,
        size_t convDesc, size_t yDesc, size_t y, int requestedAlgoCount,
        size_t workSpace, size_t workSpaceSizeInBytes):
    cdef vector.vector[ConvolutionFwdAlgoPerf] perfResults
    cdef vector.vector[int] returnedAlgoCount
    perfResults.resize(requestedAlgoCount)
    returnedAlgoCount.resize(1)
    status = cudnnFindConvolutionForwardAlgorithmEx(
        <Handle> handle, <TensorDescriptor>xDesc, <void*>x,
        <FilterDescriptor>wDesc, <void*>w, <ConvolutionDescriptor>convDesc,
        <TensorDescriptor>yDesc, <void*>y, requestedAlgoCount,
        &returnedAlgoCount[0], &perfResults[0], <void*>workSpace,
        workSpaceSizeInBytes)
    check_status(status)
    return returnedAlgoCount[0], perfResults

cpdef int getConvolutionForwardAlgorithm(
        size_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, ConvolutionFwdPreference preference,
        size_t memoryLimitInbytes) except *:
    cdef ConvolutionFwdAlgo algo
    status = cudnnGetConvolutionForwardAlgorithm(
        <Handle>handle, <TensorDescriptor>srcDesc,
        <FilterDescriptor>filterDesc, <ConvolutionDescriptor>convDesc,
        <TensorDescriptor>destDesc, <ConvolutionFwdPreference>preference,
        memoryLimitInbytes, &algo)
    check_status(status)
    return algo


cpdef size_t getConvolutionForwardWorkspaceSize(
        size_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, int algo) except *:
    cdef size_t sizeInBytes
    status = cudnnGetConvolutionForwardWorkspaceSize(
        <Handle>handle, <TensorDescriptor>srcDesc,
        <FilterDescriptor>filterDesc, <ConvolutionDescriptor> convDesc,
        <TensorDescriptor>destDesc, <ConvolutionFwdAlgo>algo, &sizeInBytes)
    check_status(status)
    return sizeInBytes


cpdef convolutionForward(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t filterDesc, size_t filterData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t destDesc, size_t destData):
    with nogil:
        status = cudnnConvolutionForward(
            <Handle>handle, <void*>alpha,
            <TensorDescriptor>srcDesc, <void*>srcData,
            <FilterDescriptor>filterDesc, <void*>filterData,
            <ConvolutionDescriptor>convDesc, <ConvolutionFwdAlgo>algo,
            <void*>workSpace, workSpaceSizeInBytes, <void*>beta,
            <TensorDescriptor>destDesc, <void*>destData)
    check_status(status)


cpdef convolutionBackwardBias(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t beta, size_t destDesc, size_t destData):
    with nogil:
        status = cudnnConvolutionBackwardBias(
            <Handle>handle, <void*>alpha,
            <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
            <TensorDescriptor>destDesc, <void*>destData)
    check_status(status)

cpdef findConvolutionBackwardFilterAlgorithm(
        size_t handle, size_t xDesc, size_t dyDesc, size_t convDesc,
        size_t dwDesc, int requestedAlgoCount):
    cdef vector.vector[ConvolutionBwdFilterAlgoPerf] perfResults
    cdef vector.vector[int] returnedAlgoCount
    perfResults.resize(requestedAlgoCount)
    returnedAlgoCount.resize(1)
    status = cudnnFindConvolutionBackwardFilterAlgorithm(
        <Handle> handle, <TensorDescriptor>xDesc, <TensorDescriptor>dyDesc,
        <ConvolutionDescriptor>convDesc, <FilterDescriptor>dwDesc,
        requestedAlgoCount, &returnedAlgoCount[0], &perfResults[0])
    check_status(status)
    return returnedAlgoCount[0], perfResults

cpdef findConvolutionBackwardFilterAlgorithmEx(
        size_t handle, size_t xDesc, size_t x, size_t dyDesc, size_t dy,
        size_t convDesc, size_t dwDesc, size_t dw, int requestedAlgoCount,
        size_t workSpace, size_t workSpaceSizeInBytes):
    cdef vector.vector[ConvolutionBwdFilterAlgoPerf] perfResults
    cdef vector.vector[int] returnedAlgoCount
    perfResults.resize(requestedAlgoCount)
    returnedAlgoCount.resize(1)
    status = cudnnFindConvolutionBackwardFilterAlgorithmEx(
        <Handle> handle, <TensorDescriptor>xDesc, <void*>x,
        <TensorDescriptor>dyDesc, <void*>dy, <ConvolutionDescriptor>convDesc,
        <FilterDescriptor>dwDesc, <void*>dw,
        requestedAlgoCount, &returnedAlgoCount[0], &perfResults[0],
        <void*>workSpace, workSpaceSizeInBytes)
    check_status(status)
    return returnedAlgoCount[0], perfResults

cpdef int getConvolutionBackwardFilterAlgorithm(
        size_t handle, size_t srcDesc, size_t diffDesc, size_t convDesc,
        size_t filterDesc, ConvolutionBwdFilterPreference preference,
        size_t memoryLimitInbytes) except *:
    cdef ConvolutionBwdFilterAlgo algo
    status = cudnnGetConvolutionBackwardFilterAlgorithm(
        <Handle>handle, <TensorDescriptor>srcDesc,
        <TensorDescriptor>diffDesc, <ConvolutionDescriptor>convDesc,
        <FilterDescriptor>filterDesc,
        <ConvolutionBwdFilterPreference>preference,
        memoryLimitInbytes, &algo)
    check_status(status)
    return algo

cpdef size_t getConvolutionBackwardFilterWorkspaceSize(
        size_t handle, size_t srcDesc, size_t diffDesc, size_t convDesc,
        size_t filterDesc, int algo) except *:
    cdef size_t sizeInBytes
    status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
        <Handle>handle, <TensorDescriptor>srcDesc,
        <TensorDescriptor>diffDesc, <ConvolutionDescriptor> convDesc,
        <FilterDescriptor>filterDesc, <ConvolutionBwdFilterAlgo>algo,
        &sizeInBytes)
    check_status(status)
    return sizeInBytes

cpdef convolutionBackwardFilter_v2(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t diffDesc, size_t diffData, size_t convDesc, size_t beta,
        size_t gradDesc, size_t gradData):
    with nogil:
        status = cudnnConvolutionBackwardFilter_v2(
            <Handle>handle, <void*>alpha,
            <TensorDescriptor>srcDesc, <void*>srcData,
            <TensorDescriptor>diffDesc, <void*>diffData,
            <ConvolutionDescriptor>convDesc, <void*>beta,
            <FilterDescriptor>gradDesc, <void*>gradData)
    check_status(status)

cpdef convolutionBackwardFilter_v3(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t diffDesc, size_t diffData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t gradDesc, size_t gradData):
    with nogil:
        status = cudnnConvolutionBackwardFilter_v3(
            <Handle>handle, <void*>alpha,
            <TensorDescriptor>srcDesc, <void*>srcData,
            <TensorDescriptor>diffDesc, <void*>diffData,
            <ConvolutionDescriptor>convDesc, <ConvolutionBwdFilterAlgo>algo,
            <void*>workSpace, workSpaceSizeInBytes, <void*>beta,
            <FilterDescriptor>gradDesc, <void*>gradData)
    check_status(status)

cpdef findConvolutionBackwardDataAlgorithm(
        size_t handle, size_t wDesc, size_t dyDesc, size_t convDesc,
        size_t dxDesc, int requestedAlgoCount):
    cdef vector.vector[ConvolutionBwdDataAlgoPerf] perfResults
    cdef vector.vector[int] returnedAlgoCount
    perfResults.resize(requestedAlgoCount)
    returnedAlgoCount.resize(1)
    status = cudnnFindConvolutionBackwardDataAlgorithm(
        <Handle> handle, <FilterDescriptor>wDesc, <TensorDescriptor>dyDesc,
        <ConvolutionDescriptor>convDesc, <TensorDescriptor>dxDesc,
        requestedAlgoCount, &returnedAlgoCount[0], &perfResults[0])
    check_status(status)
    return returnedAlgoCount[0], perfResults

cpdef findConvolutionBackwardDataAlgorithmEx(
        size_t handle, size_t wDesc, size_t w, size_t dyDesc, size_t dy,
        size_t convDesc, size_t dxDesc, size_t dx,
        int requestedAlgoCount, size_t workSpace, size_t workSpaceSizeInBytes):
    cdef vector.vector[ConvolutionBwdDataAlgoPerf] perfResults
    cdef vector.vector[int] returnedAlgoCount
    perfResults.resize(requestedAlgoCount)
    returnedAlgoCount.resize(1)
    status = cudnnFindConvolutionBackwardDataAlgorithmEx(
        <Handle> handle, <FilterDescriptor>wDesc, <void*>w,
        <TensorDescriptor>dyDesc, <void*>dy, <ConvolutionDescriptor>convDesc,
        <TensorDescriptor>dxDesc, <void*>dx,
        requestedAlgoCount, &returnedAlgoCount[0], &perfResults[0],
        <void*>workSpace, workSpaceSizeInBytes)
    check_status(status)
    return returnedAlgoCount[0], perfResults

cpdef int getConvolutionBackwardDataAlgorithm(
        size_t handle, size_t filterDesc, size_t diffDesc, size_t convDesc,
        size_t gradDesc, size_t preference,
        size_t memoryLimitInbytes) except *:
    cdef ConvolutionBwdDataAlgo algo
    status = cudnnGetConvolutionBackwardDataAlgorithm(
        <Handle>handle, <FilterDescriptor>filterDesc,
        <TensorDescriptor>diffDesc, <ConvolutionDescriptor>convDesc,
        <TensorDescriptor>gradDesc, <ConvolutionBwdDataPreference>preference,
        memoryLimitInbytes, &algo)
    check_status(status)
    return algo

cpdef size_t getConvolutionBackwardDataWorkspaceSize(
        size_t handle, size_t filterDesc, size_t diffDesc, size_t convDesc,
        size_t gradDesc, int algo) except *:
    cdef size_t sizeInBytes
    status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        <Handle>handle, <FilterDescriptor>filterDesc,
        <TensorDescriptor>diffDesc,
        <ConvolutionDescriptor>convDesc, <TensorDescriptor>gradDesc,
        <ConvolutionBwdDataAlgo>algo, &sizeInBytes)
    check_status(status)
    return sizeInBytes

cpdef convolutionBackwardData_v2(
        size_t handle, size_t alpha, size_t filterDesc, size_t filterData,
        size_t diffDesc, size_t diffData, size_t convDesc, size_t beta,
        size_t gradDesc, size_t gradData):
    with nogil:
        status = cudnnConvolutionBackwardData_v2(
            <Handle>handle, <void*>alpha,
            <FilterDescriptor>filterDesc, <void*>filterData,
            <TensorDescriptor>diffDesc, <void*>diffData,
            <ConvolutionDescriptor>convDesc, <void*>beta,
            <TensorDescriptor>gradDesc, <void*>gradData)
    check_status(status)

cpdef convolutionBackwardData_v3(
        size_t handle, size_t alpha, size_t filterDesc, size_t filterData,
        size_t diffDesc, size_t diffData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t gradDesc, size_t gradData):
    with nogil:
        status = cudnnConvolutionBackwardData_v3(
            <Handle>handle, <void*>alpha,
            <FilterDescriptor>filterDesc, <void*>filterData,
            <TensorDescriptor>diffDesc, <void*>diffData,
            <ConvolutionDescriptor>convDesc, <ConvolutionBwdDataAlgo>algo,
            <void*>workSpace, workSpaceSizeInBytes, <void*>beta,
            <TensorDescriptor>gradDesc, <void*>gradData)
    check_status(status)

###############################################################################
# Pooling
###############################################################################

cpdef size_t createPoolingDescriptor() except *:
    cdef PoolingDescriptor desc
    status = cudnnCreatePoolingDescriptor(&desc)
    check_status(status)
    return <size_t>desc


cpdef setPooling2dDescriptor_v3(
        size_t poolingDesc, int mode, int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding, int verticalStride,
        int horizontalStride):
    status = cudnnSetPooling2dDescriptor_v3(
        <PoolingDescriptor>poolingDesc, <PoolingMode>mode,
        windowHeight, windowWidth, verticalPadding, horizontalPadding,
        verticalStride, horizontalStride)
    check_status(status)


cpdef setPooling2dDescriptor_v4(
        size_t poolingDesc, int mode, int maxpoolingNanOpt, int windowHeight,
        int windowWidth, int verticalPadding, int horizontalPadding,
        int verticalStride, int horizontalStride):
    status = cudnnSetPooling2dDescriptor_v4(
        <PoolingDescriptor>poolingDesc, <PoolingMode>mode,
        <NanPropagation>maxpoolingNanOpt, windowHeight, windowWidth,
        verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    check_status(status)


cpdef setPoolingNdDescriptor_v3(
        size_t poolingDesc, int mode, int nbDims, size_t windowDimA,
        size_t paddingA, size_t strideA):
    status = cudnnSetPoolingNdDescriptor_v3(
        <PoolingDescriptor>poolingDesc, <PoolingMode>mode, nbDims,
        <int*>windowDimA, <int*>paddingA, <int*>strideA)
    check_status(status)


cpdef setPoolingNdDescriptor_v4(
        size_t poolingDesc, int mode, int maxpoolingNanOpt, int nbDims,
        size_t windowDimA, size_t paddingA, size_t strideA):
    status = cudnnSetPoolingNdDescriptor_v4(
        <PoolingDescriptor>poolingDesc, <PoolingMode>mode,
        <NanPropagation>maxpoolingNanOpt, nbDims,
        <int*>windowDimA, <int*>paddingA, <int*>strideA)
    check_status(status)


cpdef destroyPoolingDescriptor(size_t poolingDesc):
    status = cudnnDestroyPoolingDescriptor(<PoolingDescriptor>poolingDesc)
    check_status(status)


cpdef poolingForward(
        size_t handle, size_t poolingDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData):
    with nogil:
        status = cudnnPoolingForward(
            <Handle>handle, <PoolingDescriptor>poolingDesc, <void*>alpha,
            <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
            <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef poolingBackward(
        size_t handle, size_t poolingDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData,
        size_t destDesc, size_t destData, size_t beta, size_t destDiffDesc,
        size_t destDiffData):
    with nogil:
        status = cudnnPoolingBackward(
            <Handle>handle, <PoolingDescriptor>poolingDesc, <void*>alpha,
            <TensorDescriptor>srcDesc, <void*>srcData,
            <TensorDescriptor>srcDiffDesc, <void*>srcDiffData,
            <TensorDescriptor>destDesc, <void*>destData, <void*>beta,
            <TensorDescriptor>destDiffDesc, <void*>destDiffData)
    check_status(status)

###############################################################################
# Batch Normalization
###############################################################################

cpdef deriveBNTensorDescriptor(
        size_t derivedBnDesc, size_t xDesc, int mode):
    status = cudnnDeriveBNTensorDescriptor(
        <TensorDescriptor>derivedBnDesc, <TensorDescriptor>xDesc,
        <BatchNormMode> mode)
    check_status(status)

cpdef batchNormalizationForwardTraining(
        size_t handle, int mode,
        size_t alpha, size_t beta, size_t xDesc,
        size_t x, size_t yDesc, size_t y,
        size_t bnScaleBiasMeanVarDesc, size_t bnScale,
        size_t bnBias, double exponentialAverageFactor,
        size_t resultRunningMean, size_t resultRunningVariance,
        double epsilon, size_t resultSaveMean, size_t resultSaveInvVariance):
    with nogil:
        status = cudnnBatchNormalizationForwardTraining(
            <Handle>handle, <BatchNormMode> mode,
            <void*>alpha, <void*>beta, <TensorDescriptor>xDesc,
            <void*>x, <TensorDescriptor>yDesc, <void*>y,
            <TensorDescriptor>bnScaleBiasMeanVarDesc, <void*>bnScale,
            <void*>bnBias, exponentialAverageFactor,
            <void*>resultRunningMean, <void*>resultRunningVariance,
            epsilon, <void*>resultSaveMean, <void*>resultSaveInvVariance)
    check_status(status)

cpdef batchNormalizationForwardInference(
        size_t handle, int mode,
        size_t alpha, size_t beta, size_t xDesc,
        size_t x, size_t yDesc, size_t y,
        size_t bnScaleBiasMeanVarDesc, size_t bnScale,
        size_t bnBias, size_t estimatedMean, size_t estimatedVariance,
        double epsilon):
    with nogil:
        status = cudnnBatchNormalizationForwardInference(
            <Handle>handle, <BatchNormMode> mode,
            <void*>alpha, <void*>beta, <TensorDescriptor>xDesc,
            <void*>x, <TensorDescriptor>yDesc, <void*>y,
            <TensorDescriptor>bnScaleBiasMeanVarDesc, <void*>bnScale,
            <void*>bnBias, <void*>estimatedMean, <void*>estimatedVariance,
            epsilon)
    check_status(status)

cpdef batchNormalizationBackward(
        size_t handle, int mode,
        size_t alphaDataDiff, size_t betaDataDiff,
        size_t alphaParamDiff, size_t betaParamDiff,
        size_t xDesc, size_t x, size_t dyDesc,
        size_t dy, size_t dxDesc, size_t dx,
        size_t dBnScaleBiasDesc, size_t bnScale,
        size_t dBnScaleResult, size_t dBnBiasResult,
        double epsilon, size_t savedMean, size_t savedInvVariance):
    with nogil:
        status = cudnnBatchNormalizationBackward(
            <Handle>handle, <BatchNormMode>mode,
            <void*>alphaDataDiff, <void*>betaDataDiff,
            <void*>alphaParamDiff, <void*>betaParamDiff,
            <TensorDescriptor>xDesc, <void*>x,
            <TensorDescriptor>dyDesc, <void*>dy,
            <TensorDescriptor>dxDesc, <void*>dx,
            <TensorDescriptor>dBnScaleBiasDesc, <void*>bnScale,
            <void*>dBnScaleResult, <void*>dBnBiasResult,
            epsilon, <void*>savedMean, <void*>savedInvVariance)
    check_status(status)

###############################################################################
# Activation
###############################################################################

cpdef size_t createActivationDescriptor() except *:
    cdef ActivationDescriptor activationDesc
    status = cudnnCreateActivationDescriptor(&activationDesc)
    check_status(status)
    return <size_t>activationDesc

cpdef setActivationDescriptor(
        size_t activationDesc, int mode, int reluNanOpt, double reluCeiling):
    status = cudnnSetActivationDescriptor(
        <ActivationDescriptor>activationDesc, <ActivationMode>mode,
        <NanPropagation>reluNanOpt, reluCeiling)
    check_status(status)

cpdef destroyActivationDescriptor(size_t activationDesc):
    status = cudnnDestroyActivationDescriptor(
        <ActivationDescriptor>activationDesc)
    check_status(status)

cpdef softmaxForward(
        size_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData):
    with nogil:
        status = cudnnSoftmaxForward(
            <Handle>handle, <SoftmaxAlgorithm>algorithm, <SoftmaxMode>mode,
            <void*>alpha, <TensorDescriptor>srcDesc, <void*>srcData,
            <void*>beta, <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef softmaxBackward(
        size_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData, size_t beta,
        size_t destDiffDesc, size_t destDiffData):
    with nogil:
        status = cudnnSoftmaxBackward(
            <Handle>handle, <SoftmaxAlgorithm>algorithm, <SoftmaxMode>mode,
            <void*>alpha, <TensorDescriptor>srcDesc, <void*>srcData,
            <TensorDescriptor>srcDiffDesc, <void*>srcDiffData, <void*>beta,
            <TensorDescriptor>destDiffDesc, <void*>destDiffData)
    check_status(status)


cpdef activationForward_v3(
        size_t handle, int mode, size_t alpha, size_t srcDesc, size_t srcData,
        size_t beta, size_t dstDesc, size_t dstData):
    with nogil:
        status = cudnnActivationForward_v3(
            <Handle>handle, <ActivationMode>mode, <void*>alpha,
            <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
            <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef activationForward_v4(
        size_t handle, size_t activationDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData):
    with nogil:
        status = cudnnActivationForward_v4(
            <Handle>handle, <ActivationDescriptor>activationDesc, <void*>alpha,
            <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
            <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef activationBackward_v3(
        size_t handle, int mode, size_t alpha, size_t srcDesc, size_t srcData,
        size_t srcDiffDesc, size_t srcDiffData, size_t destDesc,
        size_t destData, size_t beta, size_t destDiffDesc,
        size_t destDiffData):
    with nogil:
        status = cudnnActivationBackward_v3(
            <Handle>handle, <ActivationMode>mode, <void*>alpha,
            <TensorDescriptor>srcDesc, <void*>srcData,
            <TensorDescriptor>srcDiffDesc, <void*>srcDiffData,
            <TensorDescriptor>destDesc, <void*>destData, <void*>beta,
            <TensorDescriptor>destDiffDesc, <void*>destDiffData)
    check_status(status)


cpdef activationBackward_v4(
        size_t handle, size_t activationDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData,
        size_t destDesc, size_t destData, size_t beta, size_t destDiffDesc,
        size_t destDiffData):
    with nogil:
        status = cudnnActivationBackward_v4(
            <Handle>handle, <ActivationDescriptor>activationDesc, <void*>alpha,
            <TensorDescriptor>srcDesc, <void*>srcData,
            <TensorDescriptor>srcDiffDesc, <void*>srcDiffData,
            <TensorDescriptor>destDesc, <void*>destData, <void*>beta,
            <TensorDescriptor>destDiffDesc, <void*>destDiffData)
    check_status(status)


# Dropout

cpdef size_t createDropoutDescriptor() except *:
    cdef DropoutDescriptor desc
    status = cudnnCreateDropoutDescriptor(&desc)
    check_status(status)
    return <size_t>desc


cpdef destroyDropoutDescriptor(size_t dropoutDesc):
    status = cudnnDestroyDropoutDescriptor(<DropoutDescriptor>dropoutDesc)
    check_status(status)


cpdef size_t dropoutGetStatesSize(size_t handle) except *:
    cdef size_t sizeInBytes
    status = cudnnDropoutGetStatesSize(
        <Handle>handle, &sizeInBytes)
    check_status(status)
    return sizeInBytes


cpdef setDropoutDescriptor(
        size_t dropoutDesc, size_t handle, float dropout,
        size_t states, size_t stateSizeInBytes, unsigned long long seed):
    status = cudnnSetDropoutDescriptor(
        <DropoutDescriptor>dropoutDesc, <Handle>handle, dropout,
        <void*>states, stateSizeInBytes, seed)
    check_status(status)


# RNN

cpdef size_t createRNNDescriptor() except *:
    cdef RNNDescriptor desc
    status = cudnnCreateRNNDescriptor(&desc)
    check_status(status)
    return <size_t>desc


cpdef destroyRNNDescriptor(size_t rnnDesc):
    status = cudnnDestroyRNNDescriptor(<RNNDescriptor>rnnDesc)
    check_status(status)


cpdef setRNNDescriptor(
        size_t rnnDesc, int hiddenSize, int numLayers,
        size_t dropoutDesc, int inputMode, int direction, int mode,
        int dataType):
    status = cudnnSetRNNDescriptor(
        <RNNDescriptor>rnnDesc, hiddenSize, numLayers,
        <DropoutDescriptor>dropoutDesc, <RNNInputMode>inputMode,
        <DirectionMode>direction, <RNNMode>mode, <DataType>dataType)
    check_status(status)


cpdef getRNNWorkspaceSize(
        size_t handle, size_t rnnDesc, int seqLength, size_t xDesc):
    cdef size_t sizeInBytes
    status = cudnnGetRNNWorkspaceSize(
        <Handle>handle, <RNNDescriptor>rnnDesc, seqLength,
        <TensorDescriptor*>xDesc, &sizeInBytes)
    check_status(status)
    return sizeInBytes


cpdef getRNNTrainingReserveSize(
        size_t handle, size_t rnnDesc, int seqLength, size_t xDesc):
    cdef size_t sizeInBytes
    status = cudnnGetRNNTrainingReserveSize(
        <Handle>handle, <RNNDescriptor>rnnDesc, seqLength,
        <TensorDescriptor*>xDesc, &sizeInBytes)
    check_status(status)
    return sizeInBytes


cpdef getRNNParamsSize(
        size_t handle, size_t rnnDesc, size_t xDesc, int dataType):
    cdef size_t sizeInBytes
    status = cudnnGetRNNParamsSize(
        <Handle>handle, <RNNDescriptor>rnnDesc, <TensorDescriptor>xDesc,
        &sizeInBytes, <DataType>dataType)
    check_status(status)
    return sizeInBytes


cpdef getRNNLinLayerMatrixParams(
        size_t handle, size_t rnnDesc, int layer, size_t xDesc, size_t wDesc,
        size_t w, int linLayerID, size_t linLayerMatDesc, size_t linLayerMat):
    status = cudnnGetRNNLinLayerMatrixParams(
        <Handle>handle, <RNNDescriptor>rnnDesc, layer,
        <TensorDescriptor>xDesc, <FilterDescriptor>wDesc, <void*>w,
        linLayerID, <FilterDescriptor>linLayerMatDesc, <void**>linLayerMat)
    check_status(status)


cpdef getRNNLinLayerBiasParams(
        size_t handle, size_t rnnDesc, int layer, size_t xDesc, size_t wDesc,
        size_t w, int linLayerID, size_t linLayerBiasDesc,
        size_t linLayerBias):
    status = cudnnGetRNNLinLayerBiasParams(
        <Handle>handle, <RNNDescriptor>rnnDesc, layer,
        <TensorDescriptor>xDesc, <FilterDescriptor>wDesc, <void*>w,
        linLayerID, <FilterDescriptor>linLayerBiasDesc, <void**>linLayerBias)
    check_status(status)


cpdef RNNForwardInference(
        size_t handle, size_t rnnDesc, int seqLength, size_t xDesc,
        size_t x, size_t hxDesc, size_t hx, size_t cxDesc,
        size_t cx, size_t wDesc, size_t w, size_t yDesc,
        size_t y, size_t hyDesc, size_t hy, size_t cyDesc,
        size_t cy, size_t workspace, size_t workSpaceSizeInBytes):
    with nogil:
        status = cudnnRNNForwardInference(
            <Handle>handle, <RNNDescriptor>rnnDesc, seqLength,
            <TensorDescriptor*>xDesc, <void*>x,
            <TensorDescriptor>hxDesc, <void*>hx,
            <TensorDescriptor>cxDesc, <void*>cx,
            <FilterDescriptor>wDesc, <void*>w,
            <TensorDescriptor*>yDesc, <void*>y,
            <TensorDescriptor>hyDesc, <void*>hy,
            <TensorDescriptor>cyDesc, <void*>cy,
            <void*>workspace, workSpaceSizeInBytes)
    check_status(status)


cpdef RNNForwardTraining(
        size_t handle, size_t rnnDesc, int seqLength, size_t xDesc, size_t x,
        size_t hxDesc, size_t hx, size_t cxDesc, size_t cx,
        size_t wDesc, size_t w, size_t yDesc, size_t y,
        size_t hyDesc, size_t hy, size_t cyDesc, size_t cy,
        size_t workspace, size_t workSpaceSizeInBytes, size_t reserveSpace,
        size_t reserveSpaceSizeInBytes):
    with nogil:
        status = cudnnRNNForwardTraining(
            <Handle>handle, <RNNDescriptor>rnnDesc, seqLength,
            <TensorDescriptor*>xDesc, <void*>x,
            <TensorDescriptor>hxDesc, <void*>hx,
            <TensorDescriptor>cxDesc, <void*>cx,
            <FilterDescriptor>wDesc, <void*>w,
            <TensorDescriptor*>yDesc, <void*>y,
            <TensorDescriptor>hyDesc, <void*> hy,
            <TensorDescriptor>cyDesc, <void*>cy,
            <void*>workspace, workSpaceSizeInBytes,
            <void*>reserveSpace, reserveSpaceSizeInBytes)
    check_status(status)


cpdef RNNBackwardData(
        size_t handle, size_t rnnDesc, int seqLength, size_t yDesc, size_t y,
        size_t dyDesc, size_t dy, size_t dhyDesc, size_t dhy,
        size_t dcyDesc, size_t dcy, size_t wDesc, size_t w,
        size_t hxDesc, size_t hx, size_t cxDesc, size_t cx,
        size_t dxDesc, size_t dx, size_t dhxDesc, size_t dhx,
        size_t dcxDesc, size_t dcx, size_t workspace,
        size_t workSpaceSizeInBytes, size_t reserveSpace,
        size_t reserveSpaceSizeInBytes):
    with nogil:
        status = cudnnRNNBackwardData(
            <Handle>handle, <RNNDescriptor>rnnDesc, seqLength,
            <TensorDescriptor*>yDesc, <void*>y,
            <TensorDescriptor*>dyDesc, <void*>dy,
            <TensorDescriptor>dhyDesc, <void*>dhy,
            <TensorDescriptor>dcyDesc, <void*>dcy,
            <FilterDescriptor>wDesc, <void*>w,
            <TensorDescriptor>hxDesc, <void*>hx,
            <TensorDescriptor>cxDesc, <void*>cx,
            <TensorDescriptor*>dxDesc, <void*>dx,
            <TensorDescriptor>dhxDesc, <void*>dhx,
            <TensorDescriptor>dcxDesc, <void*>dcx,
            <void*>workspace, workSpaceSizeInBytes,
            <void*>reserveSpace, reserveSpaceSizeInBytes)
    check_status(status)


cpdef RNNBackwardWeights(
        size_t handle, size_t rnnDesc, int seqLength, size_t xDesc, size_t x,
        size_t hxDesc, size_t hx, size_t yDesc, size_t y,
        size_t workspace, size_t workSpaceSizeInBytes, size_t dwDesc,
        size_t dw, size_t reserveSpace, size_t reserveSpaceSizeInBytes):
    with nogil:
        status = cudnnRNNBackwardWeights(
            <Handle>handle, <RNNDescriptor>rnnDesc, seqLength,
            <TensorDescriptor*>xDesc, <void*>x,
            <TensorDescriptor>hxDesc, <void*>hx,
            <TensorDescriptor*>yDesc, <void*>y,
            <void*>workspace, workSpaceSizeInBytes,
            <FilterDescriptor>dwDesc, <void*>dw,
            <void*>reserveSpace, reserveSpaceSizeInBytes)
    check_status(status)
