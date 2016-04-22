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
    const char* cudnnGetErrorString(Status status)

    # Version
    size_t cudnnGetVersion()

    # Initialization and CUDA cooperation
    int cudnnCreate(Handle* handle)
    int cudnnDestroy(Handle handle)
    int cudnnSetStream(Handle handle, driver.Stream stream)
    int cudnnGetStream(Handle handle, driver.Stream* stream)

    # Tensor manipulation
    int cudnnCreateTensorDescriptor(TensorDescriptor* descriptor)
    int cudnnSetTensor4dDescriptor(
        TensorDescriptor tensorDesc, TensorFormat format,
        DataType dataType, int n, int c, int h, int w)
    int cudnnSetTensor4dDescriptorEx(
        TensorDescriptor tensorDesc, DataType dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride)
    int cudnnSetTensorNdDescriptor(
        TensorDescriptor tensorDesc, DataType dataType, int nbDims,
        int* dimA, int* strideA)
    int cudnnDestroyTensorDescriptor(TensorDescriptor tensorDesc)
    int cudnnAddTensor_v2(
        Handle handle, AddMode mode, void* alpha,
        TensorDescriptor biasDesc, void* biasData, void* beta,
        TensorDescriptor srcDestDesc, void* srcDestData)
    int cudnnAddTensor_v3(
        Handle handle, void* alpha, TensorDescriptor bDesc,
        void* b, void* beta, TensorDescriptor yDesc, void* y)

    # Filter manipulation
    int cudnnCreateFilterDescriptor(FilterDescriptor* filterDesc)
    int cudnnSetFilter4dDescriptor_v3(
        FilterDescriptor filterDesc, DataType dataType,
        int n, int c, int h, int w)
    int cudnnSetFilterNdDescriptor_v3(
        FilterDescriptor filterDesc, DataType dataType, int nbDims,
        int* filterDimA)
    int cudnnGetFilterNdDescriptor(
        FilterDescriptor wDesc, int nbDimsRequested, DataType* dataType,
        TensorFormat* format, int* nbDims, int filterDimA[])
    int cudnnDestroyFilterDescriptor(FilterDescriptor filterDesc)

    # Convolution
    int cudnnCreateConvolutionDescriptor(ConvolutionDescriptor* convDesc)
    int cudnnSetConvolution2dDescriptor(
        ConvolutionDescriptor convDesc, int pad_h, int pad_w, int u,
        int v, int upscalex, int upscaley, ConvolutionMode mode)
    int cudnnSetConvolutionNdDescriptor_v2(
        ConvolutionDescriptor convDesc, int arrayLength, int* padA,
        int* filterStrideA, int* upscaleA, ConvolutionMode mode)
    int cudnnSetConvolutionNdDescriptor_v3(
        ConvolutionDescriptor convDesc, int arrayLength, int* padA,
        int* filterStrideA, int* upscaleA, ConvolutionMode mode,
        DataType dataType)
    int cudnnDestroyConvolutionDescriptor(ConvolutionDescriptor conDesc)
    int cudnnGetConvolutionForwardAlgorithm(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionFwdPreference preference,
        size_t memoryLimitInbytes, ConvolutionFwdAlgo* algo)
    int cudnnGetConvolutionForwardWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionFwdAlgo algo,
        size_t* sizeInBytes)
    int cudnnConvolutionForward(
        Handle handle, void* alpha, TensorDescriptor srcDesc,
        void* srcData, FilterDescriptor filterDesc, void* filterData,
        ConvolutionDescriptor convDesc, ConvolutionFwdAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor destDesc, void* destData)
    int cudnnConvolutionBackwardBias(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor destDesc, void* destData)
    int cudnnGetConvolutionBackwardFilterAlgorithm(
        Handle handle, TensorDescriptor srcDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor filterDesc,
        ConvolutionBwdFilterPreference preference,
        size_t memoryLimitInbytes, ConvolutionBwdFilterAlgo* algo)
    int cudnnGetConvolutionBackwardFilterWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor filterDesc,
        ConvolutionBwdFilterAlgo algo, size_t* sizeInBytes)
    int cudnnConvolutionBackwardFilter_v2(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, void* beta,
        FilterDescriptor gradDesc, void* gradData)
    int cudnnConvolutionBackwardFilter_v3(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, ConvolutionBwdFilterAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        FilterDescriptor gradDesc, void* gradData)
    int cudnnGetConvolutionBackwardDataAlgorithm(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor gradDesc,
        ConvolutionBwdDataPreference preference,
        size_t memoryLimitInbytes, ConvolutionBwdDataAlgo* algo)
    int cudnnGetConvolutionBackwardDataWorkspaceSize(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor gradDesc,
        ConvolutionBwdDataAlgo algo, size_t* sizeInBytes)
    int cudnnConvolutionBackwardData_v2(
        Handle handle, void* alpha,
        FilterDescriptor filterDesc, void* filterData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, void* beta,
        TensorDescriptor gradDesc, void* gradData)
    int cudnnConvolutionBackwardData_v3(
        Handle handle, void* alpha,
        FilterDescriptor filterDesc, void* filterData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, ConvolutionBwdDataAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor gradDesc, void* gradData)

    # Pooling
    int cudnnCreatePoolingDescriptor(PoolingDescriptor* desc)
    int cudnnSetPooling2dDescriptor_v3(
        PoolingDescriptor poolingDesc, PoolingMode mode,
        int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding,
        int verticalStride, int horizontalStride)
    int cudnnSetPoolingNdDescriptor_v3(
        PoolingDescriptor poolingDesc, PoolingMode mode, int nbDims,
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

    # Batch Normalization
    int cudnnDeriveBNTensorDescriptor(
        TensorDescriptor derivedBnDesc, TensorDescriptor xDesc,
        BatchNormMode mode)

    int cudnnBatchNormalizationForwardTraining(
        Handle handle, BatchNormMode mode,
        void* alpha, void* beta, TensorDescriptor xDesc,
        void* x, TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
        void* bnBias, double exponentialAverageFactor,
        void* resultRunningMean, void* resultRunningVariance,
        double epsilon, void* resultSaveMean, void* resultSaveInvVariance)

    int cudnnBatchNormalizationForwardInference(
        Handle handle, BatchNormMode mode,
        void* alpha, void* beta, TensorDescriptor xDesc,
        void* x, TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
        void* bnBias, void* estimatedMean, void* estimatedVariance,
        double epsilon)

    int cudnnBatchNormalizationBackward(
        Handle handle, BatchNormMode mode,
        void* alphaDataDiff, void* betaDataDiff,
        void* alphaParamDiff, void* betaParamDiff,
        TensorDescriptor xDesc, void* x,
        TensorDescriptor dyDesc, void* dy,
        TensorDescriptor dxDesc, void* dx,
        TensorDescriptor dBnScaleBiasDesc, void* bnScale,
        void* dBnScaleResult, void* dBnBiasResult,
        double epsilon, void* savedMean, void* savedInvVariance)

    # Activation
    int cudnnSoftmaxForward(
        Handle handle, SoftmaxAlgorithm algorithm, SoftmaxMode mode,
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        void* beta, TensorDescriptor dstDesc, void* dstData)
    int cudnnSoftmaxBackward(
        Handle handle, SoftmaxAlgorithm algorithm, SoftmaxMode mode,
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)
    int cudnnActivationForward_v3(
        Handle handle, ActivationMode mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData)
    int cudnnActivationBackward_v3(
        Handle handle, ActivationMode mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)

    # Dropout
    int cudnnCreateDropoutDescriptor(DropoutDescriptor* desc)
    int cudnnDestroyDropoutDescriptor(DropoutDescriptor dropoutDesc)
    int cudnnDropoutGetStatesSize(Handle handle, size_t* sizeInBytes)
    int cudnnDropoutGetReserveSpaceSize(
        TensorDescriptor xDesc, size_t* sizeInBytes)
    int cudnnSetDropoutDescriptor(
        DropoutDescriptor dropoutDesc, Handle handle, float dropout,
        void* states, size_t stateSizeInBytes, unsigned long long seed)
    int cudnnDropoutBackward(
        Handle handle, DropoutDescriptor dropoutDesc,
        TensorDescriptor dydesc, void* dy, TensorDescriptor dxdesc,
        void* dx, void* reserveSpace, size_t reserveSpaceSizeInBytes)

    # RNN
    int cudnnCreateRNNDescriptor(RNNDescriptor* rnnDesc)
    int cudnnDestroyRNNDescriptor(RNNDescriptor rnnDesc)
    int cudnnSetRNNDescriptor(
        RNNDescriptor rnnDesc, int hiddenSize, int seqLength,
        int numLayers, DropoutDescriptor dropoutDesc, RNNInputMode inputMode,
        DirectionMode direction, RNNMode mode, DataType dataType)
    int cudnnGetRNNWorkspaceSize(
        Handle handle, RNNDescriptor rnnDesc, TensorDescriptor* xDesc,
        size_t* sizeInBytes)
    int cudnnGetRNNParamsSize(
        Handle handle, RNNDescriptor rnnDesc, TensorDescriptor* xDesc,
        size_t* sizeInBytes)
    int cudnnGetRNNLinLayerMatrixParams(
        Handle handle, RNNDescriptor rnnDesc, int layer,
        TensorDescriptor* xDesc, FilterDescriptor wDesc, void* w,
        int linLayerID, FilterDescriptor linLayerMatDesc, void** linLayerMat)
    int cudnnGetRNNLinLayerBiasParams(
        Handle handle, RNNDescriptor rnnDesc, int layer,
        TensorDescriptor* xDesc, FilterDescriptor wDesc, void* w,
        int linLayerID, FilterDescriptor linLayerBiasDesc, void** linLayerBias)
    int cudnnRNNForwardInference(
        Handle handle, RNNDescriptor rnnDesc, TensorDescriptor* xDesc,
        void* x, TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc,
        void* cx, FilterDescriptor wDesc, void* w, TensorDescriptor* yDesc,
        void* y, TensorDescriptor hyDesc, void* hy, TensorDescriptor cyDesc,
        void* cy, void* workspace, size_t workSpaceSizeInBytes)
    int cudnnRNNForwardTraining(
        Handle handle, RNNDescriptor rnnDesc, TensorDescriptor* xDesc, void* x,
        TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc, void* cx,
        FilterDescriptor wDesc, void* w, TensorDescriptor *yDesc, void* y,
        TensorDescriptor hyDesc, void* hy, TensorDescriptor cyDesc, void* cy,
        void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace,
        size_t reserveSpaceSizeInBytes)
    int cudnnRNNBackwardData(
        Handle handle, RNNDescriptor rnnDesc, TensorDescriptor* yDesc, void* y,
        TensorDescriptor* dyDesc, void* dy, TensorDescriptor dhyDesc, void* dhy,
        TensorDescriptor dcyDesc, void* dcy,FilterDescriptor wDesc, void* w,
        TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc, void* cx,
        TensorDescriptor* dxDesc, void* dx, TensorDescriptor dhxDesc, void* dhx,
        TensorDescriptor dcxDesc, void* dcx, void* workspace,
        size_t workSpaceSizeInBytes, void* reserveSpace,
        size_t reserveSpaceSizeInBytes)
    int cudnnRNNBackwardWeights(
        Handle handle, RNNDescriptor rnnDesc, TensorDescriptor* xDesc, void* x,
        TensorDescriptor hxDesc, void* hx, TensorDescriptor * yDesc, void* y,
        void* workspace, size_t workSpaceSizeInBytes, FilterDescriptor dwDesc,
        void* dw, void* reserveSpace, size_t reserveSpaceSizeInBytes)

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

cpdef size_t getVersion():
    return cudnnGetVersion()


###############################################################################
# Initialization and CUDA cooperation
###############################################################################

cpdef size_t create() except *:
    cdef Handle handle
    status = cudnnCreate(&handle)
    check_status(status)
    return <size_t>handle


cpdef destroy(size_t handle):
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
        <TensorDescriptor>tensorDesc, <TensorFormat>format, <DataType>dataType,
        n, c, h, w)
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
        <TensorDescriptor>tensorDesc, <DataType>dataType, nbDims, <int*>dimA,
        <int*>strideA)
    check_status(status)


cpdef destroyTensorDescriptor(size_t tensorDesc):
    status = cudnnDestroyTensorDescriptor(<TensorDescriptor>tensorDesc)
    check_status(status)


cpdef addTensor_v2(
        size_t handle, int mode, size_t alpha, size_t biasDesc,
        size_t biasData, size_t beta, size_t srcDestDesc, size_t srcDestData):
    status = cudnnAddTensor_v2(
        <Handle>handle, <AddMode>mode, <void*>alpha,
        <TensorDescriptor>biasDesc, <void*>biasData, <void*>beta,
        <TensorDescriptor>srcDestDesc, <void*>srcDestData)
    check_status(status)


cpdef addTensor_v3(size_t handle, size_t alpha, size_t bDesc,
                   size_t b, size_t beta, size_t yDesc, size_t y):
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


cpdef getFilterNdDescriptor(size_t wDesc, int nbDimsRequested):
    cdef DataType dataType
    cdef TensorFormat format
    cdef int nbDims
    cdef vector.vector[int] filterDimA
    filterDimA.resize(nbDimsRequested)

    status = cudnnGetFilterNdDescriptor(
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


cpdef setConvolution2dDescriptor(
        size_t convDesc, int pad_h, int pad_w, int u, int v, int upscalex,
        int upscaley, int mode):
    status = cudnnSetConvolution2dDescriptor(
        <ConvolutionDescriptor>convDesc, pad_h, pad_w, u, v, upscalex,
        upscaley, <ConvolutionMode>mode)
    check_status(status)


cpdef setConvolutionNdDescriptor_v2(
        size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
        size_t upscaleA, int mode):
    status = cudnnSetConvolutionNdDescriptor_v2(
        <ConvolutionDescriptor>convDesc, arrayLength, <int*>padA,
        <int*>filterStrideA, <int*>upscaleA, <ConvolutionMode>mode)
    check_status(status)


cpdef setConvolutionNdDescriptor_v3(
        size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
        size_t upscaleA, int mode, int dataType):
    status = cudnnSetConvolutionNdDescriptor_v3(
        <ConvolutionDescriptor>convDesc, arrayLength, <int*>padA,
        <int*>filterStrideA, <int*>upscaleA, <ConvolutionMode>mode,
        <DataType>dataType)
    check_status(status)


cpdef destroyConvolutionDescriptor(size_t convDesc):
    status = cudnnDestroyConvolutionDescriptor(<ConvolutionDescriptor>convDesc)
    check_status(status)


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
    status = cudnnConvolutionBackwardBias(
        <Handle>handle, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
        <TensorDescriptor>destDesc, <void*>destData)
    check_status(status)

cpdef int getConvolutionBackwardFilterAlgorithm(
        size_t handle, size_t srcDesc, size_t diffDesc, size_t convDesc,
        size_t filterDesc, ConvolutionBwdFilterPreference preference,
        size_t memoryLimitInbytes) except *:
    cdef ConvolutionBwdFilterAlgo algo
    status = cudnnGetConvolutionBackwardFilterAlgorithm(
        <Handle>handle, <TensorDescriptor>srcDesc, <TensorDescriptor>diffDesc,
        <ConvolutionDescriptor>convDesc, <FilterDescriptor>filterDesc,
        <ConvolutionBwdFilterPreference>preference,
        memoryLimitInbytes, &algo)
    check_status(status)
    return algo

cpdef size_t getConvolutionBackwardFilterWorkspaceSize(
        size_t handle, size_t srcDesc, size_t diffDesc, size_t convDesc,
        size_t filterDesc, int algo) except *:
    cdef size_t sizeInBytes
    status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
        <Handle>handle, <TensorDescriptor>srcDesc, <TensorDescriptor>diffDesc,
        <ConvolutionDescriptor> convDesc, <FilterDescriptor>filterDesc,
        <ConvolutionBwdFilterAlgo>algo, &sizeInBytes)
    check_status(status)
    return sizeInBytes

cpdef convolutionBackwardFilter_v2(
        size_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t diffDesc, size_t diffData, size_t convDesc, size_t beta,
        size_t gradDesc, size_t gradData):
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
    status = cudnnConvolutionBackwardFilter_v3(
        <Handle>handle, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData,
        <TensorDescriptor>diffDesc, <void*>diffData,
        <ConvolutionDescriptor>convDesc, <ConvolutionBwdFilterAlgo>algo,
        <void*>workSpace, workSpaceSizeInBytes, <void*>beta,
        <FilterDescriptor>gradDesc, <void*>gradData)
    check_status(status)

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


cpdef setPoolingNdDescriptor_v3(
        size_t poolingDesc, int mode, int nbDims, size_t windowDimA,
        size_t paddingA, size_t strideA):
    status = cudnnSetPoolingNdDescriptor_v3(
        <PoolingDescriptor>poolingDesc, <PoolingMode>mode, nbDims,
        <int*>windowDimA, <int*>paddingA, <int*>strideA)
    check_status(status)


cpdef destroyPoolingDescriptor(size_t poolingDesc):
    status = cudnnDestroyPoolingDescriptor(<PoolingDescriptor>poolingDesc)
    check_status(status)


cpdef poolingForward(
        size_t handle, size_t poolingDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData):
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

cpdef softmaxForward(
        size_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData):
    status = cudnnSoftmaxForward(
        <Handle>handle, <SoftmaxAlgorithm>algorithm, <SoftmaxMode>mode,
        <void*>alpha, <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
        <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef softmaxBackward(
        size_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData, size_t beta,
        size_t destDiffDesc, size_t destDiffData):
    status = cudnnSoftmaxBackward(
        <Handle>handle, <SoftmaxAlgorithm>algorithm, <SoftmaxMode>mode,
        <void*>alpha, <TensorDescriptor>srcDesc, <void*>srcData,
        <TensorDescriptor>srcDiffDesc, <void*>srcDiffData, <void*>beta,
        <TensorDescriptor>destDiffDesc, <void*>destDiffData)
    check_status(status)


cpdef activationForward_v3(
        size_t handle, int mode, size_t alpha, size_t srcDesc, size_t srcData,
        size_t beta, size_t dstDesc, size_t dstData):
    status = cudnnActivationForward_v3(
        <Handle>handle, <ActivationMode>mode, <void*>alpha,
        <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
        <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef activationBackward_v3(
        size_t handle, int mode, size_t alpha, size_t srcDesc, size_t srcData,
        size_t srcDiffDesc, size_t srcDiffData, size_t destDesc,
        size_t destData, size_t beta, size_t destDiffDesc,
        size_t destDiffData):
    status = cudnnActivationBackward_v3(
        <Handle>handle, <ActivationMode>mode, <void*>alpha,
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


cpdef size_t dropoutGetStatesSize(size_t handle):
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
    size_t rnnDesc, int hiddenSize, int seqLength, int numLayers,
    size_t dropoutDesc, int inputMode, int direction, int mode, int dataType):
    status = cudnnSetRNNDescriptor(
        <RNNDescriptor>rnnDesc, hiddenSize, seqLength, numLayers,
        <DropoutDescriptor>dropoutDesc, <RNNInputMode>inputMode,
        <DirectionMode>direction, <RNNMode>mode, <DataType>dataType)
    check_status(status)


cpdef getRNNWorkspaceSize(
    size_t handle, size_t rnnDesc, size_t xDesc):
    cdef size_t sizeInBytes
    status = cudnnGetRNNWorkspaceSize(
        <Handle>handle, <RNNDescriptor>rnnDesc, <TensorDescriptor*>xDesc,
        &sizeInBytes)
    check_status(status)
    return sizeInBytes


cpdef getRNNParamsSize(size_t handle, size_t rnnDesc, size_t xDesc):
    cdef size_t sizeInBytes
    status = cudnnGetRNNParamsSize(
        <Handle>handle, <RNNDescriptor>rnnDesc, <TensorDescriptor*>xDesc,
        &sizeInBytes)
    check_status(status)
    return sizeInBytes


cpdef getRNNLinLayerMatrixParams(
    size_t handle, size_t rnnDesc, int layer, size_t xDesc, size_t wDesc,
    size_t w, int linLayerID, size_t linLayerMatDesc, size_t linLayerMat):
    status = cudnnGetRNNLinLayerMatrixParams(
        <Handle>handle, <RNNDescriptor>rnnDesc, layer,
        <TensorDescriptor*>xDesc, <FilterDescriptor>wDesc, <void*>w,
        linLayerID, <FilterDescriptor>linLayerMatDesc, <void**>linLayerMat)
    check_status(status)

cpdef getRNNLinLayerBiasParams(
    size_t handle, size_t rnnDesc, int layer, size_t xDesc, size_t wDesc,
    size_t w, int linLayerID, size_t linLayerBiasDesc, size_t linLayerBias):
    status = cudnnGetRNNLinLayerBiasParams(
        <Handle>handle, <RNNDescriptor>rnnDesc, layer,
        <TensorDescriptor*>xDesc, <FilterDescriptor>wDesc, <void*>w,
        linLayerID, <FilterDescriptor>linLayerBiasDesc, <void**>linLayerBias)
    check_status(status)

cpdef RNNForwardInference(
    size_t handle, size_t rnnDesc, size_t xDesc,
    size_t x, size_t hxDesc, size_t hx, size_t cxDesc,
    size_t cx, size_t wDesc, size_t w, size_t yDesc,
    size_t y, size_t hyDesc, size_t hy, size_t cyDesc,
    size_t cy, size_t workspace, size_t workSpaceSizeInBytes):
    status = cudnnRNNForwardInference(
        <Handle>handle, <RNNDescriptor>rnnDesc, <TensorDescriptor*>xDesc,
        <void*>x, <TensorDescriptor>hxDesc, <void*>hx, <TensorDescriptor>cxDesc,
        <void*>cx, <FilterDescriptor>wDesc, <void*>w, <TensorDescriptor*>yDesc,
        <void*>y, <TensorDescriptor>hyDesc, <void*>hy, <TensorDescriptor>cyDesc,
        <void*>cy, <void*>workspace, workSpaceSizeInBytes)
