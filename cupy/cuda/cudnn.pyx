"""Thin wrapper of cuDNN."""
# NOTE: This wrapper does not cover all APIs of cuDNN v4.
cimport cython


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
    int cudnnSetStream(Handle handle, Stream stream)
    int cudnnGetStream(Handle handle, Stream* stream)

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
            Handle handle, FilterDescriptor filterDesc, TensorDescriptor diffDesc,
            ConvolutionDescriptor convDesc, TensorDescriptor gradDesc,
            ConvolutionBwdDataPreference preference,
            size_t memoryLimitInbytes, ConvolutionBwdDataAlgo* algo)
    int cudnnGetConvolutionBackwardDataWorkspaceSize(
            Handle handle, FilterDescriptor filterDesc, TensorDescriptor diffDesc,
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
    status = cudnnSetStream(<Handle>handle, <Stream>stream)
    check_status(status)


cpdef size_t getStream(size_t handle) except *:
    cdef Stream stream
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
          <Handle>handle, <FilterDescriptor>filterDesc, <TensorDescriptor>diffDesc,
          <ConvolutionDescriptor> convDesc, <TensorDescriptor>gradDesc,
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
