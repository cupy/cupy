// This file is a stub header file of cudnn for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDNN_H
#define INCLUDE_GUARD_CUPY_CUDNN_H

#include "cupy_cuda.h"

#ifndef CUPY_NO_CUDA

#include <cudnn.h>

#else // #ifndef CUPY_NO_CUDA

extern "C" {

#define CUDNN_VERSION 2000

typedef enum {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
} cudnnStatus_t;

typedef enum {} cudnnActivationMode_t;
typedef enum {} cudnnConvolutionFwdAlgo_t;
typedef enum {} cudnnConvolutionFwdPreference_t;
typedef enum {} cudnnConvolutionMode_t;
typedef enum {} cudnnDataType_t;
typedef enum {} cudnnNanPropagation_t;
typedef enum {} cudnnPoolingMode_t;
typedef enum {} cudnnSoftmaxAlgorithm_t;
typedef enum {} cudnnSoftmaxMode_t;
typedef enum {} cudnnTensorFormat_t;


typedef void* cudnnConvolutionDescriptor_t;
typedef void* cudnnFilterDescriptor_t;
typedef void* cudnnHandle_t;
typedef void* cudnnPoolingDescriptor_t;
typedef void* cudnnTensorDescriptor_t;


// Error handling
const char* cudnnGetErrorString(cudnnStatus_t status) {
    return NULL;
}


// Version
size_t cudnnGetVersion() {
    return CUDNN_STATUS_SUCCESS;
}


// Initialization and CUDA cooperation
cudnnStatus_t cudnnCreate(cudnnHandle_t* handle) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t* stream) {
    return CUDNN_STATUS_SUCCESS;
}


// Tensor manipulation
cudnnStatus_t cudnnCreateTensorDescriptor(
        cudnnTensorDescriptor_t* descriptor) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(
        cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
        cudnnDataType_t dataType, int n, int c, int h, int w) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx(
        cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(
        cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
        int nbDims, int* dimA, int* strideA) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(
        cudnnTensorDescriptor_t tensorDesc) {
    return CUDNN_STATUS_SUCCESS;
}


// Filter manipulation
cudnnStatus_t cudnnCreateFilterDescriptor(
        cudnnFilterDescriptor_t* filterDesc) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(
        cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
        int k, int c, int h, int w) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilterNdDescriptor(
        cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
        int nbDims, const int filterDimA[]) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(
        cudnnFilterDescriptor_t filterDesc) {
    return CUDNN_STATUS_SUCCESS;
}


// Convolution
cudnnStatus_t cudnnCreateConvolutionDescriptor(
        cudnnConvolutionDescriptor_t* convDesc) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(
        cudnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u,
        int v, int upscalex, int upscaley, cudnnConvolutionMode_t mode) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(
        cudnnConvolutionDescriptor_t conDesc) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle_t handle, cudnnTensorDescriptor_t srcDesc,
        cudnnFilterDescriptor_t filterDesc,
        cudnnConvolutionDescriptor_t convDesc,
        cudnnTensorDescriptor_t destDesc,
        cudnnConvolutionFwdPreference_t preference,
        size_t memoryLimitInbytes, cudnnConvolutionFwdAlgo_t* algo) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle_t handle, cudnnTensorDescriptor_t srcDesc,
        cudnnFilterDescriptor_t filterDesc,
        cudnnConvolutionDescriptor_t convDesc,
        cudnnTensorDescriptor_t destDesc, cudnnConvolutionFwdAlgo_t algo,
        size_t* sizeInBytes) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward(
        cudnnHandle_t handle, void* alpha, cudnnTensorDescriptor_t srcDesc,
        void* srcData, cudnnFilterDescriptor_t filterDesc, void* filterData,
        cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        cudnnTensorDescriptor_t destDesc, void* destData) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardBias(
        cudnnHandle_t handle, void* alpha,
        cudnnTensorDescriptor_t srcDesc, void* srcData, void* beta,
        cudnnTensorDescriptor_t destDesc, void* destData) {
    return CUDNN_STATUS_SUCCESS;
}

// Pooling
cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* desc) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(
        cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
        int windowHeight, int windowWidth, int verticalPadding,
        int horizontalPadding, int verticalStride, int horizontalStride) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor(
        cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode,
        int nbDims, const int windowDimA[], const int paddingA[],
        const int strideA[]) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(
        cudnnPoolingDescriptor_t poolingDesc) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingForward(
        cudnnHandle_t handle, cudnnPoolingDescriptor_t poolingDesc,
        void* alpha, cudnnTensorDescriptor_t srcDesc, void* srcData,
        void* beta, cudnnTensorDescriptor_t dstDesc, void* dstData) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingBackward(
        cudnnHandle_t handle, cudnnPoolingDescriptor_t poolingDesc,
        void* alpha, cudnnTensorDescriptor_t srcDesc, void* srcData,
        cudnnTensorDescriptor_t srcDiffDesc, void* srcDiffData,
        cudnnTensorDescriptor_t destDesc, void* destData, void* beta,
        cudnnTensorDescriptor_t destDiffDesc, void* destDiffData) {
    return CUDNN_STATUS_SUCCESS;
}


// Activation
cudnnStatus_t cudnnSoftmaxForward(
        cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algorithm,
        cudnnSoftmaxMode_t mode,
        void* alpha, cudnnTensorDescriptor_t srcDesc, void* srcData,
        void* beta, cudnnTensorDescriptor_t dstDesc, void* dstData) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxBackward(
        cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algorithm,
        cudnnSoftmaxMode_t mode,
        void* alpha, cudnnTensorDescriptor_t srcDesc, void* srcData,
        cudnnTensorDescriptor_t srcDiffDesc, void* srcDiffData, void* beta,
        cudnnTensorDescriptor_t destDiffDesc, void* destDiffData) {
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnActivationForward(
        cudnnHandle_t handle, cudnnActivationMode_t mode, const void *alpha,
        const cudnnTensorDescriptor_t srcDesc, const void *srcData,
        const void *beta, const cudnnTensorDescriptor_t destDesc,
        void *destData) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationBackward(
        cudnnHandle_t handle, cudnnActivationMode_t mode, const void *alpha,
        const cudnnTensorDescriptor_t srcDesc, const void *srcData,
        const cudnnTensorDescriptor_t srcDiffDesc, const void *srcDiffData,
        const cudnnTensorDescriptor_t destDesc, const void *destData,
        const void *beta, const cudnnTensorDescriptor_t destDiffDesc,
        void *destDiffData) {
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"

#endif // #ifndef CUPY_NO_CUDA


///////////////////////////////////////////////////////////////////////////////
// Definitions are for compatibility with cuDNN v2, v3 v4 and v5.
///////////////////////////////////////////////////////////////////////////////

extern "C" {

#if CUDNN_VERSION < 3000
// ***_v3 functions are not declared in cuDNN v4.
// Following definitions are for compatibility with cuDNN v2 and v3.

typedef enum {} cudnnConvolutionBwdDataAlgo_t;
typedef enum {} cudnnConvolutionBwdDataPreference_t;
typedef enum {} cudnnConvolutionBwdFilterAlgo_t;
typedef enum {} cudnnConvolutionBwdFilterPreference_t;

cudnnStatus_t cudnnAddTensor_v3(
        cudnnHandle_t handle, const void* alpha,
        const cudnnTensorDescriptor_t bDesc, const void* b, const void* beta,
        cudnnTensorDescriptor_t yDesc, void* y) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_v3(
        cudnnHandle_t handle, const void* alpha,
        const cudnnTensorDescriptor_t xDesc, const void* x,
        const cudnnTensorDescriptor_t dyDesc, const void* dy,
        const cudnnConvolutionDescriptor_t convDesc,
        cudnnConvolutionBwdFilterAlgo_t algo,
        void* workSpace, size_t workSpaceSizeInBytes, const void* beta,
        const cudnnFilterDescriptor_t dwDesc, void* dw) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDes,
        const cudnnFilterDescriptor_t gradDes,
        cudnnConvolutionBwdFilterAlgo_t algo, size_t* sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnFilterDescriptor_t dwDesc,
        cudnnConvolutionBwdFilterPreference_t preference,
        size_t memoryLimitInBytes, cudnnConvolutionBwdFilterAlgo_t* algo) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardData_v3(
        cudnnHandle_t handle, const void* alpha,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const cudnnTensorDescriptor_t dyDesc, const void* dy,
        const cudnnConvolutionDescriptor_t convDesc,
        cudnnConvolutionBwdDataAlgo_t algo, void* workSpace,
        size_t workSpaceSizeInBytes, const void* beta,
        const cudnnTensorDescriptor_t dxDesc, void* dx) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
        cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        cudnnConvolutionBwdDataPreference_t preference,
        size_t memoryLimitInBytes, cudnnConvolutionBwdDataAlgo_t* algo) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        cudnnConvolutionBwdDataAlgo_t algo, size_t* sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor_v3(
        cudnnConvolutionDescriptor_t convDesc, int arrayLength,
        const int padA[], const int filterStrideA[], const int upscaleA[],
        cudnnConvolutionMode_t mode, cudnnDataType_t dataType) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // #if CUDNN_VERSION < 3000


#if CUDNN_VERSION < 4000

// ***_v2 functions are not declared in cuDNN v2 and v3.
// Following definitions are for compatibility with cuDNN v4.

#define cudnnAddTensor_v2 cudnnAddTensor
#define cudnnConvolutionBackwardData_v2 cudnnConvolutionBackwardData
#define cudnnConvolutionBackwardFilter_v2 cudnnConvolutionBackwardFilter
#define cudnnSetConvolutionNdDescriptor_v2 cudnnSetConvolutionNdDescriptor


typedef enum {} cudnnBatchNormMode_t;


cudnnStatus_t cudnnDeriveBNTensorDescriptor(
         cudnnTensorDescriptor_t derivedBnDesc,
         const cudnnTensorDescriptor_t xDesc,
         cudnnBatchNormMode_t mode) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining(
        cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
        const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
        const cudnnTensorDescriptor_t yDesc, void *y,
        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
        const void *bnScale, const void *bnBias,
        double exponentialAverageFactor, void *resultRunningMean,
        void *resultRunningVariance, double epsilon, void *resultSaveMean,
        void *resultSaveInvVariance) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(
        cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
        const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
        const cudnnTensorDescriptor_t yDesc, void *y,
        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
        const void *bnScale, const void *bnBias, const void *estimatedMean,
        const void *estimatedVariance, double epsilon) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnBatchNormalizationBackward(
        cudnnHandle_t handle, cudnnBatchNormMode_t mode,
        const void *alphaDataDiff, const void *betaDataDiff,
        const void *alphaParamDiff, const void *betaParamDiff,
        const cudnnTensorDescriptor_t xDesc, const void *x,
        const cudnnTensorDescriptor_t dyDesc, const void *dy,
        const cudnnTensorDescriptor_t dxDesc, void *dx,
        const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale,
        void *dBnScaleResult, void *dBnBiasResult, double epsilon,
        const void *savedMean, const void *savedInvVariance) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // #if CUDNN_VERSION < 4000


#if CUDNN_VERSION < 5000
// ***_v3 functions are not declared in cuDNN v2, v3 and v4.
// Following definitions are for compatibility with cuDNN v5.

#define cudnnActivationForward_v3 cudnnActivationForward
#define cudnnActivationBackward_v3 cudnnActivationBackward
#define cudnnSetFilter4dDescriptor_v3 cudnnSetFilter4dDescriptor
#define cudnnSetFilterNdDescriptor_v3 cudnnSetFilterNdDescriptor
#define cudnnSetPooling2dDescriptor_v3 cudnnSetPooling2dDescriptor
#define cudnnSetPoolingNdDescriptor_v3 cudnnSetPoolingNdDescriptor


typedef enum {} cudnnRNNMode_t;
typedef enum {} cudnnDirectionMode_t;
typedef enum {} cudnnRNNInputMode_t;

typedef void* cudnnDropoutDescriptor_t;
typedef void* cudnnRNNDescriptor_t;


cudnnStatus_t cudnnGetFilterNdDescriptor_v5(
        const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
        cudnnDataType_t* dataType, cudnnTensorFormat_t* format, int* nbDims,
        int filterDimA[]) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnCreateDropoutDescriptor(
        cudnnDropoutDescriptor_t* dropoutDesc) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(
        cudnnDropoutDescriptor_t dropoutDesc) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutGetStatesSize(
        cudnnHandle_t handle, size_t * sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize(
        cudnnTensorDescriptor_t xdesc, size_t * sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetDropoutDescriptor(
        cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
        float dropout, void* states, size_t stateSizeInBytes,
        unsigned long long seed) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutForward(
        cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
        const cudnnTensorDescriptor_t xdesc, const void* x,
        const cudnnTensorDescriptor_t ydesc, void* y,
        void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutBackward(
        cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
        const cudnnTensorDescriptor_t dydesc, const void* dy,
        const cudnnTensorDescriptor_t dxdesc, void* dx,
        void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* rnnDesc) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetRNNDescriptor(
        cudnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
        cudnnDropoutDescriptor_t dropoutDesc,
        cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction,
        cudnnRNNMode_t mode, cudnnDataType_t dataType) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength, const cudnnTensorDescriptor_t* xDesc,
        size_t* sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength, const cudnnTensorDescriptor_t* xDesc,
        size_t* sizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNParamsSize(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes,
        cudnnDataType_t dataType) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int layer, const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc,
        void** linLayerMat) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int layer, const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc,
        void** linLayerBias) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNForwardInference(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength,
        const cudnnTensorDescriptor_t* xDesc, const void* x,
        const cudnnTensorDescriptor_t hxDesc, const void* hx,
        const cudnnTensorDescriptor_t cxDesc, const void* cx,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const cudnnTensorDescriptor_t* yDesc, void* y,
        const cudnnTensorDescriptor_t hyDesc, void* hy,
        const cudnnTensorDescriptor_t cyDesc, void* cy,
        void* workspace, size_t workSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNForwardTraining(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength,
        const cudnnTensorDescriptor_t* xDesc, const void* x,
        const cudnnTensorDescriptor_t hxDesc, const void* hx,
        const cudnnTensorDescriptor_t cxDesc, const void* cx,
        const cudnnFilterDescriptor_t wDesc, const void* w,
        const cudnnTensorDescriptor_t* yDesc, void* y,
        const cudnnTensorDescriptor_t hyDesc, void* hy,
        const cudnnTensorDescriptor_t cyDesc, void* cy,
        void* workspace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNBackwardData(
        cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
        const int seqLength,
        const cudnnTensorDescriptor_t* yDesc, const void * y,
        const cudnnTensorDescriptor_t* dyDesc, const void * dy,
        const cudnnTensorDescriptor_t dhyDesc, const void * dhy,
        const cudnnTensorDescriptor_t dcyDesc, const void * dcy,
        const cudnnFilterDescriptor_t wDesc, const void * w,
        const cudnnTensorDescriptor_t hxDesc, const void * hx,
        const cudnnTensorDescriptor_t cxDesc, const void * cx,
        const cudnnTensorDescriptor_t* dxDesc, void* dx,
        const cudnnTensorDescriptor_t dhxDesc, void* dhx,
        const cudnnTensorDescriptor_t dcxDesc, void* dcx,
        void * workspace, size_t workSpaceSizeInBytes,
        const void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNBackwardWeights(
         cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
         const int seqLength, const cudnnTensorDescriptor_t* xDesc,
         const void* x, const cudnnTensorDescriptor_t hxDesc,
         const void* hx, const cudnnTensorDescriptor_t* yDesc,
         const void* y, const void* workspace, size_t workSpaceSizeInBytes,
         const cudnnFilterDescriptor_t dwDesc, void* dw,
         const void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // #if CUDNN_VERSION < 5000


#if CUDNN_VERSION >= 5000
// Some functions are renamed in cuDNN v5.
// Following definitions are for compatibility with cuDNN v5 and higher.

#define cudnnAddTensor_v3 cudnnAddTensor
#define cudnnConvolutionBackwardData_v3 cudnnConvolutionBackwardData
#define cudnnConvolutionBackwardFilter_v3 cudnnConvolutionBackwardFilter
#define cudnnSetConvolutionNdDescriptor_v3 cudnnSetConvolutionNdDescriptor
#define cudnnGetFilterNdDescriptor_v5 cudnnGetFilterNdDescriptor

#endif // CUDNN_VERSION >= 5000


#if (CUDNN_VERSION >= 5000) || defined(CUPY_NO_CUDA)
// ***_v2 functions are deleted in cuDNN v5.
// Following definitions are for compatibility with cuDNN v5 and higher.

typedef enum {} cudnnAddMode_t;

cudnnStatus_t cudnnSetConvolutionNdDescriptor_v2(
        cudnnConvolutionDescriptor_t convDesc, int arrayLength,
        const int padA[], const int filterStrideA[], const int upscaleA[],
        cudnnConvolutionMode_t mode) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor_v2(
        const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
        int* arrayLength, int padA[], int strideA[], int upscaleA[],
        cudnnConvolutionMode_t* mode) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnAddTensor_v2(
        cudnnHandle_t handle, cudnnAddMode_t mode, const void* alpha,
        const cudnnTensorDescriptor_t biasDesc, const void* biasData,
        const void* beta, cudnnTensorDescriptor_t srcDestDesc,
        void* srcDestData) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_v2(
        cudnnHandle_t handle, const void* alpha,
        const cudnnTensorDescriptor_t srcDesc, const void* srcData,
        const cudnnTensorDescriptor_t diffDesc, const void* diffData,
        const cudnnConvolutionDescriptor_t convDesc, const void* beta,
        const cudnnFilterDescriptor_t gradDesc, void* gradData) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardData_v2(
        cudnnHandle_t handle, const void* alpha,
        const cudnnFilterDescriptor_t filterDesc, const void* filterData,
        const cudnnTensorDescriptor_t diffDesc, const void* diffData,
        const cudnnConvolutionDescriptor_t convDesc, const void* beta,
        const cudnnTensorDescriptor_t gradDesc, void *gradData) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // #if (CUDNN_VERSION >= 5000) || defined(CUPY_NO_CUDA)

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUPY_CUDNN_H
