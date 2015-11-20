// This file is a stub header file of cudnn for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDNN_H
#define INCLUDE_GUARD_CUPY_CUDNN_H

#include "cupy_cuda.h"

#ifndef CUPY_NO_CUDA
#include <cudnn.h>

#else // #ifndef CUPY_NO_CUDA


///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

typedef int cudnnActivationMode_t;
typedef int cudnnAddMode_t;
typedef int cudnnConvolutionFwdAlgo_t;
typedef int cudnnConvolutionFwdPreference_t;
typedef int cudnnConvolutionMode_t;
typedef int cudnnDataType_t;
typedef int cudnnPoolingMode_t;
typedef int cudnnSoftmaxAlgorithm_t;
typedef int cudnnSoftmaxMode_t;
typedef int cudnnStatus_t;
typedef int cudnnTensorFormat_t;

typedef int ActivationMode;
typedef int AddMode;
typedef int ConvolutionFwdAlgo;
typedef int ConvolutionFwdPreference;
typedef int ConvolutionMode;
typedef int DataType;
typedef int PoolingMode;
typedef int SoftmaxAlgorithm;
typedef int SoftmaxMode;
typedef int Status;
typedef int TensorFormat;


typedef void* cudnnConvolutionDescriptor_t;
typedef void* cudnnFilterDescriptor_t;
typedef void* cudnnHandle_t;
typedef void* cudnnPoolingDescriptor_t;
typedef void* cudnnTensorDescriptor_t;

typedef void* ConvolutionDescriptor;
typedef void* FilterDescriptor;
typedef void* Handle;
typedef void* PoolingDescriptor;
typedef void* TensorDescriptor;



// Error handling
const char* cudnnGetErrorString(Status status) {
    return NULL;
}

// Initialization and CUDA cooperation
int cudnnCreate(Handle* handle) {
    return 0;
}

int cudnnDestroy(Handle handle) {
    return 0;
}

int cudnnSetStream(Handle handle, Stream stream) {
    return 0;
}

int cudnnGetStream(Handle handle, Stream* stream) {
    return 0;
}


// Tensor manipulation
int cudnnCreateTensorDescriptor(TensorDescriptor* descriptor) {
    return 0;
}

int cudnnSetTensor4dDescriptor(
        TensorDescriptor tensorDesc, TensorFormat format,
        DataType dataType, int n, int c, int h, int w) {
    return 0;
}

int cudnnSetTensor4dDescriptorEx(
        TensorDescriptor tensorDesc, DataType dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride) {
    return 0;
}

int cudnnSetTensorNdDescriptor(
        TensorDescriptor tensorDesc, DataType dataType, int nbDims,
        int* dimA, int* strideA) {
    return 0;
}

int cudnnDestroyTensorDescriptor(TensorDescriptor tensorDesc) {
    return 0;
}

int cudnnAddTensor(
        Handle handle, AddMode mode, void* alpha,
        TensorDescriptor biasDesc, void* biasData, void* beta,
        TensorDescriptor srcDestDesc, void* srcDestData) {
    return 0;
}


// Filter manipulation
int cudnnCreateFilterDescriptor(FilterDescriptor* filterDesc) {
    return 0;
}

int cudnnSetFilter4dDescriptor(
        FilterDescriptor filterDesc, DataType dataType,
        int n, int c, int h, int w) {
    return 0;
}

int cudnnSetFilterNdDescriptor(
        FilterDescriptor filterDesc, DataType dataType, int nbDims,
        int* filterDimA) {
    return 0;
}

int cudnnDestroyFilterDescriptor(FilterDescriptor filterDesc) {
    return 0;
}


// Convolution
int cudnnCreateConvolutionDescriptor(ConvolutionDescriptor* convDesc) {
    return 0;
}

int cudnnSetConvolution2dDescriptor(
        ConvolutionDescriptor convDesc, int pad_h, int pad_w, int u,
        int v, int upscalex, int upscaley, ConvolutionMode mode) {
    return 0;
}

int cudnnSetConvolutionNdDescriptor(
        ConvolutionDescriptor convDesc, int arrayLength, int* padA,
        int* filterStrideA, int* upscaleA, ConvolutionMode mode) {
    return 0;
}

int cudnnDestroyConvolutionDescriptor(ConvolutionDescriptor conDesc) {
    return 0;
}

int cudnnGetConvolutionForwardAlgorithm(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, int preference,
        size_t memoryLimitInbytes, ConvolutionFwdAlgo* algo) {
    return 0;
}

int cudnnGetConvolutionForwardWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionFwdAlgo algo,
        size_t* sizeInBytes) {
    return 0;
}

int cudnnConvolutionForward(
        Handle handle, void* alpha, TensorDescriptor srcDesc,
        void* srcData, FilterDescriptor filterDesc, void* filterData,
        ConvolutionDescriptor convDesc, ConvolutionFwdAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor destDesc, void* destData) {
    return 0;
}

int cudnnConvolutionBackwardBias(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor destDesc, void* destData) {
    return 0;
}

int cudnnConvolutionBackwardFilter(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, void* beta,
        FilterDescriptor gradDesc, void* gradData) {
    return 0;
}

int cudnnConvolutionBackwardData(
        Handle handle, void* alpha,
        FilterDescriptor filterDesc, void* filterData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, void* beta,
        TensorDescriptor gradDesc, void* gradData) {
    return 0;
}


// Pooling
int cudnnCreatePoolingDescriptor(PoolingDescriptor* desc) {
    return 0;
}

int cudnnSetPooling2dDescriptor(
        PoolingDescriptor poolingDesc, PoolingMode mode,
        int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding,
        int verticalStride, int horizontalStride) {
    return 0;
}

int cudnnSetPoolingNdDescriptor(
        PoolingDescriptor poolingDesc, PoolingMode mode, int nbDims,
        int* windowDimA, int* paddingA, int* strideA) {
    return 0;
}

int cudnnDestroyPoolingDescriptor(PoolingDescriptor poolingDesc) {
    return 0;
}

int cudnnPoolingForward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData) {
    return 0;
}

int cudnnPoolingBackward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData) {
    return 0;
}


// Activation
int cudnnSoftmaxForward(
        Handle handle, SoftmaxAlgorithm algorithm, SoftmaxMode mode,
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        void* beta, TensorDescriptor dstDesc, void* dstData) {
    return 0;
}

int cudnnSoftmaxBackward(
        Handle handle, SoftmaxAlgorithm algorithm, SoftmaxMode mode,
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData) {
    return 0;
}

int cudnnActivationForward(
        Handle handle, ActivationMode mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData) {
    return 0;
}

int cudnnActivationBackward(
        Handle handle, ActivationMode mode, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData) {
    return 0;
}


#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDNN_H
