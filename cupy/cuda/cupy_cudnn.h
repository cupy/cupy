// This file is a stub header file of cudnn for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDNN_H
#define INCLUDE_GUARD_CUPY_CUDNN_H

#include "cupy_cuda.h"

#ifndef CUPY_NO_CUDA

#include <cudnn.h>

#else // #ifndef CUPY_NO_CUDA

extern "C" {

typedef enum {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
} cudnnStatus_t;

typedef enum {} cudnnActivationMode_t;
typedef enum {} cudnnConvolutionFwdAlgo_t;
typedef enum {} cudnnConvolutionFwdPreference_t;
typedef enum {} cudnnConvolutionMode_t;
typedef enum {} cudnnDataType_t;
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
const char* cudnnGetErrorString(...) {
    return NULL;
}


// Version
size_t cudnnGetVersion() {
    return CUDNN_STATUS_SUCCESS;
}


// Initialization and CUDA cooperation
cudnnStatus_t cudnnCreate(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream(...) {
    return CUDNN_STATUS_SUCCESS;
}


// Tensor manipulation
cudnnStatus_t cudnnCreateTensorDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}


// Filter manipulation
cudnnStatus_t cudnnCreateFilterDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}


// Convolution
cudnnStatus_t cudnnCreateConvolutionDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor_v4(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardBias(...) {
    return CUDNN_STATUS_SUCCESS;
}

// Pooling
cudnnStatus_t cudnnCreatePoolingDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingForward(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingBackward(...) {
    return CUDNN_STATUS_SUCCESS;
}


// Activation
cudnnStatus_t cudnnSoftmaxForward(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSoftmaxBackward(...) {
    return CUDNN_STATUS_SUCCESS;
}


} // extern "C"

#endif // #ifndef CUPY_NO_CUDA


///////////////////////////////////////////////////////////////////////////////
// Definitions are for compatibility with cuDNN v2, v3 v4 and v5.
///////////////////////////////////////////////////////////////////////////////

extern "C" {


#if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 3000)
// ***_v3 functions are not declared in cuDNN v2.
// Following definitions are for compatibility with cuDNN v3.

typedef enum {} cudnnConvolutionBwdDataAlgo_t;
typedef enum {} cudnnConvolutionBwdDataPreference_t;
typedef enum {} cudnnConvolutionBwdFilterAlgo_t;
typedef enum {} cudnnConvolutionBwdFilterPreference_t;
typedef struct {
  cudnnConvolutionFwdAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
} cudnnConvolutionFwdAlgoPerf_t;
typedef struct {
  cudnnConvolutionBwdFilterAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
} cudnnConvolutionBwdFilterAlgoPerf_t;
typedef struct {
  cudnnConvolutionBwdDataAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
} cudnnConvolutionBwdDataAlgoPerf_t;

cudnnStatus_t cudnnAddTensor_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardData_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // #if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 3000)

#if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 3000) || (CUDNN_VERSION >= 6000)
// some ***_v3 functions are not declared in cuDNN v2 and v6.

cudnnStatus_t cudnnSetFilter4dDescriptor_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetFilterNdDescriptor_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetPooling2dDescriptor_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnActivationForward_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnActivationBackward_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // #if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 3000) || (CUDNN_VERSION >= 6000)


#if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 4000)
// ***_v2 functions are not declared in cuDNN v2 and v3.
// Following definitions are for compatibility with cuDNN v4.

#define cudnnAddTensor_v2 cudnnAddTensor
#define cudnnConvolutionBackwardData_v2 cudnnConvolutionBackwardData
#define cudnnConvolutionBackwardFilter_v2 cudnnConvolutionBackwardFilter
#define cudnnSetConvolutionNdDescriptor_v2 cudnnSetConvolutionNdDescriptor


typedef enum {} cudnnBatchNormMode_t;
typedef enum {} cudnnNanPropagation_t;


typedef void* cudnnActivationDescriptor_t;


cudnnStatus_t cudnnSetFilter4dDescriptor_v4(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetFilterNdDescriptor_v4(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetFilterNdDescriptor_v4(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnBatchNormalizationBackward(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetPooling2dDescriptor_v4(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor_v4(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnCreateActivationDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetActivationDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyActivationDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnActivationForward_v4(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnActivationBackward_v4(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // #if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 4000)


#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 5000)

#define cudnnActivationForward_v3 cudnnActivationForward
#define cudnnActivationBackward_v3 cudnnActivationBackward
#define cudnnSetFilter4dDescriptor_v3 cudnnSetFilter4dDescriptor
#define cudnnSetFilterNdDescriptor_v3 cudnnSetFilterNdDescriptor
#define cudnnSetPooling2dDescriptor_v3 cudnnSetPooling2dDescriptor
#define cudnnSetPoolingNdDescriptor_v3 cudnnSetPoolingNdDescriptor

#endif // #if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 5000)


#if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 5000)
// ***_v3 functions are not declared in cuDNN v2, v3 and v4.
// Following definitions are for compatibility with cuDNN v5.

typedef enum {} cudnnRNNMode_t;
typedef enum {} cudnnDirectionMode_t;
typedef enum {} cudnnRNNInputMode_t;

typedef void* cudnnDropoutDescriptor_t;
typedef void* cudnnRNNDescriptor_t;


cudnnStatus_t cudnnSetConvolution2dDescriptor_v5(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetFilterNdDescriptor_v5(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnCreateDropoutDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutGetStatesSize(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetDropoutDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutForward(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDropoutBackward(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnCreateRNNDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetRNNDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNParamsSize(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNForwardInference(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNForwardTraining(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNBackwardData(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNBackwardWeights(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // #if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 5000)


#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION >= 5000)
// Some functions are renamed in cuDNN v5.
// Following definitions are for compatibility with cuDNN v5 and higher.

#define cudnnAddTensor_v3 cudnnAddTensor
#define cudnnConvolutionBackwardData_v3 cudnnConvolutionBackwardData
#define cudnnConvolutionBackwardFilter_v3 cudnnConvolutionBackwardFilter
#define cudnnSetConvolutionNdDescriptor_v3 cudnnSetConvolutionNdDescriptor

#endif // #if !defined(CUPY_NO_CUDA) && CUDNN_VERSION >= 5000


#if defined(CUPY_NO_CUDA) || (CUDNN_VERSION >= 5000)
// ***_v2 functions are deleted in cuDNN v5.
// Following definitions are for compatibility with cuDNN v5 and higher.
// This section code is also used instead of cuDNN v2 stub.

typedef enum {} cudnnAddMode_t;

cudnnStatus_t cudnnSetConvolutionNdDescriptor_v2(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor_v2(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnAddTensor_v2(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardFilter_v2(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardData_v2(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // #if defined(CUPY_NO_CUDA) || (CUDNN_VERSION >= 5000)


#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 6000)

#define cudnnSetConvolution2dDescriptor_v4 cudnnSetConvolution2dDescriptor

#endif // #if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 6000)


#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION >= 6000)
// Some functions are renamed in cuDNN v6.
// Following definitions are for compatibility with cuDNN v6 and higher.

#define cudnnSetFilter4dDescriptor_v4 cudnnSetFilter4dDescriptor
#define cudnnSetFilterNdDescriptor_v4 cudnnSetFilterNdDescriptor
#define cudnnGetFilterNdDescriptor_v4 cudnnGetFilterNdDescriptor
#define cudnnSetPooling2dDescriptor_v4 cudnnSetPooling2dDescriptor
#define cudnnSetPoolingNdDescriptor_v4 cudnnSetPoolingNdDescriptor
#define cudnnActivationForward_v4 cudnnActivationForward
#define cudnnActivationBackward_v4 cudnnActivationBackward

#endif // #if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION >= 6000)


} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUPY_CUDNN_H
