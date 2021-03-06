#ifndef INCLUDE_GUARD_CUDA_CUPY_CUDNN_H
#define INCLUDE_GUARD_CUDA_CUPY_CUDNN_H

#include <cudnn.h>


///////////////////////////////////////////////////////////////////////////////
// Definitions are for compatibility with cuDNN v5 and v6.
///////////////////////////////////////////////////////////////////////////////

extern "C" {

#if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 6000)

typedef enum {} cudnnRNNAlgo_t;
typedef enum {} cudnnReduceTensorOp_t;
typedef enum {} cudnnReduceTensorIndices_t;
typedef enum {} cudnnIndicesType_t;

typedef void* cudnnPersistentRNNPlan_t;
typedef void* cudnnReduceTensorDescriptor_t;

cudnnStatus_t cudnnCreatePersistentRNNPlan(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetPersistentRNNPlan(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetRNNDescriptor_v6(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}


// Tensor reductions
cudnnStatus_t cudnnCreateReduceTensorDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetReduceTensorDescriptor(...){
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReduceTensorDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionIndicesSize(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReduceTensor(...) {
    return CUDNN_STATUS_SUCCESS;
}

#endif // #if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 6000)


#if !defined(CUPY_NO_CUDA)
// Some functions are renamed in cuDNN v5.
// Following definitions are for compatibility with cuDNN v5 and higher.

#define cudnnAddTensor_v3 cudnnAddTensor
#define cudnnConvolutionBackwardData_v3 cudnnConvolutionBackwardData
#define cudnnConvolutionBackwardFilter_v3 cudnnConvolutionBackwardFilter
#define cudnnSetConvolutionNdDescriptor_v3 cudnnSetConvolutionNdDescriptor

#endif // #if !defined(CUPY_NO_CUDA)

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


#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 7000)

#define cudnnSetRNNDescriptor_v5 cudnnSetRNNDescriptor

typedef enum {} cudnnMathType_t;
typedef enum {} cudnnCTCLossAlgo_t;
typedef void* cudnnCTCLossDescriptor_t;

// CTC
cudnnStatus_t cudnnCreateCTCLossDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}
cudnnStatus_t cudnnDestroyCTCLossDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}
cudnnStatus_t cudnnSetCTCLossDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}
cudnnStatus_t cudnnGetCTCLossDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}
cudnnStatus_t cudnnGetCTCLossWorkspaceSize(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}
cudnnStatus_t cudnnCTCLoss(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetConvolutionMathType(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionMathType(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetConvolutionGroupCount(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionGroupCount(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // #if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 7000)


#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION >= 7000)

#define cudnnSetConvolution2dDescriptor_v5 cudnnSetConvolution2dDescriptor

cudnnStatus_t cudnnSetConvolution2dDescriptor_v4(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}


#endif // #if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION >= 7000)


#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 6000)

typedef enum {} cudnnDeterminism_t;

#endif // #if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 6000)


#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 7000)

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx_v7(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}
cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx_v7(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}
cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx_v7(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

typedef struct {
  cudnnConvolutionFwdAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
  cudnnDeterminism_t determinism;
  cudnnMathType_t mathType;
  int reserved[3];
} cudnnConvolutionFwdAlgoPerf_v7_t;

typedef struct {
  cudnnConvolutionBwdFilterAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
  cudnnDeterminism_t determinism;
  cudnnMathType_t mathType;
  int reserved[3];
} cudnnConvolutionBwdFilterAlgoPerf_v7_t;

typedef struct {
  cudnnConvolutionBwdDataAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
  cudnnDeterminism_t determinism;
  cudnnMathType_t mathType;
  int reserved[3];
} cudnnConvolutionBwdDataAlgoPerf_v7_t;

#endif // #if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 7000)


#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION >= 7000)

#define cudnnFindConvolutionForwardAlgorithmEx_v7 cudnnFindConvolutionForwardAlgorithmEx
#define cudnnFindConvolutionBackwardFilterAlgorithmEx_v7 cudnnFindConvolutionBackwardFilterAlgorithmEx
#define cudnnFindConvolutionBackwardDataAlgorithmEx_v7 cudnnFindConvolutionBackwardDataAlgorithmEx

#define cudnnConvolutionFwdAlgoPerf_v7_t cudnnConvolutionFwdAlgoPerf_t
#define cudnnConvolutionBwdFilterAlgoPerf_v7_t cudnnConvolutionBwdFilterAlgoPerf_t
#define cudnnConvolutionBwdDataAlgoPerf_v7_t cudnnConvolutionBwdDataAlgoPerf_t

#endif // #if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION >= 7000)


#if defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 7200)

typedef void* cudnnRNNDataDescriptor_t;

typedef enum {} cudnnRNNDataLayout_t;
typedef enum {} cudnnRNNPaddingMode_t;

cudnnStatus_t cudnnSetRNNPaddingMode(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNPaddingMode(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnCreateRNNDataDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetRNNDataDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetRNNDataDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNForwardInferenceEx(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNForwardTrainingEx(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNBackwardDataEx(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnRNNBackwardWeightsEx(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // defined(CUPY_NO_CUDA) || (CUDNN_VERSION < 7200)

#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 8000)
// TODO: check function names when cuDNN 8 is released.

#define cudnnGetConvolutionForwardAlgorithm_v6 cudnnGetConvolutionForwardAlgorithm
#define cudnnGetConvolutionBackwardFilterAlgorithm_v6 cudnnGetConvolutionBackwardFilterAlgorithm
#define cudnnGetConvolutionBackwardDataAlgorithm_v6 cudnnGetConvolutionBackwardDataAlgorithm

#endif // #if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 8000)


#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 7000)

typedef enum {} cudnnErrQueryMode_t;
typedef struct cudnnRuntimeTag_t cudnnRuntimeTag_t;

cudnnStatus_t cudnnQueryRuntimeError(...) {
    return CUDNN_STATUS_SUCCESS;
}

#endif // !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 7000)

#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 7400)

typedef enum {} cudnnBatchNormOps_t;

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(...) {
    return CUDNN_STATUS_SUCCESS;
}

#endif // !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 7400)

#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 7600)

// fused ops
typedef void* cudnnFusedOpsConstParamPack_t;
typedef void* cudnnFusedOpsVariantParamPack_t;
typedef void* cudnnFusedOpsPlan_t;

typedef enum {} cudnnFusedOps_t;
typedef enum {} cudnnFusedOpsConstParamLabel_t;
typedef enum {} cudnnFusedOpsPointerPlaceHolder_t;
typedef enum {} cudnnFusedOpsVariantParamLabel_t;

cudnnStatus_t cudnnCreateFusedOpsConstParamPack(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnCreateFusedOpsPlan(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroyFusedOpsPlan(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnMakeFusedOpsPlan(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnFusedOpsExecute(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // !defined(CUPY_NO_CUDA) && (CUDNN_VERSION < 7600)

#if !defined(CUPY_NO_CUDA) && (CUDNN_VERSION >= 8000)

typedef enum {} cudnnConvolutionFwdPreference_t;
typedef enum {} cudnnConvolutionBwdFilterPreference_t;
typedef enum {} cudnnConvolutionBwdDataPreference_t;

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v6(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v6(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v6(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetRNNDescriptor_v5(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

#endif // !defined(CUPY_NO_CUDA) && (CUDNN_VERSION >= 8000)

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUDNN_H
