// This file is a stub header file of cudnn for Read the Docs.

#ifndef INCLUDE_GUARD_STUB_CUPY_CUDNN_H
#define INCLUDE_GUARD_STUB_CUPY_CUDNN_H


#define CUDNN_VERSION 0

#define CUDNN_BN_MIN_EPSILON 0.0

extern "C" {

typedef enum {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
} cudnnStatus_t;

typedef enum {} cudnnActivationMode_t;
typedef enum {} cudnnConvolutionFwdAlgo_t;
typedef enum {} cudnnConvolutionFwdPreference_t;
typedef enum {} cudnnConvolutionMode_t;
typedef enum {} cudnnCTCLossAlgo_t;
typedef enum {} cudnnDataType_t;
typedef enum {} cudnnPoolingMode_t;
typedef enum {} cudnnSoftmaxAlgorithm_t;
typedef enum {} cudnnSoftmaxMode_t;
typedef enum {} cudnnTensorFormat_t;
typedef enum {} cudnnErrQueryMode_t;
typedef struct cudnnRuntimeTag_t cudnnRuntimeTag_t;

typedef void* cudnnConvolutionDescriptor_t;
typedef void* cudnnCTCLossDescriptor_t;
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

// Runtime error checking
cudnnStatus_t cudnnQueryRuntimeError(...) {
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

cudnnStatus_t cudnnGetTensor4dDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnScaleTensor(...) {
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

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v6(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(...) {
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

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v6(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnConvolutionBackwardData_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v6(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor_v3(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}


typedef enum {} cudnnBatchNormMode_t;
typedef enum {} cudnnBatchNormOps_t;
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

typedef enum {} cudnnMathType_t;

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

typedef enum {} cudnnConvolutionBwdDataAlgo_t;
typedef enum {} cudnnConvolutionBwdDataPreference_t;
typedef enum {} cudnnConvolutionBwdFilterAlgo_t;
typedef enum {} cudnnConvolutionBwdFilterPreference_t;
typedef enum {} cudnnDeterminism_t;

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


// ***_v3 functions are not declared in cuDNN v2, v3 and v4.
// Following definitions are for compatibility with cuDNN v5.

typedef enum {} cudnnRNNMode_t;
typedef enum {} cudnnDirectionMode_t;
typedef enum {} cudnnOpTensorOp_t;
typedef enum {} cudnnRNNInputMode_t;

typedef void* cudnnDropoutDescriptor_t;
typedef void* cudnnRNNDescriptor_t;
typedef void* cudnnSpatialTransformerDescriptor_t;
typedef void* cudnnSamplerType_t;
typedef void* cudnnOpTensorDescriptor_t;


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

cudnnStatus_t cudnnSetRNNDescriptor_v5(...) {
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

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSpatialTfSamplerForward(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t cudnnSpatialTfSamplerBackward(...) {
    return CUDNN_STATUS_NOT_SUPPORTED;
}


// Tensor operations
cudnnStatus_t cudnnCreateOpTensorDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetOpTensorDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetOpTensorDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor(...) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpTensor(...) {
    return CUDNN_STATUS_SUCCESS;
}

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

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUDNN_H
