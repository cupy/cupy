// This file is a stub header file of cusparse for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUSPARSE_H
#define INCLUDE_GUARD_CUPY_CUSPARSE_H

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include <cuda.h>
#include <cusparse.h>

#if CUDA_VERSION < 9000
// Functions added in CUDA 9.0
cusparseStatus_t cusparseSgtsv2_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2_nopivot_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2_nopivot_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2_nopivot_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2_nopivot_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2_nopivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2_nopivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2_nopivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2_nopivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2StridedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2StridedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2StridedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2StridedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2StridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2StridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2StridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2StridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}
#endif // #if CUDA_VERSION < 9000

#if CUDA_VERSION < 9200
// Functions added in CUDA 9.2
cusparseStatus_t cusparseSgtsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgpsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgpsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgpsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgpsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgpsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgpsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgpsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgpsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}
#endif // #if CUDA_VERSION < 9200

#else  // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#ifdef CUPY_USE_HIP
#include "cupy_hip_common.h"
#else // #ifdef CUPY_USE_HIP
#include "cupy_cuda_common.h"
#endif // #ifdef CUPY_USE_HIP

extern "C" {

typedef enum {} cusparseIndexBase_t;
typedef enum {
  CUSPARSE_STATUS_SUCCESS=0,
}  cusparseStatus_t;

typedef void* cusparseHandle_t;
typedef void* cusparseMatDescr_t;
typedef void* csric02Info_t;
typedef void* bsric02Info_t;
typedef void* csrilu02Info_t;
typedef void* bsrilu02Info_t;

typedef enum {} cusparseMatrixType_t;
typedef enum {} cusparseOperation_t;
typedef enum {} cusparsePointerMode_t;
typedef enum {} cusparseAction_t;
typedef enum {} cusparseDirection_t;
typedef enum {} cusparseAlgMode_t;
typedef enum {} cusparseSolvePolicy_t;

// Version
cusparseStatus_t cusparseGetVersion(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

// cuSPARSE Helper Function
cusparseStatus_t cusparseCreate(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateMatDescr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroy(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyMatDescr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatIndexBase(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatType(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetPointerMode(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

// Stream
cusparseStatus_t cusparseSetStream(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetStream(...) {
   return CUSPARSE_STATUS_SUCCESS;
}

// cuSPARSE Level1 Function
cusparseStatus_t cusparseSgthr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgthr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgthr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgthr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

// cuSPARSE Level2 Function
cusparseStatus_t cusparseScsrmv(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmv(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrmv(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrmv(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsrmvEx_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsrmvEx(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

// cuSPARSE Level3 Function
cusparseStatus_t cusparseScsrmm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrmm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrmm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrmm2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmm2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrmm2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrmm2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}


// cuSPARSE Extra Function
cusparseStatus_t cusparseXcsrgeamNnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrgeam(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrgeam(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrgeam(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrgeam(...) {
  return CUSPARSE_STATUS_SUCCESS;
}


cusparseStatus_t cusparseXcsrgemmNnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrgemm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrgemm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrgemm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrgemm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

// cuSPARSE Format Convrsion
cusparseStatus_t cusparseXcoo2csr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsc2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsc2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsc2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsc2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsr2coo(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsr2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsr2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsr2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsr2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}


cusparseStatus_t cusparseScsr2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsr2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsr2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsr2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSdense2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDdense2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCdense2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZdense2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSdense2csr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDdense2csr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCdense2csr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZdense2csr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSnnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCnnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZnnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateIdentityPermutation(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcoosort_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcoosortByRow(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsrsort_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsrsort(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcscsort_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcscsort(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

// cuSPARSE PRECONDITIONERS

cusparseStatus_t cusparseCreateCsrilu02Info(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyCsrilu02Info(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateBsrilu02Info(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyBsrilu02Info(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateCsric02Info(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyCsric02Info(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateBsric02Info(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyBsric02Info(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrilu02_numericBoost(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrilu02_numericBoost(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrilu02_numericBoost(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrilu02_numericBoost(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsrilu02_zeroPivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrilu02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrilu02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrilu02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrilu02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrilu02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrilu02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrilu02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrilu02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrilu02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrilu02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrilu02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrilu02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSbsrilu02_numericBoost(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDbsrilu02_numericBoost(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCbsrilu02_numericBoost(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZbsrilu02_numericBoost(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXbsrilu02_zeroPivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSbsrilu02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDbsrilu02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCbsrilu02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZbsrilu02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSbsrilu02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDbsrilu02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCbsrilu02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZbsrilu02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSbsrilu02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDbsrilu02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCbsrilu02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZbsrilu02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsric02_zeroPivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsric02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsric02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsric02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsric02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsric02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsric02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsric02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsric02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsric02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsric02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsric02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsric02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXbsric02_zeroPivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSbsric02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDbsric02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCbsric02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZbsric02_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSbsric02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDbsric02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCbsric02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZbsric02_analysis(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSbsric02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDbsric02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCbsric02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZbsric02(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2_nopivot_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2_nopivot_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2_nopivot_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2_nopivot_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2_nopivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2_nopivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2_nopivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2_nopivot(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2StridedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2StridedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2StridedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2StridedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsv2StridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsv2StridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsv2StridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsv2StridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgtsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgtsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgtsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgtsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgpsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgpsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgpsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgpsvInterleavedBatch_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSgpsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgpsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCgpsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZgpsvInterleavedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

}  // extern "C"

#endif  // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)


///////////////////////////////////////////////////////////////////////////////
// Definitions are for compatibility
///////////////////////////////////////////////////////////////////////////////

extern "C" {

#if defined(CUPY_NO_CUDA) || defined(CUPY_USE_HIP)

cusparseStatus_t cusparseSnnz_compress(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnnz_compress(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCnnz_compress(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZnnz_compress(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsr2csr_compress(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsr2csr_compress(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsr2csr_compress(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsr2csr_compress(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

#endif  // #if defined(CUPY_NO_CUDA) || defined(CUPY_USE_HIP)

}  // extern "C"


#endif  // INCLUDE_GUARD_CUPY_CUSPARSE_H
