#ifndef INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H
#define INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H

extern "C" {

typedef hipsparseIndexBase_t cusparseIndexBase_t;
typedef hipsparseStatus_t cusparseStatus_t;

typedef hipsparseHandle_t cusparseHandle_t;
typedef hipsparseMatDescr_t cusparseMatDescr_t;
// TODO(leofang): how to handle them?
typedef void* bsric02Info_t;
typedef void* bsrilu02Info_t;

typedef hipsparseMatrixType_t cusparseMatrixType_t;
typedef hipsparseFillMode_t cusparseFillMode_t;
typedef hipsparseDiagType_t cusparseDiagType_t;
typedef hipsparseOperation_t cusparseOperation_t;
typedef hipsparsePointerMode_t cusparsePointerMode_t;
typedef hipsparseAction_t cusparseAction_t;
typedef hipsparseDirection_t cusparseDirection_t;
// TODO(leofang): how to handle them?
typedef enum {} cusparseAlgMode_t;
typedef hipsparseSolvePolicy_t cusparseSolvePolicy_t;

// Version
cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version) {
  return hipsparseGetVersion(handle, version);
}

// cuSPARSE Helper Function
cusparseStatus_t cusparseCreate(cusparseHandle_t* handle) {
  return hipsparseCreate(handle);
}

cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* descrA) {
  return hipsparseCreateMatDescr(descrA);
}

cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
  return hipsparseDestroy(handle);
}

cusparseStatus_t cusparseDestroyMatDescr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSetMatIndexBase(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSetMatType(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSetMatFillMode(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSetMatDiagType(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSetPointerMode(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

// Stream
cusparseStatus_t cusparseSetStream(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseGetStream(...) {
   return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

// cuSPARSE Level1 Function
cusparseStatus_t cusparseSgthr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgthr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgthr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgthr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

// cuSPARSE Level2 Function
cusparseStatus_t cusparseScsrmv(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrmv(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrmv(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrmv(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCsrmvEx_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCsrmvEx(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCsrsv2Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyCsrsv2Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrsv2_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrsv2_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrsv2_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrsv2_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrsv2_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrsv2_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrsv2_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrsv2_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrsv2_solve(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrsv2_solve(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrsv2_solve(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrsv2_solve(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsrsv2_zeroPivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

// cuSPARSE Level3 Function
cusparseStatus_t cusparseScsrmm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrmm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrmm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrmm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrmm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrmm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrmm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrmm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCsrsm2Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}
cusparseStatus_t cusparseDestroyCsrsm2Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrsm2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}
cusparseStatus_t cusparseDcsrsm2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}
cusparseStatus_t cusparseCcsrsm2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}
cusparseStatus_t cusparseZcsrsm2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrsm2_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}
cusparseStatus_t cusparseDcsrsm2_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}
cusparseStatus_t cusparseCcsrsm2_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}
cusparseStatus_t cusparseZcsrsm2_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrsm2_solve(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}
cusparseStatus_t cusparseDcsrsm2_solve(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}
cusparseStatus_t cusparseCcsrsm2_solve(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}
cusparseStatus_t cusparseZcsrsm2_solve(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsrsm2_zeroPivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

// cuSPARSE Extra Function
cusparseStatus_t cusparseXcsrgeamNnz(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrgeam(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrgeam(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrgeam(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrgeam(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrgeam2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrgeam2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrgeam2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsrgeam2Nnz(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrgeam2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrgeam2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrgeam2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrgeam2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsrgemmNnz(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrgemm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrgemm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrgemm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrgemm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCsrgemm2Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyCsrgemm2Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrgemm2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrgemm2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrgemm2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrgemm2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsrgemm2Nnz(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrgemm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrgemm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrgemm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrgemm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

// cuSPARSE Format Convrsion
cusparseStatus_t cusparseXcoo2csr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsc2dense(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsc2dense(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsc2dense(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsc2dense(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsr2coo(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsr2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsr2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsr2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsr2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}


cusparseStatus_t cusparseScsr2dense(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsr2dense(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsr2dense(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsr2dense(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSdense2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDdense2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCdense2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZdense2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSdense2csr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDdense2csr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCdense2csr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZdense2csr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSnnz(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnnz(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCnnz(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZnnz(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateIdentityPermutation(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcoosort_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcoosortByRow(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcoosortByColumn(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsrsort_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsrsort(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcscsort_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcscsort(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

// cuSPARSE PRECONDITIONERS

cusparseStatus_t cusparseCreateCsrilu02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyCsrilu02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateBsrilu02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyBsrilu02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCsric02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyCsric02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateBsric02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyBsric02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsrilu02_zeroPivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXbsrilu02_zeroPivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsric02_zeroPivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXbsric02_zeroPivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

#define CUSPARSE_VERSION -1

// cuSPARSE generic API
typedef void* cusparseSpVecDescr_t;
typedef void* cusparseDnVecDescr_t;
typedef void* cusparseSpMatDescr_t;
typedef void* cusparseDnMatDescr_t;

typedef enum {} cusparseIndexType_t;
typedef enum {} cusparseFormat_t;
typedef enum {} cusparseOrder_t;
typedef enum {} cusparseSpMVAlg_t;
typedef enum {} cusparseSpMMAlg_t;
typedef enum {} cusparseSparseToDenseAlg_t;
typedef enum {} cusparseDenseToSparseAlg_t;

cusparseStatus_t cusparseCreateSpVec(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroySpVec(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVecGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVecGetIndexBase(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVecGetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVecSetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCoo(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCooAoS(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCsr(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCsc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroySpMat(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCooGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCooAoSGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCsrGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCsrSetPointers(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatGetSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatGetFormat(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatGetIndexBase(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatGetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatSetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatGetStridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatSetStridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateDnVec(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyDnVec(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnVecGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnVecGetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnVecSetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateDnMat(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyDnMat(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnMatGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnMatGetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnMatSetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnMatGetStridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnMatSetStridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVV_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVV(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMV_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMV(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMM_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMM(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseConstrainedGeMM_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseConstrainedGeMM(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSparseToDense_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSparseToDense(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDenseToSparse_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDenseToSparse_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDenseToSparse_convert(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

typedef enum {} cusparseCsr2CscAlg_t;

cusparseStatus_t cusparseCsr2cscEx2_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCsr2cscEx2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}


///////////////////////////////////////////////////////////////////////////////
// Definitions are for compatibility
///////////////////////////////////////////////////////////////////////////////

cusparseStatus_t cusparseSnnz_compress(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnnz_compress(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCnnz_compress(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZnnz_compress(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsr2csr_compress(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsr2csr_compress(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsr2csr_compress(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsr2csr_compress(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

}  // extern "C"


#endif  // INCLUDE_GUARD_STUB_CUPY_CUSPARSE_H
