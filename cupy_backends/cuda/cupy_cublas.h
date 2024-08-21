#ifndef INCLUDE_GUARD_CUDA_CUPY_CUBLAS_H
#define INCLUDE_GUARD_CUDA_CUPY_CUBLAS_H

#include <cuda.h>
#include <cublas_v2.h>

#ifndef CUBLAS_VER_MAJOR
#define CUBLAS_VER_MAJOR 0
#endif

#if CUBLAS_VER_MAJOR < 12

typedef int intBit;
#define cublasIsamaxBit cublasIsamax
#define cublasIdamaxBit cublasIdamax
#define cublasIcamaxBit cublasIcamax
#define cublasIzamaxBit cublasIzamax
#define cublasIsaminBit cublasIsamin
#define cublasIdaminBit cublasIdamin
#define cublasIcaminBit cublasIcamin
#define cublasIzaminBit cublasIzamin
#define cublasSasumBit cublasSasum
#define cublasDasumBit cublasDasum
#define cublasScasumBit cublasScasum
#define cublasDzasumBit cublasDzasum
#define cublasSaxpyBit cublasSaxpy
#define cublasDaxpyBit cublasDaxpy
#define cublasCaxpyBit cublasCaxpy
#define cublasZaxpyBit cublasZaxpy
#define cublasSdotBit cublasSdot
#define cublasDdotBit cublasDdot
#define cublasCdotuBit cublasCdotu
#define cublasCdotcBit cublasCdotc
#define cublasZdotuBit cublasZdotu
#define cublasZdotcBit cublasZdotc
#define cublasSnrm2Bit cublasSnrm2
#define cublasDnrm2Bit cublasDnrm2
#define cublasScnrm2Bit cublasScnrm2
#define cublasDznrm2Bit cublasDznrm2
#define cublasSscalBit cublasSscal
#define cublasDscalBit cublasDscal
#define cublasCscalBit cublasCscal
#define cublasCsscalBit cublasCsscal
#define cublasZscalBit cublasZscal
#define cublasZdscalBit cublasZdscal
#define cublasSgemvBit cublasSgemv
#define cublasDgemvBit cublasDgemv
#define cublasCgemvBit cublasCgemv
#define cublasZgemvBit cublasZgemv
#define cublasSgerBit cublasSger
#define cublasDgerBit cublasDger
#define cublasCgeruBit cublasCgeru
#define cublasCgercBit cublasCgerc
#define cublasZgeruBit cublasZgeru
#define cublasZgercBit cublasZgerc
#define cublasSsbmvBit cublasSsbmv
#define cublasDsbmvBit cublasDsbmv
#define cublasSgemmBit cublasSgemm
#define cublasDgemmBit cublasDgemm
#define cublasCgemmBit cublasCgemm
#define cublasZgemmBit cublasZgemm
#define cublasSgemmBatchedBit cublasSgemmBatched
#define cublasDgemmBatchedBit cublasDgemmBatched
#define cublasCgemmBatchedBit cublasCgemmBatched
#define cublasZgemmBatchedBit cublasZgemmBatched
#define cublasSgemmStridedBatchedBit cublasSgemmStridedBatched
#define cublasDgemmStridedBatchedBit cublasDgemmStridedBatched
#define cublasCgemmStridedBatchedBit cublasCgemmStridedBatched
#define cublasZgemmStridedBatchedBit cublasZgemmStridedBatched
#define cublasStrsmBit cublasStrsm
#define cublasDtrsmBit cublasDtrsm
#define cublasCtrsmBit cublasCtrsm
#define cublasZtrsmBit cublasZtrsm
#define cublasStrsmBatchedBit cublasStrsmBatched
#define cublasDtrsmBatchedBit cublasDtrsmBatched
#define cublasCtrsmBatchedBit cublasCtrsmBatched
#define cublasZtrsmBatchedBit cublasZtrsmBatched
#define cublasSsyrkBit cublasSsyrk
#define cublasDsyrkBit cublasDsyrk
#define cublasCsyrkBit cublasCsyrk
#define cublasZsyrkBit cublasZsyrk
#define cublasSgeamBit cublasSgeam
#define cublasDgeamBit cublasDgeam
#define cublasCgeamBit cublasCgeam
#define cublasZgeamBit cublasZgeam
#define cublasSdgmmBit cublasSdgmm
#define cublasDdgmmBit cublasDdgmm
#define cublasCdgmmBit cublasCdgmm
#define cublasZdgmmBit cublasZdgmm
#define cublasSgemmExBit cublasSgemmEx
#define cublasGemmExBit cublasGemmEx
#define cublasGemmStridedBatchedExBit cublasGemmStridedBatchedEx

#else

typedef int64_t intBit;
#define cublasIsamaxBit cublasIsamax_64
#define cublasIdamaxBit cublasIdamax_64
#define cublasIcamaxBit cublasIcamax_64
#define cublasIzamaxBit cublasIzamax_64
#define cublasIsaminBit cublasIsamin_64
#define cublasIdaminBit cublasIdamin_64
#define cublasIcaminBit cublasIcamin_64
#define cublasIzaminBit cublasIzamin_64
#define cublasSasumBit cublasSasum_64
#define cublasDasumBit cublasDasum_64
#define cublasScasumBit cublasScasum_64
#define cublasDzasumBit cublasDzasum_64
#define cublasSaxpyBit cublasSaxpy_64
#define cublasDaxpyBit cublasDaxpy_64
#define cublasCaxpyBit cublasCaxpy_64
#define cublasZaxpyBit cublasZaxpy_64
#define cublasSdotBit cublasSdot_64
#define cublasDdotBit cublasDdot_64
#define cublasCdotuBit cublasCdotu_64
#define cublasCdotcBit cublasCdotc_64
#define cublasZdotuBit cublasZdotu_64
#define cublasZdotcBit cublasZdotc_64
#define cublasSnrm2Bit cublasSnrm2_64
#define cublasDnrm2Bit cublasDnrm2_64
#define cublasScnrm2Bit cublasScnrm2_64
#define cublasDznrm2Bit cublasDznrm2_64
#define cublasSscalBit cublasSscal_64
#define cublasDscalBit cublasDscal_64
#define cublasCscalBit cublasCscal_64
#define cublasCsscalBit cublasCsscal_64
#define cublasZscalBit cublasZscal_64
#define cublasZdscalBit cublasZdscal_64
#define cublasSgemvBit cublasSgemv_64
#define cublasDgemvBit cublasDgemv_64
#define cublasCgemvBit cublasCgemv_64
#define cublasZgemvBit cublasZgemv_64
#define cublasSgerBit cublasSger_64
#define cublasDgerBit cublasDger_64
#define cublasCgeruBit cublasCgeru_64
#define cublasCgercBit cublasCgerc_64
#define cublasZgeruBit cublasZgeru_64
#define cublasZgercBit cublasZgerc_64
#define cublasSsbmvBit cublasSsbmv_64
#define cublasDsbmvBit cublasDsbmv_64
#define cublasSgemmBit cublasSgemm_64
#define cublasDgemmBit cublasDgemm_64
#define cublasCgemmBit cublasCgemm_64
#define cublasZgemmBit cublasZgemm_64
#define cublasSgemmBatchedBit cublasSgemmBatched_64
#define cublasDgemmBatchedBit cublasDgemmBatched_64
#define cublasCgemmBatchedBit cublasCgemmBatched_64
#define cublasZgemmBatchedBit cublasZgemmBatched_64
#define cublasSgemmStridedBatchedBit cublasSgemmStridedBatched_64
#define cublasDgemmStridedBatchedBit cublasDgemmStridedBatched_64
#define cublasCgemmStridedBatchedBit cublasCgemmStridedBatched_64
#define cublasZgemmStridedBatchedBit cublasZgemmStridedBatched_64
#define cublasStrsmBit cublasStrsm_64
#define cublasDtrsmBit cublasDtrsm_64
#define cublasCtrsmBit cublasCtrsm_64
#define cublasZtrsmBit cublasZtrsm_64
#define cublasStrsmBatchedBit cublasStrsmBatched_64
#define cublasDtrsmBatchedBit cublasDtrsmBatched_64
#define cublasCtrsmBatchedBit cublasCtrsmBatched_64
#define cublasZtrsmBatchedBit cublasZtrsmBatched_64
#define cublasSsyrkBit cublasSsyrk_64
#define cublasDsyrkBit cublasDsyrk_64
#define cublasCsyrkBit cublasCsyrk_64
#define cublasZsyrkBit cublasZsyrk_64
#define cublasSgeamBit cublasSgeam_64
#define cublasDgeamBit cublasDgeam_64
#define cublasCgeamBit cublasCgeam_64
#define cublasZgeamBit cublasZgeam_64
#define cublasSdgmmBit cublasSdgmm_64
#define cublasDdgmmBit cublasDdgmm_64
#define cublasCdgmmBit cublasCdgmm_64
#define cublasZdgmmBit cublasZdgmm_64
#define cublasSgemmExBit cublasSgemmEx_64
#define cublasGemmExBit cublasGemmEx_64
#define cublasGemmStridedBatchedExBit cublasGemmStridedBatchedEx_64

#endif // #if CUBLAS_VER_MAJOR < 12

#if CUDA_VERSION >= 11000

#define cublasGemmExBit_v11 cublasGemmExBit
#define cublasGemmStridedBatchedExBit_v11 cublasGemmStridedBatchedExBit

#else

typedef enum{} cublasComputeType_t;
cublasStatus_t cublasGemmExBit_v11(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
cublasStatus_t cublasGemmStridedBatchedExBit_v11(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

#endif // if CUDA_VERSION >= 11000

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUBLAS_H
