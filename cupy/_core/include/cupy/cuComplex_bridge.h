/* 
   This header is to support the "translate_cucomplex" option
   that turns cuComplex function calls to their Thrust counterparts.
*/ 

/* ------------------- single complex ------------------- */
#define cuFloatComplex                complex<float>
#define cuComplex                     complex<float>
#define cuCrealf                      real
#define cuCimagf                      imag
#define make_cuFloatComplex(A, B)     complex<float>(A, B)
#define make_cuComplex(A, B)          complex<float>(A, B)
#define cuConjf                       conj
#define cuCaddf(A, B)                 (A + B)
#define cuCsubf(A, B)                 (A - B)
#define cuCmulf(A, B)                 (A * B)
#define cuCdivf(A, B)                 (A / B)
#define cuCabsf                       abs
#define cuComplexDoubleToFloat        complex<float>
#define cuCfmaf(A, B, C)              (A * B + C)

/* ------------------- double complex ------------------- */
#define cuDoubleComplex               complex<double>
#define cuCreal                       real
#define cuCimag                       imag
#define make_cuDoubleComplex(A, B)    complex<double>(A, B)
#define cuConj                        conj
#define cuCadd(A, B)                  (A + B)
#define cuCsub(A, B)                  (A - B)
#define cuCmul(A, B)                  (A * B)
#define cuCdiv(A, B)                  (A / B)
#define cuCabs                        abs
#define cuComplexFloatToDouble        complex<double>
#define cuCfma(A, B, C)               (A * B + C)
