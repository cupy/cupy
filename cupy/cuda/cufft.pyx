cimport cython


cdef extern from "cupy_cufft.h":
    ctypedef struct Complex 'cufftComplex':
        float x, y

    ctypedef struct DoubleComplex 'cufftDoubleComplex':
        double x, y

    # cuFFT Helper Function
    Result cufftDestroy(Handle plan)

    # cuFFT Plan Function
    Result cufftPlan1d(Handle *plan, int nx, Type type, int batch)

    # cuFFT Exec Function
    Result cufftExecC2C(Handle plan, Complex *idata, Complex *odata,
                        int direction)
    Result cufftExecR2C(Handle plan, Float *idata, Complex *odata)
    Result cufftExecC2R(Handle plan, Complex *idata, Float *odata)
    Result cufftExecZ2Z(Handle plan, DoubleComplex *idata,
                        DoubleComplex *odata, int direction)
    Result cufftExecD2Z(Handle plan, Double *idata, DoubleComplex *odata)
    Result cufftExecZ2D(Handle plan, DoubleComplex *idata, Double *odata)


cdef dict RESULT = {
    0: 'CUFFT_SUCCESS',
    1: 'CUFFT_INVALID_PLAN',
    2: 'CUFFT_ALLOC_FAILED',
    3: 'CUFFT_INVALID_TYPE',
    4: 'CUFFT_INVALID_VALUE',
    5: 'CUFFT_INTERNAL_ERROR',
    6: 'CUFFT_EXEC_FAILED',
    7: 'CUFFT_SETUP_FAILED',
    8: 'CUFFT_INVALID_SIZE',
    9: 'CUFFT_UNALIGNED_DATA',
    10: 'CUFFT_INCOMPLETE_PARAMETER_LIST',
    11: 'CUFFT_INVALID_DEVICE',
    12: 'CUFFT_PARSE_ERROR',
    13: 'CUFFT_NO_WORKSPACE',
    14: 'CUFFT_NOT_IMPLEMENTED',
    15: 'CUFFT_LICENSE_ERROR',
    16: 'CUFFT_NOT_SUPPORTED',
}


class CuFftError(RuntimeError):

    def __init__(self, int result):
        self.result = result
        super(CuFftError, self).__init__('%s' % (RESULT[result]))


@cython.profile(False)
cpdef inline check_result(int result):
    if result != 0:
        raise CuFftError(result)


cpdef destroy(size_t plan):
    result = cufftDestroy(plan)
    check_result(result)


cpdef plan1d(int nx, int type, int batch):
    cdef Handle plan
    result = cufftPlan1d(&plan, nx, <Type>type, batch)
    check_result(result)
    return plan


cpdef execC2C(size_t plan, size_t idata, size_t odata, int direction):
    result = cufftExecC2C(plan, <Complex*>idata, <Complex*>odata, direction)
    check_result(result)


cpdef execR2C(size_t plan, size_t idata, size_t odata):
    result = cufftExecR2C(plan, <Float*>idata, <Complex*>odata)
    check_result(result)


cpdef execC2R(size_t plan, size_t idata, size_t odata):
    result = cufftExecC2R(plan, <Complex*>idata, <Float*>odata)
    check_result(result)


cpdef execZ2Z(size_t plan, size_t idata, size_t odata, int direction):
    result = cufftExecZ2Z(plan, <DoubleComplex*>idata, <DoubleComplex*>odata,
                          direction)
    check_result(result)


cpdef execD2Z(size_t plan, size_t idata, size_t odata):
    result = cufftExecD2Z(plan, <Double*>idata, <DoubleComplex*>odata)
    check_result(result)


cpdef execZ2D(size_t plan, size_t idata, size_t odata):
    result = cufftExecZ2D(plan, <DoubleComplex*>idata, <Double*>odata)
    check_result(result)
